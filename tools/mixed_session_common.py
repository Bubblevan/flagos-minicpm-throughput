"""Per-session evidence/state/telemetry for one winning candidate; no chat timer."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

ROOT=Path(os.environ['FLAGOS_SESSION_ROOT'])
URL='http://127.0.0.1:9031'
DEADLINE=float(os.environ['FLAGOS_SESSION_DEADLINE'])
PARENT=Path(os.environ['FLAGOS_PARENT_SESSION_ROOT'])
STARTED=time.time()
INITIAL_ACCURACY=set(Path('/workspace/evalscope-datasets/level3').glob('*'))


def atomic(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def state(stage,**kwargs):
    row=dict(stage=stage,at=dt.datetime.now(dt.timezone.utc).isoformat(),**kwargs)
    atomic(ROOT/'status.json',row)
    atomic(PARENT/'status.json',dict(row,formal_root=str(ROOT)))
    print(json.dumps(row),flush=True)


def mirror():
    dest=ROOT/'benchmark_results';dest.mkdir(exist_ok=True)
    for path in Path('/workspace/benchmark_results').glob('*.csv'):
        if path.stat().st_mtime>=STARTED:shutil.copy2(path,dest/path.name)
    for path in set(Path('/workspace/evalscope-datasets/level3').glob('*'))-INITIAL_ACCURACY:
        if path.is_dir():
            shutil.copytree(path,ROOT/'level3'/path.name,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('.rerun-*','__pycache__'))
    atomic(ROOT/'remote-mirror-status.json',dict(at=time.time(),success=True))


def textfile(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def telemetry(pid,stop):
    last_mirror=0
    with (ROOT/'telemetry.jsonl').open('a') as output:
        while not stop.is_set():
            row=dict(at=time.time(),cpu_stat=textfile('/sys/fs/cgroup/cpu/cpu.stat'),
                     memory_usage=textfile('/sys/fs/cgroup/memory/memory.usage_in_bytes'),processes=[])
            try:
                parent=psutil.Process(pid)
                for process in [parent]+parent.children(recursive=True):
                    try:
                        cpu=process.cpu_times()
                        row['processes'].append(dict(pid=process.pid,name=process.name(),rss=process.memory_info().rss,
                                                     cpu_user=cpu.user,cpu_system=cpu.system))
                    except psutil.Error:pass
                row['gpu']=subprocess.check_output(['ixsmi','-q','-d','CLOCK,PERFORMANCE,POWER,UTILIZATION'],text=True,timeout=5)
                if time.monotonic()-last_mirror>=30:
                    mirror();last_mirror=time.monotonic()
            except Exception as error:row['telemetry_error']=repr(error)
            output.write(json.dumps(row)+'\n');output.flush();stop.wait(5)
    try:mirror()
    except Exception as error:atomic(ROOT/'final-mirror-error.json',dict(error=repr(error)))
