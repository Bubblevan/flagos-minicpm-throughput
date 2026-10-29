"""One reserved MetaX baseline session, with bounded workloads and mirroring."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import threading
import time

ROOT=Path(os.environ['FLAGOS_SESSION_ROOT'])
URL='http://127.0.0.1:9031'
DEADLINE=float(os.environ['FLAGOS_SESSION_DEADLINE'])
STARTED=time.time()
INITIAL_ACCURACY=set(Path('/workspace/evalscope-datasets/level3').glob('*'))


def state(stage,**details):
    value=dict(stage=stage,at=dt.datetime.now(dt.timezone.utc).isoformat(),**details)
    temporary=ROOT/'status.tmp';temporary.write_text(json.dumps(value,indent=2));temporary.replace(ROOT/'status.json')
    print(json.dumps(value),flush=True)


def mirror():
    target=ROOT/'benchmark_results';target.mkdir(exist_ok=True)
    for p in Path('/workspace/benchmark_results').glob('*.csv'):
        if p.stat().st_mtime>=STARTED:
            shutil.copy2(p,target/p.name)
    for directory in set(Path('/workspace/evalscope-datasets/level3').glob('*'))-INITIAL_ACCURACY:
        if directory.is_dir():
            shutil.copytree(directory,ROOT/'level3'/directory.name,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('.rerun-*','__pycache__'))
    (ROOT/'remote-mirror-status.json').write_text(json.dumps(dict(at=time.time(),success=True)))


def telemetry(pid,stop):
    import psutil
    while not stop.is_set():
        try:
            parent=psutil.Process(pid)
            processes=[dict(pid=p.pid,name=p.name(),rss=p.memory_info().rss) for p in [parent]+parent.children(recursive=True) if p.is_running()]
            with (ROOT/'telemetry.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=time.time(),processes=processes))+'\n')
            mirror()
        except Exception as error:
            (ROOT/'remote-mirror-error.json').write_text(json.dumps(dict(at=time.time(),error=repr(error))))
        if stop.wait(30):break
    mirror()
