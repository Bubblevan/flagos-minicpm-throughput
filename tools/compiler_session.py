"""Bounded, isolated compiler compatibility experiment. Does not start serving."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

SESSION='compiler_fix_20261116_1818'
ROOT=Path('/workspace/tokenforge-artifacts')/SESSION
SOURCES=Path('/workspace/tokenforge-sources')/SESSION
CACHE=Path('/workspace/tokenforge-caches')/SESSION
END=dt.datetime(2026,11,16,23,55,tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
START=dt.datetime(2026,11,16,18,0,tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
V2_SHA='5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325'
ZIP_SHA='ace45a56569999e010c9882d434e1d516d932790aabbc6497d703c1b23076cfc'
current=None


def write(name,data):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)


def stop(proc):
    if proc is None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)


def run(name,argv,env,seconds):
    global current
    allowed=min(seconds,END-time.time()-60)
    assert allowed>0
    with (ROOT/(name+'.log')).open('w') as f:
        current=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write('status.json',dict(stage=name,child_pid=current.pid,deadline=END,updated=time.time()))
        try:code=current.wait(timeout=allowed)
        except BaseException:stop(current);raise
    current=None
    assert code==0,(name,code)


def main():
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');a=p.parse_args()
    assert a.start and START<=time.time()<END-4*3600
    assert not ROOT.exists(), 'Do not overwrite/repeat a session'
    from tune_attention_tiles import check_exclusive
    check_exclusive()
    ROOT.mkdir(parents=True);SOURCES.mkdir(parents=True);CACHE.mkdir(parents=True)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('stop requested')))
    signal.signal(signal.SIGINT,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('stop requested')))
    base=Path(__file__).resolve().parent
    frozen=base/'FlagOS-Iluvatar-v2-full-source.zip'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==ZIP_SHA
    with zipfile.ZipFile(frozen) as z:
        assert z.testzip() is None
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(SOURCES)
    plugin=SOURCES/'vllm-plugin-FL'
    candidate=plugin/'vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==V2_SHA
    gems=Path('/workspace/tokenforge-sources/FlagGems-a7620cc')
    assert subprocess.check_output(['git','-C',str(gems),'write-tree'],text=True).strip()=='ddf3c7566aac7d23385a1b10c7c258091ef1d3a7'
    subprocess.run(['git','-C',str(gems),'diff','--quiet'],check=True)
    subprocess.run(['git','-C','/workspace/vllm-plugin-FL','diff','--quiet'],check=True)
    assert subprocess.check_output(['git','-C','/workspace/vllm-plugin-FL','rev-parse','HEAD'],text=True).strip()=='13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0'
    env=dict(os.environ,PYTHONPATH=str(plugin)+':'+str(gems/'src')+':'+str(base),
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',
             TRITON_CACHE_DIR=str(CACHE/'old/triton'),VLLM_CACHE_ROOT=str(CACHE/'old/vllm'),
             TORCHINDUCTOR_CACHE_DIR=str(CACHE/'old/inductor'),CUDA_CACHE_PATH=str(CACHE/'old/cuda'))
    # OMP cap only applies to CPU reference generation, not to any serving environment.
    preflight=dict(started=time.time(),deadline=END,flagtree=md.version('flagtree'),
                   packages={d.metadata['Name']:d.version for d in md.distributions()},
                   candidate_sha256=V2_SHA,source_zip_sha256=ZIP_SHA,
                   original_install_untouched=True,plugin=str(plugin),gems=str(gems),
                   ps=subprocess.check_output(['ps','-eo','pid,ppid,stat,etime,args'],text=True),
                   gpu=subprocess.check_output(['ixsmi'],text=True),
                   script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob('*.py')})
    write('preflight.json',preflight)
    probe=base/'compiler_attention_probe.py'
    inputs=ROOT/'inputs'
    diag_end=min(END,time.time()+2400)
    run('generate-cpu-inputs',[sys.executable,str(probe),'--inputs',str(inputs),'--generate','--deadline',str(diag_end)],env,300)
    run('old-compiler-probe',[sys.executable,str(probe),'--inputs',str(inputs),'--out',str(ROOT/'old'),
        '--candidate',str(candidate),'--deadline',str(diag_end)],env,1000)
    wheels=SOURCES/'wheels';wheels.mkdir()
    run('download-official-wheel',[sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:',
        '--dest',str(wheels),'--index-url','https://resource.flagos.net/repository/flagos-pypi-hosted/simple',
        'flagtree===0.7.0+iluvatar3.6'],env,600)
    wheel=list(wheels.glob('*.whl'));assert len(wheel)==1
    target=SOURCES/'compiler-0.7.0'
    write('wheel.json',dict(file=wheel[0].name,sha256=hashlib.sha256(wheel[0].read_bytes()).hexdigest(),size=wheel[0].stat().st_size))
    run('install-isolated-wheel',[sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(target),str(wheel[0])],env,300)
    newer=dict(env,PYTHONPATH=str(target)+':'+env['PYTHONPATH'],TRITON_CACHE_DIR=str(CACHE/'new/triton'),
               VLLM_CACHE_ROOT=str(CACHE/'new/vllm'),TORCHINDUCTOR_CACHE_DIR=str(CACHE/'new/inductor'),CUDA_CACHE_PATH=str(CACHE/'new/cuda'))
    run('new-compiler-probe',[sys.executable,str(probe),'--inputs',str(inputs),'--out',str(ROOT/'new'),
        '--candidate',str(candidate),'--deadline',str(diag_end)],newer,1000)
    old=json.loads((ROOT/'old/result.json').read_text());new=json.loads((ROOT/'new/result.json').read_text())
    assert new['flagtree']=='0.7.0+iluvatar3.6' and new['triton_file'].startswith(str(target)+'/')
    summary=dict(complete=True,old_compiler=old['flagtree'],new_compiler=new['flagtree'],
                 old_reference_pass=old['all_reference_pass'],new_reference_pass=new['all_reference_pass'],
                 old_stable=old['all_stable'],new_stable=new['all_stable'],
                 new_v2_bitwise_native=new['all_v2_bitwise_native'],
                 numerical_gate=bool(new['all_reference_pass'] and new['all_stable'] and new['all_v2_bitwise_native']),
                 formal_started=False,official_score=False,
                 measured_fix=bool(not old['all_reference_pass'] and new['all_reference_pass']))
    write('compiler-comparison.json',summary);write('status.json',dict(stage='diagnostic_complete',updated=time.time(),**summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        stop(current)
        if ROOT.exists():write('status.json',dict(stage='diagnostic_failed',error=type(e).__name__,message=str(e),updated=time.time()))
        raise
