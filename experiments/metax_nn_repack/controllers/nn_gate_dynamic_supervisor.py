"""One experiment: wait for its current A/B/A, gate one full run, then stop."""
import datetime,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import psutil

ROOT=Path('/data/tokenforge-artifacts/metax_nn_repack_20261117_2330')
BASE=Path('/data/tokenforge-project/nn_repack_20261117_2330')
END=datetime.datetime.fromisoformat('2026-11-18T01:47:00+08:00').timestamp()
SHORT_PID=int(os.environ['FLAGOS_SHORT_PID'])
SHORT_BIRTH=float(os.environ['FLAGOS_SHORT_BIRTH'])

def save(name,data):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def alive(pid,born):
    try:return psutil.Process(pid).create_time()==born and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.Error:return False

def environment():
    env=dict(os.environ)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            if b'jupyter-lab' not in (proc/'cmdline').read_bytes():continue
            for item in (proc/'environ').read_bytes().split(b'\0'):
                if b'=' in item:
                    k,v=item.split(b'=',1)
                    if k.decode() in {'PATH','LD_LIBRARY_PATH','LD_PRELOAD','MACA_HOME','MACA_PATH','CUDA_HOME'}:env[k.decode()]=v.decode()
            break
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    env['PATH']='/opt/conda/bin:/opt/mxdriver/bin:'+env.get('PATH','')
    env.setdefault('MACA_PATH','/opt/maca');env.setdefault('MACA_HOME','/opt/maca')
    env.update(PYTHONDONTWRITEBYTECODE='1',TORCHINDUCTOR_COMPILE_THREADS='1',
               FLAGOS_AUTHORIZED_DEADLINE='2026-11-18T01:47:00+08:00')
    return env

failure=None;p=None
try:
    assert os.uname().nodename=='94b96fcaa1a1'
    while alive(SHORT_PID,SHORT_BIRTH) and time.time()<END:time.sleep(2)
    status=json.loads((ROOT/'short_dynamic_v2/status.json').read_text())
    assert status['stage']=='stopped' and status.get('failure') is None,status
    summary=json.loads((ROOT/'short_dynamic_v2/summary.json').read_text())
    decision=dict(at=time.time(),short_complete=summary['complete'],short_go=summary['short_go'],
                  remaining_seconds=END-time.time(),minimum_full_budget_seconds=75*60,
                  short_summary=summary,cutoff=END)
    save('gate-dynamic-v2-decision.json',decision)
    if not summary['short_go']:raise RuntimeError('Short gate rejected; no full score established')
    assert END-time.time()>=75*60,'Insufficient authorized time for one new complete acceptance'
    for label in ['native_a1','candidate_b','native_a2']:
        assert not json.loads((ROOT/'short_dynamic_v2'/(label+'-cleanup.json')).read_text())['owned_survivors']
    assert 'no process found' in subprocess.check_output(['/opt/mxdriver/bin/mx-smi'],text=True).lower()
    env=environment()
    with (ROOT/'formal-dynamic-v2-controller.log').open('w') as f:
        p=subprocess.Popen(['/opt/conda/bin/python',str(BASE/'nn_formal_dynamic_v2.py')],env=env,cwd=BASE,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    watchenv=dict(env,FLAGOS_FORMAL_CONTROLLER_PID=str(p.pid),
                  FLAGOS_FORMAL_CONTROLLER_BIRTH=str(psutil.Process(p.pid).create_time()))
    with (ROOT/'formal-dynamic-v2-watchdog.log').open('w') as f:
        w=subprocess.Popen(['/opt/conda/bin/python',str(BASE/'nn_formal_dynamic_watchdog.py')],env=watchenv,cwd=BASE,
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save('status.json',dict(stage='formal_acceptance',at=time.time(),controller=p.pid,watchdog=w.pid,cutoff=END))
    try:p.wait(timeout=max(1,END-time.time()+55))
    except subprocess.TimeoutExpired:
        import metax_campaign as common
        common.stop(p)
        raise TimeoutError('Formal process exceeded existing authorized deadline')
    fstatus=json.loads((ROOT/'formal_dynamic_v2/status.json').read_text())
    assert fstatus['stage']=='stopped' and not fstatus.get('owned_survivors'),fstatus
    if fstatus.get('failure'):raise RuntimeError('Formal stopped with recorded failure; preserve partial artifacts')
    saved=json.loads((ROOT/'formal_dynamic_v2/independent-result.json').read_text())
    save('remote-result-needs-local-audit.json',dict(saved_result=saved,local_NN_audit_still_required=True))
except BaseException as e:
    failure=dict(error=repr(e),traceback=traceback.format_exc(),at=time.time())
    save('supervisor-dynamic-v2-failure.json',failure)
finally:
    save('status.json',dict(stage='stopped',at=time.time(),cutoff=END,failure=failure,
                           model_pass_not_claimed_before_local_audit=True))
