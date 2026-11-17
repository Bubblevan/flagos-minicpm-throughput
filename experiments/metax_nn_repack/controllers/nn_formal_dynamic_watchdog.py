"""Cleanup only this session's birth-checked groups at cutoff or parent exit."""
import datetime,json,os,signal,time
from pathlib import Path
import psutil
ROOT=Path('/data/tokenforge-artifacts/metax_nn_repack_20261117_2330/formal_dynamic_v2')
END=datetime.datetime.fromisoformat(os.environ['FLAGOS_AUTHORIZED_DEADLINE']).timestamp()
controller=int(os.environ['FLAGOS_FORMAL_CONTROLLER_PID'])
birth=float(os.environ['FLAGOS_FORMAL_CONTROLLER_BIRTH'])
def alive(pid,born):
    try:return psutil.Process(pid).create_time()==born and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.Error:return False
while time.time()<END and alive(controller,birth):time.sleep(2)
if alive(controller,birth):
    psutil.Process(controller).terminate();time.sleep(25)
rows=json.loads((ROOT/'owned-processes.json').read_text()) if (ROOT/'owned-processes.json').exists() else []
owned=[]
for row in rows:
    try:
        p=psutil.Process(row['pid'])
        if p.create_time()!=row['created'] or p.environ().get('FLAGOS_SESSION_ROOT')!=str(ROOT):continue
        owned.append((p.pid,p.create_time()))
        owned.extend((q.pid,q.create_time()) for q in p.children(recursive=True))
        if os.getpgid(p.pid)==p.pid:os.killpg(p.pid,signal.SIGTERM)
        else:p.terminate()
    except (psutil.Error,ProcessLookupError):pass
if owned:time.sleep(20)
for pid,born in set(owned):
    if alive(pid,born):
        try:psutil.Process(pid).kill()
        except psutil.Error:pass
remaining=[pid for pid,born in set(owned) if alive(pid,born)]
if ROOT.exists():(ROOT/'watchdog-final.json').write_text(json.dumps(dict(at=time.time(),cutoff=END,remaining=remaining)))
