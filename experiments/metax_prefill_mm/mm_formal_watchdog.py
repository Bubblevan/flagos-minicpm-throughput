"""Independent deadline enforcement limited to one formal session's owned groups."""
import datetime,json,os,signal,time
from pathlib import Path
import psutil
ROOT=Path('/data/tokenforge-artifacts/metax_mm_formal_20261117')
END=datetime.datetime.fromisoformat(os.environ['FLAGOS_AUTHORIZED_DEADLINE']).timestamp()
controller=int(os.environ['FLAGOS_FORMAL_CONTROLLER_PID'])
def alive(pid):
    try:return psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.Error:return False
while time.time()<END and alive(controller):time.sleep(2)
if time.time()>=END:
    try:
        if 'mm_formal.py' in ' '.join(psutil.Process(controller).cmdline()):os.kill(controller,signal.SIGTERM)
    except psutil.Error:pass
    time.sleep(25)
    rows=json.loads((ROOT/'owned-processes.json').read_text()) if (ROOT/'owned-processes.json').exists() else []
    owned=[]
    for row in rows:
        try:
            p=psutil.Process(row['pid'])
            if p.create_time()==row['created'] and p.environ().get('FLAGOS_SESSION_ROOT')==str(ROOT):
                owned.append((p.pid,p.create_time()));os.killpg(p.pid,signal.SIGTERM)
        except (psutil.Error,ProcessLookupError):pass
    time.sleep(20)
    for pid,born in owned:
        try:
            if psutil.Process(pid).create_time()==born:os.killpg(pid,signal.SIGKILL)
        except (psutil.Error,ProcessLookupError):pass
    remaining=[pid for pid,_ in owned if alive(pid)]
    if ROOT.exists():(ROOT/'watchdog-final.json').write_text(json.dumps(dict(checked=time.time(),cutoff=END,remaining=remaining)))
