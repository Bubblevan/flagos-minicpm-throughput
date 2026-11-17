"""Independent bounded cleanup of only this trial's recorded processes."""
import datetime,json,os,signal,sys,time
from pathlib import Path
import psutil
ROOT=Path('/data/tokenforge-artifacts/metax_nn_repack_20261117_2330/short_dynamic_v2')
END=datetime.datetime.fromisoformat('2026-11-18T01:47:00+08:00').timestamp()
pid=int(sys.argv[1]);born=float(sys.argv[2])
def alive(pid,born):
    try:return psutil.Process(pid).create_time()==born and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.Error:return False
while time.time()<END and alive(pid,born):time.sleep(2)
if alive(pid,born):
    psutil.Process(pid).terminate();time.sleep(25)
try:status=json.loads((ROOT/'status.json').read_text())
except (FileNotFoundError,json.JSONDecodeError):status={}
server=status.get('server_pid');sb=status.get('server_birth')
if server and sb and alive(server,sb):
    p=psutil.Process(server)
    if p.environ().get('FLAGOS_SESSION_ROOT')==str(ROOT) and 'vllm' in ' '.join(p.cmdline()):
        descendants=[(q.pid,q.create_time()) for q in p.children(recursive=True)]
        os.killpg(server,signal.SIGTERM);time.sleep(20)
        if alive(server,sb):os.killpg(server,signal.SIGKILL)
        for child,cb in descendants:
            if alive(child,cb):psutil.Process(child).kill()
if ROOT.exists():(ROOT/'watchdog.json').write_text(json.dumps(dict(at=time.time(),cutoff=END,checked_parent=pid)))
