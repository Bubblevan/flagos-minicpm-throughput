"""Independent, one-session cutoff safety net; no chat automation or GPU work."""
import json
import os
from pathlib import Path
import signal
import time
import psutil
from compiler_session import ROOT as DIAG, END

FORMAL=DIAG/'formal-v2-newcompiler'


def owned():
    found=[]
    for p in psutil.process_iter(['pid','status']):
        if p.pid==os.getpid() or p.info['status']==psutil.STATUS_ZOMBIE:continue
        try:
            # Inspect only this non-secret ownership marker; never dump full environments.
            if p.environ().get('FLAGOS_SESSION_ROOT')==str(FORMAL):found.append(p)
        except psutil.Error:pass
    return found


def save(value):
    p=DIAG/'cutoff-watchdog.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)


def main():
    pid=json.loads((DIAG/'formal-launcher.json').read_text())['controller_pid']
    try:
        controller=psutil.Process(pid);birth=controller.create_time()
        assert any('formal_compiler_validation.py' in s for s in controller.cmdline())
    except psutil.NoSuchProcess:controller=None;birth=None
    save(dict(stage='armed',controller_pid=pid,term_at=END-60,hard_cutoff=END))
    while time.time()<END-60:
        try:stage=json.loads((FORMAL/'status.json').read_text())['stage']
        except (OSError,ValueError,KeyError):stage='unknown'
        if stage=='stopped':break
        time.sleep(15)
    if controller is not None:
        try:
            if controller.create_time()==birth and controller.status()!=psutil.STATUS_ZOMBIE:controller.terminate()
        except psutil.Error:pass
    pids=[]
    # If normal controller cleanup completed, no active model matches this exact marker.
    for p in owned():
        try:pids.append(p.pid);p.terminate()
        except psutil.Error:pass
    until=min(END-10,time.time()+25)
    while time.time()<until:
        if not owned():break
        time.sleep(.5)
    for p in owned():
        try:p.kill()
        except psutil.Error:pass
    time.sleep(.3)
    save(dict(stage='finished',time=time.time(),terminated_owned_pids=pids,
              active_owned_survivors=[p.pid for p in owned()],hard_cutoff=END))


if __name__=='__main__':main()
