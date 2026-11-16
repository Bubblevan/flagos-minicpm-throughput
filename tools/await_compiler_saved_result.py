"""One bounded local completion step in this experiment's backup pipeline.

No chat/task automation, messages, GPU work, Git push or provider actions.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import paramiko

BASE=Path('D:/MyLab/FlagOS/lab-records/outputs/compiler_fix_20261116_1818')
SAVED=Path('D:/MyLab/FlagOS/lab-records/outputs/live_compiler_fix_20261116_1818/compiler_fix_20261116_1818')
REPO=Path('D:/MyLab/FlagOS/flagos-minicpm-throughput')
REMOTE='/workspace/tokenforge-artifacts/compiler_fix_20261116_1818/formal-v2-newcompiler'
OUT=REPO/'results/2026-11-16-v2-flagtree07-complete'
END=dt.datetime(2026,11,16,23,59,tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()


def save(record):
    dest=BASE/'completion-status.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(dest)


def main():
    (BASE/'completion.pid').write_text(str(os.getpid()))
    save(dict(stage='awaiting_saved_terminal_state',started=time.time(),deadline=END))
    while time.time()<END:
        try:
            status_path=SAVED/'formal-v2-newcompiler/status.json'
            status=json.loads(status_path.read_text())
            synced=json.loads((BASE/'local-sync-status.json').read_text())
            if status['stage']!='stopped' or not synced['backup_success'] or synced['last_success']<status['updated']:
                time.sleep(20);continue
            # Check all final formal files against remote hashes before finalizing.
            client=paramiko.SSHClient();client.load_host_keys('D:/MyLab/FlagOS/private/ssh/known_hosts');client.set_missing_host_key_policy(paramiko.RejectPolicy())
            try:
                client.connect('ssh.flagos.io',port=2224,username='lab_inst_01M60ANBE6KZQNS943VQF8QFFC',key_filename='D:/MyLab/FlagOS/private/ssh/flagos_20261029',look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=20,banner_timeout=20)
                code="import json,hashlib;from pathlib import Path;r=Path('"+REMOTE+"');print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}))"
                stdin,stdout,stderr=client.exec_command('python3 -',timeout=60);stdin.write(code+'\n');stdin.channel.shutdown_write()
                content=stdout.read();err=stderr.read();assert stdout.channel.recv_exit_status()==0,err.decode()[:1000]
                hashes=json.loads(content)
            finally:client.close()
            assert hashes
            for rel,digest in hashes.items():
                path=SAVED/'formal-v2-newcompiler'/rel
                assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest,rel
            if OUT.exists():raise FileExistsError('Do not overwrite prior completion audit')
            command=[sys.executable,str(REPO/'tools/finalize_compiler_validation.py'),'--root',str(SAVED/'formal-v2-newcompiler'),'--out',str(OUT)]
            finished=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=180)
            (BASE/'completion-audit.log').write_text(finished.stdout+finished.stderr,encoding='utf-8')
            assert finished.returncode==0,finished.stderr[-1200:]
            (OUT/'remote-local-hashes.json').write_text(json.dumps(dict(all_match=True,files=hashes,verified=time.time()),indent=2))
            save(dict(stage='saved_result_audited',completed=time.time(),remote_local_hashes_match=True,output=str(OUT)))
            return
        except Exception as e:
            save(dict(stage='waiting_or_audit_error',at=time.time(),error=type(e).__name__+': '+str(e)))
            if OUT.exists():return  # Preserve a partial/failed audit for human review.
            time.sleep(30)
    save(dict(stage='deadline_reached',at=time.time(),complete=False,reason='No complete hash-verified saved terminal result was audited within this pipeline window.'))


if __name__=='__main__':main()
