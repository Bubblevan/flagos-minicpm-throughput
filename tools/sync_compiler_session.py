"""Bounded incremental SFTP backup of this single authorized experiment ROOT."""
import datetime as dt
import json
import os
from pathlib import Path
import stat
import time
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
import paramiko

BASE=Path('D:/MyLab/FlagOS/lab-records/outputs/compiler_fix_20261116_1818')
DEST=Path('D:/MyLab/FlagOS/lab-records/outputs/live_compiler_fix_20261116_1818')
REMOTE='/workspace/tokenforge-artifacts/compiler_fix_20261116_1818'
END=dt.datetime(2026,11,16,23,58,tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()


def main():
    BASE.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    (BASE/'sync.pid').write_text(str(os.getpid()))
    record={}
    while time.time()<END and not (BASE/'stop-sync').exists():
        client=paramiko.SSHClient()
        record.update(last_attempt=time.time(),backup_success=False)
        try:
            client.load_host_keys('D:/MyLab/FlagOS/private/ssh/known_hosts')
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect('ssh.flagos.io',port=2224,username='lab_inst_01M60ANBE6KZQNS943VQF8QFFC',
                key_filename='D:/MyLab/FlagOS/private/ssh/flagos_20261029',
                look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=20,banner_timeout=20)
            sftp=client.open_sftp();sftp.get_channel().settimeout(90)
            copied=[];seen=[]
            def visit(remote,local):
                local.mkdir(parents=True,exist_ok=True)
                for item in sftp.listdir_attr(remote):
                    assert item.filename not in ('.','..') and '/' not in item.filename and '\\' not in item.filename
                    src=remote+'/'+item.filename;dst=local/item.filename
                    if stat.S_ISDIR(item.st_mode):visit(src,dst)
                    elif stat.S_ISREG(item.st_mode):
                        seen.append(str(dst.relative_to(DEST)))
                        changed=(not dst.exists() or dst.stat().st_size!=item.st_size
                                 or int(dst.stat().st_mtime)!=item.st_mtime
                                 or item.filename in ('status.json','compiler-comparison.json','independent-result.json'))
                        if changed:
                            partial=dst.with_name(dst.name+'.copying')
                            sftp.get(src,str(partial));os.replace(partial,dst)
                            os.utime(dst,(item.st_mtime,item.st_mtime));copied.append(str(dst.relative_to(DEST)))
                    else:raise RuntimeError('Unexpected non-regular evidence file: '+item.filename)
            visit(REMOTE,DEST/Path(REMOTE).name)
            sftp.close()
            record.update(backup_success=True,last_success=time.time(),copy_exit_code=0,
                          files_seen=len(seen),files_copied=len(copied),error=None)
        except Exception as e:
            record.update(copy_exit_code=1,error=type(e).__name__+': '+str(e))
        finally:client.close()
        tmp=BASE/'local-sync-status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(BASE/'local-sync-status.json')
        print(json.dumps(record),flush=True)
        time.sleep(60)


if __name__=='__main__':main()
