"""Archive the independently audited completed model session, without rerunning."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


def main():
    repo=Path(__file__).resolve().parents[1]
    parent=repo.parent/'lab-records/outputs/live_compiler_fix_20261116_1818/compiler_fix_20261116_1818'
    root=parent/'formal-v2-newcompiler';out=repo/'results/2026-11-16-v2-flagtree07-complete'
    result=json.loads((out/'independent-result.json').read_text())
    assert result['workloads_complete'] and result['overall_pass'] and result['accuracy_audit']['correct']==101
    hashes=json.loads((out/'remote-local-hashes.json').read_text())
    assert hashes['all_match']
    for name,digest in hashes['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert json.loads((root/'status.json').read_text())['stage']=='stopped'
    watchdog=json.loads((parent/'cutoff-watchdog.json').read_text());assert not watchdog['active_owned_survivors']
    archive=repo/'artifacts/FlagOS-Iluvatar-v2-flagtree070-full-20261116.zip';assert not archive.exists()
    files=[(p,'raw/'+p.relative_to(root).as_posix()) for p in root.rglob('*') if p.is_file()]
    files += [(p,'independent-audit/'+p.name) for p in out.iterdir() if p.is_file()]
    for name in ('preflight.json','wheel.json','compiler-comparison.json','formal-launcher.json','formal-controller.log','cutoff-watchdog.json'):
        files.append((parent/name,'session/'+name))
    files += [(p,'scripts/'+p.name) for p in (parent/'scripts-snapshot').iterdir() if p.is_file()]
    manifest={}
    for path,name in files:
        data=path.read_bytes()
        assert not re.search(rb'-----BEGIN (?:(?:OPENSSH|RSA|EC) )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|[?&](?:token|access_token|password|auth)=[A-Za-z0-9_%+/-]{12,}',data),path
        manifest[name]=hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path,name in sorted(files,key=lambda x:x[1]):z.write(path,name)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(dict(overall_pass=True,accuracy=101/105,official_submission_accepted=False,files=manifest),indent=2))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    record=dict(file=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                files=len(files)+1,crc_pass=True,high_confidence_credential_scan_pass=True,raw_hashes_rechecked=True)
    (out/'archive.json').write_text(json.dumps(record,indent=2))
    shutil.copy2(next((root/'official-benchmark/benchmark_results').glob('raw_runs_*.csv')),out/'raw.csv')
    shutil.copy2(next((root/'official-benchmark/benchmark_results').glob('summary_*.csv')),out/'summary.csv')
    print(json.dumps(record))


if __name__=='__main__':main()
