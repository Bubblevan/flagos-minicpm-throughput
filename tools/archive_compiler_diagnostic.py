"""Freeze this completed diagnostic only; exclude the still-running model session."""
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    repo=Path(__file__).resolve().parents[1]
    root=repo.parent/'lab-records/outputs/live_compiler_fix_20261116_1818/compiler_fix_20261116_1818'
    assert json.loads((root/'compiler-comparison.json').read_text())['numerical_gate']
    audit=repo/'results/2026-11-16-compiler-repair/independent-tensor-audit.json'
    assert json.loads(audit.read_text())['new_all_reference_pass']
    names=['preflight.json','wheel.json','compiler-comparison.json','native-unified-attention.py',
           'generate-cpu-inputs.log','old-compiler-probe.log','new-compiler-probe.log','install-isolated-wheel.log',
           'download-method-change.json','download-official-attempt1.py','official-wheel-range-download.log']
    files=[(root/n,n) for n in names]
    for sub in ('inputs','old','new','scripts-snapshot'):
        files.extend((p,p.relative_to(root).as_posix()) for p in (root/sub).rglob('*') if p.is_file() and not p.name.endswith('.copying'))
    files += [(audit,'independent-tensor-audit.json'),
              (repo/'results/2026-11-16-accuracy-forensics/scalar-reference-audit.json','scalar-reference-audit.json')]
    hashes={}
    for p,name in files:
        data=p.read_bytes()
        if p.suffix!='.pt':
            assert not re.search(rb'-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|[?&](?:token|access_token|password|auth)=[A-Za-z0-9_%+/-]{12,}',data),p
        hashes[name]=hashlib.sha256(data).hexdigest()
    archives=[]
    for label,predicate in [('inputs',lambda n:n.startswith('inputs/')),('outputs',lambda n:not n.startswith('inputs/'))]:
        output=repo/f'artifacts/FlagOS-Iluvatar-compiler-{label}-20261116.zip'
        assert not output.exists()
        selected=[(p,n) for p,n in files if predicate(n)]
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            for p,name in sorted(selected,key=lambda x:x[1]):z.write(p,name)
            z.writestr('ARCHIVE_MANIFEST.json',json.dumps(dict(complete_diagnostic_requires_both_archives=True,official_workload=False,
                includes_model_weights=False,inputs_are_synthetic=True,files={n:hashes[n] for _,n in selected}),indent=2))
        with zipfile.ZipFile(output) as z:assert z.testzip() is None
        assert output.stat().st_size<95*1024*1024
        archives.append(dict(file=output.name,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size,crc_pass=True,files=len(selected)+1))
    record=dict(archives=archives,credential_pattern_scan_pass=True,model_validation_not_included=True,
                split_reason='Each public GitHub file stays below the size limit; combined full ZIP also retained on D:.')
    (repo/'results/2026-11-16-compiler-repair/archive.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))


if __name__=='__main__':main()
