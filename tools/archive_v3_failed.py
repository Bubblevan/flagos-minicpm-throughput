"""Publish immutable failed-run evidence without correcting its raw JSON in place."""
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    repo=Path(__file__).resolve().parents[1]
    session=repo.parent/'lab-records/outputs/live_iluvatar_v3_complete_20261109_0000'
    root=session/'remote-roots/iluvatar_v3_complete_20261109_0000'
    archive=repo/'artifacts/FlagOS-Iluvatar-v3-failed-20261109.zip'
    assert not archive.exists()
    source=repo/'experiments/attention_tiling_v3/attention_tiles.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='6e57a2868affd9e3a5319b23300208cd3bb888f78881ea5f96b0a10c8967ee6a'
    files=[(p,'raw/'+p.relative_to(root).as_posix()) for p in root.rglob('*') if p.is_file()]
    files += [(session/'session-supervisor.py','original-session-supervisor.py'),(source,'candidate-attention-tiles-v3.py')]
    files += [(p,'derived-audit/'+p.name) for p in (repo/'results/2026-11-16-accuracy-forensics').iterdir() if p.is_file()]
    manifest={}
    for path,name in files:
        data=path.read_bytes()
        assert not re.search(rb'-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|[?&](?:token|access_token|password|auth)=[A-Za-z0-9_%+/-]{12,}',data),path
        manifest[name]=hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path,name in sorted(files,key=lambda x:x[1]):z.write(path,name)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(dict(raw_overall_pass_is_known_wrong=True,
            corrected_overall_pass=False,official_accuracy=98/105,no_resampling=True,files=manifest),indent=2))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    record=dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),size=archive.stat().st_size,
                files=len(files)+1,crc_pass=True,high_confidence_credential_scan_pass=True,overall_pass=False)
    (repo/'results/2026-11-16-accuracy-forensics/v3-archive.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__=='__main__':main()
