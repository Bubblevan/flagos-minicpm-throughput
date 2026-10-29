"""Verify the immutable day archive and recompute results without a GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from finalize_saved_result import finalize


def main():
    project = Path(__file__).resolve().parents[1]
    artifact_dir = project / 'artifacts'
    expected, name = (artifact_dir / 'SHA256SUMS').read_text().strip().split()
    archive = artifact_dir / name
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Archive SHA256 does not match the preserved snapshot')
    # Keep temporary extraction in the writable project on restricted hosts.
    with tempfile.TemporaryDirectory(prefix='.audit-', dir=project) as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive) as data:
            for item in data.infolist():
                resolved = (root / item.filename).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError('Unsafe archive member')
            if data.testzip() is not None:
                raise ValueError('Archive CRC check failed')
            data.extractall(root)
        day = root / 'outputs/repro_20261029_day'
        live = root / 'outputs/live_repro_20261029_day'
        result = finalize(
            day / 'formal-raw-snapshot.csv',
            day / 'formal-summary-snapshot.csv',
            live / 'level3/20261029_165624',
            live / 'repro_20261029_1218/aligned-runtime-environment.json',
        )
    print(json.dumps(dict(
        archive_sha256=actual,
        workloads_complete=result['workloads_complete'],
        accuracy=result['accuracy'],
        accuracy_pass=result['accuracy_pass'],
        performance_pass=result['performance_pass'],
        cases=result['cases'],
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
