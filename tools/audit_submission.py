"""CPU-only audit of an extracted TokenForge final package. Never runs a model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from build_iluvatar_candidate import ADAPTER, ADAPTER_SHA, contract_smoke
from finalize_saved_result import finalize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package-root',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=a.package_root.resolve()
    assert not a.out.exists(), 'Use a new output; do not overwrite an audit'
    a.work_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'REVIEW_MANIFEST.json').read_text())
    actual={f.relative_to(root).as_posix():digest(f) for f in root.rglob('*')
            if f.is_file() and f.name!='REVIEW_MANIFEST.json'}
    assert actual==manifest['files'], 'Package file set or hash differs'
    source=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    plugin=root/'vllm-plugin-FL'
    plugin_files={f.relative_to(plugin).as_posix():digest(f) for f in plugin.rglob('*') if f.is_file()}
    assert plugin_files==source['files'], 'Source differs from the validated frozen candidate'
    assert plugin_files[ADAPTER]==ADAPTER_SHA
    assert source['base_commit']=='13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0'
    assert set(source['changed_files'])=={'vllm_fl/platform.py',ADAPTER}
    syntax=0
    for name in plugin_files:
        if name.endswith('.py'):
            ast.parse((plugin/name).read_bytes(),filename=name); syntax+=1
    smoke=contract_smoke((plugin/ADAPTER).read_text())
    raw_archive=root/'evidence/FlagOS-Iluvatar-v2-flagtree070-full-20261116.zip'
    with tempfile.TemporaryDirectory(dir=a.work_dir,prefix='submission-audit-') as tmp:
        target=Path(tmp)
        with zipfile.ZipFile(raw_archive) as z:
            assert z.testzip() is None
            for n in z.namelist():
                path=PurePosixPath(n)
                assert not path.is_absolute() and '..' not in path.parts
            z.extractall(target)
        saved=target/'raw'
        raw=saved/'official-benchmark/benchmark_results/raw_runs_20261116_185311.csv'
        summary=saved/'official-benchmark/benchmark_results/summary_20261116_185311.csv'
        env=json.loads((saved/'runtime-environment.json').read_text())
        assert env['source_sha256']==ADAPTER_SHA
        assert env['import_proof']['flagtree']=='0.7.0+iluvatar3.6'
        assert env['numerical_gate']['numerical_gate']
        assert env['argv']==['vllm','serve','/workspace/MiniCPM5-2B','--port','9031',
            '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
            '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
        result=finalize(raw,summary,saved/'evalscope/20261116_205443',target/'independent-audit/derived-environment.json')
        overall=result['workloads_complete'] and result['performance_pass'] and result['accuracy_audit']['integrity_pass'] and result['accuracy_pass']
        assert overall and result['accuracy_audit']['correct']==101
        recorded=json.loads((root/'evidence/v2-flagtree070-complete/independent-result.json').read_text())
        assert result['cases']==recorded['cases'] and result['raw_sha256']==recorded['raw_sha256']
    output=dict(package_integrity_pass=True,source_files=len(plugin_files),python_syntax_files=syntax,
                contract_smoke=smoke,platform='iluvatar',metax_optimization_passed=False,
                cases=result['cases'],accuracy=result['accuracy'],correct=101,total=105,
                finish_reasons=result['accuracy_audit']['finish_reasons'],overall_pass=bool(overall),
                source_scope='Frozen v2 only; no MetaX experimental source',
                official_acceptance_verified=False,competition_submitted=False)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__=='__main__':
    main()
