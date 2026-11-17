"""Independent complete-session audit; extends, never relaxes the GEMM gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
from audit_metax_mm_session import run as base_audit


def run(root,source_zip,performance_only=False):
    result=base_audit(root,source_zip,performance_only)
    full=json.loads((root/'full-source-manifest.json').read_text(encoding='utf-8'))
    proof=json.loads((root/'preflight-nn-repack.json').read_text(encoding='utf-8'))
    assert proof['complete'] and len(proof['cases'])==30
    assert Counter(c['path'] for c in proof['cases'])=={
        'existing_NT_component':18,'packed_public_wrapper':6,'original_dispatch_fallback':6}
    assert all(c['bitwise_equal'] for c in proof['cases'])
    assert all(c['cpu_reference_pass'] for c in proof['cases'] if c['path']!='original_dispatch_fallback')
    assert all(c['reference_is_original_dispatch'] for c in proof['cases'] if c['path']=='original_dispatch_fallback')
    assert proof['fake_and_compile_pass'] and proof['cudagraph_pass']
    key='FlagGems/src/flag_gems/runtime/backend/_metax/ops/packed_mm.py'
    assert proof['packed_sha256']==full['files'][key]
    records=list(root.glob('nn-repack-runtime-worker-*.json'))
    assert records,'Require actual model worker fast-path evidence'
    actual=[]
    for p in records:
        d=json.loads(p.read_text(encoding='utf-8'))
        assert d['sha256']==full['files'][key]
        assert d['packed_layers']>0 and d['packed_bytes']>0
        assert d['shape']==[2048,2048]
        assert d['packed_shape'] in ([2048,2560],[2048,12288])
        assert d['records_first_attempt_not_call_coverage']
        actual.append(d)
    result.update(nn_repack_numerical_gate_pass=True,nn_repack_model_records=actual,
                  borrowed_layout_idea_attributed=True,finite_tests_not_universal_proof=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--source-zip',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--performance-only',action='store_true')
    a=ap.parse_args();assert not a.out.exists(),'Keep earlier audits immutable'
    result=run(a.root,a.source_zip,a.performance_only)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result.get('overall_pass',result['performance_pass']) else 2)
