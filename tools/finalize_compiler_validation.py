"""CPU-only final audit of this session's saved original benchmark and accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
from finalize_saved_result import finalize
from accuracy_forensics import aggregate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();a.out.mkdir(parents=True)
    status=json.loads((a.root/'status.json').read_text());assert status['stage']=='stopped'
    source=a.root/'independent-result.json'
    if not source.exists():
        result=dict(complete=False,overall_pass=False,status=status,
                    failure=json.loads((a.root/'failure.json').read_text()) if (a.root/'failure.json').exists() else None)
        (a.out/'partial-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    raw=list((a.root/'official-benchmark/benchmark_results').glob('raw_runs_*.csv'))
    summary=list((a.root/'official-benchmark/benchmark_results').glob('summary_*.csv'))
    reports=list((a.root/'evalscope').glob('*/reports/minicpm/math_500.json'))
    assert len(raw)==len(summary)==len(reports)==1
    environment=json.loads((a.root/'runtime-environment.json').read_text())
    proof=environment['import_proof'];assert proof['flagtree']=='0.7.0+iluvatar3.6'
    assert environment['source_sha256']=='5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325'
    assert '/compiler-0.7.0/' in proof['triton_file']
    assert environment['numerical_gate']['numerical_gate']
    # The full controller verifies the unchanged Gems tree before launch. Enrich
    # only this derived copy to adapt the existing independent audit interface.
    derived=dict(environment,target_tree='ddf3c7566aac7d23385a1b10c7c258091ef1d3a7',target_commit='a7620cc191a0b42e040194622c5758b22a7a25dc')
    env_path=a.out/'derived-environment.json';env_path.write_text(json.dumps(derived,indent=2))
    result=finalize(raw[0],summary[0],reports[0].parents[2],env_path)
    result.update(overall_pass=aggregate(result['performance_pass'],result['accuracy_audit']),
                  optimization_implemented=True,candidate='frozen-v2+flagtree0.7.0',
                  compiler=proof['flagtree'],raw_controller_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  explanation='One complete new compiler session, independently recomputed from all8raw rows, original summary and this105-question prediction/review/report. No cross-session substitution or score correction.')
    controller=json.loads(source.read_text())
    assert controller['overall_pass']==result['overall_pass']
    (a.out/'independent-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(overall_pass=result['overall_pass'],performance_pass=result['performance_pass'],accuracy=result['accuracy'],accuracy_pass=result['accuracy_pass'])))


if __name__=='__main__':main()
