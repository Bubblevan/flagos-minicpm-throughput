"""Audit one completed saved session with the correct hardware references."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil

from finalize_saved_result import finalize


def select_existing(paths):
    existing=[p for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError('Missing saved evidence: '+', '.join(str(p) for p in paths))
    return existing[0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--platform',choices=['iluvatar','metax'],required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    root=a.root
    prefix='optimized' if a.platform=='iluvatar' else 'metax_baseline'
    saved=json.loads((root/(prefix+'-result.json')).read_text(encoding='utf-8-sig'))
    assert saved['workloads_complete'], 'Controller has not completed all workloads'
    benchmark=json.loads((root/(prefix+'-benchmark-validation.json')).read_text(encoding='utf-8-sig'))
    raw_name=PurePosixPath(benchmark['raw_csv']).name
    summary_name=PurePosixPath(benchmark['summary_csv']).name
    raw=select_existing([root/'benchmark_results'/raw_name,root.parent/'benchmark_results'/raw_name])
    summary=select_existing([root/'benchmark_results'/summary_name,root.parent/'benchmark_results'/summary_name])
    run_name=PurePosixPath(saved['accuracy_report']).parents[2].name
    accuracy=select_existing([root/'level3'/run_name,root.parent/'level3'/run_name])
    environment=root/('optimized-runtime-environment.json' if a.platform=='iluvatar' else 'metax-baseline-environment.json')
    refs=({4096:(2028.01,11573.53,256),16384:(915.15,599262.03,128)}
          if a.platform=='iluvatar' else
          {4096:(5089.645,3199.435,256),16384:(7029.675,27197.135,128)})
    result=finalize(raw,summary,accuracy,environment,refs)
    result.update(platform=a.platform,optimization_implemented=a.platform=='iluvatar',
                  explanation='Independent audit of this controller-selected run only; no model invocation or cross-session substitution.')
    if a.platform=='iluvatar':
        result['candidate_manifest']=json.loads((root/'attention-candidate-manifest.json').read_text())
    else:
        initial=json.loads((root/'initial-environment.json').read_text())
        head=initial['repositories']['FlagGems']['head']
        result['source_git_head']=head
        result['source_git_head_is_not_target_commit']=head!=result['source_commit_equivalent_tree']
    a.out_dir.mkdir(parents=True,exist_ok=True)
    destination=a.out_dir/'independent-result.json'
    assert not destination.exists(), 'Keep independent snapshots immutable'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    shutil.copy2(raw,a.out_dir/'raw.csv')
    shutil.copy2(summary,a.out_dir/'summary.csv')
    shutil.copy2(environment,a.out_dir/'runtime-environment.json')
    print(json.dumps(dict(platform=a.platform,workloads_complete=True,accuracy=result['accuracy'],
                         accuracy_pass=result['accuracy_pass'],performance_pass=result['performance_pass'],
                         optimization_implemented=result['optimization_implemented'])))


if __name__=='__main__':main()
