"""Independently finalize completed saved workloads; never invokes a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from validate_accuracy_artifacts import audit


def finalize(raw, summary, accuracy_dir, environment):
    rows = list(csv.DictReader(raw.open(encoding='utf-8-sig')))
    summaries = list(csv.DictReader(summary.open(encoding='utf-8-sig')))
    assert len(rows) == 8 and len(summaries) == 2, 'All eight runs are required'
    references = {4096: (2028.01, 11573.53, 256), 16384: (915.15, 599262.03, 128)}
    cases = []
    for length, (ref_tps, ref_ttft, count) in references.items():
        group = [r for r in rows if int(r['Prefill']) == length]
        assert len(group) == 4
        for r in group:
            assert r['Run Status'] == 'SUCCESS' and float(r['Successful Requests']) == count
            assert int(r['Decode']) == 1024 and int(r['Conc']) == 64
            assert int(r['Num Prompts']) == count
            assert float(r['Total Input Tokens']) == count*length
            assert float(r['Total Output Tokens']) == count*1024
        tps = sum(float(r['Total tok/s']) for r in group[1:])/3
        ttft = sum(float(r['Mean TTFT (ms)']) for r in group[1:])/3
        official = [r for r in summaries if int(r['Prefill']) == length]
        assert len(official) == 1
        assert abs(tps-float(official[0]['Total tok/s'])) <= 0.015
        assert abs(ttft-float(official[0]['Mean TTFT (ms)'])) <= 0.015
        cases.append(dict(input_len=length, rounds=4, successful_requests_each=count,
                          discard_first=True, total_tokens_per_second=tps, mean_ttft_ms=ttft,
                          throughput_reference=ref_tps, ttft_reference=ref_ttft,
                          throughput_pass=tps >= ref_tps*0.99, ttft_pass=ttft <= ref_ttft*1.01))
    precision = audit(accuracy_dir)
    assert precision['integrity_pass'], precision
    pinned = json.loads(environment.read_text(encoding='utf-8-sig'))
    return dict(cases=cases, accuracy=precision['score_recomputed'],
                accuracy_reported=precision['score_reported'], accuracy_audit=precision,
                accuracy_pass=precision['accuracy_pass'],
                performance_pass=all(c['throughput_pass'] and c['ttft_pass'] for c in cases),
                source_tree=pinned['target_tree'], source_commit_equivalent_tree=pinned['target_commit'],
                source_git_head_is_not_target_commit=True, optimization_implemented=False,
                raw_csv=str(raw), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                original_summary_csv=str(summary), independent_validation=True,
                workloads_complete=True, explanation='Saved evidence audit uses EvalScope indices 0..104; '
                'it corrects the controller result-check assertion without rerunning or modifying evaluation.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--accuracy-run-dir', type=Path, required=True)
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = finalize(a.raw, a.summary, a.accuracy_run_dir, a.environment)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'accuracy_audit'}, ensure_ascii=False))
