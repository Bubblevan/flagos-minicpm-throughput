"""Audit saved EvalScope artifacts without contacting the model or changing data."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def audit(run_dir):
    report = json.loads((run_dir/'reports/minicpm/math_500.json').read_text(encoding='utf-8-sig'))
    predictions = read_rows(run_dir/'predictions/minicpm/math_500_Level 3.jsonl')
    reviews = read_rows(run_dir/'reviews/minicpm/math_500_Level 3.jsonl')
    expected = set(range(105))
    prediction_indices = Counter(x['index'] for x in predictions)
    review_indices = Counter(x['index'] for x in reviews)
    errors, empty, finish = [], [], Counter()
    for row in predictions:
        output = row.get('model_output') or {}
        choices = output.get('choices') or []
        if output.get('error') or len(choices) != 1:
            errors.append(row['index'])
        for choice in choices:
            finish[str(choice.get('stop_reason'))] += 1
            content = (choice.get('message') or {}).get('content')
            if isinstance(content, list):
                text = ''.join(str(p.get('text', '')) for p in content if p.get('type') == 'text')
            else:
                text = content or ''
            if not str(text).strip():
                empty.append(row['index'])
    score_items = [r['sample_score']['score'] for r in reviews]
    review_statuses = Counter(x.get('status') for x in score_items)
    values = [float(x['value']['accuracy']) for x in score_items]
    exact = sum(values)/len(values) if values else None
    reported = report['metrics'][0]['score']
    execution = report['execution_summary']
    indices_ok = (set(prediction_indices) == set(review_indices) == expected
                  and all(n == 1 for n in prediction_indices.values())
                  and all(n == 1 for n in review_indices.values()))
    integrity_ok = (len(predictions) == len(reviews) == 105 and indices_ok
                    and not errors and not empty
                    and review_statuses == {'success': 105}
                    and execution['requested'] == execution['succeeded'] == 105
                    and execution['errored'] == 0 and not execution['incomplete']
                    and all(v in (0.0, 1.0) for v in values)
                    and exact is not None and abs(exact-reported) < 0.0001)
    return dict(run_dir=str(run_dir), prediction_rows=len(predictions), review_rows=len(reviews),
                indices_complete_unique=indices_ok, prediction_error_indices=errors,
                empty_final_output_indices=empty, finish_reasons=dict(finish),
                review_statuses=dict(review_statuses), correct=sum(v == 1 for v in values),
                incorrect_indices=[r['index'] for r, v in zip(reviews, values) if v == 0],
                score_recomputed=exact, score_reported=reported, execution_summary=execution,
                integrity_pass=integrity_ok, accuracy_pass=bool(integrity_ok and exact >= 0.95))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_dir)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['integrity_pass'] else 2)
