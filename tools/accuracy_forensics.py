"""Read-only MATH-500 evidence review; never changes the official judge or score."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
from validate_accuracy_artifacts import audit, read_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(performance_pass, accuracy):
    return bool(performance_pass and accuracy['integrity_pass'] and accuracy['accuracy_pass'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v3-root', type=Path, required=True)
    p.add_argument('--v2-run', type=Path, required=True)
    p.add_argument('--native-run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError('Do not overwrite a completed forensic report')
    a.out.mkdir(parents=True, exist_ok=True)
    reports = list((a.v3_root/'evalscope').glob('*/reports/minicpm/math_500.json'))
    assert len(reports) == 1
    run = reports[0].parents[2]
    aa = audit(run)
    assert aa['integrity_pass'] and aa['correct'] == 98
    source = json.loads((a.v3_root/'independent-result.json').read_text(encoding='utf-8'))
    perf = source['stages']['official_eight_round_benchmark']['performance_pass']
    derived = dict(raw_result_sha256=sha(a.v3_root/'independent-result.json'),
                   raw_overall_pass=source['overall_pass'], performance_pass=perf,
                   accuracy_audit=aa, overall_pass=aggregate(perf, aa),
                   correction='Derived only: performance AND integrity AND accuracy. Raw evidence untouched.')
    assert not derived['overall_pass']
    # Regression: failed accuracy cannot pass the combined gate, even with complete records.
    for pp in (False, True):
        for ip in (False, True):
            for ap in (False, True):
                assert aggregate(pp, dict(integrity_pass=ip, accuracy_pass=ap)) == (pp and ip and ap)
    sessions = {}
    for name, folder in [('native', a.native_run), ('v2', a.v2_run), ('v3', run)]:
        ar = audit(folder)
        rows = read_rows(folder/'reviews/minicpm/math_500_Level 3.jsonl')
        sessions[name] = dict(correct=ar['correct'], wrong={
            r['sample_score']['sample_metadata']['question_id']: r['index']
            for r in rows if r['sample_score']['score']['value']['accuracy'] == 0})
    reviews = read_rows(run/'reviews/minicpm/math_500_Level 3.jsonl')
    predictions = {r['index']: r for r in read_rows(run/'predictions/minicpm/math_500_Level 3.jsonl')}
    assessments = {
        18: ('mathematically_correct_judge_dispute', 'v=(-1/3,2/3,5/3); sum=2; (z-y,x-z,y-x)=(1,-2,1). [2pt] changes spacing, not components.'),
        87: ('mathematically_correct_judge_dispute', '(a.b)/(b.b)*b = (8/49)*(2,6,3) = (16/49,48/49,24/49). [4pt] changes spacing only.'),
        99: ('mathematically_correct_extraction_dispute', 'Final text states tan A=8/15; extracted_prediction is 15. sin A=4(1-cos A), 0<A<pi gives cos A=15/17 and sin A=8/17.'),
        25: ('incorrect_final_answer', 'Division/remainders and prose list 4,2,1,0 correctly, but final boxed digits are 4120_5=535, not 555. Target 4210_5=555. Do not credit correct intermediate reasoning.'),
        7: ('incorrect_final_answer', 'Conjugate sixth powers sum to 13536 and the smaller is in (0,1), so floor is 13535; final boxed answer is 53535 despite correct preceding text.'),
        90: ('incorrect_problem_transcription', 'Prompt contains five diamonds (2T=5D); model rewrites three (2T=3D). Enumeration of the actual constraints gives uniquely (T,S,D,C)=(5,3,2,8).'),
        101: ('incorrect_geometric_reasoning', 'A radius-3 sphere requires enclosing cube side>=6, attained at6, so volume216. Removing planar tetrahedra and equating volume does not produce a sphere; 216*pi/5 is wrong.'),
    }
    # Independent exact arithmetic supporting the manual review, not a substitute judge.
    v = (F(-1,3), F(2,3), F(5,3))
    assert sum(v) == 2 and (v[2]-v[1],v[0]-v[2],v[1]-v[0]) == (1,-2,1)
    assert tuple(F(8,49)*x for x in (2,6,3)) == (F(16,49), F(48,49), F(24,49))
    assert int('4210',5)==555 and int('4120',5)==535
    solutions=[(t,s,d,c) for t in range(1,10) for s in range(1,10)
               for d in range(1,10) for c in range(1,10)
               if len({t,s,d,c})==4 and t+s==c and 2*t==5*d and 2*t==c+d]
    assert solutions==[(5,3,2,8)]
    items=[]
    for row in reviews:
        if row['sample_score']['score']['value']['accuracy'] != 0:
            continue
        i=row['index']; score=row['sample_score']['score']
        category, proof=assessments[i]
        items.append(dict(index=i, question_id=row['sample_score']['sample_metadata']['question_id'],
            question=row['messages'][0]['content'], target=row['target'],
            final_output=score['prediction'], extracted_prediction=score['extracted_prediction'],
            judge_score=score['value']['accuracy'], judge_status=score['status'],
            official_solution=row['sample_score']['sample_metadata']['solution'],
            finish_reason=predictions[i]['model_output']['choices'][0]['stop_reason'],
            classification=category, independent_reasoning=proof))
    assert {x['index'] for x in items} == set(assessments)
    truncated=[dict(index=i, question_id=r['metadata']['question_id'],
                    official_score=next(x['sample_score']['score']['value']['accuracy'] for x in reviews if x['index']==i))
               for i,r in predictions.items() if r['model_output']['choices'][0]['stop_reason']=='max_tokens']
    result=dict(official_score_unchanged=98/105, official_correct_unchanged=98,
                official_overall_pass=False, organizer_review_required=True,
                disputes=[18,87,99], sessions=sessions, truncated=truncated, cases=items,
                input_sha256={str(f.relative_to(run)):sha(f) for f in [reports[0],
                    run/'predictions/minicpm/math_500_Level 3.jsonl',run/'reviews/minicpm/math_500_Level 3.jsonl']},
                causality='Saved outputs cannot establish compiler/kernel causality; no matched logits or RNG replay exists.')
    (a.out/'forensics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.out/'derived-result.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# 11/16 v3 accuracy forensics', '',
          'Official score remains **98/105**, overall **false**. Three mathematically correct outputs have judge/extraction disputes; no score was changed. Organizer verification is required.', '',
          'Raw result was preserved. derived-result.json corrects only the combined gate to performance AND integrity AND accuracy.', '',
          '## Wrong-question sets matched by question_id', '', '```json', json.dumps(sessions,indent=2), '```', '',
          '## Seven failed reviews', '']
    for item in sorted(items,key=lambda x:x['index']):
        text += [f"### Index {item['index']}: {item['question_id']}", '',
                 item['classification'], '', item['independent_reasoning'], '',
                 'Question:', item['question'], '', 'Target: `'+item['target']+'`',
                 'Extracted: `'+item['extracted_prediction']+'`',
                 'Finish: '+item['finish_reason'], '', 'Saved final output:', '', item['final_output'], '']
    text += ['## Truncation', '', json.dumps(truncated,ensure_ascii=False), '',
             'The truncated request is distinct from the seven wrong reviews above. Do not change max_tokens or resample to select a score.', '',
             '## Limits', '', result['causality'],
             'Full untouched predictions/reviews, including reasoning parts, remain in the original D: archive. This report includes only the final-answer text; it does not alter the judge or dataset.']
    (a.out/'accuracy-forensics.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(correct=98,overall_pass=False,disputes=result['disputes'],truncated=truncated)))


if __name__=='__main__':
    main()
