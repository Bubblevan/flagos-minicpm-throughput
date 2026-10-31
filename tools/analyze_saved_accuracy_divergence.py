"""Compare saved accuracy generations without invoking any model or changing evidence."""
import argparse,hashlib,json
from pathlib import Path

def load(p):
    rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(rows)==105 and sorted(r['index'] for r in rows)==list(range(105))
    return {r['index']:r for r in rows}
def content(row):
    output=row['model_output'];choices=output['choices']
    choice=choices[0]
    value=choice.get('message',{}).get('content') or choice.get('text') or ''
    if isinstance(value,list):
        return ''.join(part.get('text','') if isinstance(part,dict) else str(part) for part in value)
    assert isinstance(value,str),type(value)
    return value
def prompt(row):
    # Per-session message UUIDs are not part of the model prompt.
    return [dict(role=m.get('role'),content=m.get('content')) for m in row['messages'] if m.get('role')!='assistant']
def main():
    p=argparse.ArgumentParser();p.add_argument('--v1',type=Path,required=True);p.add_argument('--v2',type=Path,required=True)
    p.add_argument('--v1-audit',type=Path,required=True);p.add_argument('--v2-audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    aa=load(a.v1);bb=load(a.v2);rows=[]
    for i in range(105):
        x=content(aa[i]);y=content(bb[i]);prefix=0
        for u,v in zip(x,y):
            if u!=v:break
            prefix+=1
        rows.append(dict(index=i,same_prompt=prompt(aa[i])==prompt(bb[i]),same_question_id=aa[i]['metadata']['question_id']==bb[i]['metadata']['question_id'],
            generated_text_equal=x==y,common_prefix_characters=prefix,v1_characters=len(x),v2_characters=len(y),
            v1_generation_sha256=hashlib.sha256(x.encode()).hexdigest(),v2_generation_sha256=hashlib.sha256(y.encode()).hexdigest()))
    assert all(r['same_prompt'] and r['same_question_id'] for r in rows)
    wrong1=set(json.loads(a.v1_audit.read_text())['accuracy_audit']['incorrect_indices'])
    wrong2=set(json.loads(a.v2_audit.read_text())['accuracy_audit']['incorrect_indices'])
    result=dict(scope='Saved text comparison only; no model, logits, or paired RNG-state replay.',
        examples=105,all_prompts_and_question_ids_equal=True,
        identical_generation_texts=sum(r['generated_text_equal'] for r in rows),
        additional_v2_incorrect_indices=sorted(wrong2-wrong1),shared_incorrect_indices=sorted(wrong1&wrong2),
        newly_correct_indices=sorted(wrong1-wrong2),rows=rows,
        causality='Unresolved. Temperature 1 stochastic sessions and kernel geometry differ; bitwise operator tests do not prove all model logits or RNG histories equal.',
        next_gate='In a newly authorized BI-V150 session: identical model state/input/KV cache, compare layer outputs and full logits before sampling, then fixed RNG replay. Do not rerun complete accuracy just to select a score.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()