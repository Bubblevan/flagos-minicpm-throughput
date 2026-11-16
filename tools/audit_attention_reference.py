"""Independent offline scalar check of saved CPU references, plus error localization."""
import argparse
import json
import math
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4)
    result=dict(device='cpu',scalar_reference_max_error=0.,cases=[])
    for path in sorted((a.root/'inputs').glob('q*_tail.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        actual=torch.load(a.root/'old'/(data['label']+'-outputs.pt'),map_location='cpu',weights_only=False)['native'].double()
        qlen,kvlen=data['lengths'][0];table=data['table'][0].long()
        k=data['k'][table].reshape(-1,2,128)[:kvlen].double()
        v=data['v'][table].reshape(-1,2,128)[:kvlen].double()
        q=data['q'].double();checks=[]
        for j,row in enumerate(data['samples'][0]['rows']):
            for head in (0,7,8,15):
                n=kvlen-qlen+row+1;group=head//8
                # Scalar Python fsum dot/softmax/weighted sum; independent of matrix products.
                logits=[math.fsum(float(x)*float(y) for x,y in zip(q[row,head],k[t,group]))/math.sqrt(128) for t in range(n)]
                high=max(logits);weights=[math.exp(x-high) for x in logits];normalizer=math.fsum(weights)
                scalar=torch.tensor([math.fsum(w*float(v[t,group,d]) for t,w in enumerate(weights))/normalizer for d in range(128)],dtype=torch.float64)
                stored=data['samples'][0]['reference'][j,head]
                delta=float((scalar-stored).abs().max());result['scalar_reference_max_error']=max(delta,result['scalar_reference_max_error'])
                checks.append(dict(row=row,head=head,scalar_vs_stored_max_abs=delta,old_vs_scalar_max_abs=float((actual[row,head]-scalar).abs().max())))
        result['cases'].append(dict(label=data['label'],scalar_checks=checks))
    assert result['scalar_reference_max_error']<1e-10
    result['scalar_reference_pass']=True
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(cases=len(result['cases']),scalar_reference_pass=True,scalar_reference_max_error=result['scalar_reference_max_error'])))


if __name__=='__main__':main()
