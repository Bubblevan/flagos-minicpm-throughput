"""Independently audit saved old/new tensors on CPU; no model or GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    torch.set_num_threads(4)
    inputs=json.loads((a.root/'inputs/manifest.json').read_text())
    old=json.loads((a.root/'old/result.json').read_text());new=json.loads((a.root/'new/result.json').read_text())
    assert old['complete'] and new['complete'] and len(old['cases'])==len(new['cases'])==len(inputs['cases'])==16
    assert old['candidate_sha256']==new['candidate_sha256']=='5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325'
    assert old['native_sha256']==new['native_sha256']
    assert old['flagtree']=='0.6.2a3+iluvatar3.6' and new['flagtree']=='0.7.0+iluvatar3.6'
    result=dict(scope='CPU re-audit of saved first outputs; repeat stability is independently instrumented in GPU probe, not reconstructed here.',cases=[])
    for entry in inputs['cases']:
        file=a.root/'inputs'/entry['file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==entry['sha256']
        data=torch.load(file,map_location='cpu',weights_only=False);label=data['label']
        record=dict(label=label,input_sha256=entry['sha256'],variants={})
        outputs={}
        for version in ('old','new'):
            output_file=a.root/version/(label+'-outputs.pt')
            output=torch.load(output_file,map_location='cpu',weights_only=False)
            outputs[version]=output
            assert torch.equal(output['native'],output['v2'])
            for name,tensor in output.items():
                assert tensor.shape==data['q'].shape and tensor.dtype==torch.bfloat16
                finite=bool(torch.isfinite(tensor).all());errors=[];passing=True
                for sample in data['samples']:
                    actual=tensor[sample['offset']+torch.tensor(sample['rows'])].double();reference=sample['reference']
                    errors.append(float((actual-reference).abs().max()))
                    passing &= bool(torch.allclose(actual,reference,atol=.02,rtol=.02))
                record['variants'][version+'/'+name]=dict(finite=finite,reference_pass=passing,max_abs=max(errors),
                    saved_output_sha256=hashlib.sha256(output_file.read_bytes()).hexdigest())
        record['new_vs_old_max_abs']=float((outputs['new']['native'].float()-outputs['old']['native'].float()).abs().max())
        result['cases'].append(record)
    result.update(input_hashes_pass=True,same_native_source=True,same_v2_source=True,
        old_failed_cases=[r['label'] for r in result['cases'] if not r['variants']['old/native']['reference_pass']],
        new_all_reference_pass=all(v['reference_pass'] and v['finite'] for r in result['cases'] for n,v in r['variants'].items() if n.startswith('new/')),
        within_compiler_all_v2_bitwise_native=True,
        repeated_stability_reported=bool(old['all_stable'] and new['all_stable']),
        full_model_or_accuracy_causality_proven=False)
    assert result['new_all_reference_pass'] and result['old_failed_cases']
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()
