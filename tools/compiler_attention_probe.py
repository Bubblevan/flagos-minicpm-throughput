"""Shared-input old/new compiler diagnosis of native unified attention and v2.

Never a competition score. CPU input generation and reference are compiler
independent; all input/output tensors are saved for reproducibility.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata as md
import itertools
import json
import os
from pathlib import Path
import signal
import time


def save_json(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--candidate',type=Path)
    p.add_argument('--generate',action='store_true')
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    budget=min(1000,int(a.deadline-time.time()-30))
    assert budget>120
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('probe deadline')))
    signal.alarm(budget)
    import torch
    if a.generate:
        a.inputs.mkdir(parents=True,exist_ok=False)
        generator=torch.Generator(device='cpu').manual_seed(1116)
        definitions=[(f'q{q}_tail',[(q,q+37)]) for q in (4,8,12,17,20,24,28,33,65,129,257)]
        definitions += [('ragged',[(17,191),(33,521),(9,71)]),
                        ('mixed16',[(1,512)]*15+[(113,512)]),
                        ('decode64',[(1,512)]*64),
                        ('long4k',[(1024,4096)]),('long16k',[(2048,16384)])]
        manifest=[]
        for label,lengths in definitions:
            bs=16;kh=2;heads=16;dim=128
            counts=[(kv+bs-1)//bs for _,kv in lengths]
            perm=torch.randperm(sum(counts),generator=generator)
            table=torch.zeros((len(counts),max(counts)),dtype=torch.int32)
            at=0
            for i,n in enumerate(counts):table[i,:n]=perm[at:at+n].int();at+=n
            k=torch.randn((sum(counts),bs,kh,dim),generator=generator).bfloat16()
            v=torch.randn(k.shape,generator=generator).bfloat16()
            q=torch.randn((sum(x for x,_ in lengths),heads,dim),generator=generator).bfloat16()
            samples=[];offset=0
            for i,(qlen,kvlen) in enumerate(lengths):
                rows=sorted({0,min(1,qlen-1),qlen//2,qlen-1})
                keys=k[table[i,:counts[i]].long()].reshape(-1,kh,dim)[:kvlen].double()
                values=v[table[i,:counts[i]].long()].reshape(-1,kh,dim)[:kvlen].double()
                # Explicit per-head CPU-double reference; no device matmul / native kernel.
                ref=torch.empty((len(rows),heads,dim),dtype=torch.float64)
                for h in range(heads):
                    scores=q[offset+torch.tensor(rows),h].double() @ keys[:,h//8].T * dim**-0.5
                    mask=torch.arange(kvlen)[None,:] > (kvlen-qlen+torch.tensor(rows))[:,None]
                    scores.masked_fill_(mask,float('-inf'))
                    ref[:,h]=torch.softmax(scores,dim=-1) @ values[:,h//8]
                samples.append(dict(offset=offset,rows=rows,reference=ref))
                offset+=qlen
            payload=dict(label=label,lengths=lengths,q=q,k=k,v=v,table=table,samples=samples,
                         cu=torch.tensor([0]+list(itertools.accumulate(x for x,_ in lengths)),dtype=torch.int32),
                         seq=torch.tensor([y for _,y in lengths],dtype=torch.int32))
            path=a.inputs/(label+'.pt');torch.save(payload,path)
            manifest.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),lengths=lengths))
        save_json(a.inputs/'manifest.json',dict(seed=1116,reference='CPU float64 per-head explicit GQA mapping and logical page gather',cases=manifest))
        print(json.dumps(dict(generated=len(manifest))))
        return
    from tune_attention_tiles import check_exclusive
    check_exclusive()
    import triton
    import vllm.v1.attention.ops.triton_unified_attention as native
    spec=importlib.util.spec_from_file_location('diagnostic_adapter_v2',a.candidate)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    original=native.kernel_unified_attention
    while hasattr(original,'original'):original=original.original
    result=dict(official_score=False,started=time.time(),triton=triton.__version__,triton_file=triton.__file__,
                flagtree=md.version('flagtree'),candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                native_file=native.__file__,native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                reference_atol=0.02,reference_rtol=0.02,repeats=8,cases=[])
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.inputs/'manifest.json').read_text())
    try:
        for entry in manifest['cases']:
            path=a.inputs/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
            cpu=torch.load(path,map_location='cpu',weights_only=False)
            kw={key:cpu[key].cuda() for key in ('q','k','v')}
            kw.update(out=torch.empty_like(kw['q']),cu_seqlens_q=cpu['cu'].cuda(),
                      max_seqlen_q=max(x for x,_ in cpu['lengths']),seqused_k=cpu['seq'].cuda(),
                      max_seqlen_k=max(y for _,y in cpu['lengths']),softmax_scale=128**-0.5,
                      causal=True,window_size=(-1,-1),block_table=cpu['table'].cuda(),softcap=0.0,
                      q_descale=None,k_descale=None,v_descale=None)
            row=dict(label=cpu['label'],input_sha256=entry['sha256'],variants={})
            outputs={}
            for name,kernel in [('native',original),('v2',adapter.QueryTiledKernel(original))]:
                native.kernel_unified_attention=kernel; first=None;stable=True;finite=True
                for repeat in range(8):
                    kw['out'].fill_(float('nan'));native.unified_attention(**kw);torch.cuda.synchronize()
                    out=kw['out'].cpu();finite &= bool(torch.isfinite(out).all())
                    if first is None:first=out.clone()
                    else:stable &= bool(torch.equal(first,out))
                checks=[]
                for sample in cpu['samples']:
                    actual=first[sample['offset']+torch.tensor(sample['rows'])].double()
                    ref=sample['reference']
                    checks.append(dict(max_abs=float((actual-ref).abs().max()),
                        allclose=bool(torch.allclose(actual,ref,atol=.02,rtol=.02))))
                row['variants'][name]=dict(stable=stable,finite=finite,reference_checks=checks,
                    reference_pass=all(c['allclose'] for c in checks),
                    optimized_launches=getattr(kernel,'optimized_launches',0))
                outputs[name]=first
            row['v2_bitwise_native']=bool(torch.equal(outputs['native'],outputs['v2']))
            row['v2_max_abs_native']=float((outputs['native'].float()-outputs['v2'].float()).abs().max())
            torch.save(outputs,a.out/(cpu['label']+'-outputs.pt'))
            result['cases'].append(row);save_json(a.out/'result.json',result);print(json.dumps(row),flush=True)
        result['complete']=True
        result['all_reference_pass']=all(v['reference_pass'] and v['finite'] for r in result['cases'] for v in r['variants'].values())
        result['all_stable']=all(v['stable'] for r in result['cases'] for v in r['variants'].values())
        result['all_v2_bitwise_native']=all(r['v2_bitwise_native'] for r in result['cases'])
        save_json(a.out/'result.json',result)
    finally:native.kernel_unified_attention=original


if __name__=='__main__':main()
