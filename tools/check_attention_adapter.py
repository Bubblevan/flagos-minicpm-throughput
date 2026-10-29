"""Differential test of the isolated plugin adapter, without a model service."""
import argparse
import json
from pathlib import Path
import signal
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    signal.alarm(240)
    from tune_attention_tiles import check_exclusive
    check_exclusive()
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as native
    from vllm_fl.dispatch.backends.vendor.iluvatar.impl.attention_tiles import QueryTiledKernel
    torch.manual_seed(991)
    original=native.kernel_unified_attention
    adapter=QueryTiledKernel(original)
    result=dict(official_score=False,started_at=time.time(),cases=[])
    cases=[([(1024,4096)],16,2,128), ([(2048,16384)],16,2,128),
           ([(257,513)],16,2,128), ([(128,2053)],16,2,128),
           ([(65,521),(129,2051),(17,191)],16,2,128),
           ([(1,512)]*64,16,2,128), ([(71,251)],8,2,64)]
    for lengths,heads,kvheads,dim in cases:
        bs=16;counts=[(kv+bs-1)//bs for _,kv in lengths]
        blocks=sum(counts);perm=torch.randperm(blocks,device='cuda')
        table=torch.zeros((len(lengths),max(counts)),device='cuda',dtype=torch.int32)
        at=0
        for i,c in enumerate(counts):table[i,:c]=perm[at:at+c].int();at+=c
        k=torch.randn((blocks,bs,kvheads,dim),device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k);q=torch.randn((sum(x[0] for x in lengths),heads,dim),device='cuda',dtype=torch.bfloat16)
        out=torch.empty_like(q);prefix=[0]
        for qlen,_ in lengths:prefix.append(prefix[-1]+qlen)
        kw=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=torch.tensor(prefix,device='cuda',dtype=torch.int32),
                max_seqlen_q=max(x[0] for x in lengths),seqused_k=torch.tensor([x[1] for x in lengths],device='cuda',dtype=torch.int32),
                max_seqlen_k=max(x[1] for x in lengths),softmax_scale=dim**-0.5,causal=True,window_size=(-1,-1),
                block_table=table,softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
        native.kernel_unified_attention=original
        native.unified_attention(**kw);torch.cuda.synchronize();baseline=out.clone()
        before=(adapter.optimized_launches,adapter.fallback_launches)
        native.kernel_unified_attention=adapter
        out.fill_(float('nan'));native.unified_attention(**kw);torch.cuda.synchronize()
        equal=torch.equal(baseline,out)
        row=dict(lengths=lengths,heads=heads,head_dim=dim,bitwise_equal=bool(equal),
                 max_abs_difference=float((baseline.float()-out.float()).abs().max()),
                 optimized_launches=adapter.optimized_launches-before[0],
                 fallback_launches=adapter.fallback_launches-before[1])
        result['cases'].append(row)
        a.out.write_text(json.dumps(result,indent=2))
        print(json.dumps(row),flush=True)
        assert equal, 'Candidate changed native output; do not run end-to-end'
    native.kernel_unified_attention=original
    result['complete']=True;result['all_bitwise_equal']=True
    result['scope']='Differential native equivalence; independent float32 reference discrepancy on irregular shapes remains under investigation.'
    a.out.write_text(json.dumps(result,indent=2));signal.alarm(0)


if __name__=='__main__':main()
