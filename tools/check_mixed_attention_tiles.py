"""Mixed-prefill/decode native-equivalence and event-time experiment, not a score."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import signal
import statistics
import time


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--v1-adapter',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    a=parser.parse_args()
    remaining=min(600,int(a.deadline-time.time()-30))
    if remaining<180:raise RuntimeError('Insufficient reserved diagnostic time')
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic cutoff')))
    signal.alarm(remaining)
    from tune_attention_tiles import check_exclusive
    check_exclusive()
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as native
    torch.manual_seed(1031)
    original=native.kernel_unified_attention
    while hasattr(original,'original'):original=original.original
    v1=load_module(a.v1_adapter,'isolated_v1')
    candidate=load_module(a.candidate,'isolated_mixed')
    report=dict(official_score=False,seed=1031,started_at=time.time(),cases=[],
                candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
                original_adapter_sha256=hashlib.sha256(a.v1_adapter.read_bytes()).hexdigest(),
                gpu=torch.cuda.get_device_name(0),strict_bitwise_equivalence_required=True,
                timing_scope='Isolated GPU event time; no model, serving, or graph-replay throughput inferred.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=a.out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(a.out)
    definitions=[('pure_decode',[(1,512)]*64)]
    for total in (128,256,512,1024,2048,2049):
        definitions.append((f'mixed_q{total}_kv4096',[(1,4096)]*63+[(total-63,4096)]))
    definitions.extend([
        ('mixed_q2048_kv16384',[(1,16384)]*63+[(1985,16384)]),
        ('prefill32_q2048',[(64,4096)]*32),
        ('mixed32_q256_kv4096',[(1,4096)]*31+[(225,4096)]),
        ('mixed16_q128_kv4096',[(1,4096)]*15+[(113,4096)]),
        ('mixed64_three_prefills',[(1,4096)]*61+[(662,4096),(662,4096),(663,4096)]),
    ])
    rng=random.Random(1031)
    try:
        for label,lengths in definitions:
            bs=16;heads=16;kvheads=2;dim=128
            counts=[(kv+bs-1)//bs for _,kv in lengths]
            total_blocks=sum(counts)
            physical=torch.randperm(total_blocks,device='cuda')
            table=torch.zeros((len(lengths),max(counts)),device='cuda',dtype=torch.int32)
            cursor=0
            for i,count in enumerate(counts):
                table[i,:count]=physical[cursor:cursor+count].int();cursor+=count
            k=torch.randn((total_blocks,bs,kvheads,dim),device='cuda',dtype=torch.bfloat16)
            v=torch.randn_like(k)
            q=torch.randn((sum(x[0] for x in lengths),heads,dim),device='cuda',dtype=torch.bfloat16)
            out=torch.empty_like(q)
            prefix=[0]
            for qlen,_ in lengths:prefix.append(prefix[-1]+qlen)
            kwargs=dict(q=q,k=k,v=v,out=out,cu_seqlens_q=torch.tensor(prefix,device='cuda',dtype=torch.int32),
                        max_seqlen_q=max(x[0] for x in lengths),seqused_k=torch.tensor([x[1] for x in lengths],device='cuda',dtype=torch.int32),
                        max_seqlen_k=max(x[1] for x in lengths),softmax_scale=dim**-0.5,causal=True,window_size=(-1,-1),
                        block_table=table,softcap=0.0,q_descale=None,k_descale=None,v_descale=None)
            kernels=dict(native=original,v1=v1.QueryTiledKernel(original),
                         mixed=candidate.QueryTiledKernel(original),
                         wide_diagnostic=candidate.QueryTiledKernel(original,min_average_query_rows=1,min_total_query_rows=0))
            native.kernel_unified_attention=original
            native.unified_attention(**kwargs);torch.cuda.synchronize();baseline=out.clone()
            row=dict(case=label,query_rows=q.shape[0],num_seqs=len(lengths),lengths=lengths,variants={})
            report['cases'].append(row)
            for name,kernel in kernels.items():
                native.kernel_unified_attention=kernel
                out.fill_(float('nan'))
                native.unified_attention(**kwargs);torch.cuda.synchronize()
                equal=bool(torch.equal(out,baseline))
                row['variants'][name]=dict(bitwise_equal=equal,
                    max_abs_difference=float((out.float()-baseline.float()).abs().max()),
                    optimized_launches=getattr(kernel,'optimized_launches',None))
                save()
                assert equal,f'{label}/{name} changed native output; do not deploy'
                for _ in range(3):native.unified_attention(**kwargs)
                torch.cuda.synchronize()
            samples=defaultdict(list)
            for sample in range(7):
                order=list(kernels);rng.shuffle(order)
                for name in order:
                    native.kernel_unified_attention=kernels[name]
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(3):native.unified_attention(**kwargs)
                    end.record();end.synchronize()
                    samples[name].append(begin.elapsed_time(end)/3)
            for name,times in samples.items():
                row['variants'][name].update(median_ms=statistics.median(times),samples_ms=times)
            row['mixed_relative_to_v1']=row['variants']['v1']['median_ms']/row['variants']['mixed']['median_ms']-1
            row['wide_relative_to_native']=row['variants']['native']['median_ms']/row['variants']['wide_diagnostic']['median_ms']-1
            save()
            print(json.dumps(dict(case=label,query_rows=q.shape[0],num_seqs=len(lengths),
                                  medians={n:x['median_ms'] for n,x in row['variants'].items()},
                                  mixed_relative_to_v1=row['mixed_relative_to_v1'])),flush=True)
        report.update(complete=True,all_bitwise_equal=True,finished_at=time.time())
        save();signal.alarm(0)
    finally:
        native.kernel_unified_attention=original


if __name__=='__main__':main()
