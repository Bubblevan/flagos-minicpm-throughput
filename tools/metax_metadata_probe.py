"""Bounded C500 preflight and metadata/FA2 differential microdiagnostic.

This is NOT the official benchmark. No original package is edited.
"""
import argparse, hashlib, inspect, json, os, platform, statistics, subprocess, time
from pathlib import Path

def write(root,name,data):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2,default=str));t.replace(p)
def command(args):
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=25)
    return dict(argv=args,returncode=p.returncode,output=p.stdout)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--micro',action='store_true');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    import importlib.metadata as md
    import torch
    import flash_attn
    from flash_attn import flash_attn_varlen_func
    package_names=['torch','vllm','vllm-plugin-fl','triton','flash-attn','evalscope']
    evidence=dict(at=time.time(),hostname=platform.node(),python=platform.python_version(),
        packages={n:md.version(n) for n in package_names},
        gpu=dict(name=torch.cuda.get_device_name(),properties=str(torch.cuda.get_device_properties(0))),
        flash_signature=str(inspect.signature(flash_attn_varlen_func)),flash_file=flash_attn.__file__,
        gpu_tool=command(['bash','-lc','command -v mx-smi; mx-smi']),
        processes=command(['ps','-eo','pid,ppid,stat,etime,args']))
    for repo in ['vllm-plugin-FL','FlagGems']:
        evidence[repo]=dict(head=command(['git','-C','/workspace/'+repo,'rev-parse','HEAD']),
                           status=command(['git','-C','/workspace/'+repo,'status','--porcelain']))
    native=Path('/workspace/vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py')
    evidence['attention_source_sha256']=hashlib.sha256(native.read_bytes()).hexdigest()
    (a.root/'native-flash_attn.py').write_bytes(native.read_bytes())
    interface=Path(inspect.getfile(flash_attn_varlen_func));(a.root/'installed-flash-interface.py').write_bytes(interface.read_bytes())
    write(a.root,'preflight.json',evidence)
    print(json.dumps(dict(stage='preflight_complete',hostname=evidence['hostname'],packages=evidence['packages'],
          native_attention_sha=evidence['attention_source_sha256'],flash_signature=evidence['flash_signature'])),flush=True)
    if not a.micro:return
    torch.manual_seed(5501);torch.cuda.manual_seed_all(5501)
    cases=[]
    for query_lens,seq_lens in [([2048],[4096]),([256]*8,[4096]*8),([129,257],[4096,8192]),
                                ([512]*4,[16384]*4),([256]*16,[4096]*16)]:
        n=len(query_lens);page=16;heads=16;kv_heads=2;dim=128
        lens=torch.tensor(seq_lens,device='cuda',dtype=torch.int32)
        old=torch.tensor([0]+lens.tolist(),device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
        new=torch.empty(n+1,device='cuda',dtype=torch.int32);new[0]=0
        torch.cumsum(lens,dim=0,dtype=torch.int32,out=new[1:])
        assert torch.equal(old,new),'Prefix metadata differs'
        q=torch.randn(sum(query_lens),heads,dim,device='cuda',dtype=torch.bfloat16)
        blocks=(max(seq_lens)+page-1)//page
        k=torch.randn(n*blocks,page,kv_heads,dim,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        table=torch.arange(n*blocks,device='cuda',dtype=torch.int32).reshape(n,blocks)
        cu_q=torch.tensor([0]+query_lens,device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
        args=dict(q=q,k=k,v=v,block_table=table,cu_seqlens_q=cu_q,max_seqlen_q=max(query_lens),
                  max_seqlen_k=max(seq_lens),softmax_scale=dim**-.5,causal=True,window_size=(-1,-1))
        out_a=flash_attn_varlen_func(**args,cu_seqlens_k=old)
        out_b=flash_attn_varlen_func(**args,cu_seqlens_k=new)
        assert torch.equal(out_a,out_b),'FA2 outputs differ with equal metadata'
        cases.append(dict(query_lens=query_lens,seq_lens=seq_lens,metadata_bitwise_equal=True,
                          attention_bitwise_equal=True,finite=bool(torch.isfinite(out_a).all())))
        assert cases[-1]['finite']
        del q,k,v,table,out_a,out_b
    lens=torch.full((8,),4096,device='cuda',dtype=torch.int32)
    def repeated():
        for _ in range(42):
            torch.tensor([0]+lens.tolist(),device='cuda',dtype=torch.int32).cumsum(0,dtype=torch.int32)
    def hoisted():
        once=torch.empty(9,device='cuda',dtype=torch.int32);once[0]=0
        torch.cumsum(lens,dim=0,dtype=torch.int32,out=once[1:])
        for _ in range(42):value=once
    timings={}
    for name,fn in [('old',repeated),('hoisted',hoisted),('old_repeat',repeated)]:
        for _ in range(3):fn()
        samples=[]
        for _ in range(12):
            torch.cuda.synchronize();start=time.perf_counter();fn();torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000)
        timings[name]=dict(milliseconds=samples,median_ms=statistics.median(samples))
    try:
        from torch.profiler import profile,ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
            repeated();torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.root/'metadata-only-trace.json'))
        (a.root/'metadata-only-profiler.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
        profiling=dict(captured=True,scope='synthetic metadata only, not model E2E or attention kernel time share')
    except Exception as e:profiling=dict(captured=False,error=repr(e))
    result=dict(stage='micro_complete',official_score=False,cases=cases,timing=timings,profiling=profiling,
                original_source_unchanged=True,limitations='Hoisting microbenchmark only; no serving gain established.')
    write(a.root,'metadata-micro.json',result)
    print(json.dumps(dict(stage='micro_complete',cases=len(cases),bitwise_equal=True,
          median_ms={k:v['median_ms'] for k,v in timings.items()},profiling=profiling)),flush=True)
if __name__=='__main__':main()