"""Bounded prepacked NN-layout prefill test, inspired by attributed XLANG2026 opt4."""
import ast,datetime as dt,hashlib,importlib.util,json,os,resource,signal,statistics,subprocess,time,traceback
from pathlib import Path
ROOT=Path('/data/tokenforge-artifacts/metax_nn_layout_micro_20261117_2005')
def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    assert not ROOT.exists()
    assert 'no process found' in subprocess.check_output(['/opt/mxdriver/bin/mx-smi'],text=True).lower()
    ROOT.mkdir(parents=True)
    import torch,triton
    import importlib
    native=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    source=Path(native.__file__);text=source.read_text();tree=ast.parse(text);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mm_kernel_nn')
    raw='import triton\nimport triton.language as tl\n@triton.jit\n'+ast.get_source_segment(text,fn)+'\n'
    module=ROOT/'raw_mm_nn.py';module.write_text(raw)
    spec=importlib.util.spec_from_file_location('raw_mm_nn_probe',module);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(8135);torch.cuda.manual_seed_all(8135)
    configs=[(128,128,32,8,2,'basic'),(128,256,32,8,2,'cpasync'),(256,256,32,8,2,'cpasync'),(256,128,32,8,2,'cpasync'),(128,128,64,8,2,'cpasync')]
    result=dict(complete=False,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),native_NN_kernel_body_unchanged=True,weight_repack_in_timing=False,one_time_storage_layout_change=True,shapes=[],started=time.time(),official_score=False)
    def save():
        p=ROOT/'probe.tmp';p.write_text(json.dumps(result,indent=2));p.replace(ROOT/'probe.json')
    def timing(f):
        for _ in range(3):f()
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):f()
        for _ in range(3):graph.replay()
        torch.cuda.synchronize();samples=[]
        for _ in range(5):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            for _ in range(5):graph.replay()
            b.record();b.synchronize();samples.append(a.elapsed_time(b)/50)
        return statistics.median(samples)
    for M,N,K in [(2048,2560,2048),(2048,2048,2048),(2048,12288,2048),(2048,2048,6144)]:
        a=torch.randn((M,K),device='cuda',dtype=torch.bfloat16);weight=torch.randn((N,K),device='cuda',dtype=torch.bfloat16);b=weight.t();packed=b.contiguous();c=torch.empty((M,N),device='cuda',dtype=torch.bfloat16);baseline=torch.empty_like(c)
        before=timing(lambda:native.mm_out(a,b,out=baseline));native.mm_out(a,b,out=baseline)
        row=dict(M=M,N=N,K=K,native_before_ms=before,configs=[]);result['shapes'].append(row);save()
        for BM,BN,BK,warps,stages,pipeline in configs:
            config=dict(BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,num_warps=warps,num_stages=stages,pipeline=pipeline)
            try:
                def launch():return mod.mm_kernel_nn[(triton.cdiv(M,BM)*triton.cdiv(N,BN),)](a,packed,c,M,N,K,BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,GROUP_M=8,EVEN_M=M%BM==0,EVEN_N=N%BN==0,EVEN_K=K%BK==0,num_warps=warps,num_stages=stages,pipeline=pipeline)
                launch();torch.cuda.synchronize();equal=torch.equal(c,baseline)
                diff=float((c.float()-baseline.float()).abs().max().item())
                ms=timing(launch)
                # Independent selected-row CPU float64 reference, not an adjusted tolerance.
                idx=torch.tensor([0,17,511,1023,2047],device='cuda');actual=c.index_select(0,idx).float().cpu();ref=(a.index_select(0,idx).double().cpu()@weight.double().cpu().t()).float()
                close=torch.allclose(actual,ref,atol=.02,rtol=.02)
                row['configs'].append(dict(config=config,ms=ms,bitwise_native_equal=equal,max_native_abs=diff,cpu_reference_pass=close))
            except Exception as e:row['configs'].append(dict(config=config,error=repr(e),traceback=traceback.format_exc()))
            save()
        row['native_after_ms']=timing(lambda:native.mm_out(a,b,out=baseline));save()
        eligible=[x for x in row['configs'] if x.get('bitwise_native_equal') and x.get('cpu_reference_pass')]
        row['best_bitwise']=min(eligible,key=lambda x:x['ms']) if eligible else None;save()
    result['complete']=True;result['finished']=time.time();save();print(json.dumps(dict(complete=True,official_score=False)),flush=True)
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded NN layout micro cutoff')));signal.alarm(360)
    main()
