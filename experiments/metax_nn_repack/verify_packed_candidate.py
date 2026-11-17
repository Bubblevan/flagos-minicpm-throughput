"""Actual public wrapper, native fallback, independent math, fake/graph gate."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import torch

ROOT=Path(os.environ['FLAGOS_NUM_ROOT'])
SOURCE=Path(os.environ['FLAGOS_NN_SOURCE'])
ROOT.mkdir(parents=True,exist_ok=False)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Numerical budget')))
signal.alarm(420)
torch.set_num_threads(1)
torch.cuda.set_device(0)
torch.manual_seed(20261117)
rows=[]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def save(complete=False,**kw):
    p=ROOT/'correctness.tmp'
    p.write_text(json.dumps(dict(complete=complete,cases=rows,**kw),indent=2))
    p.replace(ROOT/'correctness.json')


def reference(a,w,actual):
    idx=torch.tensor([0,a.shape[0]//2,a.shape[0]-1],device=a.device)
    ref=(a.index_select(0,idx).double().cpu()@w.double().cpu().t()).float()
    return torch.allclose(actual.index_select(0,idx).float().cpu(),ref,atol=.02,rtol=.02)


native=load('tokenforge_native_mm_reference',Path('/workspace/FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py'))
candidate=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
assert str(SOURCE/'FlagGems') in candidate.__file__
# Revalidate the unchanged guarded NT component too, on this actual new device.
for M,N,K in [(2048,2560,2048),(2048,12288,2048),(2048,2048,6144),
              (2048,2048,2048),(127,2560,2048),(64,2048,6144)]:
    for kind in ['random','cancellation','zeros']:
        a=torch.randn((M,K),device='cuda:0',dtype=torch.bfloat16)
        w=torch.randn((N,K),device='cuda:0',dtype=torch.bfloat16)
        if kind=='cancellation':
            a[:,1::2]=-a[:,::2];w[:,1::2]=w[:,::2]
        if kind=='zeros':a.zero_()
        expected=torch.empty((M,N),device=a.device,dtype=a.dtype)
        actual=torch.empty_like(expected)
        native.general_mm_nt(a,w.t(),expected,M,N,K)
        candidate.general_mm_nt(a,w.t(),actual,M,N,K)
        torch.cuda.synchronize(0)
        row=dict(path='existing_NT_component',M=M,N=N,K=K,kind=kind,
                 bitwise_equal=torch.equal(actual,expected),cpu_reference_pass=reference(a,w,actual))
        rows.append(row);save();assert row['bitwise_equal'] and row['cpu_reference_pass'],row

adapter=load('tokenforge_packed_linear_test',SOURCE/'vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/patches/packed_linear.py')
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
method=UnquantizedLinearMethod()
for N in [2560,12288]:
    for kind in ['random','cancellation','zeros']:
        a=torch.randn((2048,2048),device='cuda:0',dtype=torch.bfloat16)
        w=torch.randn((N,2048),device='cuda:0',dtype=torch.bfloat16)
        if kind=='cancellation':
            a[:,1::2]=-a[:,::2];w[:,1::2]=w[:,::2]
        if kind=='zeros':a.zero_()
        layer=torch.nn.Module();layer.weight=torch.nn.Parameter(w,requires_grad=False)
        method.process_weights_after_loading(layer)
        assert torch.equal(layer._tokenforge_nn_weight.t(),w)
        expected=torch.empty((2048,N),device=a.device,dtype=a.dtype)
        native.general_mm_nt(a,w.t(),expected,2048,N,2048)
        actual=adapter._linear(layer,a,w,None)
        torch.cuda.synchronize(0)
        row=dict(path='packed_public_wrapper',M=2048,N=N,K=2048,kind=kind,
                 bitwise_equal=torch.equal(actual,expected),cpu_reference_pass=reference(a,w,actual))
        rows.append(row);save();assert row['bitwise_equal'] and row['cpu_reference_pass'],row

for M,N,K,dtype,bias_present in [(64,2560,2048,torch.bfloat16,False),
    (127,12288,2048,torch.bfloat16,False),(2048,2048,2048,torch.bfloat16,False),
    (2048,2048,6144,torch.bfloat16,False),(64,2560,2048,torch.float32,False),
    (64,2560,2048,torch.bfloat16,True)]:
    a=torch.randn((M,K),device='cuda:0',dtype=dtype)
    w=torch.randn((N,K),device='cuda:0',dtype=dtype)
    bias=torch.randn(N,device=a.device,dtype=dtype) if bias_present else None
    layer=torch.nn.Module();layer.weight=torch.nn.Parameter(w,requires_grad=False)
    method.process_weights_after_loading(layer)
    expected=adapter._ORIGINAL_FACTORY()(layer,a,w,bias)
    actual=adapter._linear(layer,a,w,bias)
    torch.cuda.synchronize(0)
    row=dict(path='original_dispatch_fallback',M=M,N=N,K=K,dtype=str(dtype),bias=bias_present,
             bitwise_equal=torch.equal(actual,expected),cpu_reference_pass=None,
             reference_is_original_dispatch=True)
    rows.append(row);save();assert row['bitwise_equal'],row

# Eager backend tests Dynamo/fake registration without changing model settings.
a=torch.randn((2048,2048),device='cuda:0',dtype=torch.bfloat16)
w=torch.randn((2560,2048),device='cuda:0',dtype=torch.bfloat16)
packed=w.t().contiguous()
expected=adapter.packed_prefill_mm(a,packed)
compiled=torch.compile(adapter.packed_prefill_mm,backend='eager',fullgraph=True)
actual=compiled(a,packed);torch.cuda.synchronize(0)
assert torch.equal(actual,expected)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):captured=adapter.packed_prefill_mm(a,packed)
graph.replay();torch.cuda.synchronize(0)
assert torch.equal(captured,expected)
save(complete=True,fake_and_compile_pass=True,cudagraph_pass=True,
     actual_mm=candidate.__file__,packed_source=str(SOURCE/'FlagGems/src/flag_gems/runtime/backend/_metax/ops/packed_mm.py'),
     packed_sha256=hashlib.sha256((SOURCE/'FlagGems/src/flag_gems/runtime/backend/_metax/ops/packed_mm.py').read_bytes()).hexdigest(),
     finite_tests_not_universal_proof=True,official_score=False)
signal.alarm(0)
print(json.dumps(dict(complete=True,cases=len(rows),official_score=False)),flush=True)
