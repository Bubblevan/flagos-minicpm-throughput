import hashlib,importlib,importlib.util,json,signal,time
from pathlib import Path
import torch,triton
ROOT=Path('/data/tokenforge-artifacts/metax_mm_correctness_20261117_1525')
candidate=importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
assert '/TokenForge-metax-mm-20261117/FlagGems/' in candidate.__file__
spec=importlib.util.spec_from_file_location('native_mm_proof','/workspace/FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
torch.set_num_threads(1);torch.manual_seed(1143)
rows=[]
for M,N,K in [(2048,2560,2048),(2048,12288,2048),(2048,2048,6144),(2048,2048,2048),(127,2560,2048),(64,2048,6144)]:
    for kind in ['random','cancellation','zeros']:
        a=torch.randn(M,K,device='cuda',dtype=torch.bfloat16);w=torch.randn(N,K,device='cuda',dtype=torch.bfloat16)
        if kind=='cancellation':a[:,1::2]=-a[:,::2];w[:,1::2]=w[:,::2]
        if kind=='zeros':a.zero_()
        b=w.t();expected=torch.empty(M,N,device='cuda',dtype=a.dtype);actual=torch.empty_like(expected)
        native.general_mm_nt(a,b,expected,M,N,K);candidate.general_mm_nt(a,b,actual,M,N,K);torch.cuda.synchronize()
        equal=torch.equal(actual,expected)
        idx=torch.tensor([0,M//2,M-1],device='cuda');ref=(a.index_select(0,idx).double().cpu()@w.double().cpu().t()).float()
        cpu=torch.allclose(actual.index_select(0,idx).float().cpu(),ref,atol=.02,rtol=.02)
        rows.append(dict(M=M,N=N,K=K,kind=kind,bitwise_equal=equal,cpu_reference_pass=cpu))
        (ROOT/'correctness.json').write_text(json.dumps(dict(complete=False,cases=rows),indent=2))
        assert equal and cpu,rows[-1]
(ROOT/'correctness.json').write_text(json.dumps(dict(complete=True,cases=rows,actual_import=candidate.__file__,sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),finite_tests_not_universal_proof=True),indent=2));print('18 cases passed',flush=True)
