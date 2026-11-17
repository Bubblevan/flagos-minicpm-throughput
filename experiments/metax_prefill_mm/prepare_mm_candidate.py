"""Create a reversible complete FlagGems source candidate, then test its public API."""
import ast,hashlib,importlib,json,os,subprocess,tarfile,time
from pathlib import Path
BASE=Path('/workspace/FlagGems')
DEST=Path('/data/tokenforge-sources/TokenForge-metax-mm-20261117/FlagGems')
ROOT=Path('/data/tokenforge-artifacts/metax_mm_correctness_20261117_1525')
assert not DEST.exists() and not ROOT.exists()
assert subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip()=='a7620cc191a0b42e040194622c5758b22a7a25dc'
assert not subprocess.check_output(['git','-C',str(BASE),'status','--porcelain'],text=True).strip()
ROOT.mkdir(parents=True);DEST.mkdir(parents=True)
archive=ROOT/'official-base.tar'
with archive.open('wb') as f:subprocess.run(['git','-C',str(BASE),'archive','HEAD'],stdout=f,check=True)
with tarfile.open(archive) as t:t.extractall(DEST,filter='data')
p=DEST/'src/flag_gems/runtime/backend/_metax/ops/mm.py';original=p.read_text()
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2d4f80658012c1442be3fc8beee0fd1aae6f0e3f65092160fe9a4db369d77cec'
tree=ast.parse(original);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mm_kernel_nt')
kernel=ast.get_source_segment(original,fn).replace('def mm_kernel_nt(', 'def _tokenforge_mm_nt(',1)
extra='''
# Verified dense BF16 NT prefill shapes only. Unsupported shapes/layouts/dtypes
# retain the original tuner and launch. The native arithmetic body is unchanged.
_TOKENFORGE_NT_TILES = {(2560, 2048): (256, 128, 32),
                        (12288, 2048): (128, 256, 32),
                        (2048, 6144): (128, 128, 64)}

'''+ '@triton.jit\n'+kernel+'\n\n'
guard='''    tile = _TOKENFORGE_NT_TILES.get((N, K)) if M == 2048 else None
    if (tile is not None and a.dtype == b.dtype == c.dtype == torch.bfloat16
            and a.ndim == b.ndim == c.ndim == 2 and a.is_contiguous()
            and b.stride(0) == 1 and b.stride(1) == K and c.is_contiguous()
            and a.shape == (M, K) and b.shape == (K, N) and c.shape == (M, N)):
        bm, bn, bk = tile
        with torch_device_fn.device(a.device):
            _tokenforge_mm_nt[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
                a, b, c, M, N, K, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                GROUP_M=8, EVEN_M=True, EVEN_N=True, EVEN_K=True,
                num_warps=8, num_stages=2, pipeline="basic")
        return c
'''
patched=original.replace('def general_mm_nt(a, b, c, M, N, K):\n',extra+'def general_mm_nt(a, b, c, M, N, K):\n'+guard,1)
assert patched!=original;ast.parse(patched);p.write_text(patched)
newtree=ast.parse(patched);newfn=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='_tokenforge_mm_nt')
assert ast.dump(ast.Module(body=fn.body,type_ignores=[]))==ast.dump(ast.Module(body=newfn.body,type_ignores=[]))
files={f.relative_to(DEST).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(DEST.rglob('*')) if f.is_file()}
(ROOT/'source-manifest.json').write_text(json.dumps(dict(base_commit='a7620cc191a0b42e040194622c5758b22a7a25dc',source=str(DEST),kernel_ast_unchanged=True,modified_file='src/flag_gems/runtime/backend/_metax/ops/mm.py',files=files),indent=2))
# Numerical test runs in a fresh interpreter to prove actual isolated imports.
os.environ['PYTHONPATH']=str(DEST/'src')
subprocess.run(['/opt/conda/bin/python',str(Path(__file__).with_name('verify_mm_candidate.py'))],env=os.environ,check=True)
print('complete isolated FlagGems prepared and numerical gate passed',flush=True)
