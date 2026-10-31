"""Build a complete, traceable Iluvatar v2 source candidate; no GPU/import required."""
import argparse, ast, hashlib, io, json, os, subprocess, types, zipfile
from pathlib import Path
BASE = '13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0'
ADAPTER = 'vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py'
ADAPTER_SHA = '5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325'
def sha(data): return hashlib.sha256(data).hexdigest()
def git(root, *args): return subprocess.check_output(['git','-C',str(root),*args])
def contract_smoke(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QueryTiledKernel')
    env={'torch':types.SimpleNamespace(bfloat16='bf16'),'Counter':__import__('collections').Counter,
         'os':os,'time':__import__('time'),'record_runtime':lambda *a,**k:None}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'<adapter-contract>','exec'),env)
    class Kernel:
        def __getitem__(self,grid):
            return lambda *args,**kw:(grid,kw)
    query=types.SimpleNamespace(dtype='bf16',shape=(2048,16,128))
    args=dict(query_ptr=query,num_seqs=64,IS_3D=False,HEAD_SIZE=128,num_queries_per_kv=8,
              TILE_SIZE=32,KV_QUANT_MODE=0,USE_CAUSAL=True,SLIDING_WINDOW=0,CHUNK_LOOKBACK=-1,
              num_query_heads=16,BLOCK_M=16,BLOCK_Q=2,num_warps=4)
    kernel=env['QueryTiledKernel'](Kernel())
    original_grid=(1088,2)
    grid,kw=kernel[original_grid](**args)
    assert grid==(320,2) and kw['BLOCK_M']==64 and kw['BLOCK_Q']==8
    assert kw['TILE_SIZE']==32 and kw['num_warps']==4
    cases=1
    changes=[{'query_ptr':types.SimpleNamespace(dtype='bf16',shape=(64,16,128))},
             {'query_ptr':types.SimpleNamespace(dtype='bf16',shape=(256,16,128))},
             {'query_ptr':types.SimpleNamespace(dtype='fp16',shape=(2048,16,128))},
             {'IS_3D':True},{'HEAD_SIZE':64},{'num_queries_per_kv':4},{'TILE_SIZE':64},
             {'KV_QUANT_MODE':1},{'USE_CAUSAL':False},{'SLIDING_WINDOW':128},
             {'CHUNK_LOOKBACK':0}]
    changes += [{flag:True} for flag in ['USE_TD','USE_MM_PREFIX','USE_SINKS','USE_ALIBI_SLOPES',
                                       'USE_QQ_BIAS','USE_SOFTCAP','USE_PER_SEQ_CAUSAL','USE_FP8']]
    for change in changes:
        grid,kw=kernel[original_grid](**dict(args,**change))
        assert grid==original_grid and kw['BLOCK_M']==16 and kw['BLOCK_Q']==2,change
        cases+=1
    return {'cases':cases,'pass':True,'scope':'CPU launch/fallback contract only; not a GPU or mathematical test'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    assert git(a.upstream,'rev-parse','HEAD').decode().strip()==BASE
    assert not git(a.upstream,'status','--porcelain'), 'Upstream must be clean'
    assert not a.out_dir.exists(), 'Never overwrite a release candidate'
    a.out_dir.mkdir(parents=True)
    stage=a.out_dir/'vllm-plugin-FL';stage.mkdir()
    exported=zipfile.ZipFile(io.BytesIO(git(a.upstream,'archive','--format=zip',BASE)))
    baseline={}
    for entry in exported.infolist():
        if entry.is_dir():continue
        path=Path(entry.filename)
        assert not path.is_absolute() and '..' not in path.parts
        data=exported.read(entry);dest=stage/path;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data);baseline[entry.filename]=sha(data)
    patch=repo/'experiments/attention_tiling_mixed/plugin.patch'
    for mode in ['--check',None]:
        cmd=['git','-C',str(stage),'apply']+([mode] if mode else [])+[str(patch)]
        subprocess.run(cmd,check=True,capture_output=True)
    files={p.relative_to(stage).as_posix():sha(p.read_bytes()) for p in stage.rglob('*') if p.is_file()}
    changed=sorted(n for n,h in files.items() if baseline.get(n)!=h)
    assert changed==sorted(['vllm_fl/platform.py',ADAPTER]),changed
    assert files[ADAPTER]==ADAPTER_SHA, files[ADAPTER]
    assert not(set(baseline)-set(files))
    count=0
    for n in files:
        if n.endswith('.py'):
            ast.parse((stage/n).read_bytes(),filename=n);count+=1
    smoke=contract_smoke((stage/ADAPTER).read_text())
    manifest=dict(upstream_url='https://github.com/flagos-ai/vllm-plugin-FL',base_commit=BASE,
                  base_tree=git(a.upstream,'rev-parse',BASE+'^{tree}').decode().strip(),
                  patch_sha256=sha(patch.read_bytes()),changed_files=changed,files=files,
                  source_files=len(files),python_syntax_files=count,contract_smoke=smoke,
                  full_fresh_container_gpu_restore_verified=False,
                  numerical_evidence='results/2026-10-31-iluvatar-mixed-v2; packaging does not add GPU evidence')
    (a.out_dir/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (a.out_dir/'RECOVERY.md').write_text((repo/'docs/iluvatar-candidate.md').read_text(encoding='utf-8'),encoding='utf-8')
    archive=a.out_dir/'FlagOS-Iluvatar-v2-full-source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for pth in sorted(a.out_dir.rglob('*')):
            if not pth.is_file() or pth==archive:continue
            item=zipfile.ZipInfo(pth.relative_to(a.out_dir).as_posix(),(2026,10,31,22,49,16))
            item.compress_type=zipfile.ZIP_DEFLATED;item.create_system=3
            item.external_attr=(0o100755 if pth.suffix=='.sh' else 0o100644)<<16
            z.writestr(item,pth.read_bytes())
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    result=dict(archive=str(archive),sha256=sha(archive.read_bytes()),size_bytes=archive.stat().st_size,
                source_files=len(files),python_syntax_files=count,changed_files=changed,contract_smoke=smoke,
                full_fresh_container_gpu_restore_verified=False)
    (a.out_dir/'build-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()