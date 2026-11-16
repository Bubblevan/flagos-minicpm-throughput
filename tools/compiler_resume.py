"""Explicit continuation after a slow single-stream official wheel download."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import compiler_session as s


def main():
    assert s.START<=time.time()<s.END-4*3600
    assert (s.ROOT/'old/result.json').is_file() and not (s.ROOT/'new').exists()
    base=Path(__file__).resolve().parent
    plugin=s.SOURCES/'vllm-plugin-FL';gems=Path('/workspace/tokenforge-sources/FlagGems-a7620cc/src')
    candidate=plugin/'vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==s.V2_SHA
    env=dict(__import__('os').environ,PYTHONPATH=f'{plugin}:{gems}:{base}',
             PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6')
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('stop requested')))
    wheel=s.SOURCES/'wheels/flagtree-0.7.0+iluvatar3.6-cp312-cp312-linux_x86_64.whl'
    s.run('official-wheel-range-download',[sys.executable,str(base/'download_official_compiler.py'),str(wheel)],env,1150)
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='af6a7c9eddf167441fd872c3e6d673e9da572608dee79dba655b0775db740176'
    target=s.SOURCES/'compiler-0.7.0';assert not target.exists()
    s.write('wheel.json',dict(file=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),size=wheel.stat().st_size,
                             index_sha_verified=True,source='official FlagOS package index'))
    s.run('install-isolated-wheel',[sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(target),str(wheel)],env,300)
    newer=dict(env,PYTHONPATH=f'{target}:'+env['PYTHONPATH'],TRITON_CACHE_DIR=str(s.CACHE/'new/triton'),
               VLLM_CACHE_ROOT=str(s.CACHE/'new/vllm'),TORCHINDUCTOR_CACHE_DIR=str(s.CACHE/'new/inductor'),CUDA_CACHE_PATH=str(s.CACHE/'new/cuda'))
    s.run('new-compiler-probe',[sys.executable,str(base/'compiler_attention_probe.py'),'--inputs',str(s.ROOT/'inputs'),
        '--out',str(s.ROOT/'new'),'--candidate',str(candidate),'--deadline',str(min(s.END,time.time()+1000))],newer,1000)
    old=json.loads((s.ROOT/'old/result.json').read_text());new=json.loads((s.ROOT/'new/result.json').read_text())
    assert new['flagtree']=='0.7.0+iluvatar3.6' and new['triton_file'].startswith(str(target)+'/')
    summary=dict(complete=True,old_compiler=old['flagtree'],new_compiler=new['flagtree'],
        old_reference_pass=old['all_reference_pass'],new_reference_pass=new['all_reference_pass'],
        old_stable=old['all_stable'],new_stable=new['all_stable'],new_v2_bitwise_native=new['all_v2_bitwise_native'],
        numerical_gate=bool(new['all_reference_pass'] and new['all_stable'] and new['all_v2_bitwise_native']),
        formal_started=False,official_score=False,measured_fix=bool(not old['all_reference_pass'] and new['all_reference_pass']))
    s.write('compiler-comparison.json',summary);s.write('status.json',dict(stage='diagnostic_complete',updated=time.time(),**summary));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        s.stop(s.current);s.write('status.json',dict(stage='diagnostic_failed',error=type(e).__name__,message=str(e),updated=time.time()));raise
