"""Explicitly start full validation only after the preregistered short-trial gate."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import psutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');a=parser.parse_args()
    assert a.start,'Explicit --start is required; upload alone must not start a GPU workload'
    root=Path(__file__).resolve().parent
    reservation=json.loads((root/'reservation.json').read_text())
    deadline=reservation['deadline_epoch']
    assert deadline-time.time()>=4*3600,'Insufficient four-hour budget; do not squeeze or cross cutoff'
    short=json.loads((root/'short-comparison/summary.json').read_text())
    assert short['comparison_complete'] and short['go_full_validation']
    assert short['relative_gain']>=.07 and min(short['per_seed_relative_gain'])>=.05
    micro=json.loads((root/'mixed-kernels-gate8.json').read_text())
    assert micro['complete'] and micro['all_bitwise_equal']
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'A server remains active; inspect before starting'
    for process in psutil.process_iter(['pid','name','status','cmdline']):
        if process.info['status']==psutil.STATUS_ZOMBIE:continue
        tokens=process.info['cmdline'] or []
        is_api=any(Path(token).name=='vllm' for token in tokens[:3]) and 'serve' in tokens[1:4]
        is_core=(process.info['name'] or '').startswith('VLLM::Engine') or any(token.startswith('VLLM::EngineCore') for token in tokens[:1])
        assert not (is_api or is_core), 'An existing GPU model process remains; inspect first'
    source=Path('/workspace/tokenforge-sources/plugin-mixed-20261031-v2')
    files=['vllm_fl/platform.py','vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py']
    hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}
    assert hashes[files[1]]==micro['candidate_sha256']
    final=root/'formal-v2';final.mkdir(exist_ok=False)
    for name in ('mixed-kernels-gate8.json','run_official_accuracy.sh'):
        shutil.copy2(root/name,final/name)
    (final/'short-comparison.json').write_text(json.dumps(short,indent=2))
    manifest=dict(candidate_root=str(source),plugin_base_commit='13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0',
                  changes=hashes,is_git_checkout=False,revision='mixed-query-gate8',original_packages_unchanged=True)
    (final/'attention-candidate-manifest.json').write_text(json.dumps(manifest,indent=2))
    cache=Path('/workspace/tokenforge-caches/mixed_prefill_20261031_1900/formal-v2')
    environment=dict(VLLM_PLUGINS='fl',PYTHONPATH=str(source)+':/workspace/tokenforge-sources/FlagGems-a7620cc/src',
                     FLAGOS_OBSERVE_DISPATCH='0',VLLM_CACHE_ROOT=str(cache/'vllm'),
                     TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
    argv=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
          '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    pinned=dict(source_root='/workspace/tokenforge-sources/FlagGems-a7620cc',target_commit='a7620cc191a0b42e040194622c5758b22a7a25dc',
                target_tree='ddf3c7566aac7d23385a1b10c7c258091ef1d3a7',environment=environment,argv=argv,
                candidate_manifest=manifest,cutoff_epoch=deadline,short_control='v1',official_score_pending=True,
                early_stop_after_complete_4k_goal_failure=True,controller_sha256=hashlib.sha256((root/'full_mixed_validation.py').read_bytes()).hexdigest())
    probe_code="import json,pathlib,torch,triton,flag_gems,vllm_fl;pathlib.Path(__import__('sys').argv[1]).write_text(json.dumps(dict(torch=torch.__version__,triton=triton.__version__,flag_gems_file=flag_gems.__file__,plugin_file=vllm_fl.__file__,scope='Import probe under exact server environment, before model startup.'),indent=2))"
    with (final/'import-probe.log').open('w') as log:
        subprocess.run([sys.executable,'-c',probe_code,str(final/'import-proof.json')],env=dict(os.environ,**environment),
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    imports=json.loads((final/'import-proof.json').read_text())
    assert imports['flag_gems_file'].startswith('/workspace/tokenforge-sources/FlagGems-a7620cc/src/')
    assert imports['plugin_file'].startswith(str(source)+'/')
    pinned['import_probe']=imports
    (final/'optimized-runtime-environment.json').write_text(json.dumps(pinned,indent=2))
    env=dict(os.environ,FLAGOS_SESSION_ROOT=str(final),FLAGOS_PARENT_SESSION_ROOT=str(root),FLAGOS_SESSION_DEADLINE=str(deadline))
    with (final/'optimized-controller.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-u',str(root/'full_mixed_validation.py')],env=env,cwd='/workspace',
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'full-controller.pid').write_text(str(child.pid));(final/'optimized-controller.pid').write_text(str(child.pid))
    print(json.dumps(dict(started=True,controller_pid=child.pid,formal_root=str(final),cutoff=reservation['gpu_cutoff'])))


if __name__=='__main__':main()
