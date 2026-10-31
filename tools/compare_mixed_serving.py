"""Matched v1/v2 serving trial; diagnostic only, original fixed server arguments."""
import asyncio
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time

import aiohttp
import psutil
import requests

ROOT=Path(__file__).resolve().parent
DEADLINE=json.loads((ROOT/'reservation.json').read_text())['deadline_epoch']
URL='http://127.0.0.1:9031'
SERVE=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def state(stage,**details):
    row=dict(stage=stage,at=dt.datetime.now(dt.timezone.utc).isoformat(),**details)
    save(ROOT/'status.json',row)
    print(json.dumps(row),flush=True)


def stop_server(server):
    if server.poll() is None:
        os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid,signal.SIGKILL);server.wait()


async def workload(path,seed):
    length=4096;output=512;n=128;concurrency=64
    rng=random.Random(seed)
    prompts=[[rng.randrange(1000,100000) for _ in range(length)] for _ in range(n)]
    semaphore=asyncio.Semaphore(concurrency)
    timeout=aiohttp.ClientTimeout(total=600)
    rows=[]
    started=time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async def one(index,prompt):
            async with semaphore:
                begin=time.monotonic();first=None;usage=None;finish=None
                body=dict(model='minicpm',prompt=prompt,max_tokens=output,ignore_eos=True,
                          stream=True,stream_options={'include_usage':True})
                async with client.post(URL+'/v1/completions',json=body) as response:
                    if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:200]}')
                    async for line in response.content:
                        line=line.decode().strip()
                        if not line.startswith('data: '):continue
                        content=line[6:]
                        if content=='[DONE]':continue
                        chunk=json.loads(content)
                        for choice in chunk.get('choices',[]):
                            if choice.get('text') and first is None:first=time.monotonic()-begin
                            if choice.get('finish_reason'):finish=choice['finish_reason']
                        if chunk.get('usage'):usage=chunk['usage']
                assert first is not None and usage is not None
                assert usage['prompt_tokens']==length and usage['completion_tokens']==output
                assert usage['total_tokens']==length+output and finish=='length'
                rows.append(dict(index=index,ttft_s=first,elapsed_s=time.monotonic()-begin,
                                 usage=usage,finish_reason=finish,
                                 prompt_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()))
        await asyncio.gather(*(one(i,p) for i,p in enumerate(prompts)))
    elapsed=time.monotonic()-started
    assert len(rows)==n
    result=dict(official_score=False,seed=seed,requests=n,concurrency=concurrency,input_tokens_each=length,
                output_tokens_each=output,successful_requests=len(rows),elapsed_s=elapsed,
                total_tokens_per_second=n*(length+output)/elapsed,mean_ttft_s=statistics.mean(x['ttft_s'] for x in rows),
                rows=sorted(rows,key=lambda x:x['index']))
    save(path,result)
    return result


def run_condition(label,source):
    case=ROOT/'short-comparison'/label;case.mkdir(parents=True,exist_ok=False)
    with __import__('socket').socket() as sock:
        assert sock.connect_ex(('127.0.0.1',9031))!=0,'Existing service; do not duplicate'
    cache=Path('/workspace/tokenforge-caches/mixed_prefill_20261031_1900')/label
    env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONPATH=source+':/workspace/tokenforge-sources/FlagGems-a7620cc/src',
             FLAGOS_SESSION_ROOT=str(case),FLAGOS_OBSERVE_DISPATCH='1',
             VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
    module=Path(source)/'vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py'
    save(case/'environment.json',dict(argv=SERVE,source=source,adapter_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
         environment={k:env[k] for k in ('VLLM_PLUGINS','PYTHONPATH','FLAGOS_SESSION_ROOT','FLAGOS_OBSERVE_DISPATCH','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}))
    with (case/'server.log').open('w') as log:
        server=subprocess.Popen(SERVE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (case/'server.pid').write_text(str(server.pid))
    try:
        state('mixed_short_'+label+'_starting',server_pid=server.pid)
        ready_by=min(time.time()+600,PHASE_END)
        while time.time()<ready_by:
            if server.poll() is not None:raise RuntimeError(f'{label} server exited during initialization')
            try:
                if requests.get(URL+'/health',timeout=3).status_code==200:break
            except requests.RequestException:pass
            time.sleep(3)
        else:raise TimeoutError('Server readiness deadline')
        assert any(x['id']=='minicpm' for x in requests.get(URL+'/v1/models',timeout=10).json()['data'])
        measurements=[]
        for name,seed in [('warmup',4510),('measure_a',4511),('measure_b',4512)]:
            state('mixed_short_'+label+'_'+name,server_pid=server.pid)
            measurements.append(asyncio.run(workload(case/(name+'.json'),seed)))
        child_ids={p.pid for p in psutil.Process(server.pid).children(recursive=True)}
        records=[json.loads(p.read_text()) for p in case.glob('attention-tiling-runtime-*.json')]
        assert any(x['pid'] in child_ids and x['stage']=='optimized_kernel_launched' and x['source_file'].startswith(source+'/') for x in records), 'Missing actual worker launch evidence'
        result=dict(label=label,source=source,worker_ids=sorted(child_ids),measurements=measurements,
                    mean_total_tokens_per_second=statistics.mean(x['total_tokens_per_second'] for x in measurements[1:]),
                    mean_ttft_s=statistics.mean(x['mean_ttft_s'] for x in measurements[1:]))
        save(case/'summary.json',result)
        return result
    finally:
        stop_server(server)
        save(case/'cleanup.json',dict(server_pid=server.pid,server_exit_code=server.returncode,checked_at=time.time()))


def main():
    proof=json.loads((ROOT/'mixed-kernels-gate8.json').read_text())
    assert proof['complete'] and proof['all_bitwise_equal']
    assert len(proof['cases'])>=12
    source=Path('/workspace/tokenforge-sources/plugin-mixed-20261031-v2')
    relative=Path('vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py')
    candidate=(ROOT/'mixed_attention_gate8.py').read_bytes()
    assert hashlib.sha256(candidate).hexdigest()==proof['candidate_sha256']
    assert (source/relative).read_text()==(ROOT/'mixed_attention_tiles.py').read_text(), 'Unexpected preexisting source revision'
    (source/relative).write_bytes(candidate)
    save(ROOT/'candidate-gate8-manifest.json',dict(source=str(source),adapter_sha256=hashlib.sha256(candidate).hexdigest(),
         v1_original_unchanged=True,reason='Gate 4 regressed on q256/64; gate 8 retains fallback. Exact gate-8 source passed twelve mixed differential cases.'))
    results=[]
    for label,path in [('v1','/workspace/tokenforge-sources/plugin-mixed-20261031-v1_observe'),('v2',str(source))]:
        results.append(run_condition(label,path))
    for left,right in zip(results[0]['measurements'],results[1]['measurements']):
        assert left['seed']==right['seed']
        assert [x['prompt_sha256'] for x in left['rows']]==[x['prompt_sha256'] for x in right['rows']]
    relative_gain=results[1]['mean_total_tokens_per_second']/results[0]['mean_total_tokens_per_second']-1
    per_seed=[b['total_tokens_per_second']/a['total_tokens_per_second']-1 for a,b in zip(results[0]['measurements'][1:],results[1]['measurements'][1:])]
    go=relative_gain>=.07 and min(per_seed)>=.05 and results[1]['mean_ttft_s']<=results[0]['mean_ttft_s']*1.01
    summary=dict(official_score=False,comparison_complete=True,results=results,relative_gain=relative_gain,
                 per_seed_relative_gain=per_seed,go_full_validation=go,
                 preregistered_go_criteria=dict(mean_gain_min=.07,each_seed_gain_min=.05,ttft_max_ratio=1.01),
                 observation_scope='Both conditions record Python dispatch equally; graph replay is not counted as a Python dispatch. Separate new caches, full-shape warmup; fixed v1 then v2 order remains a limitation.')
    save(ROOT/'short-comparison/summary.json',summary)
    state('mixed_short_complete_needs_analysis',relative_gain=relative_gain,go_full_validation=go)


PHASE_END=min(DEADLINE,time.time()+2400)
signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(SystemExit('Session cutoff')))
watchdog=threading.Timer(max(1,PHASE_END-time.time()),lambda:os.kill(os.getpid(),signal.SIGTERM))
watchdog.daemon=True
if __name__=='__main__':
    (ROOT/'short-controller.pid').write_text(str(os.getpid()))
    watchdog.start()
    try:main()
    except Exception as error:
        state('mixed_short_failed',error=repr(error));raise
    finally:watchdog.cancel()
