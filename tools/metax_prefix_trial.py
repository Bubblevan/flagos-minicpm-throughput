"""One bounded native/candidate/native C500 trial. Diagnostic, never official score."""
import argparse, asyncio, datetime as dt, hashlib, json, os, random, shutil, signal
import socket, statistics, subprocess, tarfile, threading, time
from pathlib import Path
import aiohttp
import requests
BASE='13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0'
ATTN='vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'
URL='http://127.0.0.1:9031'

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
def run(*args,**kw):return subprocess.check_output(list(args),text=True,timeout=40,**kw).strip()
def stop_owned(server):
    import psutil
    children=[]
    try:children=psutil.Process(server.pid).children(recursive=True)
    except psutil.NoSuchProcess:pass
    if server.poll() is None:
        os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
    remaining=[]
    for child in children:
        try:
            if child.is_running() and child.status()!=psutil.STATUS_ZOMBIE:
                child.kill();remaining.append(child)
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(remaining,timeout=10)
    alive=[]
    for child in remaining:
        try:
            if child.is_running() and child.status()!=psutil.STATUS_ZOMBIE:alive.append(child.pid)
        except psutil.NoSuchProcess:pass
    return dict(server_pid=server.pid,exit_code=server.returncode,owned_survivors=alive)
async def workload(seed,n,length,output,concurrency=64):
    rng=random.Random(seed);prompts=[[rng.randrange(1000,100000) for _ in range(length)] for _ in range(n)]
    rows=[];sem=asyncio.Semaphore(concurrency);started=time.monotonic()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
        async def one(i,prompt):
            async with sem:
                start=time.monotonic();first=None;usage=None;finish=None
                body=dict(model='minicpm',prompt=prompt,max_tokens=output,ignore_eos=True,stream=True,
                          stream_options={'include_usage':True})
                async with client.post(URL+'/v1/completions',json=body) as response:
                    assert response.status==200,(response.status,(await response.text())[:300])
                    async for line in response.content:
                        if not line.startswith(b'data: '):continue
                        raw=line[6:].strip()
                        if raw==b'[DONE]':continue
                        record=json.loads(raw)
                        for choice in record.get('choices',[]):
                            if choice.get('text') and first is None:first=time.monotonic()-start
                            finish=choice.get('finish_reason') or finish
                        usage=record.get('usage') or usage
                assert first is not None and finish=='length' and usage
                assert usage['prompt_tokens']==length and usage['completion_tokens']==output
                rows.append(dict(index=i,ttft_s=first,duration_s=time.monotonic()-start,usage=usage,
                    finish_reason=finish,prompt_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()))
        await asyncio.gather(*(one(i,prompt) for i,prompt in enumerate(prompts)))
    elapsed=time.monotonic()-started
    assert len(rows)==n
    return dict(seed=seed,requests=n,input_len=length,output_len=output,concurrency=concurrency,
                elapsed_s=elapsed,total_tokens_per_second=n*(length+output)/elapsed,
                mean_ttft_s=statistics.mean(r['ttft_s'] for r in rows),rows=sorted(rows,key=lambda r:r['index']),official_score=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--micro-root',type=Path,required=True);p.add_argument('--seconds',type=int,default=2400)
    a=p.parse_args();assert 600<=a.seconds<=3300
    micro=json.loads((a.micro_root/'metadata-micro.json').read_text())
    assert micro['stage']=='micro_complete' and all(c['metadata_bitwise_equal'] and c['attention_bitwise_equal'] for c in micro['cases'])
    manifest=json.loads((Path(__file__).resolve().parents[1]/'experiments/metax_prefix/manifest.json').read_text())
    original=Path('/workspace/vllm-plugin-FL')
    assert run('git','-C',str(original),'rev-parse','HEAD')==BASE
    assert not run('git','-C',str(original),'status','--porcelain')
    assert hashlib.sha256((original/ATTN).read_bytes()).hexdigest()==manifest['native_sha256']
    assert run('git','-C','/workspace/FlagGems','rev-parse','HEAD')=='a7620cc191a0b42e040194622c5758b22a7a25dc'
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',9031))!=0,'Port already occupied; inspect live tasks'
    a.root.mkdir(parents=True,exist_ok=False);deadline=time.time()+a.seconds
    def state(stage,**data):
        row=dict(stage=stage,at=dt.datetime.now(dt.timezone.utc).isoformat(),**data)
        save(a.root/'status.json',row);print(json.dumps(row),flush=True)
    def alarm(signum,frame):raise TimeoutError('Bounded C500 deadline reached')
    signal.signal(signal.SIGTERM,alarm);signal.signal(signal.SIGALRM,alarm);signal.alarm(a.seconds)
    sources=Path('/data/tokenforge-sources')/a.root.name;sources.mkdir(parents=True,exist_ok=False)
    native=sources/'native';candidate=sources/'candidate'
    shutil.copytree(original,native,ignore=shutil.ignore_patterns('.git','__pycache__','build','dist','*.egg-info'))
    shutil.copytree(native,candidate)
    patch=Path(__file__).resolve().parents[1]/'experiments/metax_prefix/plugin.patch'
    subprocess.run(['git','-C',str(candidate),'apply','--check',str(patch)],check=True)
    subprocess.run(['git','-C',str(candidate),'apply',str(patch)],check=True)
    assert hashlib.sha256((candidate/ATTN).read_bytes()).hexdigest()==manifest['candidate_sha256']
    shutil.copy2(patch,a.root/'plugin.patch');shutil.copy2(Path(__file__),a.root/'trial-controller.py')
    shutil.copy2(a.micro_root/'preflight.json',a.root/'initial-preflight.json')
    shutil.copy2(a.micro_root/'metadata-micro.json',a.root/'metadata-micro.json')
    save(a.root/'plan.json',dict(seconds=a.seconds,deadline_epoch=deadline,order=['native_a1','candidate_b','native_a2'],
        original_serve_unchanged_except_inactive_diagnostic_profiler=True,official_score=False,
        short_go=dict(mean_4k_gain=.04,mean_16k_gain=.02,per_seed_positive=True,max_ttft_ratio=1.01,max_native_drift=.03),
        limitations='Short 256/128 output workload, sequential A/B/A, no official score or accuracy claim.'))
    serve=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--served-model-name','minicpm',
           '--gpu-memory-utilization','0.85','--max-model-len','131072']
    results={};cleanups=[];server=None
    try:
        for label,source in [('native_a1',native),('candidate_b',candidate),('native_a2',native)]:
            if time.time()>deadline-420:raise TimeoutError('Insufficient remaining startup budget')
            target=a.root/label;target.mkdir()
            cache=Path('/data/tokenforge-caches')/a.root.name/label;cache.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONPATH=str(source),
                     VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),
                     TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
            profile=dict(profiler='torch',torch_profiler_dir=str(target/'profiles'),torch_profiler_with_stack=False,
                         torch_profiler_record_shapes=True,ignore_frontend=True,max_iterations=8)
            argv=serve+['--profiler-config',json.dumps(profile)]
            imported=run('python','-c','import vllm_fl,flag_gems;print(vllm_fl.__file__);print(flag_gems.__file__)',env=env)
            assert str(source) in imported,imported
            with (target/'server.log').open('w') as log:
                server=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(target/'runtime.json',dict(argv=argv,imported=imported,source=str(source),cache=str(cache),server_pid=server.pid,
                attention_sha256=hashlib.sha256((source/ATTN).read_bytes()).hexdigest()))
            state(label+'_starting',server_pid=server.pid)
            try:
                ready_by=min(time.time()+600,deadline-60)
                while time.time()<ready_by:
                    if server.poll() is not None:raise RuntimeError(label+' server exited; read server.log')
                    try:
                        if requests.get(URL+'/health',timeout=2).status_code==200:break
                    except requests.RequestException:pass
                    time.sleep(3)
                else:raise TimeoutError(label+' readiness timeout')
                all_rows={}
                for shape,n,length,output in [('4k',64,4096,256),('16k',16,16384,128)]:
                    measurements=[]
                    for phase,seed in [('warmup',5510),('measure_1',5511),('measure_2',5512)]:
                        state(label+'_'+shape+'_'+phase)
                        result=asyncio.run(asyncio.wait_for(workload(seed,n,length,output),timeout=min(300,deadline-time.time()-30)))
                        save(target/(shape+'-'+phase+'.json'),result)
                        if phase!='warmup':measurements.append(result)
                    all_rows[shape]=measurements
                results[label]=all_rows
                if label=='native_a1':
                    state('native_model_profile')
                    requests.post(URL+'/start_profile',timeout=30).raise_for_status()
                    try:
                        record=asyncio.run(asyncio.wait_for(workload(5590,8,4096,64),timeout=min(180,deadline-time.time()-30)))
                        save(target/'profile-workload.json',record)
                    finally:requests.post(URL+'/stop_profile',timeout=45).raise_for_status()
            finally:
                clean=stop_owned(server);server=None;cleanups.append(dict(label=label,**clean));save(target/'cleanup.json',clean)
                assert not clean['owned_survivors'],clean
        gains={};ttft={};drift={};per_seed_positive=True
        for shape in ['4k','16k']:
            aa=results['native_a1'][shape];bb=results['candidate_b'][shape];a2=results['native_a2'][shape]
            for x,y,z in zip(aa,bb,a2):
                assert [r['prompt_sha256'] for r in x['rows']]==[r['prompt_sha256'] for r in y['rows']]==[r['prompt_sha256'] for r in z['rows']]
                per_seed_positive &= y['total_tokens_per_second']>statistics.mean([x['total_tokens_per_second'],z['total_tokens_per_second']])
            base=statistics.mean(r['total_tokens_per_second'] for r in aa+a2)
            gains[shape]=statistics.mean(r['total_tokens_per_second'] for r in bb)/base-1
            ttft[shape]=statistics.mean(r['mean_ttft_s'] for r in bb)/statistics.mean(r['mean_ttft_s'] for r in aa+a2)
            drift[shape]=statistics.mean(r['total_tokens_per_second'] for r in a2)/statistics.mean(r['total_tokens_per_second'] for r in aa)-1
        go=gains['4k']>=.04 and gains['16k']>=.02 and per_seed_positive and max(ttft.values())<=1.01 and max(abs(v) for v in drift.values())<=.03
        summary=dict(complete=True,official_score=False,relative_gains=gains,ttft_ratios=ttft,native_drift=drift,
                     per_seed_positive=bool(per_seed_positive),short_go=go,optimization_officially_validated=False)
        save(a.root/'short-summary.json',summary);state('short_complete',summary=summary)
    except BaseException as e:
        state('trial_failed_or_deadline',error=repr(e));raise
    finally:
        signal.alarm(0)
        if server is not None:cleanups.append(stop_owned(server))
        save(a.root/'cleanup.json',dict(checked_at=time.time(),conditions=cleanups))
        archive=a.root.with_suffix('.tar.gz');temp=archive.with_suffix('.tmp')
        with tarfile.open(temp,'w:gz') as tar:tar.add(a.root,arcname=a.root.name)
        temp.replace(archive)
        print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),size=archive.stat().st_size)),flush=True)
if __name__=='__main__':main()