"""Separate-service native/candidate/native GEMM test; never an official score."""
import datetime as dt,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import requests
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'tools'))
from metax_prefix_trial import stop_owned
ROOT=Path('/data/tokenforge-artifacts/metax_mm_model_20261117_1530')
PLUGIN=Path('/data/tokenforge-sources/metax_20261117_0118/candidate')
GEMS=Path('/data/tokenforge-sources/TokenForge-metax-mm-20261117/FlagGems')
END=dt.datetime.fromisoformat('2026-11-17T15:59:00+08:00').timestamp()
server=None;client=None
def write(n,data):
    p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def state(stage,**kw):write('status.json',dict(stage=stage,at=time.time(),deadline=END,**kw));print(stage,flush=True)
def main():
    global server,client
    proof=json.loads(Path('/data/tokenforge-artifacts/metax_mm_correctness_20261117_1525/correctness.json').read_text());assert proof['complete'] and len(proof['cases'])==18
    assert END-time.time()>18*60 and not ROOT.exists()
    assert 'no process found' in subprocess.check_output(['mx-smi'],text=True).lower()
    ROOT.mkdir(parents=True);shutil.copy2(__file__,ROOT/'controller.py')
    write('correctness.json',proof);write('gems-source-manifest.json',json.loads(Path('/data/tokenforge-artifacts/metax_mm_correctness_20261117_1525/source-manifest.json').read_text()))
    write('plan.json',dict(official_score=False,hypothesis='Prefill trace73.2852% native NT GEMM. Change only three verified M2048 BF16 NT tilings; retain O projection, all other shapes and original arithmetic.',kernel_ast_unchanged=True,prior_micro_weighted_mm_time_reduction=.09540919489,prior_micro_not_e2e=True,sampler='Original frozen native filter in both conditions',plugin=str(PLUGIN),order=['native_a1','candidate_b','native_a2'],warmup=[4096,1024,64,64],measured=[4096,1024,64,128],gates=dict(min_gain=.03,max_native_drift=.03,max_ttft_ratio=1.0),deadline=END))
    bench=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py');spec=importlib.util.spec_from_file_location('original_bench',bench);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    rows={}
    for label,gemsrc in [('native_a1',Path('/workspace/FlagGems/src')),('candidate_b',GEMS/'src'),('native_a2',Path('/workspace/FlagGems/src'))]:
        assert END-time.time()>5*60
        cache=Path('/data/tokenforge-caches')/ROOT.name/label
        for n in ['triton','vllm','inductor']:
            prior=Path('/data/tokenforge-caches/metax_prefill_warm_20261117_1450')/n
            if prior.exists():shutil.copytree(prior,cache/n)
        env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONPATH=str(PLUGIN)+':'+str(gemsrc),TORCHINDUCTOR_COMPILE_THREADS='1',FLAGOS_SESSION_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',VLLM_SERVER_DEV_MODE='1',VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
        for key in ['FLAGOS_METAX_TOPP_MODE','FLAGOS_METAX_TOPP_DIAGNOSTIC_CONTROL']:env.pop(key,None)
        imported=subprocess.check_output(['/opt/conda/bin/python','-c','import importlib,json,hashlib;from pathlib import Path;m=importlib.import_module("flag_gems.runtime.backend._metax.ops.mm");p=Path(m.__file__);print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'],env=env,text=True)
        write(label+'-import.json',json.loads(imported.strip().splitlines()[-1]))
        assert str(gemsrc) in json.loads(imported.strip().splitlines()[-1])['path']
        argv=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
        write(label+'-argv.json',dict(argv=argv,PYTHONPATH=env['PYTHONPATH']))
        with (ROOT/(label+'-server.log')).open('w') as f:server=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd='/workspace')
        state(label+'_starting',server_pid=server.pid)
        limit=min(time.time()+420,END-240)
        while time.time()<limit:
            assert server.poll() is None,'Server exited'
            try:
                if requests.get('http://127.0.0.1:9031/health',timeout=2).status_code==200:break
            except requests.RequestException:pass
            time.sleep(2)
        else:raise TimeoutError('Readiness budget')
        clientenv=dict(env,VLLM_PLUGINS='',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
        def run(suffix,n):
            global client
            requests.post('http://127.0.0.1:9031/reset_prefix_cache',timeout=20).raise_for_status();state(label+'_'+suffix,server_pid=server.pid)
            cmd=['vllm','bench','serve','--backend','vllm','--model','minicpm','--tokenizer','/workspace/MiniCPM5-2B','--endpoint','/v1/completions','--host','localhost','--port','9031','--dataset-name','random','--ignore-eos','--random-input-len','4096','--random-output-len','1024','--max-concurrency','64','--num-prompts',str(n)]
            p=ROOT/(label+'-'+suffix+'.log')
            with p.open('w') as f:client=subprocess.Popen(cmd,env=clientenv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:code=client.wait(timeout=min(260,END-time.time()-30))
            finally:
                if client.poll() is None:stop_owned(client)
            assert code==0
            result=original.extract_metrics(p.read_text());assert result['successful_requests']==n and result['total_input_tokens']==4096*n and result['total_output_tokens']==1024*n
            write(label+'-'+suffix+'.json',result);return result
        run('warmup',64);rows[label]=run('measure',128)
        write(label+'-gpu.txt.json',dict(output=subprocess.check_output(['mx-smi'],text=True)))
        write(label+'-cleanup.json',stop_owned(server));server=None
    a,b,c=[rows[k] for k in ['native_a1','candidate_b','native_a2']]
    gain=b['total_token_throughput']/((a['total_token_throughput']+c['total_token_throughput'])/2)-1;drift=c['total_token_throughput']/a['total_token_throughput']-1;ttft=b['mean_ttft_ms']/((a['mean_ttft_ms']+c['mean_ttft_ms'])/2)
    write('summary.json',dict(complete=True,official_score=False,gain=gain,native_drift=drift,ttft_ratio=ttft,short_go=gain>=.03 and abs(drift)<=.03 and ttft<=1,rows=rows))
if __name__=='__main__':
    def interrupted(*_):raise TimeoutError('15:59 hard cutoff')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(max(1,int(END-time.time())))
    failure=None
    try:main()
    except BaseException as e:
        failure=dict(error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        if ROOT.exists():write('failure.json',failure)
    finally:
        signal.alarm(0)
        if client and client.poll() is None:stop_owned(client)
        cleanup=stop_owned(server) if server else {}
        if ROOT.exists():write('cleanup.json',cleanup);state('stopped',failure=failure)
