"""One new complete acceptance. Requires a passed short gate and explicit time budget."""
import csv,datetime as dt,hashlib,json,os,shutil,signal,socket,subprocess,sys,threading,time,traceback,zipfile
from pathlib import Path
import requests
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'tools'))
import metax_campaign as m
from validate_accuracy_artifacts import audit
ROOT=Path('/data/tokenforge-artifacts/metax_mm_formal_20261117')
SHORT=Path('/data/tokenforge-artifacts/metax_mm_model_20261117_1530')
NUMERICAL=Path('/data/tokenforge-artifacts/metax_mm_correctness_20261117_1525')
# Change only after the user explicitly confirms a later authorized deadline.
END=dt.datetime.fromisoformat(os.environ.get('FLAGOS_AUTHORIZED_DEADLINE','2026-11-17T15:59:00+08:00')).timestamp()
m.ROOT=ROOT;m.END=END
halt=threading.Event()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def telemetry():
    while not halt.wait(5):
        row=dict(at=time.time(),memory={n:(Path('/sys/fs/cgroup')/n).read_text() for n in ['memory.current','memory.events','memory.pressure','cpu.stat']})
        with (ROOT/'telemetry.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def main():
    assert END-time.time()>70*60,'Need at least70minutes of explicitly authorized time'
    summary=json.loads((SHORT/'summary.json').read_text());assert summary['complete'] and summary['short_go']
    assert json.loads((SHORT/'status.json').read_text())['stage']=='stopped'
    assert not json.loads((SHORT/'status.json').read_text()).get('failure')
    proof=json.loads((NUMERICAL/'correctness.json').read_text());assert proof['complete'] and len(proof['cases'])==18
    assert all(c['bitwise_equal'] and c['cpu_reference_pass'] for c in proof['cases'])
    assert json.loads((BASE/'cpu-client-http-smoke.json').read_text())['pass']
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',9031))!=0
    gpu=m.command(['mx-smi']);assert 'no process found' in gpu.lower()
    assert not m.command(['git','-C','/workspace/vllm-plugin-FL','status','--porcelain'])
    assert not m.command(['git','-C','/workspace/FlagGems','status','--porcelain'])
    bench=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py');assert sha(bench)=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    archive=NUMERICAL/'complete-candidate-source.zip';archive_proof=json.loads((NUMERICAL/'source-archive.json').read_text());assert sha(archive)==archive_proof['sha256']
    restore=Path('/data/tokenforge-sources/TokenForge-mm-formal-restored-20261117');assert not restore.exists()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(restore)
    source=restore/'TokenForge';manifest=json.loads((source/'source-manifest.json').read_text())
    assert len(manifest['files'])==5185
    for n,h in manifest['files'].items():assert sha(source/n)==h,n
    plugin=source/'vllm-plugin-FL';gems=source/'FlagGems/src'
    assert sha(gems/'flag_gems/runtime/backend/_metax/ops/mm.py')==proof['sha256']
    ROOT.mkdir(parents=True,exist_ok=False)
    original_spawn=m.spawn
    def tracked_spawn(argv,env,log,cwd):
        proc=original_spawn(argv,env,log,cwd)
        records=[]
        for child in m.children:
            try:records.append(dict(pid=child.pid,created=m.psutil.Process(child.pid).create_time()))
            except m.psutil.Error:pass
        m.write('owned-processes.json',records)
        return proc
    m.spawn=tracked_spawn
    for name,path in [('full-source-manifest.json',source/'source-manifest.json'),('preflight-short.json',SHORT/'summary.json'),('preflight-numerical.json',NUMERICAL/'correctness.json'),('source-archive.json',NUMERICAL/'source-archive.json'),('cpu-client-http-smoke.json',BASE/'cpu-client-http-smoke.json'),('controller.py',Path(__file__))]:shutil.copy2(path,ROOT/name)
    cache=Path('/data/tokenforge-caches')/ROOT.name
    for n in ['triton','vllm','inductor']:
        old=Path('/data/tokenforge-caches')/SHORT.name/'candidate_b'/n
        if old.exists():shutil.copytree(old,cache/n)
    env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONPATH=str(plugin)+':'+str(gems),TORCHINDUCTOR_COMPILE_THREADS='1',FLAGOS_SESSION_ROOT=str(ROOT),PYTHONDONTWRITEBYTECODE='1',VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
    for k in ['FLAGOS_METAX_TOPP_DIAGNOSTIC_CONTROL','FLAGOS_METAX_TOPP_MODE','VLLM_SERVER_DEV_MODE']:env.pop(k,None)
    imported=m.command([sys.executable,'-c','import importlib,json;from pathlib import Path;import hashlib,vllm_fl;mm=importlib.import_module("flag_gems.runtime.backend._metax.ops.mm");p=Path(mm.__file__);print(json.dumps(dict(plugin=vllm_fl.__file__,mm=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))'],env=env)
    imported=json.loads(imported.splitlines()[-1]);assert str(plugin) in imported['plugin'] and str(gems) in imported['mm'] and imported['sha256']==proof['sha256']
    argv=['vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    m.write('plan.json',dict(platform='metax_c500_64gb',cutoff_epoch=END,official_serve_argv=argv,benchmark_sha256=sha(bench),candidate='guarded_native_NT_GEMM_geometry_plus_prefix_metadata_hoist',full_source_restoration=True,sampling_unchanged=True,benchmark_model_data_unchanged=True,abort_partial_on_complete_4k_failure=True,short_gate=summary,budget_forecast=dict(original_eight_minutes=40,startup_and_client_margin_minutes=10,accuracy_reserved_minutes=20),environment_repairs_not_optimization=['Single compiler thread','HTTP-only clients use CPU platform to avoid redundant GPU runtime']))
    m.write('runtime.json',dict(import_proof=imported,environment={k:env[k] for k in ['VLLM_PLUGINS','PYTHONPATH','TORCHINDUCTOR_COMPILE_THREADS','FLAGOS_SESSION_ROOT']},gpu=gpu,hostname=os.uname().nodename))
    m.write('packages.json',json.loads(m.command([sys.executable,'-c',"import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['torch','vllm','vllm-plugin-FL','flag-gems','triton','evalscope']}))"],env=env)))
    config=Path('/workspace/MiniCPM5-2B/config.json');m.write('model-config.json',dict(path=str(config),sha256=sha(config),config=json.loads(config.read_text())))
    threading.Thread(target=telemetry,daemon=True).start()
    server=m.spawn(argv,env,'server.log',Path('/workspace'));m.state('server_starting',server_pid=server.pid)
    ready_by=min(time.time()+480,END-60*60)
    while time.time()<ready_by:
        assert server.poll() is None,'Server exited'
        try:
            if requests.get('http://127.0.0.1:9031/health',timeout=2).status_code==200:break
        except requests.RequestException:pass
        time.sleep(2)
    else:raise TimeoutError('No readiness within full acceptance budget')
    client_env=dict(env,VLLM_PLUGINS='',CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
    folder=ROOT/'official-benchmark';folder.mkdir()
    args=[sys.executable,str(bench),'--model','/workspace/MiniCPM5-2B','--served-model-name','minicpm','--port','9031','--test-cases','[[4096,1024,64,256],[16384,1024,64,128]]']
    m.state('official-benchmark',argv=args);client=m.spawn(args,client_env,'official-benchmark.log',folder)
    limit=min(time.time()+3000,END-18*60);checked=False
    while client.poll() is None:
        if time.time()>limit:raise TimeoutError('Performance exceeds completion budget')
        raws=list((folder/'benchmark_results').glob('raw_runs_*.csv'))
        if len(raws)==1 and not checked:
            four=[r for r in csv.DictReader(raws[0].open()) if r.get('Prefill')=='4096']
            if len(four)==4 and all(r.get('Total tok/s') and r.get('Mean TTFT (ms)') for r in four):
                assert all(r['Run Status']=='SUCCESS' and int(float(r['Successful Requests']))==256 and int(float(r['Total Input Tokens']))==4096*256 and int(float(r['Total Output Tokens']))==1024*256 for r in four)
                tps=sum(float(r['Total tok/s']) for r in four[1:])/3;ttft=sum(float(r['Mean TTFT (ms)']) for r in four[1:])/3;passed=tps>=5089.645*.99 and ttft<=3199.435*1.01
                m.write('4k-interim-validation.json',dict(complete_four_rounds=True,total_tokens_per_second=tps,mean_ttft_ms=ttft,performance_4k_pass=passed,overall_pass_established=False));checked=True
                if not passed:raise RuntimeError('Full4Kfails: stop partial, never splice results')
        time.sleep(3)
    assert client.returncode==0
    perf=m.perf(folder);m.write('performance-validation.json',perf);assert perf['performance_pass']
    assert END-time.time()>16*60,'Insufficient accuracy time'
    m.run('official-accuracy',['bash',str(BASE/'tools/run_official_accuracy.sh')],END-time.time()-45,dict(client_env,FLAGOS_EVAL_WORK_DIR=str(ROOT/'evalscope')),ROOT)
    reports=list((ROOT/'evalscope').glob('*/reports/minicpm/math_500.json'));assert len(reports)==1
    accuracy=audit(reports[0].parents[2]);m.write('accuracy-audit.json',accuracy)
    m.write('independent-result.json',dict(platform='metax_c500_64gb',performance=perf,accuracy_audit=accuracy,full_source_manifest=manifest,performance_pass=perf['performance_pass'],accuracy_pass=accuracy['accuracy_pass'],workloads_complete=accuracy['integrity_pass'],overall_pass=bool(perf['performance_pass'] and accuracy['accuracy_pass'] and accuracy['integrity_pass']),completed=time.time()))
    m.state('complete_needs_local_audit')
if __name__=='__main__':
    def interrupted(*_):raise TimeoutError('Explicit authorized cutoff')
    signal.signal(signal.SIGALRM,interrupted);signal.signal(signal.SIGTERM,interrupted);signal.alarm(max(1,int(END-time.time())))
    failure=None
    try:main()
    except BaseException as e:
        failure=dict(error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
        if ROOT.exists():m.write('failure.json',failure)
    finally:
        signal.alarm(0);survivors=[]
        for p in reversed(m.children):survivors+=m.stop(p)
        halt.set()
        if ROOT.exists():m.state('stopped',failure=failure,owned_survivors=survivors)
