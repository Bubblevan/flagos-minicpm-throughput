"""One fresh full v2+isolated compiler validation, after numerical gates only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import psutil
from compiler_session import ROOT as DIAG, SOURCES, CACHE, START, END, V2_SHA
from validate_accuracy_artifacts import audit
from accuracy_forensics import aggregate

ROOT=DIAG/'formal-v2-newcompiler'
PLUGIN=SOURCES/'vllm-plugin-FL'
COMPILER=SOURCES/'compiler-0.7.0'
GEMS=Path('/workspace/tokenforge-sources/FlagGems-a7620cc')
MODEL=Path('/workspace/MiniCPM5-2B')
BENCH=Path('/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py')
REFS={4096:(2028.01,11573.53,256),16384:(915.15,599262.03,128)}
children=[]


def write(name,value):
    dest=ROOT/name;tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(dest)


def status(stage,**kwargs):
    write('status.json',dict(stage=stage,updated=time.time(),hard_cutoff=END,**kwargs))


def clean(proc):
    if proc is None:return
    owned=[]
    try:
        root=psutil.Process(proc.pid)
        owned=[(p.pid,p.create_time()) for p in root.children(recursive=True)]
    except psutil.Error:pass
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    for pid,born in owned:
        try:
            p=psutil.Process(pid)
            if p.create_time()==born:p.terminate()
        except psutil.Error:pass
    deadline=time.time()+20
    while time.time()<deadline:
        remaining=[]
        for pid,born in owned:
            try:
                p=psutil.Process(pid)
                if p.create_time()==born and p.status()!=psutil.STATUS_ZOMBIE:remaining.append(p)
            except psutil.Error:pass
        if proc.poll() is not None and not remaining:return
        time.sleep(.2)
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    for p in remaining:
        try:p.kill()
        except psutil.Error:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass


def spawn(argv,env,log,cwd):
    with log.open('w') as f:
        p=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=cwd,start_new_session=True)
    children.append(p);return p


def run(name,argv,env,cwd,budget):
    allowed=min(budget,END-time.time()-60)
    assert allowed>0
    p=spawn(argv,env,ROOT/(name+'.log'),cwd);status(name,child_pid=p.pid)
    try:code=p.wait(timeout=allowed)
    except BaseException:clean(p);raise
    assert code==0,(name,code)


def validate(folder,cases,formal):
    files=list(folder.glob('benchmark_results/raw_runs_*.csv'));assert len(files)==1
    rows=list(csv.DictReader(files[0].open(encoding='utf-8-sig',newline='')))
    assert len(rows)==len(cases)*4
    summaries=[]
    for inp,out,con,count in cases:
        group=[r for r in rows if int(r['Prefill'])==inp];assert len(group)==4
        for r in group:
            assert r['Run Status']=='SUCCESS' and int(float(r['Successful Requests']))==count
            assert int(float(r['Total Input Tokens']))==inp*count and int(float(r['Total Output Tokens']))==out*count
        tps=sum(float(r['Total tok/s']) for r in group[1:])/3
        ttft=sum(float(r['Mean TTFT (ms)']) for r in group[1:])/3
        record=dict(input_len=inp,rounds=4,discard_first=True,total_tokens_per_second=tps,mean_ttft_ms=ttft,
                    successful_requests_each=count,input_tokens_each=inp*count,output_tokens_each=out*count)
        if formal:
            ref,lat,_=REFS[inp];record.update(throughput_reference=ref,ttft_reference=lat,
                throughput_pass=tps>=ref*.99,ttft_pass=ttft<=lat*1.01)
        summaries.append(record)
    return dict(raw_csv=str(files[0]),cases=summaries,
                performance_pass=bool(formal and all(x['throughput_pass'] and x['ttft_pass'] for x in summaries)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');a=p.parse_args()
    assert a.start and START<=time.time()<END-4*3600
    assert not ROOT.exists(), 'Never overwrite or resample a completed validation'
    gate=json.loads((DIAG/'compiler-comparison.json').read_text());assert gate['complete'] and gate['numerical_gate']
    candidate=PLUGIN/'vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==V2_SHA
    assert subprocess.check_output(['git','-C',str(GEMS),'write-tree'],text=True).strip()=='ddf3c7566aac7d23385a1b10c7c258091ef1d3a7'
    subprocess.run(['git','-C',str(GEMS),'diff','--quiet'],check=True)
    from tune_attention_tiles import check_exclusive
    check_exclusive()
    ROOT.mkdir()
    status('preflight')
    base=Path(__file__).resolve().parent;cache=CACHE/'formal'
    env=dict(os.environ,VLLM_PLUGINS='fl',PYTHONPATH=f'{COMPILER}:{PLUGIN}:{GEMS}/src:{base}',
             PYTHONDONTWRITEBYTECODE='1',FLAGOS_SESSION_ROOT=str(ROOT),FLAGOS_OBSERVE_DISPATCH='0',
             TRITON_CACHE_DIR=str(cache/'triton'),VLLM_CACHE_ROOT=str(cache/'vllm'),
             TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),CUDA_CACHE_PATH=str(cache/'cuda'))
    env.pop('FLAGOS_TILED_ATTN_MODE',None)  # v2's frozen source, not the unselected v3 adapter.
    proof="import importlib.metadata as m,json,triton,torch,flag_gems,vllm_fl;print(json.dumps(dict(flagtree=m.version('flagtree'),triton=triton.__version__,triton_file=triton.__file__,torch=torch.__version__,plugin_file=vllm_fl.__file__,gems_file=flag_gems.__file__)))"
    result=subprocess.run([sys.executable,'-c',proof],env=env,capture_output=True,text=True,timeout=120,check=True)
    lines=[x for x in result.stdout.splitlines() if x.startswith('{')];proof_json=json.loads(lines[-1])
    assert proof_json['flagtree']=='0.7.0+iluvatar3.6' and proof_json['triton_file'].startswith(str(COMPILER)+'/')
    assert proof_json['plugin_file'].startswith(str(PLUGIN)+'/') and proof_json['gems_file'].startswith(str(GEMS)+'/src/')
    argv=['vllm','serve',str(MODEL),'--port','9031','--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY"}',
          '--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    write('runtime-environment.json',dict(import_proof=proof_json,argv=argv,environment={k:env[k] for k in (
        'PYTHONPATH','VLLM_PLUGINS','TRITON_CACHE_DIR','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH')},
        source_sha256=V2_SHA,compiler_wheel=json.loads((DIAG/'wheel.json').read_text()),numerical_gate=gate,
        fixed_serve_parameters_unchanged=True,profiler_enabled=False,hard_cutoff=END))
    result=dict(session=ROOT.name,candidate='frozen-v2+isolated-flagtree0.7.0',started=time.time(),complete=False,overall_pass=False)
    server=spawn(argv,env,ROOT/'server.log',Path('/workspace'));status('server_starting',server_pid=server.pid)
    ready_by=min(time.time()+900,END-3*3600-60)
    while time.time()<ready_by:
        if server.poll() is not None:raise RuntimeError('server startup failed')
        try:
            with urllib.request.urlopen('http://127.0.0.1:9031/health',timeout=3) as r:
                if r.status==200:break
        except OSError:pass
        time.sleep(5)
    else:raise TimeoutError('server not ready in budget')
    smoke=ROOT/'smoke';smoke.mkdir();cases=[[4096,128,8,8],[16384,128,4,4]]
    benchmark=lambda c:[sys.executable,str(BENCH),'--model',str(MODEL),'--served-model-name','minicpm','--port','9031','--test-cases',json.dumps(c,separators=(',',':'))]
    run('smoke',benchmark(cases),env,smoke,600);result['smoke']=validate(smoke,cases,False)
    assert END-time.time()>3*3600+60, 'Insufficient budget for original full evaluation'
    bench=ROOT/'official-benchmark';bench.mkdir();cases=[[4096,1024,64,256],[16384,1024,64,128]]
    run('official-benchmark',benchmark(cases),env,bench,END-time.time()-3600-120)
    perf=validate(bench,cases,True);result['performance']=perf;write('performance-validation.json',perf)
    marker_files=list(ROOT.glob('attention-tiling-runtime-*.json'))
    markers=[json.loads(f.read_text()) for f in marker_files]
    assert any(m['stage']=='optimized_kernel_launched' and m['source_file']==str(candidate) for m in markers)
    result['runtime_markers']=markers
    if not perf['performance_pass']:
        result.update(complete=False,decision='Stop after failed full performance; preserve v2 original result, skip accuracy.')
        write('partial-result.json',result);return
    acc_env=dict(env,FLAGOS_EVAL_WORK_DIR=str(ROOT/'evalscope'))
    run('official-accuracy',['bash',str(base/'run_official_accuracy.sh')],acc_env,ROOT,min(3600,END-time.time()-90))
    reports=list((ROOT/'evalscope').glob('*/reports/minicpm/math_500.json'));assert len(reports)==1
    accuracy=audit(reports[0].parents[2]);write('accuracy-audit.json',accuracy)
    result.update(accuracy_audit=accuracy,performance_pass=perf['performance_pass'],accuracy_pass=accuracy['accuracy_pass'],
                  complete=bool(accuracy['integrity_pass']),overall_pass=aggregate(perf['performance_pass'],accuracy),completed=time.time())
    write('independent-result.json',result);status('complete',overall_pass=result['overall_pass'])


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('stop requested')))
    signal.signal(signal.SIGINT,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('stop requested')))
    error=None
    try:main()
    except BaseException as e:
        error=dict(type=type(e).__name__,message=str(e),at=time.time())
        if ROOT.exists():write('failure.json',error)
        raise
    finally:
        for child in reversed(children):clean(child)
        if ROOT.exists():status('stopped',failure=error,process_returncodes={str(p.pid):p.poll() for p in children})
