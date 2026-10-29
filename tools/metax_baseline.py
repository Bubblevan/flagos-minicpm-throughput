"""Unchanged full official baseline on MetaX C500; no candidate optimization."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
import csv

import requests
from metax_common import ROOT, URL, DEADLINE, state, telemetry, mirror
from validate_accuracy_artifacts import audit


def wait_process(p, stage):
    while p.poll() is None:
        if time.time()>=DEADLINE:
            os.killpg(p.pid,signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            state('investigation_cutoff',workload=stage,exit_code=p.returncode)
            return False
        time.sleep(5)
    if p.returncode:
        state(stage+'_failed',exit_code=p.returncode)
        return False
    return True


def run(cmd, env, stage, logname):
    state(stage,argv=cmd)
    with (ROOT/logname).open('w') as f:
        p=subprocess.Popen(cmd,cwd='/workspace',env=dict(env,PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/(stage+'.pid')).write_text(str(p.pid))
    return wait_process(p,stage)


def main():
    if time.time()>DEADLINE-2*3600:
        raise RuntimeError('Less than two hours left: review time budget before full reproduction')
    pinned=json.loads((ROOT/'metax-baseline-environment.json').read_text())
    source=Path(pinned['source_root'])
    subprocess.run(['git','-C',str(source),'diff','--quiet'],check=True)
    tree=subprocess.check_output(['git','-C',str(source),'write-tree'],text=True).strip()
    assert tree==pinned['target_tree']
    try:
        requests.get(URL+'/health',timeout=2)
    except requests.RequestException:
        pass
    else:
        raise RuntimeError('A server is already active; inspect before starting another')
    env=dict(os.environ,**pinned['environment'])
    cmd=pinned['argv']
    assert '--profiler-config' not in cmd
    with (ROOT/'metax_baseline-server.log').open('w') as f:
        server=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    (ROOT/'metax_baseline-server.pid').write_text(str(server.pid))
    stop=threading.Event()
    monitor=threading.Thread(target=telemetry,args=(server.pid,stop),daemon=True)
    monitor.start()
    try:
        state('metax_baseline_server_starting',server_pid=server.pid,source_tree=tree)
        ready_by=min(time.time()+900,DEADLINE-60)
        while time.time()<ready_by:
            if server.poll() is not None:
                raise RuntimeError('Formal server exited during initialization')
            try:
                if requests.get(URL+'/health',timeout=3).status_code==200:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise RuntimeError('Formal server readiness timed out')
        models=requests.get(URL+'/v1/models',timeout=10).json()
        assert any(x['id']=='minicpm' for x in models['data'])
        smoke=requests.post(URL+'/v1/chat/completions',json={'model':'minicpm',
            'messages':[{'role':'user','content':'introduce llm'}],'max_tokens':256},timeout=180)
        smoke.raise_for_status();assert smoke.json().get('choices')
        (ROOT/'metax_baseline-smoke.json').write_text(json.dumps(smoke.json(),indent=2))
        before=set(Path('/workspace/benchmark_results').glob('raw_runs_*.csv'))
        benchmark=['python3','/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py',
            '--model','/workspace/MiniCPM5-2B','--served-model-name','minicpm',
            '--port','9031','--test-cases','[[4096,1024,64,256],[16384,1024,64,128]]']
        if not run(benchmark,env,'metax_baseline_benchmark_running','metax_baseline-benchmark.log'):
            return
        files=set(Path('/workspace/benchmark_results').glob('raw_runs_*.csv'))-before
        assert len(files)==1,files
        raw=files.pop()
        rows=list(csv.DictReader(raw.open()))
        assert len(rows)==8 and all(x['Run Status']=='SUCCESS' for x in rows)
        for x in rows:
            expected=256 if int(x['Prefill'])==4096 else 128
            assert float(x['Successful Requests'])==expected
            assert float(x['Total Output Tokens'])==expected*1024
            assert float(x['Total Input Tokens'])==expected*int(x['Prefill'])
        summary=raw.with_name(raw.name.replace('raw_runs_','summary_'))
        sums=list(csv.DictReader(summary.open()));assert len(sums)==2
        recomputed=[]
        for length in (4096,16384):
            case=[r for r in rows if int(r['Prefill'])==length]
            assert len(case)==4
            values={key:sum(float(r[key]) for r in case[1:])/3
                    for key in ('Total tok/s','Mean TTFT (ms)')}
            reported=next(r for r in sums if int(r['Prefill'])==length)
            assert all(abs(values[k]-float(reported[k]))<=0.015 for k in values)
            recomputed.append(dict(prefill=length,discard_first=True,averages=values))
        (ROOT/'metax_baseline-benchmark-validation.json').write_text(json.dumps(dict(
            raw_csv=str(raw),summary_csv=str(summary),all_eight_runs_success=True,
            source_tree=tree,source_commit=pinned['target_commit'],summary=sums,
            independent_averages=recomputed),indent=2))
        if DEADLINE-time.time()<1200:
            state('metax_baseline_benchmark_done_accuracy_deferred',summary=sums)
            return
        existing=set(Path('/workspace/evalscope-datasets/level3').glob('*'))
        if not run(['bash',str(ROOT/'run_official_accuracy.sh')],env,
                   'metax_baseline_accuracy_running','metax_baseline-accuracy.log'):
            return
        folders=set(Path('/workspace/evalscope-datasets/level3').glob('*'))-existing
        reports=[p for f in folders for p in f.glob('reports/minicpm/math_500.json')]
        assert len(reports)==1,reports
        report=json.loads(reports[0].read_text());execution=report['execution_summary']
        accuracy=report['metrics'][0]['score']
        accuracy_root=reports[0].parents[2]
        precision_audit=audit(accuracy_root)
        (ROOT/'metax-independent-accuracy-audit.json').write_text(json.dumps(precision_audit,indent=2))
        assert precision_audit['integrity_pass'], 'Review saved accuracy artifacts'

        prediction_files=list(accuracy_root.glob('predictions/minicpm/*.jsonl'))
        review_files=list(accuracy_root.glob('reviews/minicpm/*.jsonl'))
        assert len(prediction_files)==len(review_files)==1
        predictions=[json.loads(s) for s in prediction_files[0].read_text().splitlines() if s]
        reviews=[json.loads(s) for s in review_files[0].read_text().splitlines() if s]
        expected=set(range(105))
        assert len(predictions)==len(reviews)==105
        assert {x['index'] for x in predictions}=={x['index'] for x in reviews}==expected
        scores=[x['sample_score']['score'] for x in reviews]
        assert all(x['status']=='success' for x in scores)
        exact_accuracy=sum(float(x['value']['accuracy']) for x in scores)/105
        assert abs(exact_accuracy-accuracy)<0.0001
        (ROOT/'metax_baseline-accuracy-validation.json').write_text(json.dumps(dict(
            prediction_rows=len(predictions),review_rows=len(reviews),indices_complete=True,
            review_errors=0,score_reported=accuracy,score_recomputed=exact_accuracy),indent=2))
        accuracy_ok=(execution['requested']==105 and execution['succeeded']==105
            and execution['errored']==0 and not execution['incomplete'] and accuracy>=0.95)
        refs={4096:(5089.645,3199.435),16384:(7029.675,27197.135)}
        checks=[]
        for row in sums:
            a,b=refs[int(row['Prefill'])]
            checks.append(dict(input_len=int(row['Prefill']),
                total_tokens_per_second=float(row['Total tok/s']),mean_ttft_ms=float(row['Mean TTFT (ms)']),
                throughput_pass=float(row['Total tok/s'])>=a*0.99,
                ttft_pass=float(row['Mean TTFT (ms)'])<=b*1.01))
        result=dict(cases=checks,accuracy=accuracy,execution_summary=execution,
            accuracy_pass=accuracy_ok,performance_pass=all(x['throughput_pass'] and x['ttft_pass'] for x in checks),
            accuracy_report=str(reports[0]),source_tree=tree,source_commit=pinned['target_commit'],
            optimization_implemented=False,workloads_complete=True,platform='metax_c500_64gb',plugin_commit=pinned['plugin_commit'])
        (ROOT/'metax_baseline-result.json').write_text(json.dumps(result,indent=2))
        state('metax_baseline_complete_needs_analysis',result=result)
    finally:
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGKILL);server.wait()
        stop.set();monitor.join(timeout=8);mirror()


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        state('metax_baseline_reproduction_failed',error=repr(e))
        raise
