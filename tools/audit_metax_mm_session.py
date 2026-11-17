"""Independent saved-artifact audit for the guarded MetaX GEMM acceptance."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
from validate_accuracy_artifacts import audit
REFS={4096:(5089.645,3199.435,256),16384:(7029.675,27197.135,128)}
def sha(data):return hashlib.sha256(data).hexdigest()
def run(root,source_zip,performance_only=False):
    raws=list((root/'official-benchmark/benchmark_results').glob('raw_runs_*.csv'))
    assert len(raws)==1,'One newly completed original session required'
    raw=raws[0];summary=raw.with_name(raw.name.replace('raw_runs_','summary_'))
    rows=list(csv.DictReader(raw.open(encoding='utf-8-sig',newline='')))
    sums=list(csv.DictReader(summary.open(encoding='utf-8-sig',newline='')))
    assert len(rows)==8 and len(sums)==2
    assert [int(r['Prefill']) for r in rows]==[4096]*4+[16384]*4
    cases=[]
    for length,(ref,lat,n) in REFS.items():
        rs=[r for r in rows if int(r['Prefill'])==length]
        for row in rs:
            assert row['Run Status']=='SUCCESS'
            for key,value in [('Decode',1024),('Conc',64),('Num Prompts',n),('Successful Requests',n),('Total Input Tokens',length*n),('Total Output Tokens',1024*n)]:assert int(float(row[key]))==value,(key,row[key],value)
        tps=sum(float(r['Total tok/s']) for r in rs[1:])/3
        ttft=sum(float(r['Mean TTFT (ms)']) for r in rs[1:])/3
        reported=next(r for r in sums if int(r['Prefill'])==length)
        assert abs(tps-float(reported['Total tok/s']))<=.015
        assert abs(ttft-float(reported['Mean TTFT (ms)']))<=.015
        cases.append(dict(input_len=length,rounds=4,discard_first=True,requests_each=n,total_tokens_per_second=tps,mean_ttft_ms=ttft,throughput_reference=ref,ttft_reference=lat,throughput_pass=tps>=ref*.99,ttft_pass=ttft<=lat*1.01))
    full=json.loads((root/'full-source-manifest.json').read_text(encoding='utf-8'))
    assert full['gems_base']=='a7620cc191a0b42e040194622c5758b22a7a25dc'
    assert full['plugin_base']=='13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0'
    with zipfile.ZipFile(source_zip) as z:
        assert z.testzip() is None
        assert set(full['files'])==set(json.loads(z.read('TokenForge/source-manifest.json'))['files'])
        for name,h in full['files'].items():assert sha(z.read('TokenForge/'+name))==h,name
    numerical=json.loads((root/'preflight-numerical.json').read_text(encoding='utf-8'))
    assert numerical['complete'] and len(numerical['cases'])==18
    assert all(c['bitwise_equal'] and c['cpu_reference_pass'] for c in numerical['cases'])
    mm='FlagGems/src/flag_gems/runtime/backend/_metax/ops/mm.py'
    assert numerical['sha256']==full['files'][mm]
    short=json.loads((root/'preflight-short.json').read_text(encoding='utf-8'))
    assert short['complete'] and short['short_go']
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    assert plan['official_serve_argv']==['vllm','serve','/workspace/MiniCPM5-2B','--port','9031','--served-model-name','minicpm','--gpu-memory-utilization','0.85','--max-model-len','131072']
    assert plan['benchmark_sha256']=='0813eba574e9d1eb50c297c515baa426da1bcb61ce8c78a398b482f246005565'
    runtime=json.loads((root/'runtime.json').read_text(encoding='utf-8'))
    assert runtime['environment']['VLLM_PLUGINS']=='fl'
    assert runtime['environment']['TORCHINDUCTOR_COMPILE_THREADS']=='1'
    plugin,gems=runtime['environment']['PYTHONPATH'].split(':')
    assert runtime['import_proof']['plugin'].startswith(plugin+'/')
    assert runtime['import_proof']['mm'].startswith(gems+'/')
    assert runtime['import_proof']['sha256']==full['files'][mm]
    result=dict(platform='metax_c500_64gb',cases=cases,performance_workloads_complete=True,performance_pass=all(c['throughput_pass'] and c['ttft_pass'] for c in cases),source_hashes_match=True,full_source_files=len(full['files']),source_archive_sha256=sha(source_zip.read_bytes()),raw_sha256=sha(raw.read_bytes()),original_benchmark_unchanged=True)
    if performance_only:return result
    reports=list((root/'evalscope').glob('*/reports/minicpm/math_500.json'));assert len(reports)==1
    precision=audit(reports[0].parents[2]);result['accuracy_audit']=precision
    result['accuracy_pass']=precision['accuracy_pass'];result['workloads_complete']=bool(result['performance_workloads_complete'] and precision['integrity_pass'])
    result['overall_pass']=bool(result['performance_pass'] and result['accuracy_pass'] and result['workloads_complete'])
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    assert status['stage']=='stopped' and not status['owned_survivors'] and status.get('failure') is None
    result['model_stopped_verified']=True
    saved=json.loads((root/'independent-result.json').read_text(encoding='utf-8'))
    assert saved['overall_pass']==result['overall_pass'] and saved['accuracy_pass']==result['accuracy_pass']
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-zip',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--performance-only',action='store_true');a=p.parse_args()
    assert not a.out.exists(),'Do not overwrite audits'
    result=run(a.root,a.source_zip,a.performance_only);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));raise SystemExit(0 if result.get('overall_pass',result['performance_pass']) else 2)
