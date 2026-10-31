"""Summarize saved MetaX torch profiler traces, without running a model."""
import argparse,collections,gzip,json
from pathlib import Path

def summarize(root):
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or not (p.name.endswith('.trace.json') or p.name.endswith('.trace.json.gz')):continue
        raw=gzip.open(p,'rt') if p.suffix=='.gz' else p.open()
        with raw:data=json.load(raw)
        events=data.get('traceEvents',[]);cats=collections.Counter(e.get('cat','') for e in events)
        kernels=collections.defaultdict(lambda:[0,0.0]);cpu=collections.defaultdict(lambda:[0,0.0]);copies=collections.defaultdict(lambda:[0,0.0])
        for e in events:
            if e.get('ph')!='X':continue
            name=e.get('name','');cat=e.get('cat','');dur=float(e.get('dur',0))
            if cat=='kernel':kernels[name][0]+=1;kernels[name][1]+=dur
            if name.startswith('aten::'):cpu[name][0]+=1;cpu[name][1]+=dur
            if 'memcpy' in cat.lower() or 'memcpy' in name.lower():copies[name][0]+=1;copies[name][1]+=dur
        def ordered(table):return [dict(name=k,calls=v[0],sum_duration_us=v[1]) for k,v in sorted(table.items(),key=lambda x:-x[1][1])[:15]]
        rows.append(dict(file=p.relative_to(root).as_posix(),categories=dict(cats),kernel_top=ordered(kernels),cpu_operator_top=ordered(cpu),copy_top=ordered(copies),
            gpu_kernel_duration_sum_us=sum(v[1] for v in kernels.values())))
    return dict(scope='Bounded native_a1 profile workload only, max 8 iterations; not entire benchmark. CPU durations are nested and cannot be added as wall time.',traces=rows,
                no_model_invocation=True,gpu_capture_present=any(r['kernel_top'] for r in rows))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=summarize(a.root);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(trace_count=len(result['traces']),gpu_capture_present=result['gpu_capture_present'],kernel_top=[r['kernel_top'][:3] for r in result['traces']],copy_top=[r['copy_top'][:3] for r in result['traces']])))
if __name__=='__main__':main()