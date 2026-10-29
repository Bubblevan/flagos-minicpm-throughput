"""Optional bounded source comparison, AFTER all official workloads finish.

Diagnostic results are never competition scores. Original packages are unchanged.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import requests
from session_common import ROOT, URL, DEADLINE, workload, state


async def bounded_workload(seed):
    budget = min(600, DEADLINE-time.time()-30)
    if budget < 60:
        raise TimeoutError('Reserved time nearly exhausted')
    return await asyncio.wait_for(workload(64, 4096, 256, seed), timeout=budget)


def main():
    if time.time() > DEADLINE-1800:
        raise RuntimeError('Do not start: less than thirty reserved minutes remain')
    validated = json.loads((ROOT/'prior-independent-validation.json').read_text())
    assert validated['workloads_complete'] and validated['independent_validation']
    try:
        requests.get(URL+'/health', timeout=2)
    except requests.RequestException:
        pass
    else:
        raise RuntimeError('A service is still active; do not start a duplicate')
    target = ROOT/'source-comparison'
    target.mkdir(exist_ok=False)
    (target/'prior-controller-status.json').write_text((ROOT/'status.json').read_text())
    pin = json.loads((ROOT/'prior-runtime-environment.json').read_text())
    results = []
    for label, source in [('required_tag_a1', pin['source_root']+'/src'),
                          ('preinstalled_b', '/usr/local/lib/python3.12/site-packages'),
                          ('required_tag_a2', pin['source_root']+'/src')]:
        if DEADLINE-time.time() < 600:
            state('source_compare_diagnostics_time_limited', completed_sources=len(results))
            break
        cache = Path('/workspace/tokenforge-caches')/ROOT.name/label
        assert not cache.exists(), 'Caches must start fresh for both source conditions'
        cache.mkdir(parents=True)
        env = dict(os.environ, VLLM_PLUGINS='fl', PYTHONPATH=source,
                   VLLM_CACHE_ROOT=str(cache/'vllm'), TRITON_CACHE_DIR=str(cache/'triton'),
                   TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'))
        imported = subprocess.check_output(['python', '-c',
            'import flag_gems; print(flag_gems.__file__)'], env=env, text=True)
        assert source in imported
        with (target/(label+'-server.log')).open('w') as f:
            server = subprocess.Popen(pin['argv'], env=env, stdin=subprocess.DEVNULL,
                                      stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (target/(label+'-environment.json')).write_text(json.dumps(dict(
            source=source, imported=imported.strip(), cache=str(cache), argv=pin['argv'],
            server_pid=server.pid, official_score=False), indent=2))
        try:
            state('source_compare_'+label+'_starting', server_pid=server.pid)
            ready_by = min(time.time()+600, DEADLINE-60)
            while time.time() < ready_by:
                if server.poll() is not None:
                    raise RuntimeError('Diagnostic server exited')
                try:
                    if requests.get(URL+'/health', timeout=2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError('Diagnostic readiness budget exhausted')
            phase_results = []
            # Identical seeds and full shapes for both fresh-cache conditions.
            for phase, seed in [('warmup', 4100), ('measure_a', 4101), ('measure_b', 4102)]:
                state('source_compare_'+label+'_'+phase)
                result = asyncio.run(bounded_workload(seed))
                assert len(result['rows']) == 64
                assert all(r['usage']['prompt_tokens'] == 4096 and
                           r['usage']['completion_tokens'] == 256 for r in result['rows'])
                result['gpu_after_measurement'] = subprocess.check_output(
                    ['ixsmi', '-q', '-d', 'CLOCK,PERFORMANCE,POWER,UTILIZATION'], text=True, timeout=5)
                (target/(label+'-'+phase+'.json')).write_text(json.dumps(result, indent=2))
                if phase != 'warmup':
                    phase_results.append(dict(phase=phase, seconds=result['elapsed_seconds'],
                        total_tokens_per_second=64*(4096+256)/result['elapsed_seconds'],
                        mean_ttft_seconds=sum(r['ttft_seconds'] for r in result['rows'])/64))
            results.append(dict(source=label, measurements=phase_results,
                mean_total_tokens_per_second=sum(x['total_tokens_per_second'] for x in phase_results)/2))
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    summary = dict(results=results, official_score=False, comparison_complete=len(results) == 3,
                   limitations='Sequential comparison with separate initially fresh caches; '
                   'one warmup and two measurements each. Time/order effects remain possible.')
    if len(results) == 3:
        a1=results[0]['mean_total_tokens_per_second']; b=results[1]['mean_total_tokens_per_second']; a2=results[2]['mean_total_tokens_per_second']
        summary['required_tag_relative_to_preinstalled']=(a1+a2)/2/b-1
        summary['a2_relative_to_a1_drift']=a2/a1-1
        summary['condition_order']='required_tag_a1, preinstalled_b, required_tag_a2'
    (target/'summary.json').write_text(json.dumps(summary, indent=2))
    state('source_compare_diagnostics_complete', summary=summary)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        state('source_compare_diagnostics_failed', error=repr(e))
        raise
