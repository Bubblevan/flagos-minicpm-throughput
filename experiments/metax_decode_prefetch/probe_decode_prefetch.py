"""Bounded borrowed-idea test; requires an idle GPU and a current deadline."""
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import time
import traceback

ROOT = Path(os.environ['FLAGOS_PROBE_ROOT'])
END = dt.datetime.fromisoformat(os.environ['FLAGOS_AUTHORIZED_DEADLINE']).timestamp()
RESULT = dict(complete=False, official_score=False, shapes=[], started=time.time())


def save():
    p = ROOT / 'probe.tmp'
    p.write_text(json.dumps(RESULT, indent=2))
    p.replace(ROOT / 'probe.json')


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    assert END - time.time() >= 7 * 60, 'Need a bounded7minute budget'
    assert not ROOT.exists()
    gpu = subprocess.check_output(['/opt/mxdriver/bin/mx-smi'], text=True)
    assert 'no process found' in gpu.lower(), 'Do not run alongside a model'
    ROOT.mkdir(parents=True)
    RESULT['gpu_before'] = gpu
    import torch
    import triton
    from prefetch_kernel import tokenforge_nt_prefetch
    native = importlib.import_module('flag_gems.runtime.backend._metax.ops.mm')
    RESULT['native_source'] = str(native.__file__)
    RESULT['native_sha256'] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    assert RESULT['native_sha256'] == '2d4f80658012c1442be3fc8beee0fd1aae6f0e3f65092160fe9a4db369d77cec'
    RESULT['candidate_sha256'] = hashlib.sha256(Path(__file__).with_name('prefetch_kernel.py').read_bytes()).hexdigest()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(91217)
    torch.cuda.manual_seed_all(91217)

    def timing(f):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                f()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                graph.replay()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) / 50)
        return statistics.median(samples)

    configs = [(64, 32, 32, 4), (64, 64, 32, 4), (64, 64, 64, 4),
               (64, 64, 128, 4), (64, 32, 128, 8)]
    for N, K in [(2560, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]:
        M = 64
        a = torch.randn((M, K), device='cuda:0', dtype=torch.bfloat16)
        weight = torch.randn((N, K), device='cuda:0', dtype=torch.bfloat16)
        b = weight.t()
        c = torch.empty((M, N), device='cuda:0', dtype=torch.bfloat16)
        baseline = torch.empty_like(c)
        before = timing(lambda: native.mm_out(a, b, out=baseline))
        native.mm_out(a, b, out=baseline)
        row = dict(M=M, N=N, K=K, native_before_ms=before, configs=[])
        RESULT['shapes'].append(row)
        save()
        for BM, BN, BK, warps in configs:
            config = dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=warps)
            try:
                def launch():
                    tokenforge_nt_prefetch[(triton.cdiv(M, BM)*triton.cdiv(N, BN),)](
                        a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN,
                        BLOCK_K=BK, GROUP_M=8, EVEN_K=K % BK == 0, num_warps=warps,
                        num_stages=2, pipeline='basic')
                launch()
                torch.cuda.synchronize()
                equal = torch.equal(c, baseline)
                maximum = float((c.float()-baseline.float()).abs().max().item())
                ms = timing(launch)
                idx = torch.tensor([0, 7, 17, 31, 63], device='cuda:0')
                actual = c.index_select(0, idx).float().cpu()
                ref = (a.index_select(0, idx).double().cpu() @ weight.double().cpu().t()).float()
                row['configs'].append(dict(config=config, ms=ms,
                    bitwise_native_equal=equal, max_native_abs=maximum,
                    cpu_reference_pass=torch.allclose(actual, ref, atol=.02, rtol=.02)))
            except Exception as e:
                row['configs'].append(dict(config=config, error=repr(e), traceback=traceback.format_exc()))
            save()
        row['native_after_ms'] = timing(lambda: native.mm_out(a, b, out=baseline))
        valid = [x for x in row['configs'] if x.get('bitwise_native_equal') and x.get('cpu_reference_pass')]
        row['best_bitwise'] = min(valid, key=lambda x: x['ms']) if valid else None
        save()
    chosen = [s['best_bitwise'] for s in RESULT['shapes']]
    native_ms = sum((s['native_before_ms']+s['native_after_ms'])/2 for s in RESULT['shapes'])
    candidate_ms = sum(c['ms'] for c in chosen) if all(chosen) else None
    RESULT.update(complete=True, finished=time.time(), weighted_native_ms=native_ms,
                  weighted_candidate_ms=candidate_ms,
                  micro_go=bool(candidate_ms is not None and candidate_ms <= native_ms*.90),
                  boundary_and_model_validation_still_required=True)
    save()


if __name__ == '__main__':
    def timeout(*_):
        raise TimeoutError('Bounded experimental cutoff')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1, min(420, int(END-time.time()))))
    try:
        main()
    except BaseException as e:
        RESULT['failure'] = dict(error=repr(e), traceback=traceback.format_exc())
        if ROOT.exists():
            save()
        raise
    finally:
        signal.alarm(0)
