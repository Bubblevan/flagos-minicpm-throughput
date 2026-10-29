"""Bounded, isolated launch-tiling experiment; never a competition score.

Reuses the installed native kernel and makes in-memory copies of its Python
launch wrapper. Does not edit vLLM, the plugin, FlagGems, model, or benchmark.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import time


def check_exclusive():
    import psutil
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1', 9031)) == 0:
            raise RuntimeError('A model service is active; skip GPU diagnostic')
    for process in psutil.process_iter(['pid', 'status', 'cmdline']):
        if process.info['status'] == psutil.STATUS_ZOMBIE:
            continue
        command = ' '.join(process.info['cmdline'] or [])
        if 'vllm serve' in command or 'VLLM::EngineCore' in command:
            raise RuntimeError(f'Active model process {process.pid}; skip diagnostic')


def wrapper(native, block_m, tile, warps):
    source = inspect.getsource(native.unified_attention)
    marker = '    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs'
    assert source.count(marker) == 1
    source = source.replace(marker, f'''    if max_seqlen_q > 1:
        BLOCK_M = {block_m}
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = {warps}
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs''')
    marker = '    if use_td:\n'
    assert source.count(marker) == 1
    source = source.replace(marker, f'''    if max_seqlen_q > 1:
        TILE_SIZE_PREFILL = {tile}
    if use_td:
''')
    namespace = dict(native.__dict__)
    exec(compile(source, f'<isolated-tiling-{block_m}-{tile}-{warps}>', 'exec'), namespace)
    return namespace['unified_attention']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    budget = min(600, int(args.deadline - time.time() - 30))
    if budget < 180:
        raise RuntimeError('Insufficient reserved time for isolated diagnostic')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Diagnostic cutoff')))
    signal.alarm(budget)
    check_exclusive()
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as native

    torch.manual_seed(117)
    torch.cuda.set_device(0)
    device = 'cuda'
    dtype = torch.bfloat16
    source_path = Path(inspect.getsourcefile(native))
    report = dict(official_score=False, modifies_installed_source=False,
                  source_path=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(0), started_at=time.time(), results=[])

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))

    variants = [(16, 32, 4), (32, 32, 4), (64, 32, 4), (64, 64, 4), (64, 64, 8), (128, 64, 8)]
    # First two approximate chunked prefill; irregular lengths check tails and
    # batch indexing. Other head grouping checks that this is not a fixed-case trick.
    shapes = [([(1024, 4096)], 16, 2, 128), ([(2048, 16384)], 16, 2, 128),
              ([(17, 191), (33, 521), (9, 71)], 16, 2, 128),
              ([(71, 251)], 8, 2, 64)]
    for lengths, heads, kv_heads, dim in shapes:
        if args.deadline-time.time() < 90:
            raise TimeoutError('Reservation nearly exhausted')
        block_size = 16
        blocks_per_seq = [(kv+block_size-1)//block_size for _, kv in lengths]
        num_blocks = sum(blocks_per_seq)
        # Random physical block order prevents accidental contiguous-cache assumptions.
        physical = torch.randperm(num_blocks, device=device, dtype=torch.int64)
        table = torch.zeros((len(lengths), max(blocks_per_seq)), device=device, dtype=torch.int32)
        cursor = 0
        for i, count in enumerate(blocks_per_seq):
            table[i, :count] = physical[cursor:cursor+count].to(torch.int32)
            cursor += count
        k = torch.randn((num_blocks, block_size, kv_heads, dim), device=device, dtype=dtype)
        v = torch.randn_like(k)
        q = torch.randn((sum(x[0] for x in lengths), heads, dim), device=device, dtype=dtype)
        out = torch.empty_like(q)
        cu = torch.tensor([0]+list(__import__('itertools').accumulate(x[0] for x in lengths)), device=device, dtype=torch.int32)
        seq = torch.tensor([x[1] for x in lengths], device=device, dtype=torch.int32)
        kwargs = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
                      max_seqlen_q=max(x[0] for x in lengths), seqused_k=seq,
                      max_seqlen_k=max(x[1] for x in lengths), softmax_scale=dim**-0.5,
                      causal=True, window_size=(-1,-1), block_table=table, softcap=0.0,
                      q_descale=None, k_descale=None, v_descale=None)
        native.unified_attention(**kwargs)
        torch.cuda.synchronize()
        baseline = out.clone()
        q_cpu = q.cpu().float()
        k_cpu = k.cpu().float()
        v_cpu = v.cpu().float()
        table_cpu = table.cpu().long()
        baseline_cpu = baseline.cpu().float()
        # Independent float32 reference on selected query rows, covering first,
        # last and interior positions of every sequence (with cached context).
        reference_checks = []
        reference_tensors = []
        offset = 0
        for i, (qlen, kvlen) in enumerate(lengths):
            positions = sorted({0, 1, qlen//3, qlen//2, qlen-2, qlen-1})
            positions = [x for x in positions if 0 <= x < qlen]
            pos = torch.tensor(positions)
            cache_indices = table_cpu[i, :blocks_per_seq[i]]
            keys = k_cpu[cache_indices].reshape(-1, kv_heads, dim)[:kvlen]
            values = v_cpu[cache_indices].reshape(-1, kv_heads, dim)[:kvlen]
            keys = keys.repeat_interleave(heads//kv_heads, dim=1).transpose(0,1).float()
            values = values.repeat_interleave(heads//kv_heads, dim=1).transpose(0,1).float()
            queries = q_cpu[offset+pos].transpose(0,1)
            # CPU float32 reference is independent of the CoreX matmul path.
            keys = keys.cpu()
            values = values.cpu()
            scores = torch.matmul(queries, keys.transpose(-1,-2))*dim**-0.5
            mask = torch.arange(kvlen)[None,:] > (kvlen-qlen+pos.cpu())[:,None]
            scores.masked_fill_(mask[None,:,:], float('-inf'))
            reference = torch.matmul(torch.softmax(scores,dim=-1), values).transpose(0,1)
            actual = baseline_cpu[offset+pos]
            try:
                torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
                reference_pass = True
            except AssertionError:
                reference_pass = False
            reference_checks.append(dict(sequence=i, rows=positions,
                reference_device='cpu', baseline_reference_pass=reference_pass,
                baseline_max_abs_error=float((actual-reference).abs().max())))
            reference_tensors.append((offset, pos, reference))
            offset += qlen
        shape = dict(lengths=lengths, query_heads=heads, kv_heads=kv_heads, head_dim=dim,
                     dtype='bfloat16', block_size=16, randomized_block_table=True,
                     reference_checks=reference_checks, variants=[])
        report['results'].append(shape)
        save()
        for block_m, tile, warps in variants:
            row = dict(block_m=block_m, tile=tile, warps=warps)
            shape['variants'].append(row)
            try:
                function = wrapper(native, block_m, tile, warps)
                out.fill_(float('nan'))
                function(**kwargs)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, baseline, atol=0.02, rtol=0.02)
                row['max_abs_difference_from_native'] = float((out.float()-baseline.float()).abs().max())
                candidate_cpu = out.cpu().float()
                for reference_offset, reference_positions, reference_output in reference_tensors:
                    torch.testing.assert_close(candidate_cpu[reference_offset+reference_positions],
                                               reference_output, atol=0.02, rtol=0.02)
                row['correctness_pass'] = True
                # Separate compilation/warmup from event measurements.
                for _ in range(3):
                    function(**kwargs)
                torch.cuda.synchronize()
                samples=[]
                for _ in range(5):
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(5):
                        function(**kwargs)
                    end.record(); end.synchronize()
                    samples.append(start.elapsed_time(end)/5)
                row['median_ms']=statistics.median(samples)
                row['event_samples_ms']=samples
            except (AssertionError, RuntimeError) as error:
                row['correctness_pass']=False
                row['error']=str(error)[:1200]
                torch.cuda.synchronize()
            save()
            print(json.dumps(dict(shape=lengths, **row)), flush=True)
    report['complete']=True
    report['finished_at']=time.time()
    save()
    signal.alarm(0)


if __name__ == '__main__':
    main()
