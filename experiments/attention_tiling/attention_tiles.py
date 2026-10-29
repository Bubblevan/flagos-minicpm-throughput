# SPDX-License-Identifier: Apache-2.0
"""Iluvatar prefill query tiling, keeping native dispatch and KV reduction order.

This adapter reuses the selected native Triton kernel. It increases query rows
per CTA for the validated BF16 GQA layout; decode and unvalidated features keep
their original launch. It does not inspect benchmark names or prompt lengths.
"""
import json
import os
from pathlib import Path
import torch


def record_runtime(adapter, stage, **details):
    root = os.environ.get('FLAGOS_SESSION_ROOT')
    if not root:
        return
    destination = Path(root) / f'attention-tiling-runtime-{os.getpid()}.json'
    data = dict(pid=os.getpid(), stage=stage, source_file=__file__,
                optimized_launches=adapter.optimized_launches,
                fallback_launches=adapter.fallback_launches, **details)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(destination)


class QueryTiledKernel:
    def __init__(self, original):
        self.original = original
        self.optimized_launches = 0
        self.fallback_launches = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            query = kwargs.get('query_ptr')
            count = kwargs.get('num_seqs', 0)
            eligible = (
                query is not None
                and query.dtype == torch.bfloat16
                and count > 0
                and query.shape[0] >= count * 64
                and not kwargs.get('IS_3D', True)
                and kwargs.get('HEAD_SIZE') == 128
                and kwargs.get('num_queries_per_kv') == 8
                and kwargs.get('TILE_SIZE') == 32
                and int(kwargs.get('KV_QUANT_MODE', -1)) == 0
                and kwargs.get('USE_CAUSAL', False)
                and kwargs.get('SLIDING_WINDOW') == 0
                and kwargs.get('CHUNK_LOOKBACK', -1) == -1
                and not any(kwargs.get(key, False) for key in (
                    'USE_TD', 'USE_MM_PREFIX', 'USE_SINKS', 'USE_ALIBI_SLOPES',
                    'USE_QQ_BIAS', 'USE_SOFTCAP', 'USE_PER_SEQ_CAUSAL', 'USE_FP8',
                ))
            )
            if eligible:
                # Keep TILE_SIZE=32 and four warps: the KV accumulation order
                # stays identical while eight query tokens share each CTA.
                kwargs = dict(kwargs, BLOCK_M=64, BLOCK_Q=8, num_warps=4)
                selected_grid = (
                    query.shape[0] // 8 + count,
                    kwargs['num_query_heads'] // kwargs['num_queries_per_kv'],
                )
                self.optimized_launches += 1
                if self.optimized_launches == 1:
                    record_runtime(self, 'optimized_kernel_launched',
                                   query_shape=list(query.shape), block_m=64,
                                   block_q=8, kv_tile=kwargs['TILE_SIZE'])
            else:
                selected_grid = grid
                self.fallback_launches += 1
            return self.original[selected_grid](*args, **kwargs)
        return launch


def install():
    import vllm.v1.attention.ops.triton_unified_attention as native
    from vllm.platforms import current_platform
    if getattr(current_platform, 'vendor_name', None) != 'iluvatar':
        return False
    if isinstance(native.kernel_unified_attention, QueryTiledKernel):
        return True
    native.kernel_unified_attention = QueryTiledKernel(native.kernel_unified_attention)
    record_runtime(native.kernel_unified_attention, 'installed')
    return True
