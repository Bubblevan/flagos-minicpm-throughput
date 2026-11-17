# SPDX-License-Identifier: Apache-2.0
# Weight-layout idea credited to XLANG2026 / xy200303 opt4 (see README.md).
# TokenForge: two-shape guard, unchanged original dispatch fallback, separate
# opaque op/fake implementation and one-time runtime provenance record.
import hashlib
import json
import os
from pathlib import Path

import torch
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.layers.utils as layer_utils
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

_ORIGINAL_PROCESS = UnquantizedLinearMethod.process_weights_after_loading
_ORIGINAL_FACTORY = layer_utils.dispatch_unquantized_gemm
_STATS = dict(layers=0, bytes=0)
_RECORDED = False


@torch.library.custom_op('tokenforge_metax::packed_prefill_mm', mutates_args=())
def packed_prefill_mm(a: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    # Defer the Gems import until runtime to avoid plugin/Gems import cycles.
    from flag_gems.runtime.backend._metax.ops.packed_mm import packed_mm
    global _RECORDED
    if not _RECORDED:
        root = os.environ.get('FLAGOS_SESSION_ROOT')
        if root:
            import inspect
            source = Path(inspect.getfile(packed_mm))
            data = dict(pid=os.getpid(), packed_layers=_STATS['layers'],
                        packed_bytes=_STATS['bytes'], source=str(source),
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        shape=list(a.shape), packed_shape=list(packed.shape),
                        records_first_attempt_not_call_coverage=True)
            dest = Path(root)/('nn-repack-runtime-worker-'+str(os.getpid())+'.json')
            tmp = dest.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(dest)
        _RECORDED = True
    return packed_mm(a, packed)


@packed_prefill_mm.register_fake
def _fake_packed_prefill_mm(a, packed):
    return torch.empty((a.shape[0], packed.shape[1]), device=a.device, dtype=a.dtype)


def _process_weights(self, layer):
    _ORIGINAL_PROCESS(self, layer)
    if '_tokenforge_nn_weight' not in layer._buffers:
        layer.register_buffer('_tokenforge_nn_weight', None, persistent=False)
    else:
        layer._tokenforge_nn_weight = None
    w = getattr(layer, 'weight', None)
    if (w is not None and w.ndim == 2 and w.is_cuda
            and w.dtype == torch.bfloat16 and w.is_contiguous()
            and tuple(w.shape) in ((2560, 2048), (12288, 2048))):
        # One-time lossless layout copy; retain the original Parameter for
        # decode, unsupported shapes and the original dispatch path.
        packed = w.detach().t().contiguous()
        layer._tokenforge_nn_weight = packed
        _STATS['layers'] += 1
        _STATS['bytes'] += packed.numel()*packed.element_size()


def _linear(layer, x, weight, bias):
    packed = getattr(layer, '_tokenforge_nn_weight', None)
    if (packed is not None and bias is None and x.ndim == 2
            and x.shape[0] == 2048 and x.shape[1] == 2048
            and x.dtype == torch.bfloat16 and x.is_cuda
            and x.is_contiguous() and x.device == packed.device):
        return packed_prefill_mm(x, packed)
    # Retain the framework's existing backend selection and every unsupported
    # behavior. No direct vendor library, sampler or Attention replacement.
    return _ORIGINAL_FACTORY()(layer, x, weight, bias)


def _factory():
    return _linear


UnquantizedLinearMethod.process_weights_after_loading = _process_weights
layer_utils.dispatch_unquantized_gemm = _factory
linear_mod.dispatch_unquantized_gemm = _factory
