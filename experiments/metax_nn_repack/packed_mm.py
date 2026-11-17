# SPDX-License-Identifier: Apache-2.0
# Original NN arithmetic/indexing: FlagGems a7620cc, Apache-2.0.
# Weight-layout idea: XLANG2026 / xy200303 opt4; see README.md.
# TokenForge integration is bounded to two validated BF16 prefill shapes.
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn


@triton.jit
def _tokenforge_packed_nn(A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        ram = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    else:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    if EVEN_N:
        rbn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    else:
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rbn[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C + rm[:, None] * N + rn[None, :]
    result = acc.to(C.dtype.element_ty)
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, result)
    else:
        tl.store(c_ptrs, result, mask=(rm < M)[:, None] & (rn < N)[None, :])


def packed_mm(a, packed):
    """Internal fast path; caller must retain the native unsupported fallback."""
    assert a.ndim == packed.ndim == 2
    M, K = a.shape
    K2, N = packed.shape
    assert M == 2048 and K == K2 == 2048 and N in (2560, 12288)
    assert a.dtype == packed.dtype == torch.bfloat16
    assert a.device == packed.device and a.is_cuda
    assert a.is_contiguous() and packed.is_contiguous()
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    with torch_device_fn.device(a.device):
        _tokenforge_packed_nn[(triton.cdiv(M, 256)*triton.cdiv(N, 256),)](
            a, packed, out, M, N, K, BLOCK_M=256, BLOCK_N=256, BLOCK_K=32,
            GROUP_M=8, EVEN_M=True, EVEN_N=True, EVEN_K=True,
            num_warps=8, num_stages=2, pipeline='basic')
    return out
