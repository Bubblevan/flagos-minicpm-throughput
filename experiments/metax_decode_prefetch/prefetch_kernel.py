# SPDX-License-Identifier: Apache-2.0
# Adapted from XLANG2026 / xy200303 FlagGems mm_kernel_nt_db, commit
# aff68555e266582f68c6554898f4c795c869578d. See README.md for provenance.
# TokenForge changes: explicit reduction precision, K-tail masks and bounded
# experimental launch selection. This file is not installed by default.
import triton
import triton.language as tl


@triton.jit
def tokenforge_nt_prefetch(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
):
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
    rk = tl.arange(0, BLOCK_K)
    ap = A + rm[:, None] * K + rk[None, :]
    bp = B + rk[:, None] + rn[None, :] * K
    a = tl.load(ap, mask=(rm[:, None] < M) & (EVEN_K | (rk[None, :] < K)), other=0.0)
    b = tl.load(bp, mask=(rn[None, :] < N) & (EVEN_K | (rk[:, None] < K)), other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for step in range(tl.cdiv(K, BLOCK_K) - 1):
        ap += BLOCK_K
        bp += BLOCK_K
        remaining = K - (step + 1) * BLOCK_K
        # Explicit next-iteration loads precede the current dot. No split-K
        # or atomic partial sums: the serial accumulation order is retained.
        a_next = tl.load(ap, mask=(rm[:, None] < M) & (EVEN_K | (rk[None, :] < remaining)), other=0.0)
        b_next = tl.load(bp, mask=(rn[None, :] < N) & (EVEN_K | (rk[:, None] < remaining)), other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a = a_next
        b = b_next
    acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
    cp = C + rm[:, None] * N + rn[None, :]
    tl.store(cp, acc.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))
