# MetaX decode explicit-prefetch experiment (2026-11-17)

Experimental only. No numerical or performance pass is implied by these files.
Never run alongside a model or the healthy formal acceptance session.

Primary source: [XLANG2026 opt2 investigation](https://github.com/xy200303/minicpm-flagserve/blob/58bd3714275b5b2539a7e924a1d38580ae6d7c26/docs/baselines/metax-c500/opt2-gemm.md).
The kernel is adapted from `mm_kernel_nt_db` in
[xy200303/FlagGems at aff68555](https://github.com/xy200303/FlagGems/blob/aff68555e266582f68c6554898f4c795c869578d/src/flag_gems/runtime/backend/_metax/ops/mm.py),
licensed Apache-2.0 there. Preserve that attribution and the complete FlagGems
LICENSE in any distributed integration. Explicit prefetch is a borrowed idea,
not TokenForge's original invention. Changes here add explicit FP32/TF32 policy,
K-tail masks and a bounded probe against our actual native public function.

Hypothesis: skinny-M NT GEMM on this MACA compiler has serialized loads in its
K loop; moving next-tile loads before the current dot can overlap memory latency.
This changes the computation kernel instead of merely switching a backend.
The experiment retains BF16 inputs/output, FP32 serial accumulation, model,
sampling, fixed serve parameters and the original attention/sampler paths.

Test four actual M64 layer GEMMs, with a small fixed geometry set. Require all
selected configs bitwise equal to native and passing the existing selected-row
CPU float64 reference tolerance; record negative cases as well. A weighted
micro-time reduction of at least10% is only a gate to a matched model A/B/A,
not a prediction or substitute for original eight rounds plus a new105.

The other team's complete-package result cannot be claimed as ours. Its package
also uses approximate histogram sampling and a direct vendor GEMM route; those
are not adopted by this experiment. No external install or shell scripts run.
