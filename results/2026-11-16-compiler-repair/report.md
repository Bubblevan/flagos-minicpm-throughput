# Shared-input compiler repair: observed and independently re-audited

**The tested native attention numerical discrepancies are fixed by the isolated
compiler change. This does not yet prove full-model accuracy or v3 failure causality.**

The plugin v2 file and native unified-attention Python source have identical
hashes across both conditions. The original installed compiler was not changed;
new target imports were verified. Both conditions use fresh separate caches.
All16synthetic inputs were CPU-seeded once, saved, and hash-checked. Q/K/V,
page tables, lengths, dtype, scale and causal positions are shared exactly.

| Check | Old0.6.2a3+iluvatar3.6 | New0.7.0+iluvatar3.6 |
|---|---|---|
|16cases, repeated8times | stable and finite | stable and finite |
|v2 vs native within compiler | all bitwise equal | all bitwise equal |
|Independent CPU-float64 reference, atol/rtol0.02 |13cases have failures | all16pass |
|q8tail selected-row max absolute error |0.6259114 |0.00411239 |
|q17tail selected-row max absolute error |0.4673803 |0.00328941 |
|ragged selected-row max absolute error |0.1629281 |0.00322469 |
|decode64(2D diagnostic) selected-row max absolute error |0.0546059 |0.00203472 |

The independent local CPU audit reloads all saved first outputs, validates each
input hash, recomputes reference comparisons, and verifies native/v2 bitwise
equality. A second scalar Python-fsum dot/softmax/weighted-sum implementation
checks11short cases independently of CPU matrix multiplication; maximum scalar
vs stored reference difference is7.77e-16. Eight-repeat stability is instrumented
in the GPU probe; only its first full output tensors are archived for local
recomputation. No repeated-output tensors have been invented.

Reference coverage is selected first/interior/last rows per sequence, not every
possible input. decode64 here exercises the same wrapper's2D branch without
3D segment buffers; it does not certify full graph-replayed3D decode. No full
model logits or matched stochastic RNG replay was captured. The mathematical
reference cannot attribute any specific MATH-500 answer change to the compiler.

The official wheel359492273bytes was verified against the official index SHA:
af6a7c9eddf167441fd872c3e6d673e9da572608dee79dba655b0775db740176.
Bounded range transfer was used after slow/short HTTP bodies. No unrelated
dependencies were upgraded, original packages and installed sources remain intact.

One v2+new-compiler full model validation started11/16 about18:44, with frozen
official serve arguments, no profiler, new cache and ROOT/formal-v2-newcompiler.
It first runs a fixed-argument smoke, then original eight performance rounds and
one fresh105-question accuracy session. No resampling is allowed. The hard GPU
cutoff is23:55 with an independent exact-session ownership watchdog. Results are
pending; no official throughput/accuracy score is inferred from the microtests.

The raw diagnostic ZIP is separate from the running full-model session, with
per-file hashes and CRC. See archive.json and independent-tensor-audit.json.
