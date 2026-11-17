# Two-shape prepacked-NN candidate

**Work in progress; no model pass.** The basic-pipeline operator probe found
M2048 QKV and Gate/Up shapes faster and bitwise equal to native. O projection
has negligible gain; Down has a bitwise difference and is not repacked.
The earlier explicit-prefetch decode candidate was slower and rejected.

Idea attribution: [XLANG2026 / opt4](https://github.com/xy200303/minicpm-flagserve/blob/58bd3714275b5b2539a7e924a1d38580ae6d7c26/docs/baselines/metax-c500/opt4-nn-gemm.md).
The computational NN kernel body comes from frozen FlagGems a762, Apache-2.0.
Weight repacking is borrowed, not claimed as a TokenForge invention. The bounded
two-shape integration, preservation of original dispatch, evidence and validation
must be reported separately. Keep the existing FlagGems LICENSE in any full package.

After original weight postprocessing, store one contiguous [K,N] BF16 copy for
two selected projections, retaining the original Parameter. M2048 contiguous
BF16 inputs use the validated NN geometry (256,256,32),8warps,2stages,basic;
all other paths call the original GEMM dispatch factory. Original Attention,
sampler, weights' values, official serve parameters and sampling are unchanged.
No lm_head/embedding repack or direct vendor GEMM switch is included.

Extra weight memory is real: record packed bytes and the resulting KV pool;
do not raise gpu-memory-utilization to compensate. Installation is isolated:
copy packed_mm.py into FlagGems' MetaX ops directory and packed_linear.py into
the MetaX plugin patches directory, then import it at the end of that vendor's
patch initializer. Do not modify installed packages or the protected Tian ZIP.

Before model use: compare the extracted NN kernel AST with frozen source;
rerun public-function random/cancellation/zero and fallback checks on a healthy
C500, test fake/compiled execution, then matched model A/B/A. Only after those
gates and sufficient authorized time run original8rounds plus a new105.

## Current evidence (23:45 China time)

On the healthy replacement C500/driver3.8.30, all30 checks passed:18 existing
NT cases,6 packed-wrapper random/cancellation/zero cases,6 original-dispatch
fallback cases. The first24 use independent selected-row CPU float64 reference;
fallback equality is against original dispatch, not mislabeled independent math.
Fake/Dynamo and CUDA Graph tests passed. See correctness-20261117-new-device.json,
SHA256454b1f4579d98a893fbb9f760c7fda11d539578bd48cf29ea086fe934f4826d3.

The NN micro result at basic(256,256,32),8warps/2stages is0.23945->0.17904ms
for QKV and0.88665->0.70726ms for Gate/Up, bitwise-native. Down fails bitwise
equality and is not repacked. Earlier cpasync and skinny-prefetch negative
configurations remain recorded. These times are not end-to-end scores.

Complete5188-file source archive SHA256:
42231f2e471cc73c9f8c511238a5a1567b35320874644abf5ef8bd1ad46f87ed.
Kernel math/index AST equals frozen NN; protected Tianv2 files remain exact.
The model A/B/A compares this NN adapter against the otherwise identical
GEMM-only control, not against original stock.64 full-output warmup requests
precede128 measured4K/1024 requests per condition,64 concurrency, original
random-text generator and sampler. Require>=3%gain,<=3%control drift and no
TTFT regression before considering original full acceptance. Gate pending.

## Dynamic-shape correction (11/18, 00:07)

The first model trial completed its A1 control, but B failed before readiness:
vLLM's compiled graph rejected a Python `M==2048` branch that specialized its
dynamic token dimension. This is an integration failure, not a GPU fault or a
completed performance run. That trial is preserved and its A1 is not reused.

The corrected adapter tests M inside the opaque custom op. Its traced wrapper
keeps the token dimension symbolic; every other M calls original dispatch using
the retained layer/weight owner. Kernel arithmetic and the two packed shapes
are unchanged. The new complete source archive is
`artifacts/FlagOS-dual-nn-repack-dynamic-candidate-source-20261118.zip`, SHA256
`e4f40c1d482a85eaea070a75a4f725999a129b37722558837aafdc92b40e9f1f`.
All5188 restored files match their manifest. Protected Tian files are unchanged.

Fresh checks on the current C500 pass all30 numerical/fallback cases, fake/
compiled execution, CUDA Graph replay, and explicit dynamic-token lengths
2048/63/64/127/2047. See `correctness-dynamic-20261118.json`, SHA256
`b3ba34a55d56650b51753b16832d274f8d43c832897ff7765262afd8d3dec0c4`.
These finite tests are not a universal proof. A new matched A/B/A is running;
there is still no NN full eight-round or new105 acceptance result.
