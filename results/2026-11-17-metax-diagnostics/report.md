> 最新状态（11/18 00:31）：GEMM-only完整4K未通过；显式prefetch微测更慢；NN重排在动态形状修复及30项检查后完成同机A/B/A，吞吐−7.1153%，基线漂移−0.3870%，已否决并停止模型。未为NN启动完整八轮/新105。排序边界分桶精修仅在微测，尚无沐曦新整体通过结论。下文保留历史失败与阶段证据。

# C500 decode bottleneck and guarded nucleus candidate

This is diagnostic evidence, not a passing competition result. The successful
Iluvatar submission candidate remains unchanged.

The previous C500 attempt saved4complete4K rounds and only2of4required16K
rounds. Its4K last-three mean4426.5533tok/s /3449.0667ms TTFT fails the
5089.645 /3199.435 references at1% tolerance. There is no new105-question
result for that interrupted session. The current container was rebuilt at
11:01 on11/17; the prior termination cause is unproved. No rounds are spliced.

The CPU12core quota coexists with128visible affinity CPUs. PyTorch consequently
defaults to32compiler processes. Cgroup memory previously reached62.47GiB
against64GiB before OOM. Setting `TORCHINDUCTOR_COMPILE_THREADS=1` allowed
initialization, but is an operational repair, not original algorithm credit.
Actual API and EngineCore private memory together exceed40GiB. The original
HTTP benchmark CLI can use its supported CPU platform without loading the GPU
plugin; this is another environment measure to be recorded separately.

## Evidence that changed the optimization target

Twelve paged BF16/GQA decode tensor cases tested native KV-split launch choices
and exact GQA head packing. Common8–64request batches show no stable improvement;
the native heuristic remains. A direct-output-buffer experiment failed the
native C++ output-layout check and was rejected before any model integration.

Next, a real-model decode trace captured8steps at64requests. Recorded kernel
duration totals366406.84microseconds. Sort sweep plus global histogram account
for55.54%; native FA2 split-KV plus combine account for19.58%; matrix-multiply
kernels account for14.39%. This is a bounded decode window, not a full-workload
time fraction. Parent graph/operator durations are not added to child kernels.

The frozen MetaX plugin explicitly patches vLLM's unavailable Triton filtering
path to full-vocabulary PyTorch sort. This connects the profile to an actual
source path instead of assuming attention dominates every platform.

## Candidate and correctness boundary

[Mass-certified nucleus shortlist](../../experiments/metax_nucleus_shortlist/README.md)
keeps original sampling settings, original logits and downstream RNG. Top-p-only
rows use a bounded shortlist only when its mass covers the nucleus and boundary
checks pass. Unsupported top-k, uncertain cutoffs, split ties and uncovered mass
use the original full sort. It does not inspect benchmark identity or round.

Thirty synthetic cases passed bitwise filtered-logit comparison and same-noise
sample comparison. The cases include flat/masked distributions, quantized ties,
different top-p values and adversarial cutoffs. This is not a universal floating-
point proof or a model-accuracy result. Full new105-question acceptance remains
mandatory for a final candidate.

The original topk-based prototype's plain-Torch measurements did not reproduce
worker registration; real-model shadow triggered FlagGems large-K compilation
and timed out before comparison. It is rejected. The current custom compact
implementation passed the same30differentials with worker-selected FlagGems
registration enabled.

Current micro timings illustrate the coverage risk: batch64 scale5shows+56.9%
filtering throughput, scale10shows+111.7%, while scale2falls back and regresses
43.6%; batch8 scale5regresses61.1%. These are filtering-only wall-time measurements,
not model or competition gains. Real-model shadow and matched long-output A/B/A
must decide whether the candidate is worth a new full run.

The final candidate now returns to native below64rows and fuses scratch
initialization. Actual vocabulary is130560, not the earlier synthetic130304.
The final30cases at130560again pass with worker-selected FlagGems registration;
7/64row default/non-default stream checks also pass. Current batch64 scale5/10
filtering throughput gains are58.7%/124.0%; scale2still falls back and regresses
41.6%. These remain operator diagnostics. Real-model acceptance is pending.

Real-model shadow subsequently completed1040sampling steps with bitwise-identical
filtered logits, covering65551sampling rows;38255rows used the shortlist and the
rest used native fallback. These counters describe filter invocations, not a
distinct generated-token count. Shadow returns original outputs, so it is a
correctness experiment, not a timed improvement or new MATH score. Matched long-output A/B/A subsequently completed. The 64-request 4K/1024
candidate improved 8.9028% against the mean of native A1/A2; native drift was
0.9935%. The initial 16-request long-context diagnostic bypassed this candidate
and had 67.16% native drift, so its apparent gain is rejected. A new shape-warmed
64-concurrency 16K/1024 A/B/A found native 7019.6228/7025.1484 versus candidate
6874.1354 total tok/s: -2.1111%, with only 0.0787% native drift. The candidate
therefore fails its long-context gate. These are matched diagnostics, not official
four-round scores. No official MetaX pass is established.

A separate immutable candidate now tests a coverage-based cost hint: low recent
certified-row coverage skips seven costly shortlist attempts, then probes again.
Every bypass calls the original filter; no request length, benchmark identity,
round number, sampling setting or RNG rule is used. Its new 30-case differential
passed, but its model A/B/A regressed 11.1721% at4K and1.9028% at16K, with native drift below1%. Its gate is false and the implementation is rejected. No full run was launched. All negative results are retained; gates were not retrospectively changed. A separate scheduler-context cost guard is now under bounded validation; it has no full-pass claim.

The first short-test wrapper stopped on a missing cache-reset developer route
before comparisons. Its failed setup and clean service exit are preserved.
The corrected diagnostic enables that route, clears prompt caches between
conditions and leaves original model/serve parameters unchanged. Formal runs
must remove both developer mode and the diagnostic mode-control file.

See[evidence-manifest.json](evidence-manifest.json) for the diagnostic archive
hashes. Full raw private records stay on D; no credentials or model weights are
included in this public diagnostic package.

The context-cost guard subsequently passed its predeclared matched gate:4K
+8.8492%,16K+0.0772%; native drift -0.4137%/-0.1822%. All measured64requests
and1024outputtokens each are complete. New integration shadow recorded1008
bitwise-equal filter calls and actual long-context metadata proves native fallback.
The full408-file source was unpacked into a fresh isolated source directory and
used for these model runs; no new container was created. Its source ZIP is in
artifacts, with SHA25608fb615171af9262b2554643e059bf67848a2580bde0d85d846dd2648ecc3008.
The source preserves frozen Iluvatarv2 and materializes one upstream MoE symlink
with the exact target bytes for portable ZIP restoration. FlagGems is unchanged.

At13:58on11/17, a new original eight-round performance run plus one new105-question
acceptance was launched using exactly that source, with no profiler/developer
route/diagnostic mode file. Its hard deadline is15:59; no pass is established yet.
The original script, model, data, sampling settings and fixed server arguments
are unchanged. Single-process compilation and CPU-only HTTP benchmark/accuracy
clients avoid extra GPU runtime memory; these are operational measures, not
original optimization credit. A completed failing4Kcase may abort a partial
session to preserve budget; such a session must never be called eight-round
completion or stitched into another run.

The context candidate's full4Kcase failed at14:21:four rounds complete,
last-three mean4807.27tok/s and3471.4467ms TTFT. It fails both reference tolerances.
Per the registered stop policy the partial session ended before16K/new105, with
no owned survivors. It is NOT an eight-round or accuracy completion. The raw
four-round record is archived and must not be reused in another session.

A diagnostic-design limitation was also made explicit: earlier custom trials
submitted token-ID arrays, while the original benchmark generates random text
and continuously replenishes64requests. Thus the observed+8.85% was not a
full-workload prediction. The next bounded test uses the original vllm bench
random generator and128continuous requests. A separate fused-threshold/CDF
candidate passed35actual-vocabulary differential/stream cases; operator-only
concentrated-distribution filtering improved from~28.9ms native to10.8/6.0ms.
Model gains and a new full pass are still unestablished. No scoring settings,
model, data or original benchmark code were changed.


## 15:25: warmed sampler rejected; isolated prefill GEMM candidate

The full-shape-warmed original-generator A/B/A found candidate5021.59tok/s,
gain -0.0454%, native drift -0.1023%, and TTFT ratio1.00905. Its gate is false.
The earlier +14.6% versus a cold first baseline was invalidated by14.92% drift.
No fused-sampler full pass exists.

The separate actual8-step prefill trace attributes73.2852% of GPU-kernel time
to1344 native mm_kernel_nt calls; sort kernels account for2.5824% in that window.
This is a different window from the earlier decode sort hotspot and is not an
end-to-end time fraction. Four real GEMM shapes atM2048 were observed. Installed
MetaX mm.py matches clean official a762 bytes, SHA256:
2d4f80658012c1442be3fc8beee0fd1aae6f0e3f65092160fe9a4db369d77cec.

A complete isolated FlagGems source changes only three BF16 contiguous NT
prefill launch geometries. The native arithmetic AST is identical; unsupported
shapes/layouts/dtypes, decode and the unimproved O projection retain the native
path. Selected (N,K)->(BM,BN,BK), all8warps/2stages/basic:
(2560,2048)->(256,128,32);(12288,2048)->(128,256,32);
(2048,6144)->(128,128,64). A bounded8-geometry probe selected these configurations.
Graph-timed weighted native-MM time reduction is9.54%, not an end-to-end gain.

The actual isolated public function passed18 cases: random, cancellation and
zeros across those shapes plus O-projection and smaller-M fallback shapes.
All outputs were bitwise equal to native; selected rows passed independent
float64 CPU references using existing tolerances. Finite checks are not a
universal exactness proof. Candidate mm.py SHA256:
600f5f02b03ff5a96968ca82dbafaf3e24e1203cc637579d74f808421dbe375f.

The next separate-service native/candidate/native trial uses the original
random-text generator,64request full-output warmup and128measured requests per
condition. Only the FlagGems source changes; native sampling and the existing
prefix metadata patch are common. Its predeclared gate requires>=3%gain,
<=3%native drift and no TTFT regression. Hard cutoff remains15:59; an asked
extension is not assumed. No new official eight-round/105-question pass exists.
If accepted, this candidate must ship complete FlagGems as well as the plugin.


The separate-service GEMM A/B/A completed15:46 with all tokens and requests:
native5021.24 / candidate5177.19 / native4994.61tok/s. The candidate improves3.3799%
against the native mean, TTFT ratio0.9143438 (-8.5656%), native drift-0.5303%.
The fixed short gate passes; all three services exited and final remote/local
backup hashes match. This is128request diagnostics, not the official256request
four-round average. Three new generated kernel configurations and actual isolated
imports are recorded. Candidate source remained frozen across the trial.

The user explicitly extended compute to16:59. At15:47 a new original eight-round
plus new105-question acceptance was started from a freshly restored complete
5185-file source archive (both plugin and modified FlagGems). No profiler,
developer route, diagnostic mode selector, sampling/model/data change or hardware
tuning is present. The same operational single compiler thread and CPU-only HTTP
client prevent extra runtime memory. This is a source restoration in the current
container, not a claim of a fresh container. Overall acceptance remains PENDING.


### Evening recovery: keep model diagnostics separate from device faults

The15:47formal attempt stopped producing evidence at15:52:34and the container
was later rebuilt. Recovered15files have matching remote/local SHA256; no raw
CSV or new105report exists. Last sampled host memory was49,366,605,824bytes,
with no OOM event up to that sample; the termination cause after the sample is
unknown. The new cgroup's zero counters do not exclude an earlier OOM.

At18:19the rebuilt container used a different GPU atPCI0f. Native Torch, without
vLLM/FlagGems/our candidate, repeatedly logged ioctl create queue block timeout
and could not complete a one-element allocation/synchronization probe. A user
power-cycle still assignedPCI0fand reproduced the failure; the bounded probes
were stopped. The candidate has no new precision/performance conclusion from
those failures. An intermediate48GiBhost-memory resource also did not complete
its initial native synchronization; its cause remains separate and unconfirmed.

The user then supplied a new64GiBresource: driver3.8.30, C500PCI12, original frozen
plugin/Gems image. Explicit logical-device0selection, allocation, synchronization
and scalar readback completed. All5186integrated source files were restored;
the exact protected Iluvatarv2files and MetaX/Gems changes are retained. Numerical
revalidation and CPU-only client smoke precede a fresh isolated acceptance ROOT.
New compute cutoff20:44, backup cutoff20:49. No old rounds or accuracy are reused.

### 19:40: complete4K rejection and native-attention isolation

The new64GiB machine passed all18 numerical checks. The GEMM+prefix full4K
attempt completed four256-request rounds with exact input/output totals:
3263.50/16470.70,3907.30/3718.88,4822.68/3028.14,4782.12/3118.05
(total tok/s / mean TTFT ms). Original discard-first averaging yields4504.0333
tok/s and3288.3567ms: both fail. It stopped cleanly19:27:29, without complete
16K or new105. All21 stopped ROOT files have matching remote/local hashes.
See gemm-prefix-four-round-audit.json and its immutable failed-session archive.

The next source restores frozen native MetaX Attention exactly and changes only
the guarded GEMM on this platform; protected Tianv2 files remain byte-identical.
Its complete5186-file archive SHA256 is
c3d8ad41789bbbf64f30e47ad291cf6ffdc5c207d7d0e52b873659e1d3895146.
Original acceptance began19:34:25 in metax_gemm_only_formal_20261117_1930,
with the same20:44 hard cutoff. Prior warm caches are copied, declared as an
operational condition. No result or accuracy is spliced. The prefix regression
hypothesis is unproven; removing an unvalidated full-workload variable narrows
the experiment. Prior A/B/A on another C500 had prefix common to all conditions.

Read-only comparison found2607 installed Python/YAML/JSON Gems files identical
to the clean official checkout, with no differences or missing files among them.
Thus unexpected changes to those installed files are not supported as a cause.
The prior full4K sampled no OOM; no persistent JIT activity or CPU quota cause
has been established. GPU clock observations are not normalized into scores.

### 20:00: GEMM-only rejects; investigate a cited structural rewrite

The four native-attention/GEMM-only rounds completed4299.50,3656.02,4129.31,
3892.55tok/s; original discard-first mean3892.6267tok/s and5121.9467ms fails
both gates. All request/token totals are intact. Cleanup19:58:25 has no owned
survivors; all20 stopped files match remote hashes. No complete16K/new105 is
claimed. See gemm-only-four-round-audit.json and the immutable failed ZIP.

Public research found XLANG2026's matching-model C500 eight-row CSV and summary.
Independent arithmetic confirms its reported12399.3133/11046.19tok/s means;
its aggregate accuracy report says105 successes and0.981. This is external
evidence, not our score, official acceptance or a full per-question audit here.
Its approximate sampling/direct vendor routing are not imported. A separately
attributed Apache-2.0 explicit-prefetch NT kernel is prepared for an idle-GPU
bounded test; see experiments/metax_decode_prefetch/research.md. Numerical,
matched model and full official gates remain mandatory. Deadline stays20:44.
