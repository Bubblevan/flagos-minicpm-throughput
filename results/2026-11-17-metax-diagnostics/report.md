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
passed, but its model A/B/A and full acceptance remain pending. The previous
negative result is retained; this is not a retrospective change to its gate.

The first short-test wrapper stopped on a missing cache-reset developer route
before comparisons. Its failed setup and clean service exit are preserved.
The corrected diagnostic enables that route, clears prompt caches between
conditions and leaves original model/serve parameters unchanged. Formal runs
must remove both developer mode and the diagnostic mode-control file.

See[evidence-manifest.json](evidence-manifest.json) for the diagnostic archive
hashes. Full raw private records stay on D; no credentials or model weights are
included in this public diagnostic package.
