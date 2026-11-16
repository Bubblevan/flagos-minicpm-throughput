# 11/16 v2 + FlagTree 0.7.0: complete validation passed

Completed21:28:36 Beijing; service cleanup completed21:28:40. This is one new
complete session, not a splice of10/31v2 or11/09v3 results. The unchanged v2
source was restored from the full frozen source ZIP and loaded with the isolated
official0.7.0+iluvatar3.6 wheel. Original installations were preserved.

| Workload | Total tok/s, last3mean | Mean TTFT ms, last3mean | Existing published reference tok/s | Both thresholds |
|---|---:|---:|---:|---|
|4K/1024output/64concurrency/256requests |2435.35 |5639.32 |2028.01 |pass |
|16K/1024output/64concurrency/128requests |1890.7267 |268099.60 |915.15 |pass |

All8raw runs are SUCCESS, each with the expected256/128requests and complete
4096/16384input and1024output tokens. The independent means match the original
benchmark summary. Against the existing published references, throughput is
+20.09%/+106.60%; these are observations against reference numbers, not proof
that all gains are caused by our tiling or by the compiler alone.

Fresh MATH-500 Level3: **101/105=96.1905%**,105unique complete prediction/review
records,105successful reviews,0execution errors,0missing/empty final answers.
Finish reasons:104stop,1max_tokens(index83,incorrect). Wrong indices18,87,101,83.
No official score has been edited; there was no resampling. Precision passes0.95,
but one stochastic run does not guarantee all future runs pass.

The matrix outputs at18/87 are mathematically equal to the targets but contain
LaTeX row-spacing commands. They remain scored0and are separately documented for
organizer review. Index101has an incorrect geometric answer;83reaches the fixed
output limit without completing the expected final answer. Do not change the
generation limit or count intermediate mentions of the correct number as a pass.

`overall_pass=true` now correctly combines full workload integrity, performance
and accuracy. The CPU finalizer rechecked all raw counts/tokens, original summary
and this session's105reviews; it did not simply trust the controller result.
All saved formal files were also checked against remote SHA256 before this audit.
An additional archive-time check reconfirmed the local files against that manifest.

Actual import proof pins new FlagTree target, unchanged plugin v2 SHA
5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325,
Gems official equivalent tree ddf3c7566aac7d23385a1b10c7c258091ef1d3a7.
Fixed serve arguments were unchanged; no external profiler, benchmark/model/data/
sampling changes, main merge or device clock/power changes were used.

The separate16-case shared-input experiment proves a numerical repair on those
attention cases. This full model run demonstrates compatibility and passes the
required observed thresholds. It does not prove every logit is mathematically
exact or causally attribute a particular wrong answer. Compiler repair is an
organizer-provided environment update; our code contribution remains v2 tiling.

Remote read-only process inspection at23:27 found no model, benchmark, accuracy,
controller or cutoff-watchdog process remaining. Watchdog recorded no active
owned survivors. This confirms model cleanup, not provider power-off.

Full raw archive, CRC and credential-pattern scan: see archive.json. V2 old/v3
evidence remains immutable. The new complete source/readme/report package is a
validated local candidate requiring user review, not official submission acceptance.
