# Iluvatar mixed-query v2: complete validated result

This run finished on 2026-10-31 at 22:49 (Asia/Shanghai). It meets the project's
unchanged task thresholds: throughput >=99% of reference, Mean TTFT <=101%,
and complete 105-question accuracy >=0.95. This is a local full validation,
not a competition submission or an official organizer acceptance.

| Workload | Total tokens/s | Reference | Difference | Mean TTFT ms | Reference TTFT ms |
|---|---:|---:|---:|---:|---:|
| 4K/1024 output | 2167.5333 | 2028.01 | +6.88% | 8050.3533 | 11573.53 |
| 16K/1024 output | 1323.2367 | 915.15 | +44.59% | 398010.0933 | 599262.03 |

All eight original benchmark rounds succeeded. Every 4K round has 256/256
requests, 1,048,576 input and 262,144 output tokens. Every 16K round has 128/128
requests, 2,097,152 input and 131,072 output tokens. Each case discards round one
and averages the remaining three, matching the original summary CSV. No rounds
were borrowed from another session.

This session's MATH-500 Level 3 accuracy is **100/105 = 95.238095%**. All 105
prediction/review records have unique indices 0..104; there are no execution
errors, missing records, or empty final outputs, and all 105 finish reasons
are stop. Incorrect indices: 18, 70, 87, 90, 101. Accuracy passes with a small
margin; do not describe this as unchanged accuracy or rerun to select a higher
score. The official temperature/top_p/output cap were unchanged. One stochastic
run does not establish identical results for every future run.

The isolated adapter SHA256 is
`5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325`.
It uses the same native Triton kernel and KV reduction order, widening query
tiles for supported BF16/GQA mixed batches. No backend switch, benchmark/model/
data/sampling/serve-argument change, clock/power tuning, dependency upgrade, or
original-install modification was used. The exact environment, import probe,
worker runtime records, manifests, and raw data are retained. The candidate
source is a snapshot based on plugin 13eb9be69, not itself a Git checkout.
The FlagGems worktree/index tree equals official a7620cc while its old local
HEAD remains distinct; the audit does not misreport that HEAD.

The complete 111-file evidence ZIP is
`artifacts/FlagOS-Iluvatar-mixed-v2-20261031.zip`. Its CRC and SHA256 were
verified. A stalled recursive local SCP backup had last succeeded at 22:30;
a separate complete remote tar snapshot was downloaded, hash-matched and safely
extracted, then the independent audit was run on the repaired final copy.
The stale partial copy is retained locally for diagnosis. No model was rerun to
repair the backup or audit results.

The model server exited with code 0; no owned process-group survivors were
reported. The controller's zombie entry is not an active GPU workload.
Platform power-off has not been claimed. The source-restoration patch is
`experiments/attention_tiling_mixed/plugin.patch`; it passed git apply --check
against the unchanged frozen 13eb9be69 base and was not applied to the original.
The separate small-irregular-shape mathematical-reference discrepancy seen in
the original kernel remains unresolved; native bitwise tests and task validation
are not a proof over all mathematical inputs.
