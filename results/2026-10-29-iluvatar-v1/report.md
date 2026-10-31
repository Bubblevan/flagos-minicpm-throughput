# Iluvatar query-tiling v1: completed full validation

All eight original benchmark rounds and this session's 105-question accuracy
run completed on 2026-10-29. Each case discards round one and averages rounds
2–4; input/output token totals and all 256/128 requests per round were audited.

| Case | Total tokens/s | Official reference | TTFT ms | Official TTFT ms | Pass |
|---|---:|---:|---:|---:|---|
| 4K | 1909.77 | 2028.01 | 9616.5033 | 11573.53 | Throughput fails, TTFT passes |
| 16K | 1318.2233 | 915.15 | 399266.5033 | 599262.03 | Both pass |

Accuracy is 102/105 (0.97142857): 105 unique prediction/review records, no errors
or empty final outputs, 105 stop. Incorrect indices are 18, 87, and 90. This is
the new 20261029_223313 run, not the day session reused.

Overall performance still fails because 4K throughput is 5.83% below reference.
Versus the separate day session's observed means, 4K throughput is about +1.90%
and 16K about +52.92%; this is not a simultaneous causal A/B comparison.
The candidate preserves the selected native backend and KV accumulation order,
with isolated query tiling and fallback. Full manifest and audit are alongside
this report. Pre-test manifest flags are retained as original evidence.

Raw full records are preserved in the migrated local workspace on D:.
The existing public ZIP archives cover the day run, A/B/A, and MetaX baseline;
a final Iluvatar evening ZIP has not yet been published. Do not claim otherwise.
