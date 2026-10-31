# Mixed prefill/decode query tiling: 2026-10-31 experiment

This continues the implemented Iluvatar query-tiling v1, rather than rerunning
an unchanged original full baseline. V1's complete 16K and 105-question accuracy
already passed; its 4K throughput remained 5.83% below official reference.

The v1 predicate requires `query_rows >= 64 * num_seqs`. A 2048-row batch with
62–64 sequences falls back even though its long prefill chunk can benefit from
the already validated wider query tile. Post-ready dispatch snapshots during
real requests have now observed that fallback. Counts are Python kernel calls,
not graph replays, a fraction of end-to-end runtime, or a count of user requests.

The candidate keeps the same native Triton kernel, BF16/GQA/layout feature
checks, KV tile 32, four warps, and KV reduction order. It permits a wider query
tile when rows exceed the sequence count and `rows >= max(8 * num_seqs, 128)`.
Pure decode and unsupported features fall back. It does not recognize benchmark
names, prompt lengths, or score files and does not switch backends.

The first gate-4 experiment demonstrated the tradeoff: 2048 rows/64 sequences
was about 16.59 -> 9.46 ms, but 256 rows/64 sequences regressed about 2.5%.
Gate 8 was selected to avoid that observed small-batch regression. Its 12 mixed,
boundary, multiple-prefill, long-context, and pure-decode differential cases
were all bitwise equal to native. The preexisting irregular-shape discrepancy
against an independent float32 reference has not been resolved; native
agreement must not be described as proving every mathematical reference check.
Raw event measurements are in results/2026-10-31-mixed-dispatch.

## This reservation's decision procedure

The provider UI confirmed 2026-10-31 18:00–2026-11-01 00:00. Own GPU workload
cutoff is 23:55; evidence synchronization stops at 23:58. No chat heartbeat was
created. New source/cache/ROOT paths isolate this session from all day-29 data.

1. Run v1 and candidate with the unchanged official serve argv, separate new
   caches, identical seeds 4510/4511/4512, 128 requests, concurrency 64,
   4096 input tokens and 512 output tokens: one full-shape warmup + two measured
   runs. Both conditions enable matching low-frequency dispatch observation.
2. Require every request's tokens, finish_reason=length and prompt hashes to
   match; average throughput gain >=7%, each measured seed >=5%, and TTFT <=101%
   of v1. This is a short diagnostic, not an official score. Fixed v1-then-v2
   order, caching drift and shorter output remain limitations.
3. Only a passing candidate with at least four hours left may start unchanged
   original 4K/16K four-round validation; dispatch profiling is disabled there.
   Code/source hashes and actual worker launch records are retained.
4. After the complete original 4K four rounds, independently recompute the last
   three. If 4K still fails the official 99% throughput / 101% TTFT goal, stop
   the remaining workload and save an explicitly partial result rather than
   spend hours on a candidate already unable to pass overall. No splicing.
5. If 4K passes, finish all four 16K rounds, then the original new 105-question
   accuracy session. Audit report, predictions, reviews and finish reasons.
   A generation reaching its fixed cap is disclosed, not called a natural stop.

A one-session remote advancement process enforces these gates. It is not a
recurring app task and cannot exceed the confirmed reservation. Full-controller
finally blocks stop only owned process groups and mirror final evidence. The
local CPU collector independently rechecks saved artifacts; it never launches
or retries a GPU workload. A candidate that does not pass is reported honestly.

The exact gate-8 adapter tested by the second matrix and serving comparison has
SHA256 `5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325`.
The original preflight snapshots and gate-4 input retain their original bytes;
CRLF/LF normalization explains their different raw file hashes, not a numerical
algorithm difference. Each measured/tested file's own SHA remains recorded.
