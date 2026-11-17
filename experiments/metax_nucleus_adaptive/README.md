# Coverage-based cost guard for nucleus shortlist

This is a separate candidate, not a passing competition submission. The previous
non-adaptive implementation and its negative long-context evidence are retained.

The real C500 decode trace showed full-vocabulary sorting as the largest sampled
kernel cost. A bounded, mass-certified shortlist can avoid that work, but failed
rows still pay for statistics, compaction and sorting before native fallback.
The first model A/B/A improved 4K throughput by 8.90% while a properly warmed
64-concurrency long-context A/B/A regressed 2.11%. This motivated a cost guard.

The guard uses recent certified-row coverage only, with an exponential moving
average (new observation weight 1/4). Below half coverage it directly invokes
the original filter for the next seven otherwise-eligible calls, then probes
again. A current batch with fewer than half potentially eligible rows skips
compaction/sorting immediately. No benchmark name, round, prompt contents,
input length, dataset identity or scoring outcome selects this path. Distribution
changes can recover at the next probe. Cold state starts at one-half coverage.
This may trade away profitable calls after a distribution shift; it is a cost
heuristic rather than a correctness requirement or a claimed global optimum.

All bypasses use the original filter. The shortlist path retains the same
probability-mass, boundary and tie guards, original retained logits, and downstream
RNG. Unsupported top-k, small batches below64, layouts and numerical uncertainty
fall back. No temperature, top-p, top-k setting, model, weights, precision,
request length or output cap is changed. The frozen MetaX attention backend
remains in use; prefix metadata hoisting is a separate four-file candidate change.

Thirty actual-vocabulary (130560) synthetic differential cases passed under the
worker's FlagGems registration, including same-noise sampled-token comparisons.
This finite suite is not a universal floating-point proof. New real-model shadow,
matched A/B/A and original eight-round plus new105-question acceptance are required.
The model gate was written before its results: 4K gain>=2%,16K gain>=-0.5%,
native drift<=8%,TTFT ratio<=1.05. These gates decide whether to spend time on
full acceptance, not whether a competition result passes its published thresholds.

Install these three files into the matching frozen MetaX `patches/` directory,
plus the hash-locked prefix metadata candidate documented in `../metax_prefix/`.
Set `FLAGOS_METAX_TOPP_MODE=candidate` before starting the fixed official server.
Default is native. Remove that variable to roll back filtering; restore the
frozen attention source separately to roll back metadata hoisting. Diagnostic
mode files and developer routes must be absent in formal acceptance. The full
source bundle additionally preserves the vendor-guarded Iluvatar v2 code.

Final matched result: 4K -11.1721% (native drift -0.6914%),16K -1.9028% (drift +0.5575%). Gate false, service exited0 with no owned survivors. Operator equality is insufficient for a useful optimization. Do not deploy this candidate.
