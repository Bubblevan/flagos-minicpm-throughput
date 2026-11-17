# MetaX mass-certified nucleus shortlist (candidate; no official result yet)

The 2026-11-17 bounded real-model decode trace contains 8 steps at batch64.
Sort kernels account for55.54% of recorded GPU kernel duration; native FA2
split-KV plus combine account for19.58%. This is a decode sampling window,
not an end-to-end benchmark percentage. The frozen MetaX plugin explicitly
disables vLLM's incompatible Triton sampler and uses PyTorch full-vocabulary
sort. Optimizing attention alone misses the largest recorded bottleneck.

The candidate changes filtering, retaining the existing downstream softmax
and exponential-noise sampler. It does not change temperature, top-p, top-k,
model, precision, request generation, or RNG calls. It does not select another
attention backend. Prefix-metadata hoisting remains a separate existing change.

For top-p-only requests, small custom Triton kernels test monotone pivots below
the row maximum, certify probability mass/count, and compact at most4096 logits.
When their probability mass exceeds p, all omitted values precede this shortlist in ascending order.
Add the omitted probability mass to the shortlist's ascending cumulative sum
to recover the nucleus support. Retained logits are copied verbatim. Rows with
insufficient mass, uncertain floating-point boundaries, or a tie split at the
cutoff use the original full-sort implementation. Top-k and unsupported layouts
also retain that implementation. Capacity is fixed and independent of benchmark
names, sequence lengths, question content, or round number.

The mass identity is mathematical; it is not a universal floating-point proof.
The1e-5 ambiguity guard and real-model shadow comparison are correctness gates.
Do not reduce accuracy requirements or claim universally lossless sampling from
these finite tests. Real-model shadow mode returns original outputs and compares
all filtered logits before timed native/candidate/native workloads.

The frozen guarded implementation uses the original filter for batches smaller
than64; measured small-batch overhead does not justify the new path. Scratch
initialization is fused into a small kernel, avoiding the observed pathological
FlagGems zero-allocation compile for irregular batches. The final synthetic audit
at the actual130560vocabulary with worker-selected FlagGems registration passed30
cases, including quantized ties, flat and
masked distributions, multiple batch sizes/top-p values, and near-boundary cases.
Filtered logits are bitwise equal and samples match under the same noise. An
additional7/64row test passes on both default and non-default CUDA streams.
Micro performance depends on shortlist coverage: high-entropy rows may fall back
and become slower. These tests do not establish a model speedup or official pass.

The first prototype used `torch.topk(4096)`. Plain-Torch tests passed, but real
worker registration routed this operation through FlagGems' large-K iterative
selection implementation and caused a compilation timeout. That attempt is
rejected. The present statistics/compaction/restore kernels avoid this path;
FlagGems source and operator selection settings are not modified.

The files replace/add only
`vllm_fl/dispatch/backends/vendor/metax/patches/{topk_topp_sampler,top_p_shortlist,shortlist_kernels}.py`
in an isolated frozen-plugin source copy. Default mode is native; candidate mode
requires `FLAGOS_METAX_TOPP_MODE=candidate` before service start. The diagnostic
control-file option is exclusively for bounded same-model A/B/A tests; it must
be absent during the original full8rounds and new105-question acceptance run.

Before any measured workload in a same-server comparison, reset the prefix cache
using the verified vLLM developer endpoint. Otherwise identical prompts in later
conditions would receive an unfair cache advantage. Developer routes are enabled
only for diagnostics and removed from the formal environment. Record this setup
difference and require a new fixed-argument full run before submission claims.
