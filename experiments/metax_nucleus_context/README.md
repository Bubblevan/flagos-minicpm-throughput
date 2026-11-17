# Context-cost guarded exact-support nucleus shortlist

Status: matched model gate passed; original full acceptance running, not an official passing result. Preserve
the validated Iluvatar package and all earlier negative MetaX diagnostics.

The static mass-certified shortlist improved one matched4K diagnostic8.90%,
but regressed the warmed64-concurrency16K case2.11%. A coverage-EMA attempt
regressed both and is explicitly rejected. This candidate keeps the static
numerical path and adds a conservative cost guard from existing CPU scheduler
metadata: unknown max_seq_len, max_seq_len>8192, or batch<64 uses native filtering.
The boundary is empirical between the tested regimes, not a global optimum or
performance claim for every intermediate context. Metadata is refreshed in the
attention metadata builder, including graph-replay iterations; no GPU sync is
added to read it. Source/runtime logs must prove the actual context and mode.

This branch chooses between numerically checked filtering implementations.
It does not change sampling parameters, RNG, model, precision, backend, served
request length or official serve arguments. It does not inspect benchmark name,
round, dataset, prompt content or scoring. Long contexts retain the same original
PyTorch filter. It is not a new attention algorithm. The implemented computation
is the custom mass statistics, bounded compaction and guarded shortlisted sort;
the context bound protects workloads where that work did not pay off.

Install the five hash-locked files into their matching frozen MetaX paths. The
complete source bundle contains all408files, including the vendor-guarded
Iluvatarv2 implementation. In the official mirror use PYTHONPATH to load the
complete source with its existing fl entry point; do not set VLLM_VENDOR=metax
for setup.py, whose optional C++ build supports only CUDA. No external package
upgrade is needed. Set FLAGOS_METAX_TOPP_MODE=candidate before service start;
default is native. Remove the variable for sampling rollback. Restore attention
source separately to remove prefix metadata hoisting and the cost-hint publisher.
Formal runs must unset developer mode and diagnostic-control paths.

The static shortlist has30actual-vocabulary differential cases and real-model
shadow evidence. New context integration must also pass real-model shadow,
shape-warmed matched4K/16K long-output A/B/A, and one original eight-round plus
new105-question acceptance. Short numbers cannot replace those final checks.

Completed short gate:4K +8.8492% (native drift -0.4137%),16K +0.0772% (drift -0.1822%). TTFT ratios1.00329/0.99887; all64requests and65536outputtokens per measured stage complete. Service exit0,no owned survivors. Real shadow1008calls bitwise equal; long-context runtime metadata confirms native fallback. These are diagnostic results, not four-round competition scores or a new accuracy result.
