# MetaX prefix-metadata hoisting v1 (candidate, not a measured optimization)

Frozen file: vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py.
Native MacaFlashAttentionBackend selects MACA FlashAttention v2; Prefill uses
flash_attn_varlen_func on paged KV, Decode uses flash_attn_with_kvcache.
The installed library's actual GPU kernel names/configuration still require sampling.
Old baseline logs identify dispatch; they do not establish an attention time share.

Candidate 1 hoists int32 prefix KV lengths from every attention layer into the
metadata builder once. Device cumsum avoids the native GPU .tolist() host roundtrip.
Attention arguments/math/backend and KV cache writes stay unchanged. Pure decode
has no new prefix work. Optional metadata field retains native fallback for callers
constructing metadata outside the builder. Validate graph reuse/mixed batches before GO.
Candidate 2 is deliberately not proposed until real profiling identifies another hotspot.

First validation: tools/metax_metadata_probe.py --root NEW_ROOT --micro.
It verifies exact prefix equality and the same native FA2 outputs for 5 prefill/
paged/long-context cases, then measures 42-layer synthetic metadata old/hoisted/old.
A CPU/CUDA micro trace is metadata-only, never a full-model hotspot percentage.
Only successful differential micro evidence permits a matched serving experiment.
This candidate has not yet passed GPU tests or demonstrated an end-to-end gain.

Do not replace the native attention backend or copy Iluvatar's adapter. Stop if
actual source/version differs; never overwrite /workspace's installed plugin.
New C500 is user-started; only Lab is authorized for connection. First session is
bounded to one hour, with no automatic full official run. No new rental/cash transaction
or credential changes are performed by these scripts.