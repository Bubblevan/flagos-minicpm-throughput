# MetaX prefill NT GEMM geometry candidate (2026-11-17)

Experimental; **not yet an official acceptance pass**. Original matrix arithmetic
is retained; three measured BF16 NT prefill shapes select fixed launch geometry.
Other shapes, layouts and dtypes keep the native path. Decode is unchanged.

The separate prefill trace attributes73.2852% of GPU kernel time to native NT GEMM;
this is not an end-to-end percentage. The bounded geometry probe selected three
bitwise-native configurations and the isolated public function passed18 random,
cancellation,zero and fallback cases plus independent selected-row CPU references.
Finite tests are not a universal exactness proof. No full accuracy is reused.

Base: official FlagGems a7620cc191a0b42e040194622c5758b22a7a25dc.
Actual installed/native source SHA256 is recorded in geometry-probe.json.
Apply flaggems.patch to a clean isolated full checkout; never modify the original
installation. Use that checkout's src on PYTHONPATH, together with the plugin.
The complete two-repository source snapshot is in artifacts. It is a C500
engineering candidate, not a replacement for the protected validated Iluvatar ZIP.
Any final submission using this modification must contain the complete FlagGems
repository alongside the complete vllm-plugin-FL source, readme and report.pdf.

Validation order: numerical gate -> matched original-generator model A/B/A ->
original eight performance rounds -> one new105-question official accuracy run.
The diagnostic control script contains a historical fixed deadline; inspect and
replace ROOT/cache/time authorization before any future execution. Do not run it
blindly. The model script changes only FlagGems between separate services; the
existing prefix metadata patch and original sampler remain common to both.

Revert by removing the isolated FlagGems src from PYTHONPATH. No installed package,
model,benchmark,data,sampling setting or hardware frequency/power is changed.

## 19:40 recovery and isolation update

On the new64GiB C500 (driver3.8.30, PCI12), the18 numerical cases were rerun
and passed. GEMM plus the historical prefix metadata patch then completed four
original4K rounds:3263.50,3907.30,4822.68,4782.12tok/s. Discarding the first gives
4504.0333tok/s and3288.3567ms TTFT: both gates fail. There are no completed16K
rounds or new105-question report in that session. The complete21-file stopped
ROOT has matching remote/local hashes; its failed record is not discarded.

A new source archive restores MetaX flash_attn.py exactly to frozen13eb bytes,
retaining only the guarded GEMM optimization on C500 and the protected Tianv2
files. The5186-file ZIP is FlagOS-dual-gemm-only-candidate-source-20261117.zip,
SHA256 c3d8ad41789bbbf64f30e47ad291cf6ffdc5c207d7d0e52b873659e1d3895146.
Its independent full acceptance began19:34:25 under the unchanged20:44 cutoff.
This is isolation, not proof of a prefix bug. The prior short A/B/A had prefix
common to all conditions and ran on another C500; it is not a new GEMM-only
end-to-end gain measurement. Warm caches were copied as an operational measure,
not as a fresh-cold-container reproduction or an optimization contribution.

Read-only package comparison found2607 installed Python/YAML/JSON files match
the clean official Gems checkout, with no differing or missing files in that
comparison. No OOM event was recorded during the complete failed4K session.
These checks narrow hypotheses; they do not establish the performance cause.
