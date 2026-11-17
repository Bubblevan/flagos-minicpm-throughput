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
