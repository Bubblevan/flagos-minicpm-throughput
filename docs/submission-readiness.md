# Submission readiness and competitiveness, checked11/16

Only Iluvatar has completed optimized full validation. MetaX's archived native
4K4945.3633/3255.5733ms fails its throughput/TTFT thresholds;16K6979.6633/
27379.3633passes,accuracy101/105passes. Prefix metadata hoisting has only an
observed short A/B/A and no complete new accuracy or independently saved raw
short artifacts. It is not a passed MetaX competition candidate and is not
enabled in the submitted Iluvatar source tree.

The locally validated TokenForge.zip has one top-level TokenForge directory,
complete405-filevllm-plugin-FLsource including build scripts/other vendors/license,
readme.md,report.pdf and evidence/tools. No FlagGems operator source was modified;
the pinned unmodified Gems dependency is documented rather than misrepresented
as a new operator contribution. Thus no modified FlagGems tree is being omitted.
The package is below300MB and has not been uploaded to the competition platform.

The report now contains a guarded-launch flow diagram, normalized full-throughput
plot with absolute token rates, and all16shared-input compiler-error cases. These
figures are generated from audited JSON, with PNG/SVG exports; captions distinguish
reference comparisons from causal A/B and compiler repair from original code.

**Engineering insight:** ragged mixed batches can have many decode sequences but
enough query rows to benefit from wider query tiles. The guard uses existing
query rows/sequence count and feature conditions, preserves native fallback and
the same KV-reduction path, and is validated with mixed-shape tests and full
end-to-end workloads. This is launch-geometry optimization, not a new attention
algorithm or an established first-of-its-kind research result.

**Limits of competitiveness:** Iluvatar throughput is20.09%/106.60% above the
existing published references in this run, but some improvement is associated
with the organizer-recommended compiler environment and must not all be claimed
as our code contribution. A single stochastic101/105run gives no universal
precision guarantee. MetaX is incomplete and originality is judged by experts.

The logged-in official problem page was rechecked11/16: total score includes
Iluvatar performance, MetaX performance and innovation; performance weighting70%
and expert innovation30% are stated. The leaderboard explicitly says post-event
uniform review/reproduction and expected release in mid-December2026; it shows
no current rival scores. A prize/rank guarantee is therefore unsupported.

An unresolved rule question must not be guessed: if one platform misses a basic
threshold, does only that platform lose points or is the whole entry invalid?
The displayed text does not settle this case explicitly. The user should obtain
an organizer ruling before relying on a one-platform submission. No group message,
issue or email has been sent by the agent.

Priority before the11/19 23:59:59submission-page deadline: preserve the validated
Iluvatar candidate, clarify that rule, and—only with valid authorized C500 access—
recover the existing metadata-hoisting candidate and complete its original8rounds
plus105questions after correctness/short gates. Do not spend the remaining time
inventing a new sparse design or rerunning passing Iluvatar accuracy to select a score.

Official source: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
