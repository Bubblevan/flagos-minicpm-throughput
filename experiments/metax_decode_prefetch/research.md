# 2026-11-17 public-source review

These are external results and borrowed ideas, not TokenForge measurements or
proof that organizers accepted another team's entry. We contacted nobody.

## Evidence with real C500 measurements

[XLANG2026 / MiniCPM-FlagServe](https://github.com/xy200303/minicpm-flagserve)
was inspected at report commit58bd3714275b5b2539a7e924a1d38580ae6d7c26.
Its [final audit directory](https://github.com/xy200303/minicpm-flagserve/tree/58bd3714275b5b2539a7e924a1d38580ae6d7c26/docs/experiments/runs/c500-final-audit)
contains the original eight-row CSV, summary, logs and aggregate accuracy report.
We independently recomputed the request/token checks and discard-first means:
4K12399.3133tok/s /1719.1767ms;16K11046.19/18141.4733. These match its summary.
The aggregate report says105 successful,0 errors, score0.981; individual predictions
and reviews have not been independently audited here. Our derived external audit
is results/2026-11-17-metax-diagnostics/external-research-audit.json.

The published manifest has the same MACA3.7.0.38/driver3.8.30 generation as our
current card, but128GiB host RAM instead of64GiB. Its result is a combined stack,
not proof that any one component yields the advertised complete speedup.

Relevant ideas, in priority order:

1. [Explicit skinny-NT double buffering](https://github.com/xy200303/minicpm-flagserve/blob/58bd3714275b5b2539a7e924a1d38580ae6d7c26/docs/baselines/metax-c500/opt2-gemm.md):
   prefetch the next K tile before the current dot rather than relying entirely
   on compiler software pipelining. Test this directly against our native public
   mm_out, using actual four layer shapes, original precision and strict gates.
2. [One-time weight-layout repacking](https://github.com/xy200303/minicpm-flagserve/blob/58bd3714275b5b2539a7e924a1d38580ae6d7c26/docs/baselines/metax-c500/opt4-nn-gemm.md):
   align weight storage with an efficient NN prefill pipeline, retaining skinny
   decode fallback. Its duplicate-weight memory and KV capacity cost must be
   measured. This is a second candidate, not yet adopted or benchmarked here.

Not imported: approximate histogram top-p (different distribution-boundary/RNG
questions), direct vendor mcBLAS replacement (not our kernel optimization), the
whole integration branch, new dependencies, hardware tuning or external scripts.
Their prefill metadata sync reduction resembles our previous prefix candidate;
our full4K combined failure means we cannot claim that component solved ours.

The adapted kernel's source repository has an Apache-2.0 LICENSE. It is explicitly
credited in prefetch_kernel.py and README.md. Borrowed double buffering is not
reported as TokenForge's original invention. Any additional integration, guard,
validation and measurement are described separately.

## Other sources do not establish this competition's C500 pass

- [Shivamdube007 kernel suite](https://github.com/Shivamdube007/FlagOS-Season2-MiniCPM5-2B-Optimization)
  has numerical claims and a split-KV implementation/roadmap, but the inspected
  files do not provide this model's complete C500 eight-round+new105 evidence.
- [ApexForge-cz](https://github.com/ApexForge-cz/apexinfer-minicpm) has an active
  fork and plans. Its inspected branch/readme is not evidence of a C500 pass.
- [FlagGems issue5144](https://github.com/flagos-ai/FlagGems/issues/5144) records
  a Qwen3-4B/C500 case where faster isolated operators did not translate into
  whole-model gains. Versions/model differ; this supports testing composition,
  not importing its scores or assigning our root cause.
- [MetaX mcoplib](https://github.com/MetaX-MACA/mcoplib) is a primary vendor kernel
  reference, but its current release targets newer versions and is not installed.

No claim that these searches cover every participant or private group message.
No claim of a guaranteed rank, universally exact kernel or official acceptance.

## Actual adoption decisions

The explicit-prefetch test completed and is rejected: our current public native
M64 matrix path is already approximately22-61us, and every tested new geometry
is slower. The external early110us bottleneck does not transfer to this path.
The complete negative probe is preserved with remote/local hash match.

NN repacking at the documented basic256x256x32 geometry gives bitwise-equal
QKV/GateUp improvement on this compiler; cpasync samples were slower and are
preserved. Down has bitwise differences and stays on our prior native-layout
path. The resulting two-shape guarded integration is in ../metax_nn_repack,
with its own new-device numerical and model gates. No approximate sampler or
direct vendor library route is added. No claimed model pass yet.
