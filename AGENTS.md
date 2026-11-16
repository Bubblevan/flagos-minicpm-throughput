# FlagOS project instructions (public, credential-free)

For this user's migrated local checkout, first read `D:/MyLab/FlagOS/AGENTS.md`
if it exists. That parent file is the complete private handoff with login and
web-operation procedures. **Never copy its contents into this public repository,
commit it, upload it, or print its credentials in logs.** Do not initialize Git
in the parent workspace. Other clones without that file should use README/docs
and ask the user for current authorized access when actually needed.

## Current verified outcome

The Iluvatar mixed-query v2 run completed 2026-10-31 at 22:49. Its unchanged
original full workloads yielded 4K 2167.5333 total tok/s / 8050.3533 ms TTFT,
16K 1323.2367 / 398010.0933. Both pass the reference thresholds. New accuracy
is 100/105 (95.2381%), 105 stop, no errors/missing/empty final outputs; it passes
0.95 with a small margin. This is one validated run, not official submission
acceptance or a claim of no accuracy loss. Do not rerun just to select a better
score. Do not repeat the unchanged original baseline.

Read results/2026-10-31-iluvatar-mixed-v2/independent-result.json and report.md
first. The complete ZIP and restoration patch are linked there. The final
backup was repaired from a hash-verified remote snapshot after an SCP stall;
raw evidence and the audit remain complete. The server exited with no owned
process-group survivors. A zombie controller is not an active GPU workload.

The older Iluvatar v1 and day baseline results are historical; their failing
4K/progress notes are not the current v2 outcome. MetaX remains unchanged:
its original 4K fails, 16K passes, accuracy 101/105, 104 stop + one max_tokens.
V2 was not applied to MetaX. See its separate final report and use its own refs.

## Local paths and reservations

The repository is now D:/MyLab/FlagOS/flagos-minicpm-throughput. Full raw records
are in D:/MyLab/FlagOS/lab-records. New environments, caches, temporary files,
backups, and downloads belong on D:, not the removed C: project paths.
Historical C: strings inside immutable evidence must not be rewritten.
The 10/29 reservation and all old deadlines/PIDs/terminal session IDs are stale.
Check the provider UI and obtain the user's new time/budget before starting
GPU work. Model shutdown is not provider power-off or proof of stopped billing.
The user canceled this chat's heartbeat automation; do not recreate it.

## Experiment workflow

Preserve original benchmark, model, data, accuracy sampling, official fixed serve
arguments and hardware conditions. Use each platform's own references. Never
modify clocks/power, CPU governor/cgroup, or upgrade dependencies to manufacture
scores. Keep original installs untouched; use isolated, reversible source copies.
Do not merely switch backends or remove selection logic to claim optimization.

Before starting, inspect live processes, health, logs, import paths, source tree,
argv, and the actual reservation. Do not duplicate or restart a healthy task or
add GPU diagnostics on its card. Use a new session ROOT and caches outside ROOT.
Set a cutoff five minutes before the authorized reservation ends and implement
TERM/wait/KILL cleanup only for the session's own workloads and server.

Validate a real hypothesis with differential tests and matched bounded trials,
then run the original complete 4K/16K four rounds and a new 105-question accuracy
session. Discard round one and average the last three; verify all 256/128 requests
and complete input/output tokens. Review report, predictions, indices 0..104,
review records, errors, empty outputs, and finish reasons. Exit 0 is insufficient.
Never splice interrupted runs or reuse another session's accuracy as this one.

Keep raw results immutable. Label short trials separately; no unmeasured gains
or causal claims from clock ratios. The v1 mixed-dispatch fallback has now been observed and addressed in v2.
Further changes require new evidence; preserve successful v2 and its raw results.
The small-irregular-shape native mathematical-reference discrepancy remains
unresolved; do not describe differential bitwise checks as a universal proof.

Back up source/version/import/argv manifests, CSVs, logs, and the corresponding
accuracy artifacts before provider shutdown. Verify hashes and scan credentials.
The user permits public code and complete experiment records, but never private
keys, tokens, passwords, cookies, models, datasets, or compilation caches.
Use normal Git credential helpers without extracting credentials. Confirm the
actual remote commit after push. Do not send issues/comments/email or competition
submissions without explicit user authorization.

## MetaX recovery update

The user reports the old C500 instance has been released. Do not use its old
endpoint as current access. The local saved session was independently re-audited
and the 34-file public ZIP was downloaded again with matching SHA256. Logs, all
eight rounds and 105 prediction/review records remain available. See
docs/next-steps.md before any new development or rental; no new GPU time is
authorized by this recovery check.

## 11/01 C500 short trial update

The user started a new C500 and explicitly requires Lab only; do not try SSH
again or change credentials. Prefix metadata hoisting v1 has completed one
bounded native/candidate/native short trial. Controller-reported gains are
4K +11.52%, 16K +7.64%; original drift +1.01%/+0.45%. All three services
exited with no owned survivors. These are short diagnostics, not official
scores. Original raw tar downloads are blocked by the browser download tool;
user was asked to download the two named archives to D. Read the new report
before any claim of independent audit, full backup, or new GPU work.
The complete Iluvatar v2 frozen source ZIP and recovery instructions are in
artifacts/FlagOS-Iluvatar-v2-full-source.zip and docs/iluvatar-candidate.md.
No fresh BI-V150 container end-to-end restoration has been performed.

Final user decision: stop download attempts and commit/push the honest state.
User then explicitly authorized provider shutdown; the C500 UI confirmed stopped
with about 36 hours of data retention. The new raw logs are remote-only, not
GitHub-backed. The full remote archive's SHA and size are recorded in the short
report. Do not claim a complete local backup, restart the provider without new
authorization, attempt SSH again, or upload temporary authentication documents.

## 11/16 active session

Read docs/session-20261116.md and docs/competition-intelligence.md. User booked
18:00-24:00; portal verified11/16, own GPU cutoff23:55. V3's full11/09 run is
98/105 and fails overall despite a wrong raw overall_pass=true. Preserve the raw
JSON and use the corrected derived result. V2 remains the candidate. User supplied
the organizer's recommendation for FlagTree0.7.0+iluvatar3.6 and permission to
attach judge disputes for organizer verification; this is not permission to
change official scores or submit a competition entry automatically.
New compiler testing is isolated; do not overwrite the old installation, relax
reference tolerances, retry full accuracy to select a score, or launch a second
GPU workload. No heartbeat has been recreated. Original uncommittedv3/MetaX/tools
work from another chat was preserved; do not reset/clean it.

## 11/16 final outcome (overrides the active-session paragraph)

Full v2+isolatedFlagTree0.7.0 finished21:28:36; cleanup21:28:40. The new original
eight rounds and new105-question run independently pass overall:4K2435.35tok/s,
TTFT5639.32ms;16K1890.7267/268099.60;accuracy101/105,104stop+1max_tokens(index83,
incorrect),0errors/missing/empty final. Two matrix-format disputes remain scored0.
Do not rejudge or resample. Full evidence and remote/local hash match are in
results/2026-11-16-v2-flagtree07-complete. No owned model processes survived;
23:27read-only SSH found no remaining workload. Provider power state is unverified.
Local sync was gracefully stopped after final archive verification; completion
step and remote cutoff watchdog had already ended. No chat automation exists.

Current candidate package is artifacts/TokenForge-v2-flagtree070-20261116.zip,
with complete source,updated readme,5-page visually checked PDF and full raw evidence.
It requires user review and has NOT been uploaded to the competition platform.
The original v2 and failedv3 records remain immutable. Preserve other chat's
uncommitted code. No new GPU work is needed just to obtain another score.
