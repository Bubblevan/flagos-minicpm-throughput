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
