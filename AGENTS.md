# FlagOS project instructions (public, credential-free)

For this user's migrated local checkout, first read `D:/MyLab/FlagOS/AGENTS.md`
if it exists. That parent file is the complete private handoff with login and
web-operation procedures. **Never copy its contents into this public repository,
commit it, upload it, or print its credentials in logs.** Do not initialize Git
in the parent workspace. Other clones without that file should use README/docs
and ask the user for current authorized access when actually needed.

## Current verified outcome

Both 2026-10-29 full sessions finished. Iluvatar query-tiling v1 produced
4K 1909.77 total tok/s / 9616.5033 ms TTFT and 16K 1318.2233 / 399266.5033.
Its accuracy was 102/105 with 105 stop. The 4K throughput still fails the official
1% tolerance; overall performance does not pass. MetaX's original baseline was
4K 4945.3633 / 3255.5733 and 16K 6979.6633 / 27379.3633, accuracy 101/105.
MetaX has 104 stop and one max_tokens at index 83. It also fails overall due to
4K. The Iluvatar candidate was not applied to MetaX.

Read the final independent-result.json in results/2026-10-29-iluvatar-v1 and
results/2026-10-29-metax-baseline. Older progress notes and pre-test manifest flags
are historical snapshots, not final status. The configs/versions.json baseline
optimization flag does not negate the later candidate implementation.

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
or causal claims from clock ratios. Next candidate work should examine actual
mixed prefill/decode dispatch coverage: q_rows >= num_seqs*64 can fall back for
2048 rows/64 sequences. This is a static hypothesis, not a measured histogram;
blindly lowering the gate can waste compute on short decode rows.

Back up source/version/import/argv manifests, CSVs, logs, and the corresponding
accuracy artifacts before provider shutdown. Verify hashes and scan credentials.
The user permits public code and complete experiment records, but never private
keys, tokens, passwords, cookies, models, datasets, or compilation caches.
Use normal Git credential helpers without extracting credentials. Confirm the
actual remote commit after push. Do not send issues/comments/email or competition
submissions without explicit user authorization.
