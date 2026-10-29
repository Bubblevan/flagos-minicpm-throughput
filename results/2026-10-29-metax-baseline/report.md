# MetaX C500 original baseline, 2026-10-29

The unchanged official workload completed at 20:45. All eight rounds succeeded,
with 256/256 requests in each 4K round and 128/128 in each 16K round; input and
output token counts are complete. Each case discards its first round and averages
the remaining three.

| Case | Total tokens/s | MetaX reference | Mean TTFT ms | Reference TTFT ms | Within 1% |
|---|---:|---:|---:|---:|---|
| 4K | 4945.3633 | 5089.645 | 3255.5733 | 3199.435 | No |
| 16K | 6979.6633 | 7029.675 | 27379.3633 | 27197.135 | Yes |

This session's Level 3 accuracy is 101/105 = 96.1905%, above 0.95. All 105
prediction and review records exist, with no execution errors or empty final
outputs. Finish reasons are **104 stop and 1 max_tokens**; index 83 reached the
fixed output cap and was incorrect. This does not establish 105 natural stops.
Sampling and the output cap were not modified.

Overall performance does not pass because the 4K case is outside tolerance.
This is the original MetaX baseline, without the Iluvatar attention candidate.
The installed FlagGems HEAD is the actual official a7620cc commit; the plugin is
the frozen 13eb9be69 commit.

The controller and model service were confirmed absent at 21:23, and port 9031
refused connections. The provider UI still showed the instance running at 21:24;
stopping the model has not powered off the provider instance.

The complete saved session, including this run's predictions and reviews, is
in `artifacts/FlagOS-MetaX-baseline-20261029.zip`. SHA256 and the bounded
credential scan result are recorded alongside the independent audit. No
additional model invocation was used to audit these artifacts.
