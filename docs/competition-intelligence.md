# Competition intelligence (checked 2026-11-16)

This is an evidence register, not permission to change rules or contact anyone.
No group messages have been sent and no private chat access has been bypassed.

| Source | Status and precise implication |
|---|---|
| [FlagTree Iluvatar official installation documentation](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/multi-backend-prebuilt-docker-image-install/install-iluvatar.html) | Read 11/16. Lists BI-V150 / CoreX4.5 / torch2.10 and `flagtree===0.7.0+iluvatar3.6` from the official package index. This establishes availability, not proof of our model correctness or competition acceptance. |
| User-transcribed original Feishu discussion, supplied11/16 | The user supplied an announcement recommending `flagtree==0.7.0+iluvatar3.6`, explicitly saying existing submissions are unaffected and acceptance retesting will use a uniform environment. Chen Zhuo's reply permits attaching a report for EvalScope formatting/equivalence disputes, for organizer verification. This is direct user-supplied transcription, not an independently accessed group export; message timestamps and the announcement author's identity were not included. |
| [FlagTree issue1490](https://github.com/flagos-ai/FlagTree/issues/1490) | Opened10/31; reports nondeterministic fused-attention corruption for certain q-length residues on old0.6.2a3 with Gems5.4/Qwen/PrivateUse1. Different source path and environment from MiniCPM's native vLLM unified attention. Worth boundary coverage, not direct proof of our failure. |
| [FlagTree issue1492](https://github.com/flagos-ai/FlagTree/issues/1492) | Opened11/01; explicitly reports wrong partial-M-tile 3D matmul with program_id(2) **on0.7.0**. Therefore new version does not establish blanket numerical correctness. It is not direct proof against our 2D attention path either. |

## Relevant user-supplied organizer text

> 天数的Causal attention 数值错误通过更新flagtree可以解决（已经提交的成绩不受影响，验收复测时会统一环境）

The announcement supplies the official package index and exact0.7.0+iluvatar3.6
version. It does not enumerate all other dependencies in the uniform image.

Chen Zhuo's response, as transcribed by the user:

> 精度评测使用evalscope 工具，可能存在这样的问题。如果在最终提交的代码里问题依然存在，在提交的压缩包里附上报告，赛方会核实验证

The same supplied discussion includes permission for sparse attention. This does
not establish correctness or benefit for a new sparse implementation. No new
sparse candidate is planned in this bounded session. A participant's question
about learning from warmup data was not accompanied by a ruling in the supplied
text; it is not authorization to adapt to benchmark data.

Keep the plugin13eb9be69 and Gemsa762/tree ddf3c756 frozen. Compiler testing uses
an isolated official wheel target, separate caches, exact shared input tensors,
independent CPU references and repeat-stability checks. Original installation and
the validated v2 evidence are preserved. Do not merge upstream main or upgrade
unrelated dependencies.

The logged-in submission page was rechecked11/16: it still shows **2026-11-19
23:59:59 UTC+8**, while the problem prose shows11/20 23:59. Plan to the earlier
deadline. The page requires team-name.zip (<=300MB), complete vllm-plugin-FL/,
readme.md and report.pdf, plus FlagGems/ if its operators were changed; maximum5
submissions per team per day. No file was uploaded and no submission button was
pressed. No competition submission has been authorized or performed. Verify the actual
uniform evaluation environment and compiler version requirement.

Prepared question for the user to ask organizers if the existing announcement
does not already answer it (not sent):

> 赛道二天数复测的统一环境是否固定为FlagTree0.7.0+iluvatar3.6，其他依赖仍为原冻结镜像？若提交插件依赖该编译器修复，应在作品中如何声明和安装？MATH-500数学等价/答案提取争议是否可附逐题原始prediction/review供核实？
