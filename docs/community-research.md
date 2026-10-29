# 官方与社区复核

截至2026-10-29 18:30的初步结果；后续继续更新。这里只记录已读证据，未向任何人发送消息。

- [官方赛题](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)：18:20公共API复核仍列固定版本、服务/评测命令、1%性能容差及精度>=0.95。内容快照保存在完整证据及晚间会话中。图像内容和线上实验室公告仍需进一步比对。
- [另一参赛项目S1基线计划](https://github.com/ApexForge-cz/apexinfer-minicpm/issues/2)：目前看到计划和验收条件，评论为空；没有看到可证明同样约6%偏差的实测报告。其README还是插件介绍，不能据此推定已复现成功或失败。
- [官方插件issue298](https://github.com/flagos-ai/vllm-plugin-FL/issues/298)：涉及另一硬件/版本的RotaryEmbedding启动问题，不是本次正常运行后的吞吐差距。
- [官方插件issue521](https://github.com/flagos-ai/vllm-plugin-FL/issues/521)：另一模型在BI-V150上初始化挂起，评论为空；不能解释本次MiniCPM完整负载性能偏差。

公开搜索尚未找到与本次条件相同、提供原始结果的直接反馈。这只描述当前检索范围，不能证明社区无人遇到。转载聚合文章不作为技术结论依据。

18:36复核：官方正文HTML及天数/沐曦两张基线图与日间快照逐字节一致，详见 results/2026-10-29-evening/official-comparison.json。实验室管理页跳到了登录页，尚未读到其内部公告；已有IDE保留使用，已请用户方便时完成管理页登录。

用户完成登录后，18:46–18:50已读[实验室管理页](https://flagos.io/OnlineLab)和[赛道二完整系统通知](https://flagos.io/MessageDetail?id=8)。页面确认当前预约18:00–次日00:00；环境释放另列11月21日00:00，不可将释放日期当作持续使用GPU的授权。资源BI-V150×1、CPU6核、内存20GB，与容器记录一致。赛道二通知发表于9月11日，重申固定硬件/模型和精度约束，未提供新基线数值、GPU频率或镜像修正说明；同一预览中的其他两条为会议邀请和旧维护通知。详细核对记录见本次results目录。

19:10追加核实：[issue539](https://github.com/flagos-ai/vllm-plugin-FL/issues/539)记录另一环境的capture_begin失败，不能套用本机已有FULL decode捕获完成及cudaGraphLaunch的情况；[issue263](https://github.com/flagos-ai/vllm-plugin-FL/issues/263)涉及其他模型的MoE/编译与分发问题。两者都没有证明本次约6%性能差距的根因。

作品提交tab实际显示11月19日23:59:59，正文仍写11月20日23:59；采用较早日期安排，11月18日争取首个可提交版本。另[沐曦券官方页](https://developer.metax-tech.com/activities/32)顶部发行数量200、正文仍写300，页面自身存在不一致；100元券需报名凭证审核，未将券数量冲突当作已解决。
