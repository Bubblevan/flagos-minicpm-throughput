# 官方与社区复核

截至2026-10-29 18:30的初步结果；后续继续更新。这里只记录已读证据，未向任何人发送消息。

- [官方赛题](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)：18:20公共API复核仍列固定版本、服务/评测命令、1%性能容差及精度>=0.95。内容快照保存在完整证据及晚间会话中。图像内容和线上实验室公告仍需进一步比对。
- [另一参赛项目S1基线计划](https://github.com/ApexForge-cz/apexinfer-minicpm/issues/2)：目前看到计划和验收条件，评论为空；没有看到可证明同样约6%偏差的实测报告。其README还是插件介绍，不能据此推定已复现成功或失败。
- [官方插件issue298](https://github.com/flagos-ai/vllm-plugin-FL/issues/298)：涉及另一硬件/版本的RotaryEmbedding启动问题，不是本次正常运行后的吞吐差距。
- [官方插件issue521](https://github.com/flagos-ai/vllm-plugin-FL/issues/521)：另一模型在BI-V150上初始化挂起，评论为空；不能解释本次MiniCPM完整负载性能偏差。

公开搜索尚未找到与本次条件相同、提供原始结果的直接反馈。这只描述当前检索范围，不能证明社区无人遇到。转载聚合文章不作为技术结论依据。
