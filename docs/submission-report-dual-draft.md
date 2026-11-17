# TokenForge 双平台技术报告草稿 - 沐曦完整验收待定

本草稿不代表双平台通过，不替代已通过的天数 v2 提交候选。最终 PDF 和作品包必须引用本次独立审计后的完整结果，不能把短测或历史精度拼入新结果。

## 1. 问题与设计依据

MiniCPM5-2B 使用 BF16、16 个 Query 头与 2 个 KV 头、Head Dimension 128。相同模型在两个平台上的真实瓶颈不同。天数采样窗口主要落在原生 attention；沐曦的 64 请求 decode 窗口则有 55.54% 的 GPU kernel 时长落在全词表排序，attention 分裂 KV 及合并约 19.58%。这些是有限窗口的 kernel 时长占比，不是端到端 Amdahl 上限。

本作品据此采用两个受保护的执行路径优化，保留原 backend、模型权重、精度与采样规则。设计重点是减少无效计算和数据移动，而不是更换整套 Attention 或更改比赛测试。

## 2. 天数：混合批次 Query Tiling

原生 Triton unified attention 的 Query Tile 从 16 扩大到 64，仅用于经过验证的 BF16/GQA8/D128、plain causal 和合适混合批次形状。v2 根据 Q_rows >= max(8*num_seqs,128) 且 Q_rows>num_seqs 的公开元数据放行；不支持特性与纯 decode 保留原 launch。KV tile、寻址与归约顺序不变。

核心 insight：混合 prefill/decode 中许多单 token 序列会拉低平均 query 行数；只按纯 prefill 门槛分派会漏掉有收益的计算形状。这属于 shape-aware launch geometry，不是新的 Attention 数学算法。

主办方推荐的 FlagTree 0.7.0 修复与原创优化分开归因。11/16 完整八轮及新 105 题已独立通过：4K 2435.35 tok/s、5639.32 ms TTFT；16K 1890.7267 tok/s、268099.60 ms；精度 101/105，104 stop 与 1 max_tokens，无错误、缺失和空 final。它不是官方复测或获奖保证。

## 3. 沐曦：受保护的原生NT GEMM分块

采样shortlist系列未在充分预热的原负载上获得稳定收益，完整4K反例及无收益A/B/A均保留，不纳入当前候选。新的实际8步prefill窗口显示1344次NT GEMM占GPU kernel时长73.2852%；对应8步×42层×4矩阵乘。这不是全程比例，也不与decode窗口直接混为一个性能上限。

实际M=2048的QKV、Gate+Up与Down投影使用受保护配置(BM,BN,BK)分别为(256,128,32)、(128,256,32)、(128,128,64)，均8warps/2stages/basic。复用原kernel计算AST，保留float32累加与BF16输出。O投影、其他M/形状/layout/dtype及decode回退原路径。改变K分块仍有浮点风险，相同AST不是普遍逐位相同的证明。

8组微测配置筛选后，18项public函数随机/抵消/零与回退检查同原生逐位一致，选取行独立CPU float64参考通过，未放宽容差；新64GiB机器重新通过18项。微测加权MM耗时减少9.54%不是端到端收益。原随机文本128请求匹配A/B/A为5021.24/5177.19/4994.61tok/s，吞吐+3.38%、TTFT−8.57%、原生漂移−0.53%；该试验在此前另一C500上进行，prefix补丁在三段中相同。

新机器GEMM＋prefix组合四轮完整4K均值4504.0333tok/s、TTFT3288.3567ms，两项失败；没有继续16K或新105。现将MetaX Attention恢复冻结原版，只保留GEMM候选独立验收。不能认定prefix是性能原因，也不能将旧A/B/A包装为GEMM-only的新端到端结论。原sampler、采样、模型与固定服务参数不变。

## 4. 环境恢复与工程可信度

提交候选保留冻结插件13eb9be69及FlagGems a762的完整源码与构建文件，FlagGems仅修改MetaX mm.py。天数两源文件与已通过v2精确一致，MetaX不走其vendor guard。当前5186文件完整GEMM-only源码ZIP SHA256为c3d8ad41789bbbf64f30e47ad291cf6ffdc5c207d7d0e52b873659e1d3895146。ZIP跨系统恢复物化一处NVIDIA MoE配置符号链接为目标原始字节并记录，MiniCPM不使用该MoE配置。运行缓存复制单独披露，不声称全新冷容器复现。

本 C500 的 CPU 配额为 12 核但可见 128 核，默认 32 个编译进程曾造成系统内存 OOM。单编译进程和纯 HTTP 评测客户端隐藏 GPU 是运行环境修复，单独记账，不作为原创算法收益。服务参数与原版性能脚本保持不变。

## 5. 验收与限制

每个平台各 4K/16K 四轮，丢首轮平均后三轮，核对每轮请求与完整输入/输出 token。吞吐至少参考 99%、TTFT 至多参考 101%；本次新 105 题完整无错误缺失且精度至少 0.95。精度一次运行，不挑分重跑，不复用历史题目输出。

源码、原始 CSV、预测、review、finish reason、实际 import/argv、依赖、设备信息和控制器清理均归档。短测、官方参考、独立会话成绩和主办方编译器修复严格区分。当前尚无沐曦整体通过结果，不能保证双卡有效性、专家创新分或排名；最终作品须用户审阅后才能提交。
