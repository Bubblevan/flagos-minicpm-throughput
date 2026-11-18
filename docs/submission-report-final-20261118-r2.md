# TokenForge: MiniCPM5-2B 混合批次 Query Tiling

面向天数 BI-V150 的形状感知 Attention 分块优化。TokenForge，2026年11月18日。

## 1. 方案概述与实测结果

本方案面向 MiniCPM5-2B 的混合 prefill/decode 批次，依据 query 行数、序列数和算子特性选择 Attention 的 query 分块。实现基于冻结版 vllm-plugin-FL，复用原生 Triton unified attention kernel，通过 launch adapter 将适用批次的 BLOCK_M 从16扩大至64，提高同一分块内的 query 计算并行度与 KV 数据复用。

天数 BI-V150 完整性能测试结果：4K场景为2435.35 total tok/s、Mean TTFT 5639.32ms；16K场景为1890.7267 total tok/s、Mean TTFT 268099.60ms。相对官方参考2028.01/915.15 tok/s，吞吐分别提升20.09%/106.60%。两场景吞吐及TTFT均达到赛题门槛。

性能测试使用官方4K与16K配置，每组四轮，丢首轮后取后三轮均值。八轮全部SUCCESS；每轮4K为256请求、16K为128请求，输入和输出token总量完整，独立重算结果与原版summary一致。测试于2026年11月16日21:28:36完成。

同次 MATH-500 Level3 精度为101/105，即96.1905%，达到95%要求。105条预测和评审记录完整唯一，执行错误、缺失和空final均为0；104题自然停止，1题达到max_tokens。性能、工作负载完整性和精度三项联合验收通过。

最终配置为 mixed-query v2 与 FlagTree 0.7.0+iluvatar3.6。Query tiling及混合批次分派由本作品实现；FlagTree版本更新采用主办方推荐的编译器修复。图2分别展示官方参考、旧编译器v2与新编译器v2的实测结果。

## 2. 核心观察与实现架构

混合批次由长query的prefill序列和单token的decode序列共同组成。大量decode序列会降低batch的平均query行数，导致以纯prefill为假设的严格门槛频繁回退。v1要求query_rows至少为num_seqs的64倍；在混合批次观测与合成测试基础上，v2将分派条件调整为query_rows>=max(8*num_seqs,128)，且query_rows>num_seqs，以覆盖更多具有计算收益的混合形状。

完整插件基于冻结提交13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0。实现涉及两个文件：vllm_fl/platform.py安装launch adapter；新增vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py负责特性检查、分块选择、原生回退和运行记录。PlatformFL.import_kernels完成接入，执行内核仍为kernel_unified_attention。

适用条件为BF16、head_dim128、16Q/2KV对应GQA8、原KV TILE_SIZE32、plain causal和2D路径。命中时BLOCK_M由16变64，BLOCK_Q由2变8，num_warps保持4，KV分块及归约顺序保持原配置。纯decode、滑窗、量化及其它特性沿用原生launch。分派输入来自已有shape与特性元数据。

优化复用现有Q/K/V张量和KV页面，沿用模型、采样、benchmark及官方固定服务与设备配置。候选源码由SOURCE_MANIFEST.json逐文件hash锁定。

## 3. 分层验证与效果分析

验证流程分为算子差分、数学参考、匹配短测和完整模型验收。v2的12类合成batch覆盖纯decode、63条单token decode加1条prefill、多个prefill、16/32/64序列、4K/16K上下文及query行尾部；所测输入与同环境原生输出逐位一致。

数学参考采用CPU float64逐head计算，并通过标量fsum交叉核验。11类短形状上两份CPU参考的最大差约7.8e-16。保持源码、Q/K/V和分页表相同，对比旧版与FlagTree 0.7.0后，全部16组在固定atol=0.02、rtol=0.02下通过；v2与新环境原生输出逐位一致，八次重复结果稳定。q8尾块最大绝对误差由0.6259降至0.0041，保存张量重新载入后的独立核算通过。

匹配短测采用128请求、64并发、4096输入与512输出。同配置下v1均值2412.24 tok/s，v2为3046.62 tok/s，提升26.30%。该实验验证混合形状覆盖调整的实际效果；最终性能采用第1节4096/16384输入、1024输出的官方完整测试。

运行时worker记录及编译产物确认优化launch实际执行。实现同时保留原生回退路径，分别以launch契约、张量差分和端到端工作负载检查分派行为与计算结果。完整源码中的287个Python文件语法检查和20项CPU分派/回退契约均通过。

## 4. 精度结果与逐题分析

本次精度保持temperature=1.0、top_p=0.95、max_tokens=32768、batch_size=8。105题按索引0至104逐条核对，正确101题，错误索引为18、87、101、83。原始评分及finish reason完整保存。

题18与87的矩阵数值分别为(-1/3,2/3,5/3)和(16/49,48/49,24/49)，独立数学复核支持与标准答案等价；输出中的[4pt]行距命令与评分结果存在格式兼容问题。两题原始prediction、review及数学证明附于accuracy-disputes.json，供赛方核实。报告精度采用原始101/105。

题101输出错误的几何答案216π；题83达到固定32768 token上限，最终答案未完成。两题均按原始评审判错。本轮记录为104stop、1max_tokens。

历史对照中，10月31日旧编译器v2为2167.5333/1323.2367 tok/s、100/105；11月9日v3为2141.3967/1320.1767 tok/s、98/105。最终采用完成本轮联合验收的v2。历史成绩、评分争议和原始记录分别归档。

## 5. 实验环境与复现流程

硬件为天数BI-V150单卡，driver4.5.0，CPU6核、内存20GiB。软件为Python3.12、torch2.10/CoreX4.5、vLLM0.24，编译器使用隔离安装的FlagTree0.7.0+iluvatar3.6。该版本来自主办方推荐及官方软件源。

FlagGems锁定官方提交a7620cc191a0b42e040194622c5758b22a7a25dc，对应完整tree ddf3c7566aac7d23385a1b10c7c258091ef1d3a7。实验中以完整tree与文件hash核对源码身份。FlagGems算子保持官方版本；作品改动位于插件中的launch adapter。

完整插件源码包含405个文件，保留构建脚本、依赖声明、LICENSE、测试和各平台实现。原完整源码ZIP SHA256为ace45a56569999e010c9882d434e1d516d932790aabbc6497d703c1b23076cfc，两次独立构建hash一致。本轮从该ZIP恢复到新的隔离目录，使用独立PYTHONPATH和新缓存完成端到端验收。

服务参数为FULL_DECODE_ONLY、port9031、served-model-name minicpm、gpu-memory-utilization0.85、max-model-len131072。性能配置为[[4096,1024,64,256],[16384,1024,64,128]]；精度为MATH-500 Level3、temperature1、top_p0.95、max_tokens32768、batch8、timeout3600。完整命令、依赖准备、源码恢复与审计步骤见readme.md。

离线审计工具核对交付文件hash、源码清单、分派契约、原始8轮CSV、原版summary与105题预测和评审。整体判定为工作负载完整性、性能与精度的合取。源码恢复时核对实际import和版本；回退时移除本次隔离路径及缓存环境变量，重新加载原安装。

## 6. 平台结果与资料索引

天数成果为混合prefill/decode批次的形状感知query tiling：扩大适用形状中的query计算分块，保持原生KV组织与归约路径，通过原生回退覆盖其余特性。该方案将批次结构观察落实为明确的分派条件，并以分层数值检查和完整负载验收验证效果。

沐曦原始基线4K为4945.3633 tok/s、TTFT3255.5733ms，16K为6979.6633 tok/s、TTFT27379.3633ms，精度101/105。4K未达到赛题门槛。GEMM/prefix、NN重排与分桶实验已结束，最终交付源码中的MetaX路径保持冻结原生实现。原始基线、NN试验及收尾说明附于evidence/。

官网赛题：https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

FlagTree天数安装文档：https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/multi-backend-prebuilt-docker-image-install/install-iluvatar.html

完整源码、原始记录与独立审计：https://github.com/Bubblevan/flagos-minicpm-throughput
