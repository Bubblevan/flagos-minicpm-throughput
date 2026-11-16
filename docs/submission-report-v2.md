# TokenForge: MiniCPM5-2B 混合批次 Query Tiling

2026-11-16 可审查草稿。尚未正式提交，不代表组委会接收或复测通过。

## 1. 当前候选与结果

本作品选择天数 BI-V150 上的 mixed-query v2。它保持原 Attention Backend、原生 Triton unified attention kernel、KV 分页寻址和 KV 归约路径，只对受支持形状改变 query 分块。不是重写整套 Attention 算法，也不是通过切换 backend 获益。

2026-10-31 完整会话：4K/1024 输出平均2167.5333 total tok/s，Mean TTFT8050.3533ms；16K/1024 输出平均1323.2367 total tok/s，Mean TTFT398010.0933ms。每组原版四轮，丢首轮平均后三轮；4K每轮256成功、16K每轮128成功，完整输入输出 tokens 已独立核对。

本次新105题 MATH-500 Level3 精度100/105=95.2381%，105自然stop、零执行错误、零缺失、零空final。按当时既定门槛，性能与精度均通过。正确数恰为105题的最低过线数，再错一题即不达标；不能声称无损、普遍稳定通过或把其它会话精度拼入本次。

天数官方参考：4K2028.01tok/s、TTFT11573.53ms；16K915.15tok/s、TTFT599262.03ms。相对参考的吞吐增幅约6.88%和44.59%，TTFT降低约30.44%和33.58%。这些是对官方参考的观察比值，不是同机同时A/B因果增幅。

## 2. 实现与边界

完整插件以冻结提交13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0为基点，只改变vllm_fl/platform.py并新增vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py。PlatformFL.import_kernels安装launch adapter，仍调用原生kernel_unified_attention。

条件包括：BF16、head_dim128、16Q/2KV对应GQA8、原KV TILE_SIZE32、plain causal、2D、无量化/滑窗/TD等未覆盖特性；query_rows>=max(8*num_seqs,128)且query_rows>num_seqs。命中时BLOCK_M16变64、BLOCK_Q2变8，仍为4warps。其它调用和纯decode走原生launch，不移除主要算子选择逻辑。

v1要求平均每序列至少64query行，在混合prefill/decode中回退较多。v2用合成混合批次和匹配短测验证后将门槛调整为8，增加已测混合形状覆盖。平均query行只是形状代理，不证明每条序列都有8个query token。

该优化复用Q/K/V与KV页面，不增加模型权重副本，不改变模型、tokenizer、数据、采样、benchmark、固定serve参数或设备频率/功耗。候选adapter SHA256为5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325。

## 3. 实验与正确性证据

v2的12类合成batch涵盖纯decode、63条单token decode加1条prefill、多个prefill、16/32/64序列、4K/16K上下文及query行尾部。已测输入下与同环境原生输出逐位一致。这个差分只证明这些输入与原生相同，不能外推为所有shape、完整logits或所有生成结果的数学正确性证明。

此前独立CPU参考检查在长上下文通过，短不规则形状原生也存在偏差。11/16新增共享输入、独立CPU float64逐head参考及标量fsum复核；11类短形状的两份CPU参考最大差约7.8e-16。旧编译器原生与v2同样偏离该参考；保持源码和输入不变，隔离改用0.7.0后，全部16组在固定atol/rtol0.02下通过，v2仍与原生逐位一致、八次重复稳定。q8尾块最大绝对误差从0.6259降至0.0041，D盘重载张量独立重算也通过。此证据限定所测算子路径，不证明某道MATH错题的原因或所有完整模型logits均正确。新编译器模型验收另存新会话，不覆盖10/31成绩。

128请求、64并发、4K输入/512输出的匹配短测：v1均值2412.24，v2为3046.62tok/s，约+26.30%。这是诊断负载，不能替代原版1024输出完整四轮。正式成绩只采用第1节完整会话。

运行时worker记录及编译证据支持优化launch实际执行。首次marker不是全程覆盖率，CUDA Graph外的采样也不代表图内decode全部耗时。不能把此前prefill局部约93%的kernel时间统计当作全模型占比。

## 4. 失败候选与精度争议

11/09的v3完整会话：4K2141.3967tok/s、TTFT8188.6533ms；16K1320.1767、TTFT398614.3267ms；精度98/105，104stop加1max_tokens。性能通过但精度失败，整体false，因此没有替代v2，也没有重跑同一精度任务挑分。

v3原始远端overall_pass=true与accuracy_pass=false矛盾。原文件保留，本地派生结果修正为performance AND integrity AND accuracy。真实v3记录及8种布尔组合均用于聚合判定回归检查。

v3错题18(precalculus/1300)与87(precalculus/625)的矩阵数值正确，分别带[2pt]/[4pt]排版标记；题99(precalculus/1082)最终文本为8/15，评分器只提取15。这三题有独立数学核验和原始prediction/review，附在evidence/accuracy-forensics/供赛方核实。未经赛方确认，不修改官方98/105。

其它四题有真实最终答案错误：题25把正确中间位序4210写成4120；题7正文推得13535但boxed写53535；题90把五个diamond误抄为三个；题101把球与截角多面体混淆。不能把这些错误都归因编译器。唯一截断题83被官方评审判对，并非上述七题之一。

## 5. 环境、恢复与验收

硬件BI-V150单卡，driver4.5.0，CPU6核、内存20GiB；torch2.10/CoreX4.5，vLLM0.24。10/31已验证结果使用FlagTree0.6.2a3+iluvatar3.6。FlagGems锁定官方a7620cc191a0b42e040194622c5758b22a7a25dc，完整tree ddf3c7566aac7d23385a1b10c7c258091ef1d3a7；隔离副本HEAD仍为旧基点，不能将HEAD伪报为a762。

主办方群消息由用户11/16转录：更新到FlagTree0.7.0+iluvatar3.6可解决天数Causal attention数值问题，已有成绩不受影响，验收复测统一环境。官方安装文档也列出该版本。今晚在隔离target和全新cache对照；本草稿尚不把新编译器成绩写成已完成。

完整源码包405个文件，287个Python语法检查和20项CPU分派/回退契约通过，独立两次构建ZIP哈希一致。完整源码ZIP SHA256为ace45a56569999e010c9882d434e1d516d932790aabbc6497d703c1b23076cfc。此证据不等于已经在全新容器中完成GPU端到端恢复。

恢复时先检查无活跃模型，核对设备、源码/文件hash、实际import、依赖版本和固定argv；使用独立PYTHONPATH/源码/cache，保留原安装。回退时停止自己的服务并移除隔离路径及其环境变量，重新从原安装启动；不得把不同环境轮次或精度拼接。

serve保持FULL_DECODE_ONLY、port9031、served-model-name minicpm、gpu-memory-utilization0.85、max-model-len131072。性能原版用[[4096,1024,64,256],[16384,1024,64,128]]。精度保持temperature1、top_p0.95、max_tokens32768、batch8、timeout3600，核对105条prediction/review、索引0..104和finish reason；exit0不代表通过。

## 6. 来源和提交边界

官网赛题与提交页：https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

11/16已登录页面复核：作品提交截止2026-11-19 23:59:59，北京时间；正文仍写11/20，按较早日期规划。完整插件、readme.md及report.pdf为必需，FlagGems源码仅在修改其算子时必需。本候选没有修改FlagGems算子源文件，只有必要版本对齐。

官方编译器文档：https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/multi-backend-prebuilt-docker-image-install/install-iluvatar.html

社区问题1490/1492是不同模型或kernel的故障报告，仅作边界测试线索，不冒充本项目根因证据。完整原始证据和源码恢复材料见公开仓库https://github.com/Bubblevan/flagos-minicpm-throughput。沐曦候选只有短测观察、缺完整官方验收和新原始数据离机审计，本包不申报其优化成绩。

本报告是可审查草稿，仍需填入本次新编译器最终结论、完成环境恢复验证并由用户审阅后，才可决定正式提交。未向赛事平台上传作品，未发送issue、评论或群消息。
