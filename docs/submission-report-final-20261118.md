# TokenForge: MiniCPM5-2B 混合批次 Query Tiling

2026-11-18 最终整理版。天数v2已完成本地完整验收，沐曦探索已结束且未通过。本包待用户审阅与上传，不代表组委会接收或复测通过。

## 1. 本轮候选与完整结果

候选为天数BI-V150的冻结mixed-query v2加主办方推荐的FlagTree0.7.0+iluvatar3.6。v2代码未再改动，保留原Backend、原生unified attention kernel及KV寻址与归约路径，只对受支持形状改变query分块。新编译器修复是主办方提供的环境更新，不归入我们的原创优化。

本轮于11/16 21:28:36完成：4K为2435.35 total tok/s、Mean TTFT5639.32ms；16K为1890.7267tok/s、Mean TTFT268099.60ms。原版每组四轮，丢首轮平均后三轮；全部八轮SUCCESS，每轮4K256/16K128请求及完整输入输出tokens均通过独立重算，与原版summary一致。

本轮新MATH-500 Level3为101/105=96.1905%，105条prediction/review完整唯一、无执行错误/缺失/空final，104stop、1max_tokens（题83，判错）。原始评分不改，精度超过0.95；整体判定为性能AND完整性AND精度，本轮true。它仍只是一轮temperature1的随机采样记录，不保证所有未来运行过线。

对既有官方参考2028.01/915.15tok/s，本轮吞吐分别高20.09%/106.60%；TTFT均低于各参考11573.53/599262.03ms的101%上限。这是对参考的观察比值，不是同时A/B单因增幅。不能将新编译器带来的收益全部包装为原创tiling收益。

10/31旧编译器v2的2167.5333/1323.2367tok/s、100/105作为历史对照保留。11/09 v3完整98/105失败也保留；本轮未借用任何旧轮次或旧精度、未重跑挑分。模型及评测服务21:28:40正常停止，D盘完整副本与远端逐文件哈希匹配。

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

## 4. 本轮精度争议与失败候选历史

本轮错题为18、87、101、83。18和87矩阵的数值分别为(-1/3,2/3,5/3)和(16/49,48/49,24/49)，满足题设，但输出带[4pt]行距命令后被判错；独立数学复核支持等价，原始prediction/review及证明已附accuracy-disputes.json，交赛方核实，不自行改写101/105。101为错误的几何答案216π，83触及固定32768token上限且未完成最终答案，不提高上限、不补采样。

11/09的v3完整会话：4K2141.3967tok/s、TTFT8188.6533ms；16K1320.1767、TTFT398614.3267ms；精度98/105，104stop加1max_tokens。性能通过但精度失败，整体false，因此没有替代v2，也没有重跑同一精度任务挑分。

v3原始远端overall_pass=true与accuracy_pass=false矛盾。原文件保留，本地派生结果修正为performance AND integrity AND accuracy。真实v3记录及8种布尔组合均用于聚合判定回归检查。

v3错题18(precalculus/1300)与87(precalculus/625)的矩阵数值正确，分别带[2pt]/[4pt]排版标记；题99(precalculus/1082)最终文本为8/15，评分器只提取15。这三题有独立数学核验和原始prediction/review，附在evidence/accuracy-forensics/供赛方核实。未经赛方确认，不修改官方98/105。

其它四题有真实最终答案错误：题25把正确中间位序4210写成4120；题7正文推得13535但boxed写53535；题90把五个diamond误抄为三个；题101把球与截角多面体混淆。不能把这些错误都归因编译器。唯一截断题83被官方评审判对，并非上述七题之一。

## 5. 环境、恢复与验收

硬件BI-V150单卡，driver4.5.0，CPU6核、内存20GiB；torch2.10/CoreX4.5，vLLM0.24。本轮使用隔离FlagTree0.7.0+iluvatar3.6；10/31历史成绩使用0.6.2a3。FlagGems锁定官方a7620cc191a0b42e040194622c5758b22a7a25dc，完整tree ddf3c7566aac7d23385a1b10c7c258091ef1d3a7；隔离副本HEAD仍为旧基点，不能将HEAD伪报为a762。

主办方群消息由用户11/16转录：更新到FlagTree0.7.0+iluvatar3.6可解决天数Causal attention数值问题，已有成绩不受影响，验收复测统一环境。官方安装文档也列出该版本。11/16在隔离target和全新cache完成算子对照及一次完整模型验收，原安装未改；本轮成绩均取自新会话，历史成绩与参考单独标注。

完整源码包405个文件，287个Python语法检查和20项CPU分派/回退契约通过，独立两次构建ZIP哈希一致。完整源码ZIP SHA256为ace45a56569999e010c9882d434e1d516d932790aabbc6497d703c1b23076cfc。本轮已从该ZIP恢复到新的隔离源码目录并完成端到端验收；没有进行不含任何历史文件的全新镜像重建，不把静态构建检查冒充这一验证。

恢复时先检查无活跃模型，核对设备、源码/文件hash、实际import、依赖版本和固定argv；使用独立PYTHONPATH/源码/cache，保留原安装。回退时停止自己的服务并移除隔离路径及其环境变量，重新从原安装启动；不得把不同环境轮次或精度拼接。

serve保持FULL_DECODE_ONLY、port9031、served-model-name minicpm、gpu-memory-utilization0.85、max-model-len131072。性能原版用[[4096,1024,64,256],[16384,1024,64,128]]。精度保持temperature1、top_p0.95、max_tokens32768、batch8、timeout3600，核对105条prediction/review、索引0..104和finish reason；exit0不代表通过。

## 6. 来源和提交边界

官网赛题与提交页：https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

11/18已登录提交页再次复核：作品提交截止2026-11-19 23:59:59，北京时间；正文仍写11/20，按较早日期规划。完整插件、readme.md及report.pdf为必需，FlagGems源码仅在修改其算子时必需。本候选没有修改FlagGems算子源文件，只有必要版本对齐。

官方编译器文档：https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/multi-backend-prebuilt-docker-image-install/install-iluvatar.html

社区问题1490/1492是不同模型或kernel的故障报告，仅作边界测试线索，不冒充本项目根因证据。完整原始证据和源码恢复材料见公开仓库https://github.com/Bubblevan/flagos-minicpm-throughput。11/18结束沐曦探索：GEMM/prefix完整4K及NN重排A/B/A未达标，分桶候选在新卡数值门禁失败，未合入作品。完整插件的MetaX路径保持冻结原生实现；本包不申报沐曦优化通过。原始基线、NN否决证据及收尾说明附于evidence/。

本报告对应已完成本地验收的新编译器v2候选，正式提交前需用户审阅压缩包及恢复说明。未向赛事平台上传作品，未发送issue、评论或群消息。
