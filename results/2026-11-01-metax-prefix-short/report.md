# C500 prefix metadata hoisting v1：短测阶段结果

用户11/01新租并自行开机的C500，未重新跑原始完整基线。用户明确只用Lab；
最初BatchMode测试拒绝后没有继续SSH，也没有改密码/公钥/认证。原安装未修改。

## 改动与已观察证据

冻结插件13eb9be69，MacaFlashAttentionBackend：Prefill调用MACA原生
flash_attn_varlen_func，Decode调用flash_attn_with_kvcache；BF16/GQA16Q:2KV/
head128。没有搬天数适配器，没有切换Attention backend。

唯一候选把Prefill KV长度前缀和从每层forward移入metadata builder，一次
GPU int32 cumsum、各层复用；去掉重复GPU .tolist()及H2D重建。纯decode无新增
前缀计算；外部构造metadata字段为None时回退原生实现。属于元数据/同步开销
优化，不是重写FlashAttention计算kernel。

5组GPU微诊断包含paged4K、16K、多prefill、不规则129/257查询，前缀元数据
和同一原生FA2输出均逐位一致。合成42层元数据路径约2–3ms降到约0.055ms，
仅为微路径，不是模型吞吐倍数。仍需更完整的元数据生命周期/graph复用验证。

## 有界模型 A/B/A

原生A1→候选B→原生A2，每条件新独立缓存，两种形状均预热+同种子两测：
4K为64请求/4096输入/256输出；16K为16请求/16384输入/128输出。
保留固定官方serve参数，三个条件均增加相同诊断profiler配置；仅A1测后单独
执行8请求profile，计时短测未开启采样。无完整官方分数或新精度声明。

| 对照两次原生均值 | 4K | 16K |
|---|---:|---:|
| 报告吞吐收益 | +11.5209% | +7.6359% |
| TTFT比值B/A | 0.82951 | 0.90976 |
| 原生A2/A1漂移 | +1.0119% | +0.4527% |

控制器报告每seed均正收益、prompt hashes/完整tokens匹配，预设短测GO通过。
这张表目前来自Lab终端所读控制器summary；**尚未从本机原始逐请求文件独立审计**。
不能将短输出分数与官方1024输出参考直接比较或推算必过。没有运行新105题。

三条件模型exit0，cleanup无己方存活进程组；控制器03点前已完成。
源文件hash见experiments/metax_prefix/manifest.json，原版和候选各自实际import与
argv/hash/cache信息保存在远端runtime.json，未冒充旧基线环境。

## 模型 profiler：已观察到什么

8步上限原生模型采样中有真实GPU kernel记录（非只有CPU）：主要名称包括
mm_kernel_nt、MACA flash_fwd_splitkv_kernel（head128、不同query/KV分块traits）。
采样中mm_kernel_nt为第一大项；Device→Pageable的DtoH事件336次，与42层×8步
的重复路径一致。不能据此把所有拷贝都唯一归因，也不能当作完整benchmark占比。
完整trace和解析JSON仍在远端，tools/analyze_metax_trace.py可离线重算。

## 备份阻塞：不要误说已全部异地保存

源码、补丁、控制器及天数完整源码候选已在D盘/GitHub。
新沐曦原始归档目前仍在远端/data/tokenforge-artifacts：

- metax-prefix-micro-20261101.tar.gz（约56.8KB）
- metax_prefix_trial_20261101_0210.tar.gz（约1.4MB，含逐请求、服务日志和模型trace）

自动下载事件和downloadMedia均超时；无登录态HTTP请求只得到登录HTML，已经在
D盘明确命名failed-download-response.html，没有当成成功tar。CUA无法向D盘写文件，
即使本轮文件系统授权已申请也返回EPERM。未改用SSH、未关闭认证或公开数据盘。
已经请用户正常从Lab下载上述两份归档到
D:/MyLab/FlagOS/lab-records/outputs/metax_prefix_20261101/，等待完成后核验。

临时BACKUPS.md/PRIVATE-DOWNLOAD.md含本实例现有认证链接，禁止上传、打包或
复制入公开项目。既有tar在这些文件创建前生成，打包的是特定会话目录，
不含外层临时下载文档。关机前会清除自己生成的下载文档中的认证参数。

## 下一步门禁

先完成原始归档离机、hash/CRC/成员路径与凭证检查，并独立重算所有A/B/A逐请求
和均值/漂移/原生kernel路径。然后在明确的新使用窗口内，用候选完整原版八轮
与新105题验收，禁profiler，保留原始采样。沐曦参考为5089.645/7029.675tok/s；
目标高参考1%为5140.54145/7099.97175；这不是当前候选已取得的完整成绩。
若未过门槛则保留反例，不反复重抽精度，不破坏天数已验证实现。
## 本轮最终处理（用户确认）

用户要求停止下载尝试，按现状提交并推送Git；本轮不再以下载进度100%冒充
文件已落盘。随后用户明确授权“现在关机保留数据”，平台实际确认该实例已关机、
开关unchecked、Lab已禁用；没有销毁实例。首次关机曾被自动审批拒绝，取得本次
明确授权后才完成，没有用Linux或其他方式绕过拒绝。

平台关机时显示数据保留1天11小时59分；这不是永久备份。新原始记录仍仅在远端，
当前GitHub只有源码、工具、本文及已读汇总，没有完整新沐曦逐请求日志/trace。
10/29原始沐曦基线归档和天数旧完整归档均已异地核验，不受本轮下载问题影响。

额外生成了覆盖两个会话目录、全部服务/控制器日志和profiler解析的完整单包：
/data/tokenforge-artifacts/metax-prefix-session-20261101-complete.tar.gz
大小1,534,665bytes，远端gzip CRC通过，SHA256：
`41d3f1738870fa5ec722873f568683252fec27c2a38dcb4458a468bcd713db63`。
原来的两份tar未改动；没有将源码、缓存、模型或临时认证文档混入完整证据包。

自己创建的BACKUPS.md和PRIVATE-DOWNLOAD.md中的认证参数已在关机前清除，
没有改变服务器认证。CUA写D的EPERM没有被用C盘临时落地绕开，也未公开认证链接。
后续如用户决定补取，只用新开机后的正常Lab Download入口；先验证上述SHA、
安全解压，再独立重算原始逐请求和trace。不延用旧PID，也不恢复聊天定时任务。
