# 天数 v2 完整源码候选：恢复与验收

该候选在冻结插件13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0上仅修改
vllm_fl/platform.py、新增iluvatar/impl/attention_tiles.py。保留全部上游源码、
LICENSE、构建文件、各vendor与benchmark。不是另一个Attention backend，
属于shape-aware query tiling / launch geometry；不宣称重写Attention算法。

已验证：冻结tree导出、补丁应用、两文件差集、适配器SHA、全体Python语法、
CPU分派/回退契约及ZIP CRC。完整GPU记录来自10/31隔离候选；本源码包尚未在
全新比赛容器端到端恢复验收。提交候选不是已提交作品。

## 构建

在现有项目用标准库Python执行：

```powershell
python tools/build_iluvatar_candidate.py --upstream D:/MyLab/FlagOS/sources/vllm-plugin-FL-frozen --out-dir D:/MyLab/FlagOS/releases/iluvatar-v2-20261101
```

输出ZIP含完整vllm-plugin-FL/、SOURCE_MANIFEST.json、此说明。源目录必须是
干净冻结commit，输出目录必须不存在，避免覆盖旧候选。zip不含.git、凭证、
模型或编译缓存。manifest逐文件hash可在任意系统独立复核。

## 新比赛容器恢复（需新的有效预约）

1. 解压到新隔离路径，确认已有服务和GPU负载，保留原安装与原Gems。
2. 以SOURCE_MANIFEST.json逐文件SHA核对，禁止只检查同名tag；Gems必须等价
   官方a7620cc191a0b42e040194622c5758b22a7a25dc/tree
   ddf3c7566aac7d23385a1b10c7c258091ef1d3a7。硬件需为BI-V150。
3. 在比赛已有依赖环境使用候选，不升级torch/vLLM/Triton等。
   最小恢复采用PYTHONPATH指向解压后的插件根与官方Gems根，沿用已有安装提供的
   fl入口点；这是已测隔离加载方法。先导入并记录vllm_fl.__file__、flag_gems.__file__，
   不能把已有安装路径冒充候选。若需要安装源码包，先审阅冻结setup.py；用已装构建
   依赖并禁依赖解析/构建隔离，VLLM_VENDOR=iluvatar，setuptools-scm无.git时显式
   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_PLUGIN_FL=0.3.0rc1+tokenforge.v2。
   安装路径尚未在新容器验收，不把这段说明当作安装成功证明。
4. 用同一固定serve参数启动隔离服务，先查health与实际worker加载路径，做最小
   请求冒烟和差分矩阵。纯decode/不支持特性保留原始launch。适配器SHA应为
   5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325。
5. 固定服务参数：

```bash
VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

6. 先用有界冒烟发现恢复错误；只有新容器恢复门禁通过且时间足够才做正式8轮和
   新105题。不为精度100/105的边缘分数反复抽样挑高分，采样设置不变。
7. 回退：停止自己进程组，移除该会话PYTHONPATH覆盖，重新使用原冻结安装；
   不在模型运行时切换文件，不覆盖原安装，不全机pkill。

## 解释范围

10/31完整验收：4K2167.53、16K1323.24tok/s，精度100/105。此为同次完整记录，
不是全新容器恢复成绩。12组与native逐位一致不构成所有形状的数学证明；短不规则
形状原native也有独立reference差异，仍待最小复现。精度余量小，不声称无损。

沐曦路径保持上游原样。该适配器有iluvatar guard，不能声称它优化了C500。