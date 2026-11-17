# TokenForge 双平台候选复现说明（待沐曦完整验收）

本说明对应包含完整插件和 FlagGems 的集成候选。天数 v2＋FlagTree0.7.0 的完整验收已经通过；沐曦 GEMM 候选目前只有数值与匹配 A/B/A 门禁通过，不能把此草稿当成双平台已通过作品。原始完整天数包继续保留。

## 1. 文件与版本

最终目录应为 `TokenForge/readme.md`、`TokenForge/report.pdf`、`TokenForge/vllm-plugin-FL/`、`TokenForge/FlagGems/`，另附审计工具、逐文件 manifest 和原始证据。所有文件以提交包 manifest 为准；不包含模型权重、数据集、凭证或编译缓存。

- 插件冻结基点：`13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`。
- FlagGems 基点：`a7620cc191a0b42e040194622c5758b22a7a25dc`。
- 集成源码共 5,186 文件。MetaX 仅修改原生 NT GEMM 的受保护分块；MetaX Attention 恢复冻结原版，插件保留天数 v2 的两个精确源文件。
- 当前集成源码 ZIP：`artifacts/FlagOS-dual-gemm-only-candidate-source-20261117.zip`，SHA256 `c3d8ad41789bbbf64f30e47ad291cf6ffdc5c207d7d0e52b873659e1d3895146`。上一GEMM＋prefix组合四轮4K失败；本次GEMM-only尚待完整验收，不宣称prefix已被证明是原因。
- 基点提交、源码快照和已安装 distribution 版本是不同概念；完整快照没有伪造新的 Git HEAD。

## 2. 指定运行环境

使用比赛对应的官方镜像及平台。不要在两种硬件之间混装厂商运行库。

| 项目 | 天数 BI-V150 | 沐曦 C500 64GB |
|---|---|---|
| Torch | 2.10 / CoreX4.5 | 2.8.0+metax3.7.0.7 |
| vLLM | 0.24 | 0.24.0+empty |
| 插件 | 冻结13eb9be69 | 冻结13eb9be69 |
| Triton/编译器 | FlagTree0.7.0+iluvatar3.6 | Triton3.0.0+metax3.7.0.7 |
| EvalScope | 使用验收记录的镜像版本 | 1.11.1 |

天数新编译器采用主办方指定版本，按已验收的隔离安装方案恢复；不要升级镜像中其他依赖。C500 不安装天数的 FlagTree。当前正式 C500 资源实际为12核/64GiB宿主内存、driver3.8.30、MACA3.7.0.38；前一台48GiB/driver3.9.35资源的诊断不混为本次结果。

## 3. 恢复源码和基础检查

以下均为远端 Linux 命令，不能直接贴入 Windows PowerShell。先确认新的有效预约/预算、进程与健康状态，不重启已有健康服务。每次使用新的证据 ROOT、缓存和提前5分钟截止。

```bash
cd /path/to/TokenForge
export SUBMISSION_ROOT="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=fl
export PYTHONPATH="$SUBMISSION_ROOT/vllm-plugin-FL:$SUBMISSION_ROOT/FlagGems/src"
python - <<'PY'
import hashlib, json
from pathlib import Path
root=Path.cwd()
manifest=json.loads((root/'source-manifest.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
print('all source hashes match')
PY
python - <<'PY'
import importlib, torch, vllm_fl, flag_gems
print(vllm_fl.__file__)
print(flag_gems.__file__)
print(torch.__version__)
torch.cuda.set_device(0)
x=torch.ones(1,device='cuda:0')
torch.cuda.synchronize(0)
print(torch.cuda.get_device_name(0),x.item())
PY
```

运行路径必须指向包内源文件。C500 还需核对 `flag_gems.runtime.backend._metax.ops.mm` 的实际路径与 manifest；天数核对 `attention_tiles.py`。若最小原生同步出现 MACA 队列创建超时，先恢复可用设备，不启动正式模型，也不使用 CPU 结果替代 GPU 成绩。

官方镜像已提供 `fl` entrypoint、vLLM/FlashAttention等匹配二进制依赖。本作品的改动使用 Python/Triton，首次运行由 Triton JIT 编译。上游 `setup.py`、`pyproject.toml` 和 `csrc` 构建脚本完整保留；本候选无需重编译外部 vLLM/FlashAttention。上游可选 C++ 构建分支仅支持 CUDA，不要设置 `VLLM_VENDOR=metax` 运行该分支。本说明不声称完成了任意裸系统从零安装。

## 4. 模型和数据

使用镜像中的 `/workspace/MiniCPM5-2B` 和 `/workspace/evalscope-datasets/math_500`，保留原始配置、权重、分词器和数据。模型为42层、hidden2048、16Q/2KV头、D128、BF16。保存 `config.json` 的 SHA256、实际包版本、设备信息及运行命令。不要把权重或数据集加入提交包。

## 5. 固定服务命令

C500 启动前设置 `TORCHINDUCTOR_COMPILE_THREADS=1`。这是避免宿主编译并发OOM的运行修复，不算原创收益。源码和缓存位于证据ROOT之外；不要调整频率、功耗、CPU governor/cgroup或官方服务参数。

天数：

```bash
VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

C500：

```bash
TORCHINDUCTOR_COMPILE_THREADS=1 VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

确认 `/health` 返回200及实际加载路径。正式验收禁用 profiler、developer route 和历史诊断模式文件。

## 6. 性能和本次新精度

在新ROOT下的空 `official-benchmark` 目录运行冻结原脚本，不覆盖旧CSV：

```bash
VLLM_PLUGINS='' CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES='' python /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

上述环境只作用于发送HTTP的CPU客户端，服务端仍为 `fl` GPU平台。这样避免测试客户端重复初始化GPU运行时；请求生成、原脚本、采样与统计均不变。客户端 CPU-only 冒烟必须先通过。

性能完成后串行运行所附 `tools/run_official_accuracy.sh`，用 `FLAGOS_EVAL_WORK_DIR` 指定本ROOT的新目录。脚本固定为 MATH-500 Level3 105题、temperature1.0、top_p0.95、max_tokens32768、batch8、timeout3600、openai_api。HTTP精度客户端同样隐藏GPU；不能修改评分器、输出上限或采样，也不能因低分反复挑分重跑。

## 7. 独立验收和回退

- 两组各四轮；每4K轮256成功、每16K轮128成功，输入/输出token分别精确为请求数乘4096或16384、乘1024。
- 丢首轮平均后三轮，与原版summary交叉核对。C500参考为5089.645/7029.675 total tok/s，TTFT3199.435/27197.135ms。吞吐至少参考99%，TTFT至多参考101%，两组均需满足。
- 精度读取report、prediction、review；索引0..104完整唯一、无错误/缺失/空final，重算精度>=0.95，并报告真实stop/length等finish reason。exit0不代表通过。
- 用 `tools/audit_metax_mm_session.py --root <保存的ROOT> --source-zip <完整源码ZIP> --out <新的审计JSON>` 做CPU独立审计。它拒绝不完整八轮，不借用旧会话105题，也不覆盖既有审计。
- 回退C500 GEMM：移除修改后的 `FlagGems/src` 的PYTHONPATH，恢复冻结原生包。回退天数：使用受保护的已验收源码包。保留失败反例和所有原始记录。

完成或截止时只停止自身进程组，完整备份原CSV、精度report/预测/review、日志、环境、实际import和源码hash。最终文件对照远端哈希并扫描凭证后再Git推送。停止模型不等于供应商实例关机；供应商操作按用户当次授权执行。
