# TokenForge / MiniCPM5-2B / 天数 v2

**已完成本地完整验收的候选，待用户审阅，尚未正式提交。**
11/16冻结v2+FlagTree0.7.0完整八轮和本次105题已完成：4K2435.35tok/s、TTFT5639.32ms；
16K1890.7267tok/s、TTFT268099.60ms；精度101/105=96.1905%，整体通过既定门槛。
104stop、1max_tokens，0执行错误/缺失/空final。完整结果不与旧会话拼接。
report.pdf为技术报告，evidence包含本轮完整原始记录、独立审计、评分争议及历史对照。

## 包内文件

平台范围：本包声明天数优化；MetaX源码保留原生路径，没有承诺沐曦性能达标。
两平台使用各自官方镜像；不要在沐曦环境安装下面的Iluvatar专用编译器wheel。

- vllm-plugin-FL/：完整冻结插件源码，保留全部vendor、LICENSE、构建脚本和benchmark。
- report.pdf、report-source.md：当前技术报告和可编辑源文件。
- tools/validate_accuracy_artifacts.py：不运行模型的逐题完整性与精度审计。
- evidence/：已完成结果、原始完整v2归档、v3错题法证及编译器差分摘要。
- REVIEW_MANIFEST.json：本包逐文件SHA256。没有权重、数据集、密钥或编译缓存。

## 环境与源码身份

单卡BI-V150，CoreX4.5/driver4.5.0，torch2.10，Python3.12，vLLM0.24。
插件基点13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0，基于官方flagos-2026-s2。
只修改platform.py并新增iluvatar/impl/attention_tiles.py；adapter SHA256：
5bafdb0bf547e0567154ebf6a9d57465e677ea707b5f019832aaf63d77a4d325。
FlagGems使用v5.3.5官方commit a7620cc191a0b42e040194622c5758b22a7a25dc，完整tree：
ddf3c7566aac7d23385a1b10c7c258091ef1d3a7。未改其算子，不申报Gems原创优化。

旧成绩编译器为FlagTree0.6.2a3+iluvatar3.6。主办方建议并将在验收复测统一使用
更新环境；11/16按其推荐对0.7.0+iluvatar3.6做隔离测试。原安装不要覆盖；新版本
完整模型已在本次新会话通过，不能把旧精度借给新编译器。

## 恢复与导入冒烟（在官方Linux容器中执行）

先确认有效预约与截止、GPU无其它模型/评测。解压本包到新目录，进入TokenForge/。
以下变量均为本次新目录，GEMS_ROOT必须指向已经核对完整tree的Gems源码：

```bash
export PACKAGE_ROOT="$PWD"
export PLUGIN_ROOT="$PACKAGE_ROOT/vllm-plugin-FL"
export GEMS_ROOT=/workspace/tokenforge-sources/FlagGems-a7620cc
test -f "$PLUGIN_ROOT/vllm_fl/platform.py"
test "$(git -C "$GEMS_ROOT" write-tree)" = ddf3c7566aac7d23385a1b10c7c258091ef1d3a7
git -C "$GEMS_ROOT" diff --quiet
sha256sum "$PLUGIN_ROOT/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_tiles.py"
```

如Gems目录不存在或tree不同，先按官方冻结版本恢复并重新核对，不直接升级main。
基镜像已有冻结插件的fl入口。只通过PYTHONPATH加载包内完整源码，避免改原安装。
编译器也放隔离target，保持torch等依赖不变：

```bash
python3 -m pip install --no-deps --no-compile \
  --target "$PACKAGE_ROOT/compiler-0.7.0" \
  --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
  'flagtree===0.7.0+iluvatar3.6'
export VLLM_PLUGINS=fl
export PYTHONPATH="$PACKAGE_ROOT/compiler-0.7.0:$PLUGIN_ROOT:$GEMS_ROOT/src"
python3 - <<'PY'
import importlib.metadata as m, triton, vllm_fl, flag_gems
print(m.version('flagtree'),triton.__file__,vllm_fl.__file__,flag_gems.__file__)
assert m.version('flagtree') == '0.7.0+iluvatar3.6'
PY
```

人工确认三个import文件来自对应隔离目录，而非原安装。新建RUN和cache，分别保存
argv、依赖版本、源码hash及日志：

```bash
export RUN="/workspace/tokenforge-artifacts/tokenforge-v2-$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN"
export TRITON_CACHE_DIR="/workspace/tokenforge-caches/$(basename "$RUN")/triton"
export VLLM_CACHE_ROOT="/workspace/tokenforge-caches/$(basename "$RUN")/vllm"
export TORCHINDUCTOR_CACHE_DIR="/workspace/tokenforge-caches/$(basename "$RUN")/inductor"
python3 -m pip freeze > "$RUN/pip-freeze.txt"
cd "$RUN"
```

只在用户确认的预约中启动，给负载设置提前5分钟
截止和进程组清理。此处列出官方命令本体，不能脱离截止封装直接无人值守运行。

## 固定服务和评测

```bash
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
```

等待health200、Graph捕获完成，确认加载本包v2。另一个终端继承同一环境；进入新RUN：

```bash
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

使用基镜像原版benchmark，不能修改它。每组4轮、丢首轮取后三轮均值；原始CSV必须
8行SUCCESS、每轮4K256/16K128成功，请求输入/输出tokens完整。两组total tok/s分别
>=2028.01*0.99、915.15*0.99；TTFT分别<=11573.53*1.01、599262.03*1.01。

性能结束后再串行一次精度；只改变证据保存目录，固定采样参数不变：

```bash
evalscope eval --model minicpm --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets math_500 \
  --dataset-args '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}' \
  --eval-batch-size 8 --timeout 3600 \
  --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' \
  --work-dir "$RUN/level3" --ignore-errors
```

以实际生成的时间戳目录替换下面RUN_DIR，再运行CPU审计：

```bash
python3 "$PACKAGE_ROOT/tools/validate_accuracy_artifacts.py" \
  --run-dir "$RUN_DIR" --out "$RUN/accuracy-audit.json"
```

审计退出0仅代表记录完整，仍须检查accuracy_pass。105索引0..104唯一完整、无错误
或空final，正确数至少100才达>=0.95。记录每题stop/max_tokens，不修改评分器、不
挑分重跑。整体pass必须是完整性、性能、精度三者同时通过。评分争议原始分数不改，
附原始prediction/review和独立数学报告，交赛方核实。

## 回退与数据保护

停止本次自己启动的评测和模型进程组，确认无存活worker。保留新证据再移除当前
shell的隔离PYTHONPATH及cache环境变量，从原安装重新加载；不要删原环境/旧缓存。
因为没有覆盖site-packages，回退不需要卸载或重新安装旧编译器。不同会话不可拼分。
完整备份并验证hash后才在平台关机；不要销毁实例。

10/31历史v2完整成绩：4K2167.5333/8050.3533ms，16K1323.2367/398010.0933ms，
精度100/105。它不是新编译器的成绩，也不是官方接收确认。本包已包含本轮独立结果及完整证据，正式比赛上传仍需用户审阅确认，不自动提交。
