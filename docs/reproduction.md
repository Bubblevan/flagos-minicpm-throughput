# 复现与版本

官方来源：[赛题页面](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)。本机天数 BI-V150，driver4.5.0；torch2.10.0+corex4.5，vLLM0.24.0+empty，FlagTree0.6.2a3+iluvatar3.6，Triton3.6。

插件固定分支 flagos-2026-s2，实测SHA见 configs/versions.json。官方 FlagGems v5.3.5 的完整tree已在隔离目录还原。该副本HEAD仍为旧基点，必须同时检查tree和实际import路径，不能把HEAD误报成官方commit。

```bash
VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B --port 9031 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

精度完整原命令保存在 tools/run_official_accuracy.sh；105题 MATH-500 Level3，temperature1、top_p0.95、max_tokens32768、batch8。吞吐与精度串行运行，不叠加GPU诊断负载。

每个负载运行四轮，丢首轮后平均后三轮；吞吐 >= 天数参考99%，TTFT <= 参考101%，精度完整无请求错误且 >=0.95，才判定本项目复现目标通过。程序exit0本身不代表达标。
