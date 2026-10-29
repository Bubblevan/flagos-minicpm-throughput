# Iluvatar query tile candidate

目标是已经采样确认的 native Triton unified_attention prefill 热点。这里不替换Attention backend，而是在原kernel的launch接口中调整query并行分块：BLOCK_M16→64、BLOCK_Q2→8。KV tile32、4warps和KV归约顺序保持原样。只对已验证的BF16/GQA8/head128/plain causal prefill启用，其余特性和decode保留原launch。

候选通过平台的Iluvatar import_kernels路径加载。源码补丁适用于插件冻结commit13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0，不需要修改原vLLM安装。实际独立副本及源码hash保存在实验manifest，正式测试时禁止改动。

## 已有证据

| 测试 | 原版 | 候选 | 范围 |
|---|---:|---:|---|
| 单算子：1024 query / 4096 KV | 8.962ms | 4.194ms | BF16，输出逐位相同 |
| 单算子：2048 query / 16384 KV | 75.192ms | 33.386ms | BF16，输出逐位相同 |
| 4K短测 total tok/s | 2910.17 | 3410.04 | 64请求，4096输入/256输出 |
| 16K短测 total tok/s | 1064.47 | 2016.48 | 8请求，16384输入/128输出 |

短测先同形状预热，然后平均两轮，不是官方满负载结果。7组张量差分覆盖非整齐尾部、混合长度、decode及head64回退，均与原版逐位一致。原版短不规则shape也未通过独立数学参考，差异仍待查；未通过放宽容差隐藏该问题。更快的KV tile64方案存在数值差异，未用于模型候选。

编译IR确认实际产生64×128 query矩阵；worker独立JSON记录installed和optimized_kernel_launched，避免仅凭命令行或INFO日志判断已启用。

## 正式验收

候选的完整官方8轮benchmark及本次105题精度已于2026-10-29 19:43启动，结果尚未完成。固定serve、原版benchmark、模型、数据与采样不变。必须核验四轮完整请求/tokens、丢首轮平均后三轮、各平台参考阈值及本次精度，才能给出正式收益结论。
