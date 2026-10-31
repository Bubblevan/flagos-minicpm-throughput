# 天数 v1/v2 正确性定点审查（不重抽完整精度）

## 已核验的保存记录

离线读取两次完整105题预测，按index0..104一一对应，比较实际角色/内容，
排除EvalScope每会话随机message UUID。105题question_id与实际prompt全部一致。
全部105条生成全文均不同；这里比较的是文本字符，不是token logits或随机状态。
v1错误18/87/90，v2错误18/70/87/90/101，新增错误70/101，无新增正确题。
所有请求均自然stop，没有执行错误或缺失。原记录未修改，没有新调用模型。
详见saved-generation-comparison.json及tools/analyze_saved_accuracy_divergence.py。

这不能证明额外两道错题由优化导致，也不能证明全部来自随机性。
官方temperature1的不同会话不是相同RNG状态的逐步重放；没有保存完整logits。
12组算子逐位一致仅覆盖既定矩阵，不是所有模型真实shape或所有logits的证明。

## 短不规则独立参考：本次审阅到哪里

复查tools/tune_attention_tiles.py：物理page随机置换，按block_table收集K/V，
截断到真实KV长度，GQA按连续8个Q head映射一个KV head，causal位置为
kv_len-query_len+query_position，CPU float32计算softmax(QK*scale)V。
旧记录的两个长上下文参考通过；短不规则原native和候选均有差异。
本次没有拿沐曦代替天数重现，也没有改容差、改原数据或宣布该差异已解决。
旧记录只保留种子/源码/误差摘要，未保存失败输入张量，无法仅从摘要确定
GPU RNG输入、原native掩码还是CPU参考哪一侧出错。

## 下一次BI-V150授权窗口的最小验证

1. 冻结已通过的v2，另开诊断副本；保存原生/候选实际文件hash。
2. 先复现原native短不规则失败样本，保存完整Q/K/V、block_table、cu_q、seq_len。
   核验页边界、stride、dtype、GQA head映射、causal offset、所有输出是否写入。
3. 对同一已保存输入做CPU显式逐head循环与向量参考对照，再与native和候选比；
   对首/尾/页边界及失败坐标定位，不能降低容差将问题改成通过。
4. 用真实MiniCPM输入和相同KV状态，在采样前比较逐层Attention输出及完整logits。
   先定位第一处分歧；若logits相同，再用同一RNG状态比较下一token。
   只比较相同token前缀，不能在生成已分叉后把后续差异都归于kernel。
5. 覆盖prefill、混合批次、纯decode及回退特性；保留任何反例。
   在证据明确前不称“无损”，不反复完整MATH-500挑选更高分。

本次无新BI-V150时段授权，以上GPU/logits步骤尚未执行。C500上的同值元数据
验证属于另一个平台的候选，不能补足天数的这项正确性证据。