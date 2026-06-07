# Track 1 实验结果记录

NLPCC 2026 Shared Task 10 - Track 1：面向实验结果的陈述级忠实性判定

## 方法

用 BERT 做句子级 5 分类。脚本：`baseline_prompting/track1_bert_trainer.py`

- 预训练模型：`bert-base-uncased`
- 任务：对 claim 段落中的每个句子，判断属于 5 类标签中的哪一类
- 标签：Supported / Unsupported Causal Mechanistic / Unsupported Entity / Scope Overgeneralization / Contradiction
- 输入：句子文本（**本版本未拼接证据材料**）
- 数据：`data/traindev-track-1.jsonl`，按 8:2 划分训练 / 验证集

## 训练配置

| 项 | 值 |
|---|---|
| epochs | 6 |
| train batch size | 8 |
| eval batch size | 16 |
| warmup steps | 200 |
| weight decay | 0.01 |
| max length | 512 |

## 验证集结果

| epoch | accuracy | macro-F1 |
|---|---|---|
| 1 | 0.9322 | 未统计 |
| 6 | 0.9322 | 未统计 |

## 已知问题与后续改进

1. **accuracy 虚高**：约 93% 样本为 `Supported`，模型倾向于全预测 Supported，
   导致 accuracy 高但实际区分能力弱。比赛真正指标是 **macro-F1**，应以此为准。
2. **当前未统计 macro-F1**：`compute_metrics` 只算了 accuracy，需补上 macro-F1。
3. **未使用证据材料**：当前只用句子文本，模型看不到 evidence。
   下一步应改为句对输入 `(evidence, sentence)`，让模型基于证据判断。
4. **类别不平衡**：可考虑加权损失或对少数类过采样。
