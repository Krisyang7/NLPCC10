# BERT 模型训练结果

## 模型信息
- **基础模型**: BERT-base-uncased
- **任务类型**: 5分类文本分类
- **标签**: Supported, Unsupported Causal Mechanistic, Unsupported Entity, Scope Overgeneralization, Contradiction

## 训练参数

| 参数 | 值 |
|------|-----|
| 批次大小 (训练) | 8 |
| 批次大小 (评估) | 16 |
| 最大序列长度 | 512 tokens |
| Weight Decay | 0.01 |
| Warmup Steps | 200 |
| Early Stopping Patience | 3 |
| 监控指标 | macro-F1 |
| 类别不平衡处理 | 加权交叉熵损失 |

## 类别分布与权重

| 标签 | 样本数 | 权重 |
|------|--------|------|
| Supported | 16438 | 0.21 |
| Unsupported Causal Mechanistic | 140 | 25.07 |
| Unsupported Entity | 213 | 16.48 |
| Scope Overgeneralization | 540 | 6.50 |
| Contradiction | 216 | 16.25 |
| **总计** | **17547** | - |

## 训练结果

| Epoch | Loss | Accuracy | F1 | Precision | Recall |
|-------|------|----------|-----|-----------|--------|
| 1 | 2.0262 | 0.9365 | 0.3052 | 0.2849 | 0.3352 |
| 2 | 1.8974 | 0.9393 | 0.3144 | 0.2885 | 0.3587 |
| 3 | 2.2565 | 0.9447 | 0.2949 | 0.3173 | 0.2821 |

**最佳模型**: Epoch 2 (F1 = 0.3144)

## 模型保存
- **路径**: `./my_bert_model/`
