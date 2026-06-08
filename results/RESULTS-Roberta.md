# DeBERTa 模型训练结果

## 模型信息
- **基础模型**: DeBERTa-base
- **任务类型**: 5分类文本分类
- **标签**: Supported, Unsupported Causal Mechanistic, Unsupported Entity, Scope Overgeneralization, Contradiction

## 训练参数

| 参数 | 值 |
|------|-----|
| 批次大小 (训练) | 4 |
| 批次大小 (评估) | 8 |
| 最大序列长度 | 512 tokens |
| 学习率 | 1e-5 |
| Weight Decay | 0.01 |
| Warmup Steps | 200 |
| Early Stopping Patience | 3 |
| 监控指标 | macro-F1 |
| 混合精度训练 | FP16 |
| 梯度累积步数 | 2 |
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
| 1 | 1.5945 | 0.9550 | 0.3447 | 0.3205 | 0.3789 |
| 2 | 1.6402 | 0.9558 | 0.4431 | 0.4053 | 0.5202 |
| 3 | 1.4756 | 0.9581 | 0.4647 | 0.4203 | 0.5315 |
| 4 | 1.1910 | 0.9476 | 0.4511 | 0.4131 | 0.5367 |

**最佳模型**: Epoch 3 (F1 = 0.4647)

## 模型保存
- **路径**: `./my_deberta_model/`
