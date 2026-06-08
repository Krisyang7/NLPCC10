import os
import sys
import json
import logging
import datasets
import evaluate
import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn

from transformers import AutoTokenizer

# ===================== 【关键】你的所有标签（全部写在这里）=====================
LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {v: k for k, v in label2id.items()}

# ===================== 读取你的 JSON 数据 =====================
def load_all_json_data(json_path="C:/Users/Lenovo1/Desktop/NLPCC-2026-Task10-Science-main (3)/NLPCC-2026-Task10-Science-main/data/traindev-track-1.jsonl"):
    sentences = []
    labels = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # 读取 sentence_label
            for item in data.get("sentence_label", []):
                sentence = item["sentence"]
                tag = item.get("types", [None])[0]  # 取第一个标签

                # 只保留我们定义的标签
                if tag and tag in label2id:
                    sentences.append(sentence)
                    labels.append(label2id[tag])

    df = pd.DataFrame({
        "text": sentences,
        "label": labels
    })
    return df

# 加载数据
df = load_all_json_data()

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("start training")

    # 划分训练/验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 转 HuggingFace Dataset
    train_dataset = datasets.Dataset.from_pandas(train_df)
    val_dataset = datasets.Dataset.from_pandas(val_df)

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ===================== 模型：多分类 =====================
    model = AutoModelForSequenceClassification.from_pretrained(
        'microsoft/deberta-base',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # 计算类别权重（处理类别不平衡）
    class_counts = df['label'].value_counts().sort_index().values
    total_samples = len(df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights)

    print("\n=== 类别分布 ===")
    for idx, (label, count) in enumerate(zip(LABELS, class_counts)):
        print(f"{label}: {count} samples, weight: {class_weights[idx]:.2f}")
    print(f"总样本数: {total_samples}\n")

    # 自定义Trainer，添加加权损失
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # 使用加权交叉熵损失
            loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # 多分类评估指标
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)
        macro_f1 = f1_score(labels, predictions, average='macro')
        macro_precision = precision_score(labels, predictions, average='macro')
        macro_recall = recall_score(labels, predictions, average='macro')
        return {**accuracy, "f1": macro_f1, "precision": macro_precision, "recall": macro_recall}

    class EpochCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            epoch = state.epoch
            if epoch is not None and metrics:
                print(f"\nEpoch {int(epoch)} 完成:")
                for key, value in metrics.items():
                    if key.startswith("eval_"):
                        print(f"  {key.replace('eval_', '')}: {value:.4f}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir='./checkpoint_deberta',
        num_train_epochs=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), EpochCallback()]
    )

    # 开始训练
    trainer.train()

    # 保存模型到指定位置
    trainer.save_model('./my_deberta_model')
    tokenizer.save_pretrained('./my_deberta_model')

    # 训练完成
    logger.info("训练完成！DeBERTa模型已支持多分类！")
