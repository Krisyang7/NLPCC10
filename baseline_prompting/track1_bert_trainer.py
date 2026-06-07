import os
import sys
import json
import logging
import datasets
import evaluate
import pandas as pd
import numpy as np

from transformers import BertForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from transformers.models.bert.tokenization_bert import BertTokenizerFast

# ===================== 【关键】你的所有标签（全部写在这里）=====================
# 把你所有7种标签全部写进去，我先按常见科学论文证据标签给你补齐
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
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ===================== 模型：多分类 =====================
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label2id),  # 自动变成 7 分类
        id2label=id2label,
        label2id=label2id
    )

    # 多分类评估指标
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 训练参数
    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 保存模型到指定位置
    trainer.save_model('./my_bert_model')
    tokenizer.save_pretrained('./my_bert_model')

    # 训练完成
    logger.info("训练完成！模型已支持多分类！")
