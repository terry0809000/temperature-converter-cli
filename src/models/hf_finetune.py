from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


@dataclass
class HFFinetuneConfig:
    model_name: str
    max_length: int
    num_labels: int
    lr: float
    batch_size: int
    epochs: int
    patience: int
    fp16: bool
    output_dir: str


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    metrics = {"macro_f1": macro_f1, "micro_f1": micro_f1}
    if logits.shape[1] == 2:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        metrics["auroc"] = roc_auc_score(labels, probs)
    return metrics


def train_hf_model(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    config: HFFinetuneConfig,
) -> Trainer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name, num_labels=config.num_labels
    )

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, config.max_length)

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=config.fp16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )

    trainer.train()
    return trainer
