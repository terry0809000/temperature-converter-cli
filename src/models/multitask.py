from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from transformers import AutoModel


@dataclass
class MultitaskConfig:
    model_name: str
    num_labels: Dict[str, int]


class MultitaskClassifier(nn.Module):
    def __init__(self, config: MultitaskConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict(
            {task: nn.Linear(hidden_size, num_labels) for task, num_labels in config.num_labels.items()}
        )

    def forward(self, input_ids, attention_mask, task: str):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.heads[task](pooled)
        return logits
