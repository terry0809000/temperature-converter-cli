from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer


@dataclass
class EmbeddingConfig:
    model_name: str
    pooling: str
    max_length: int


def _pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(texts: List[str], config: EmbeddingConfig, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for idx in range(0, len(texts), 8):
            batch = texts[idx : idx + 8]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = _pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"], config.pooling)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def train_embedding_classifier(embeddings: np.ndarray, labels: List[int], model_type: str):
    if model_type == "logreg":
        model = LogisticRegression(max_iter=2000, solver="liblinear")
    elif model_type == "svm":
        model = LinearSVC()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.fit(embeddings, labels)
    return model
