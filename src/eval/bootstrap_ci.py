from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import f1_score


def bootstrap_macro_f1(labels: List[int], preds: List[int], n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    preds = np.array(preds)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(labels), len(labels))
        score = f1_score(labels[idx], preds[idx], average="macro")
        scores.append(score)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {"macro_f1_lower": float(lower), "macro_f1_upper": float(upper)}
