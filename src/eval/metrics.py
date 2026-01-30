from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(labels: List[int], preds: List[int], probs: Optional[np.ndarray] = None) -> Dict:
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    metrics = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_f1": {k: v["f1-score"] for k, v in report.items() if k.isdigit()},
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    if probs is not None and probs.shape[1] == 2:
        metrics["auroc"] = roc_auc_score(labels, probs[:, 1])
    return metrics
