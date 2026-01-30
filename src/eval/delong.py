from typing import Dict

import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_x = np.sort(x)
    idx = np.argsort(x)
    midranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midrank = 0.5 * (i + j - 1) + 1
        midranks[idx[i:j]] = midrank
        i = j
    return midranks


def _fast_delong(preds: np.ndarray, labels: np.ndarray):
    positive_preds = preds[labels == 1]
    negative_preds = preds[labels == 0]
    m = len(positive_preds)
    n = len(negative_preds)
    tx = _compute_midrank(positive_preds)
    ty = _compute_midrank(negative_preds)
    tz = _compute_midrank(np.concatenate([positive_preds, negative_preds]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    sx = np.var(v01, ddof=1)
    sy = np.var(v10, ddof=1)
    s = sx / m + sy / n
    return auc, s


def delong_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    auc_a, var_a = _fast_delong(preds_a, labels)
    auc_b, var_b = _fast_delong(preds_b, labels)
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "z": float(z), "p_value": float(p_value)}
