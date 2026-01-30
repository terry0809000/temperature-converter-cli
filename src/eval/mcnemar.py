from typing import Dict, List

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(labels: List[int], preds_a: List[int], preds_b: List[int]) -> Dict[str, float]:
    labels = np.array(labels)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    correct_a = preds_a == labels
    correct_b = preds_b == labels
    table = [[0, 0], [0, 0]]
    table[0][0] = int(np.sum(correct_a & correct_b))
    table[0][1] = int(np.sum(correct_a & ~correct_b))
    table[1][0] = int(np.sum(~correct_a & correct_b))
    table[1][1] = int(np.sum(~correct_a & ~correct_b))
    result = mcnemar(table, exact=False, correction=True)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
