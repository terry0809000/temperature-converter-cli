from typing import Dict, Tuple

import numpy as np
import pandas as pd


def split_by_subject(
    data: pd.DataFrame,
    subject_col: str = "subject_id",
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    subjects = data[subject_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_total = len(subjects)
    n_train = int(n_total * train_size)
    n_val = int(n_total * val_size)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train : n_train + n_val])
    test_subjects = set(subjects[n_train + n_val :])

    train_df = data[data[subject_col].isin(train_subjects)].copy()
    val_df = data[data[subject_col].isin(val_subjects)].copy()
    test_df = data[data[subject_col].isin(test_subjects)].copy()

    return {"train": train_df, "val": val_df, "test": test_df}


def split_fallback_random(
    data: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    shuffled = data.sample(frac=1.0, random_state=seed)
    n_total = len(shuffled)
    n_train = int(n_total * train_size)
    n_val = int(n_total * val_size)
    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train : n_train + n_val].copy()
    test_df = shuffled.iloc[n_train + n_val :].copy()
    return {"train": train_df, "val": val_df, "test": test_df}
