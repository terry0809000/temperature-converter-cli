from typing import Dict, Tuple

import pandas as pd

from src.data.loader import align_notes_labels, load_mimic_notes, load_sbdh_labels
from src.data.split import split_by_subject, split_fallback_random


def build_dataset(data_cfg: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    notes = load_mimic_notes(data_cfg["mimic_notes_path"], data_cfg.get("note_columns", {}))
    labels = load_sbdh_labels(data_cfg["sbdh_labels_path"], data_cfg.get("label_columns", {}))
    aligned = align_notes_labels(notes, labels, data_cfg.get("join_cols"))

    if "subject_id" in aligned.columns:
        splits = split_by_subject(
            aligned,
            subject_col="subject_id",
            train_size=data_cfg.get("train_size", 0.7),
            val_size=data_cfg.get("val_size", 0.15),
            seed=data_cfg.get("seed", 42),
        )
        split_note = "patient-level split applied"
    else:
        splits = split_fallback_random(
            aligned,
            train_size=data_cfg.get("train_size", 0.7),
            val_size=data_cfg.get("val_size", 0.15),
            seed=data_cfg.get("seed", 42),
        )
        split_note = "subject_id missing; random split used"

    return splits, {"split_note": split_note}
