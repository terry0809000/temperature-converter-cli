from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def load_mimic_notes(path: str, column_map: Dict[str, str]) -> pd.DataFrame:
    notes_path = Path(path)
    if not notes_path.exists():
        raise FileNotFoundError(f"Missing MIMIC-III notes file: {notes_path}")
    notes = pd.read_csv(notes_path)
    category_col = column_map.get("category", "CATEGORY")
    text_col = column_map.get("text", "TEXT")
    subject_col = column_map.get("subject_id", "SUBJECT_ID")
    hadm_col = column_map.get("hadm_id", "HADM_ID")
    notes = notes[notes[category_col].str.lower() == "discharge summary"].copy()
    notes = notes[[subject_col, hadm_col, text_col]].rename(
        columns={subject_col: "subject_id", hadm_col: "hadm_id", text_col: "text"}
    )
    return notes


def load_sbdh_labels(path: str, column_map: Dict[str, str]) -> pd.DataFrame:
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing MIMIC-SBDH labels file: {label_path}")
    labels = pd.read_csv(label_path)
    subject_col = column_map.get("subject_id", "SUBJECT_ID")
    hadm_col = column_map.get("hadm_id", "HADM_ID")
    labels = labels.rename(columns={subject_col: "subject_id", hadm_col: "hadm_id"})
    return labels


def align_notes_labels(
    notes: pd.DataFrame,
    labels: pd.DataFrame,
    join_cols: Optional[list] = None,
) -> pd.DataFrame:
    join_cols = join_cols or ["subject_id", "hadm_id"]
    aligned = labels.merge(notes, on=join_cols, how="inner")
    return aligned
