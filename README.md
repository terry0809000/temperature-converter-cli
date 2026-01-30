# SBDH/SDoH Benchmarking Framework (MIMIC-III + MIMIC-SBDH)

This repository provides a modular, CLI-driven codebase to benchmark traditional ML, frozen transformer embeddings, and end-to-end transformer fine-tuning for Social and Behavioral Determinants of Health (SBDH/SDoH) classification using MIMIC-III discharge summaries and MIMIC-SBDH annotations.

## Repository Structure

```
src/
  data/
    loader.py
    dataset.py
    split.py
  preprocess/
    phi.py
    section.py
    chunking.py
    prepare.py
  models/
    tfidf_baselines.py
    frozen_embeddings.py
    hf_finetune.py
    multitask.py
  eval/
    metrics.py
    bootstrap_ci.py
    mcnemar.py
    delong.py
    checklist_tests.py
  utils/
    config.py
    logging_utils.py
    seeding.py
configs/
  tfidf.yaml
  frozen_embeddings.yaml
  hf_finetune.yaml
scripts/
  train.py
  evaluate.py
  run_benchmark.py
  benchmark_report.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> NOTE: `medspacy` is optional but recommended for robust section extraction.

## Data Inputs

You must provide local file paths in the YAML configs for:

- MIMIC-III `NOTEEVENTS.csv` (or a filtered discharge summary export).
- MIMIC-SBDH label file with SBDH annotations.

Column names can be mapped via the `note_columns` and `label_columns` entries in the config.

## Key Features

- Patient-level splitting by `SUBJECT_ID` with fallback random split if missing.
- PHI handling (replace or remove).
- Social history extraction via medSpaCy sectionizer with regex fallback.
- Fixed preprocessing and split protocol for fair comparisons.
- Random-search hyperparameter tuning with a fixed trial budget.
- Macro-F1 primary metric, plus micro-F1 and AUROC (binary).
- Bootstrap confidence intervals and statistical tests (McNemar, DeLong).
- CheckList-style behavioral tests (negation, historical phrasing, attribution, misspellings).

## Example Commands

### Build/Train Models

```bash
python scripts/train.py --config configs/tfidf.yaml
python scripts/train.py --config configs/frozen_embeddings.yaml
python scripts/train.py --config configs/hf_finetune.yaml
```

### Evaluate + CheckList Tests

```bash
python scripts/evaluate.py \
  --config configs/tfidf.yaml \
  --model_path outputs/tfidf/community_present_tfidf.joblib \
  --model_family tfidf \
  --task community_present
```

### Run Benchmark Across Multiple Configs

```bash
python scripts/run_benchmark.py --configs configs/tfidf.yaml configs/frozen_embeddings.yaml configs/hf_finetune.yaml
```

### Build Final Benchmark Table

```bash
python scripts/benchmark_report.py \
  --metrics_dir outputs/tfidf \
  --output_csv outputs/benchmark.csv \
  --output_md outputs/benchmark.md
```

## Notes on Full Note Processing

Enable `preprocess.full_note: true` to activate sliding-window chunking for long notes. Chunk-level predictions are aggregated by max pooling across chunks.

## Reproducibility

All scripts set deterministic seeds (configurable in `training.seed`). Use fixed train/val/test splits for fair comparison.

## Limitations

- If `SUBJECT_ID` is unavailable, the framework logs a fallback to random splitting.
- Chunked training replicates labels across chunks, which may slightly bias class balance.
