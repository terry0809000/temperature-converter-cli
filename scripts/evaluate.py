import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.dataset import build_dataset
from src.eval.checklist_tests import build_checklist
from src.eval.metrics import compute_metrics
from src.models.frozen_embeddings import embed_texts
from src.preprocess.prepare import prepare_texts
from src.utils.config import load_config
from src.utils.logging_utils import log_jsonl, setup_logging
from src.utils.seeding import get_torch_device, set_seed


def _aggregate_predictions(preds: List[int], mapping: List[int]) -> List[int]:
    doc_preds = {}
    for pred, idx in zip(preds, mapping):
        doc_preds.setdefault(idx, []).append(pred)
    return [max(vals) for _, vals in sorted(doc_preds.items())]


def _predict_hf(texts: List[str], model_dir: str, max_length: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for idx in range(0, len(texts), 8):
            batch = texts[idx : idx + 8]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            preds.extend(outputs.logits.argmax(dim=1).cpu().numpy().tolist())
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models and run checklist tests")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_family", required=True, choices=["tfidf", "frozen_embeddings", "hf_finetune"])
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.output.get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir), name="evaluate")
    set_seed(config.training.get("seed", 42))

    splits, _ = build_dataset(config.data)
    task = next(t for t in config.modeling.get("tasks", []) if t["name"] == args.task)
    label_col = task["label_col"]

    test_df = splits["test"]
    test_texts = test_df["text"].tolist()
    y_test = test_df[label_col].tolist()

    model_name = config.modeling.get("base_model_name", "emilyalsentzer/Bio_ClinicalBERT")
    test_texts, test_map = prepare_texts(test_texts, config.preprocess, model_name)

    if args.model_family == "tfidf":
        model = joblib.load(args.model_path)
        preds = model.predict(test_texts)
    elif args.model_family == "frozen_embeddings":
        payload = joblib.load(args.model_path)
        device = get_torch_device(config.training.get("prefer_gpu", True))
        embeddings = embed_texts(test_texts, payload["config"], device)
        preds = payload["model"].predict(embeddings)
    else:
        device = get_torch_device(config.training.get("prefer_gpu", True))
        preds = _predict_hf(test_texts, args.model_path, config.preprocess.get("max_length", 256), device)

    if config.preprocess.get("full_note", False):
        preds = _aggregate_predictions(preds, test_map)

    metrics = compute_metrics(y_test, preds)
    logger.info("Test metrics: %s", metrics)
    log_jsonl(str(output_dir / f"{args.task}_evaluation.jsonl"), {"task": args.task, "metrics": metrics})

    checklist_texts = test_df["text"].sample(
        n=min(len(test_df), config.evaluation.get("checklist_samples", 50)),
        random_state=config.training.get("seed", 42),
    ).tolist()
    checklist = build_checklist(checklist_texts)
    checklist_results = {}
    for name, texts in checklist.items():
        processed, mapping = prepare_texts(texts, config.preprocess, model_name)
        if args.model_family == "tfidf":
            preds = model.predict(processed)
        elif args.model_family == "frozen_embeddings":
            embeddings = embed_texts(processed, payload["config"], device)
            preds = payload["model"].predict(embeddings)
        else:
            preds = _predict_hf(processed, args.model_path, config.preprocess.get("max_length", 256), device)
        if config.preprocess.get("full_note", False):
            preds = _aggregate_predictions(preds, mapping)
        checklist_results[name] = {"positives": int(np.sum(np.array(preds) > 0))}
    log_jsonl(
        str(output_dir / f"{args.task}_checklist.jsonl"),
        {"task": args.task, "checklist": checklist_results},
    )


if __name__ == "__main__":
    main()
