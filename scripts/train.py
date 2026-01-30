import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import joblib

from src.data.dataset import build_dataset
from src.eval.metrics import compute_metrics
from src.models.frozen_embeddings import EmbeddingConfig, embed_texts, train_embedding_classifier
from src.models.hf_finetune import HFFinetuneConfig, train_hf_model
from src.models.tfidf_baselines import TfidfConfig, train_tfidf_model
from src.preprocess.prepare import prepare_texts
from src.utils.config import load_config
from src.utils.logging_utils import log_jsonl, setup_logging
from src.utils.seeding import get_torch_device, set_seed


def _sample(config_list: List, rng: np.random.Generator):
    return rng.choice(config_list)


def random_search_tfidf(texts_train, y_train, texts_val, y_val, search_cfg, rng):
    best = None
    trials = search_cfg.get("trials", 5)
    for _ in range(trials):
        tfidf_cfg = TfidfConfig(
            ngram_range=tuple(_sample(search_cfg.get("ngram_range", [(1, 2)]), rng)),
            analyzer=_sample(search_cfg.get("analyzer", ["word"]), rng),
            max_features=int(_sample(search_cfg.get("max_features", [20000]), rng)),
        )
        model_type = _sample(search_cfg.get("model_type", ["logreg"]), rng)
        model = train_tfidf_model(texts_train, y_train, tfidf_cfg, model_type)
        preds = model.predict(texts_val)
        metrics = compute_metrics(y_val, preds)
        score = metrics["macro_f1"]
        if best is None or score > best["score"]:
            best = {"model": model, "metrics": metrics, "score": score, "config": tfidf_cfg, "type": model_type}
    return best


def random_search_embeddings(texts_train, y_train, texts_val, y_val, search_cfg, device, rng):
    best = None
    trials = search_cfg.get("trials", 3)
    for _ in range(trials):
        emb_cfg = EmbeddingConfig(
            model_name=_sample(search_cfg.get("model_name", ["emilyalsentzer/Bio_ClinicalBERT"]), rng),
            pooling=_sample(search_cfg.get("pooling", ["cls"]), rng),
            max_length=int(_sample(search_cfg.get("max_length", [256]), rng)),
        )
        model_type = _sample(search_cfg.get("model_type", ["logreg"]), rng)
        train_emb = embed_texts(texts_train, emb_cfg, device)
        val_emb = embed_texts(texts_val, emb_cfg, device)
        clf = train_embedding_classifier(train_emb, y_train, model_type)
        preds = clf.predict(val_emb)
        metrics = compute_metrics(y_val, preds)
        score = metrics["macro_f1"]
        if best is None or score > best["score"]:
            best = {"model": clf, "metrics": metrics, "score": score, "config": emb_cfg, "type": model_type}
    return best


def random_search_hf(texts_train, y_train, texts_val, y_val, search_cfg, output_dir, rng):
    best = None
    trials = search_cfg.get("trials", 2)
    for trial in range(trials):
        cfg = HFFinetuneConfig(
            model_name=_sample(search_cfg.get("model_name", ["emilyalsentzer/Bio_ClinicalBERT"]), rng),
            max_length=int(_sample(search_cfg.get("max_length", [256]), rng)),
            num_labels=int(_sample(search_cfg.get("num_labels", [2]), rng)),
            lr=float(_sample(search_cfg.get("lr", [2e-5]), rng)),
            batch_size=int(_sample(search_cfg.get("batch_size", [8]), rng)),
            epochs=int(_sample(search_cfg.get("epochs", [3]), rng)),
            patience=int(_sample(search_cfg.get("patience", [2]), rng)),
            fp16=bool(_sample(search_cfg.get("fp16", [False]), rng)),
            output_dir=str(Path(output_dir) / f"trial_{trial}"),
        )
        trainer = train_hf_model(texts_train, y_train, texts_val, y_val, cfg)
        metrics = trainer.evaluate()
        score = metrics["eval_macro_f1"]
        if best is None or score > best["score"]:
            best = {"trainer": trainer, "metrics": metrics, "score": score, "config": cfg}
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SBDH models")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.output.get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir), name="train")
    set_seed(config.training.get("seed", 42))

    splits, meta = build_dataset(config.data)
    logger.info("Dataset alignment complete: %s", meta)

    tasks = config.modeling.get("tasks", [])
    rng = np.random.default_rng(config.training.get("seed", 42))

    for task in tasks:
        task_name = task["name"]
        label_col = task["label_col"]
        logger.info("Training task: %s", task_name)

        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        train_texts = train_df["text"].tolist()
        val_texts = val_df["text"].tolist()
        test_texts = test_df["text"].tolist()
        y_train = train_df[label_col].tolist()
        y_val = val_df[label_col].tolist()
        y_test = test_df[label_col].tolist()

        model_family = config.modeling.get("family", "tfidf")
        model_name = config.modeling.get("base_model_name", "emilyalsentzer/Bio_ClinicalBERT")
        train_texts, _ = prepare_texts(train_texts, config.preprocess, model_name)
        val_texts, _ = prepare_texts(val_texts, config.preprocess, model_name)
        test_texts, test_map = prepare_texts(test_texts, config.preprocess, model_name)

        metrics_path = output_dir / f"{task_name}_metrics.jsonl"

        if model_family == "tfidf":
            start = time.time()
            best = random_search_tfidf(train_texts, y_train, val_texts, y_val, config.modeling.get("tfidf", {}), rng)
            duration = time.time() - start
            infer_start = time.time()
            preds = best["model"].predict(test_texts)
            infer_seconds = time.time() - infer_start
            metrics = compute_metrics(y_test, preds)
            model_path = output_dir / f"{task_name}_tfidf.joblib"
            joblib.dump(best["model"], model_path)
            coef = best["model"].named_steps["clf"].coef_
            param_count = int(coef.size)
            payload = {
                "task": task_name,
                "family": model_family,
                "val_metrics": best["metrics"],
                "test_metrics": metrics,
                "train_seconds": duration,
                "inference_seconds": infer_seconds,
                "param_count": param_count,
                "model_path": str(model_path),
                "gpu_max_mem_mb": float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0,
            }
            log_jsonl(str(metrics_path), payload)
        elif model_family == "frozen_embeddings":
            device = get_torch_device(config.training.get("prefer_gpu", True))
            start = time.time()
            best = random_search_embeddings(
                train_texts, y_train, val_texts, y_val, config.modeling.get("frozen", {}), device, rng
            )
            duration = time.time() - start
            test_emb = embed_texts(test_texts, best["config"], device)
            infer_start = time.time()
            preds = best["model"].predict(test_emb)
            infer_seconds = time.time() - infer_start
            metrics = compute_metrics(y_test, preds)
            model_path = output_dir / f"{task_name}_frozen.joblib"
            joblib.dump({"model": best["model"], "config": best["config"]}, model_path)
            coef = best["model"].coef_
            param_count = int(coef.size)
            payload = {
                "task": task_name,
                "family": model_family,
                "val_metrics": best["metrics"],
                "test_metrics": metrics,
                "train_seconds": duration,
                "inference_seconds": infer_seconds,
                "param_count": param_count,
                "model_path": str(model_path),
                "gpu_max_mem_mb": float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0,
            }
            log_jsonl(str(metrics_path), payload)
        elif model_family == "hf_finetune":
            start = time.time()
            best = random_search_hf(
                train_texts,
                y_train,
                val_texts,
                y_val,
                config.modeling.get("hf", {}),
                str(output_dir / task_name),
                rng,
            )
            duration = time.time() - start
            infer_start = time.time()
            test_metrics = best["trainer"].evaluate()
            infer_seconds = time.time() - infer_start
            model_path = output_dir / f"{task_name}_hf_finetune"
            best["trainer"].save_model(str(model_path))
            param_count = int(sum(p.numel() for p in best["trainer"].model.parameters()))
            payload = {
                "task": task_name,
                "family": model_family,
                "val_metrics": best["metrics"],
                "test_metrics": test_metrics,
                "train_seconds": duration,
                "inference_seconds": infer_seconds,
                "param_count": param_count,
                "model_path": str(model_path),
                "gpu_max_mem_mb": float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0,
            }
            log_jsonl(str(metrics_path), payload)
        else:
            raise ValueError(f"Unknown model family: {model_family}")


if __name__ == "__main__":
    main()
