from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


@dataclass
class TfidfConfig:
    ngram_range: tuple
    analyzer: str
    max_features: int


def build_vectorizer(config: TfidfConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=config.ngram_range,
        analyzer=config.analyzer,
        max_features=config.max_features,
        min_df=2,
    )


def build_classifier(model_type: str):
    if model_type == "logreg":
        return LogisticRegression(max_iter=2000, solver="liblinear")
    if model_type == "svm":
        return LinearSVC()
    raise ValueError(f"Unknown model type: {model_type}")


def train_tfidf_model(
    texts: List[str],
    labels: List[int],
    config: TfidfConfig,
    model_type: str,
) -> Pipeline:
    vectorizer = build_vectorizer(config)
    classifier = build_classifier(model_type)
    pipeline = Pipeline([(f"tfidf", vectorizer), ("clf", classifier)])
    pipeline.fit(texts, labels)
    return pipeline


def predict_tfidf(model: Pipeline, texts: List[str]) -> np.ndarray:
    return model.predict(texts)
