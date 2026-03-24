import logging
from dataclasses import dataclass, field
from typing import Any, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
  test_size: float
  val_size: float
  random_state: int
  model_type: str
  tfidf_ngram_range: tuple[int, int]
  tfidf_max_features: int | None
  # Logistic Regression
  logistic_c: float = 1.0
  logistic_max_iter: int = 1000
  logistic_solver: str = "lbfgs"
  # Random Forest
  rf_n_estimators: int = 100
  rf_max_depth: int | None = None
  # Linear SVC
  svm_c: float = 1.0
  svm_max_iter: int = 2000
  # Naive Bayes
  nb_alpha: float = 1.0
  cv_folds: int = 5


def _build_model(cfg: TrainConfig) -> Any:
  logger.info("Building model of type %s", cfg.model_type)
  if cfg.model_type == "logistic_regression":
    return LogisticRegression(
      C=cfg.logistic_c,
      max_iter=cfg.logistic_max_iter,
      solver=cfg.logistic_solver,
      random_state=cfg.random_state,
    )
  elif cfg.model_type == "random_forest":
    return RandomForestClassifier(
      n_estimators=cfg.rf_n_estimators,
      max_depth=cfg.rf_max_depth,
      random_state=cfg.random_state,
      n_jobs=-1,
    )
  elif cfg.model_type == "linear_svc":
    return LinearSVC(
      C=cfg.svm_c,
      max_iter=cfg.svm_max_iter,
      random_state=cfg.random_state,
    )
  elif cfg.model_type == "naive_bayes":
    return MultinomialNB(alpha=cfg.nb_alpha)
  logger.error("Unsupported model type: %s", cfg.model_type)
  raise ValueError(f"Unsupported model type: {cfg.model_type}")


def train_and_evaluate(
  X,
  y,
  cfg: TrainConfig,
) -> Tuple[Pipeline, dict]:
  """Train text classifier and return fitted pipeline + metrics."""
  logger.info("Splitting data into train / val / test sets")
  stratify = y if len(np.unique(y)) > 1 else None
  X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=stratify,
  )
  val_ratio = cfg.val_size / (1.0 - cfg.test_size)
  X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=cfg.random_state,
    stratify=y_temp if len(np.unique(y_temp)) > 1 else None,
  )
  logger.info("Split sizes — train: %d  val: %d  test: %d", len(X_train), len(X_val), len(X_test))

  logger.info("Training model")
  model = _build_model(cfg)
  pipeline = Pipeline(
    steps=[
      (
        "tfidf",
        TfidfVectorizer(
          ngram_range=cfg.tfidf_ngram_range,
          max_features=cfg.tfidf_max_features,
          lowercase=True,
        ),
      ),
      ("model", model),
    ]
  )

  pipeline.fit(X_train, y_train)
  logger.info("Model training completed")

  # K-Fold CV on training data — strong overfitting signal
  cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cfg.cv_folds, scoring="accuracy")
  logger.info("CV scores (%d folds): %s  mean=%.4f  std=%.4f", cfg.cv_folds, cv_scores, cv_scores.mean(), cv_scores.std())

  y_val_pred = pipeline.predict(X_val)
  y_test_pred = pipeline.predict(X_test)

  metrics = {
    "cv_mean_accuracy": float(cv_scores.mean()),
    "cv_std_accuracy": float(cv_scores.std()),
    "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
    "val_macro_f1": float(f1_score(y_val, y_val_pred, average="macro", zero_division=0)),
    "accuracy": float(accuracy_score(y_test, y_test_pred)),
    "macro_precision": float(precision_score(y_test, y_test_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y_test, y_test_pred, average="macro", zero_division=0)),
    "macro_f1": float(f1_score(y_test, y_test_pred, average="macro", zero_division=0)),
  }
  logger.info("Evaluation metrics: %s", metrics)

  mlflow.log_params(
    {
      "model_type": cfg.model_type,
      "tfidf_ngram_range": str(cfg.tfidf_ngram_range),
      "tfidf_max_features": cfg.tfidf_max_features,
      "test_size": cfg.test_size,
      "random_state": cfg.random_state,
      # model-specific params
      "logistic_c": cfg.logistic_c,
      "logistic_max_iter": cfg.logistic_max_iter,
      "logistic_solver": cfg.logistic_solver,
      "rf_n_estimators": cfg.rf_n_estimators,
      "rf_max_depth": cfg.rf_max_depth,
      "svm_c": cfg.svm_c,
      "svm_max_iter": cfg.svm_max_iter,
      "nb_alpha": cfg.nb_alpha,
    }
  )
  mlflow.log_metrics(metrics)

  return pipeline, metrics

