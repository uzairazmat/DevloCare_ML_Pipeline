import json
import logging
import os
from pathlib import Path

import dagshub
import mlflow
import yaml
from dagshub import dagshub_logger  # type: ignore
from dotenv import load_dotenv

from src.data.load_dataset import load_dataset
from src.features.preprocess import build_preprocessor
from src.models.train import TrainConfig, train_and_evaluate


logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
  logger.info("Loading parameters from %s", params_path)
  with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)
  logger.info("Parameters loaded successfully")
  return params


def configure_mlflow(mlflow_cfg: dict, training_cfg: dict):
  tracking_uri = mlflow_cfg.get("tracking_uri", "local")
  if tracking_uri == "dagshub":
    owner = mlflow_cfg["dagshub_repo_owner"]
    name = mlflow_cfg["dagshub_repo_name"]
    dagshub.init(repo_owner=owner, repo_name=name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{name}.mlflow")
    logger.info("Configured MLflow to use DagsHub tracking for %s/%s", owner, name)
  else:
    mlflow.set_tracking_uri("file:./mlruns")
    logger.info("Using local MLflow tracking URI")
  mlflow.set_experiment(training_cfg["experiment_name"])
  logger.info("MLflow experiment set to %s", training_cfg["experiment_name"])


def run_training():
  try:
    load_dotenv()
    params = load_params()

    data_cfg = params["data"]
    feat_cfg = params["features"]
    model_cfg = params["model"]
    training_cfg = params["training"]
    mlflow_cfg = params["mlflow"]

    configure_mlflow(mlflow_cfg, training_cfg)

    df = load_dataset(data_cfg["raw_path"])
    X, y = build_preprocessor(
      df,
      target_column=data_cfg["target_column"],
      text_column=data_cfg["text_column"],
      drop_columns=data_cfg.get("drop_columns", []),
    )

    ngram_range_cfg = feat_cfg["tfidf_ngram_range"]
    train_cfg = TrainConfig(
      test_size=data_cfg["test_size"],
      val_size=data_cfg.get("val_size", 0.15),
      random_state=data_cfg["random_state"],
      model_type=model_cfg["type"],
      tfidf_ngram_range=(ngram_range_cfg[0], ngram_range_cfg[1]),
      tfidf_max_features=feat_cfg.get("tfidf_max_features"),
      logistic_c=model_cfg.get("logistic_c", 1.0),
      logistic_max_iter=model_cfg.get("logistic_max_iter", 1000),
      logistic_solver=model_cfg.get("logistic_solver", "lbfgs"),
      rf_n_estimators=model_cfg.get("rf_n_estimators", 100),
      rf_max_depth=model_cfg.get("rf_max_depth"),
      svm_c=model_cfg.get("svm_c", 1.0),
      svm_max_iter=model_cfg.get("svm_max_iter", 2000),
      nb_alpha=model_cfg.get("nb_alpha", 1.0),
      cv_folds=training_cfg.get("cv_folds", 5),
    )

    with mlflow.start_run(run_name=training_cfg["run_name"]):
      # Optional DagsHub logger wrapper
      with dagshub_logger() as dag_logger:
        logger.info("Starting model training run %s", training_cfg["run_name"])
        model, metrics = train_and_evaluate(X, y, train_cfg)
        dag_logger.log_metrics(metrics)

      # Write metrics to file for DVC tracking
      metrics_path = Path("metrics/scores.json")
      metrics_path.parent.mkdir(parents=True, exist_ok=True)
      with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
      logger.info("Saved metrics to %s", metrics_path)

      # Save trained model
      model_path = Path(training_cfg["model_output_path"])
      model_path.parent.mkdir(parents=True, exist_ok=True)

      import joblib

      joblib.dump(model, model_path)
      logger.info("Saved trained model to %s", model_path)
      mlflow.sklearn.log_model(model, name="model")

  except Exception as exc:
    logger.exception("Training pipeline failed with an exception")
    raise exc


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  )
  run_training()

