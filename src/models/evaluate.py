from typing import Any, Dict

import mlflow
from sklearn.metrics import classification_report


def evaluate_model(model: Any, X, y) -> Dict[str, float]:
  """Log detailed evaluation report to MLflow."""
  y_pred = model.predict(X)
  report = classification_report(y, y_pred, output_dict=True)
  # Log macro avg metrics
  macro = report.get("macro avg", {})
  for key, value in macro.items():
    if isinstance(value, (int, float)):
      mlflow.log_metric(f"macro_{key}", float(value))
  return {"macro_" + k: float(v) for k, v in macro.items() if isinstance(v, (int, float))}

