import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
  """Load dataset from CSV."""
  csv_path = Path(path)
  logger.info("Loading dataset from %s", csv_path)
  if not csv_path.exists():
    logger.error("Dataset not found at %s", csv_path.resolve())
    raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")
  df = pd.read_csv(csv_path)
  logger.info("Loaded dataset with %d rows and %d columns", df.shape[0], df.shape[1])
  return df

