import logging
from typing import List, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


def build_preprocessor(
  df: pd.DataFrame,
  target_column: str,
  text_column: str,
  drop_columns: List[str] | None = None,
) -> Tuple[pd.Series, pd.Series]:
  """Prepare text and target columns for text classification."""
  columns_to_drop = [col for col in (drop_columns or []) if col in df.columns and col != target_column]
  if columns_to_drop:
    logger.info("Dropping columns: %s", columns_to_drop)
    df = df.drop(columns=columns_to_drop)

  if text_column not in df.columns:
    raise ValueError(f"Text column '{text_column}' not found in dataset")
  if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset")

  X = df[text_column].astype(str)
  y = df[target_column].astype(str)
  logger.info("Prepared text dataset with %d samples", len(df))
  return X, y

