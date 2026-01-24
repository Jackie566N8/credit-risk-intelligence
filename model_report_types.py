from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ModelReport:
    name: str
    label: str
    path: Path
    raw_text: str
    data_summary: dict[str, str]
    cv_metrics: pd.DataFrame
    holdout_metrics: pd.DataFrame
    default_threshold_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
