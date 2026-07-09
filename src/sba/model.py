"""Train, save, and load the MLB home-win probability model."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sba.config import MODEL_PATH
from sba.features import FEATURE_COLUMNS, LABEL_COLUMN


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train(features: pd.DataFrame) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(features[FEATURE_COLUMNS], features[LABEL_COLUMN])
    return pipeline


def save(pipeline: Pipeline, path=MODEL_PATH) -> None:
    joblib.dump(pipeline, path)


def load(path=MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_proba(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return P(home team wins) for each row."""
    proba = pipeline.predict_proba(features[FEATURE_COLUMNS])
    home_win_col = list(pipeline.classes_).index(1)
    return pd.Series(proba[:, home_win_col], index=features.index)
