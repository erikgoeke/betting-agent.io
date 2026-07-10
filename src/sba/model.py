"""Train, save, and load the MLB home-win probability model.

LightGBM is the primary model -- it handles the mix of raw and engineered
features without needing scaling, and trains fast enough for frequent
retraining. XGBoost is available as a drop-in alternative (`model_type="xgboost"`)
for comparison; both are gradient-boosted trees so the same feature set works
for either. See tuning.py for Optuna hyperparameter search and importance.py
for SHAP-based feature importance.
"""

from __future__ import annotations

from typing import Literal

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from sba.config import MODEL_PATH
from sba.features import FEATURE_COLUMNS, LABEL_COLUMN

ModelType = Literal["lightgbm", "xgboost"]

# Conservative defaults for a few thousand rows and a handful of features --
# tuning.py's Optuna search should be preferred over these for real use.
DEFAULT_PARAMS: dict[ModelType, dict] = {
    "lightgbm": dict(
        n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15, min_child_samples=20, verbose=-1
    ),
    "xgboost": dict(
        n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=20, eval_metric="logloss"
    ),
}


def build_pipeline(model_type: ModelType = "lightgbm", params: dict | None = None) -> Pipeline:
    if model_type not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown model_type: {model_type!r} (expected 'lightgbm' or 'xgboost')")
    merged = {**DEFAULT_PARAMS[model_type], **(params or {})}
    clf = lgb.LGBMClassifier(**merged) if model_type == "lightgbm" else xgb.XGBClassifier(**merged)
    return Pipeline([("clf", clf)])


def train(features: pd.DataFrame, *, model_type: ModelType = "lightgbm", params: dict | None = None) -> Pipeline:
    pipeline = build_pipeline(model_type=model_type, params=params)
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
