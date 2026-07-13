"""Train, save, and load the MLB home-win probability model.

LightGBM is the primary model -- it handles the mix of raw and engineered
features without needing scaling, and trains fast enough for frequent
retraining. XGBoost is available as a drop-in alternative (`model_type="xgboost"`),
and "ensemble" averages both -- typically a modest variance-reduction gain over
either alone. See tuning.py for Optuna hyperparameter search and importance.py
for SHAP-based feature importance.

Calibration: gradient-boosted trees are well known to be overconfident at the
extremes (a raw 90% often isn't really a 90-in-100 outcome). `calibrate=True`
(the default) wraps the classifier in isotonic calibration via cross-validated
folds of the training data, which matters more than raw accuracy for a model
whose output feeds a Kelly stake calculation.
"""

from __future__ import annotations

from typing import Literal

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from sba.config import MODEL_PATH
from sba.features import FEATURE_COLUMNS, LABEL_COLUMN

ModelType = Literal["lightgbm", "xgboost", "ensemble"]

# Conservative defaults for a few thousand rows and a handful of features --
# tuning.py's Optuna search should be preferred over these for real use.
DEFAULT_PARAMS: dict[str, dict] = {
    "lightgbm": dict(
        n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15, min_child_samples=20, verbose=-1
    ),
    "xgboost": dict(
        n_estimators=200, learning_rate=0.05, max_depth=4, min_child_weight=20, eval_metric="logloss"
    ),
}

CALIBRATION_CV_FOLDS = 5


class EnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Averages a LightGBM and an XGBoost classifier's predicted probabilities.

    Written as a proper sklearn estimator (not just a plain wrapper class) so
    it can be cloned by CalibratedClassifierCV/Optuna the same way a single
    model can -- __init__ only stores its arguments verbatim, per sklearn's
    convention, and get_params/set_params come for free from BaseEstimator.
    """

    def __init__(self, lightgbm_params: dict | None = None, xgboost_params: dict | None = None):
        self.lightgbm_params = lightgbm_params
        self.xgboost_params = xgboost_params

    def fit(self, X, y):
        lgb_params = {**DEFAULT_PARAMS["lightgbm"], **(self.lightgbm_params or {})}
        xgb_params = {**DEFAULT_PARAMS["xgboost"], **(self.xgboost_params or {})}
        self.lgb_ = lgb.LGBMClassifier(**lgb_params).fit(X, y)
        self.xgb_ = xgb.XGBClassifier(**xgb_params).fit(X, y)
        self.classes_ = self.lgb_.classes_
        return self

    def predict_proba(self, X):
        return (self.lgb_.predict_proba(X) + self.xgb_.predict_proba(X)) / 2

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


def _build_classifier(model_type: ModelType, params: dict | None):
    if model_type == "lightgbm":
        merged = {**DEFAULT_PARAMS["lightgbm"], **(params or {})}
        return lgb.LGBMClassifier(**merged)
    if model_type == "xgboost":
        merged = {**DEFAULT_PARAMS["xgboost"], **(params or {})}
        return xgb.XGBClassifier(**merged)
    if model_type == "ensemble":
        params = params or {}
        return EnsembleClassifier(lightgbm_params=params.get("lightgbm"), xgboost_params=params.get("xgboost"))
    raise ValueError(f"Unknown model_type: {model_type!r} (expected 'lightgbm', 'xgboost', or 'ensemble')")


def build_pipeline(model_type: ModelType = "lightgbm", params: dict | None = None, *, calibrate: bool = True) -> Pipeline:
    clf = _build_classifier(model_type, params)
    if calibrate:
        clf = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=CALIBRATION_CV_FOLDS)
    return Pipeline([("clf", clf)])


def train(
    features: pd.DataFrame, *, model_type: ModelType = "lightgbm", params: dict | None = None, calibrate: bool = True
) -> Pipeline:
    pipeline = build_pipeline(model_type=model_type, params=params, calibrate=calibrate)
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
