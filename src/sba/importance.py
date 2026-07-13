"""SHAP feature importance for the win-probability model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from sba.features import FEATURE_COLUMNS
from sba.model import EnsembleClassifier


def _unwrap_tree_model(clf):
    """SHAP's TreeExplainer needs an actual tree model, not a calibration
    wrapper or the ensemble's averaging wrapper -- pick a representative
    underlying tree model for importance purposes. Approximate for both
    wrappers (one calibration fold, or just the LightGBM half of an ensemble),
    but SHAP importance is about relative feature ranking, not exact values,
    so this is a reasonable simplification rather than something worth the
    complexity of averaging across every fold/sub-model.
    """
    if isinstance(clf, CalibratedClassifierCV):
        clf = clf.calibrated_classifiers_[0].estimator
    if isinstance(clf, EnsembleClassifier):
        clf = clf.lgb_
    return clf


def shap_importance(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Mean |SHAP value| per feature, largest first."""
    clf = _unwrap_tree_model(pipeline.named_steps["clf"])
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(features[FEATURE_COLUMNS])
    if isinstance(shap_values, list):  # binary classifiers that return [neg_class, pos_class]
        shap_values = shap_values[1]
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS)
    return importance.sort_values(ascending=False)
