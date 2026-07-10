"""SHAP feature importance for the win-probability model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from sba.features import FEATURE_COLUMNS


def shap_importance(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Mean |SHAP value| per feature, largest first."""
    clf = pipeline.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(features[FEATURE_COLUMNS])
    if isinstance(shap_values, list):  # binary classifiers that return [neg_class, pos_class]
        shap_values = shap_values[1]
    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS)
    return importance.sort_values(ascending=False)
