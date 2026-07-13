"""Optuna hyperparameter search for the win-probability model.

Validates against a single held-out season (the same split backtest.py uses)
rather than random k-fold CV: shuffling games across time would leak future
team form into the training fold, since features.py's rolling stats are
computed in chronological order.
"""

from __future__ import annotations

import optuna
import pandas as pd
from sklearn.metrics import log_loss

from sba.features import FEATURE_COLUMNS, LABEL_COLUMN
from sba.model import ModelType, build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial: optuna.Trial, model_type: ModelType) -> dict:
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if model_type == "lightgbm":
        return {
            **common,
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "verbose": -1,
        }
    return {
        **common,
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "eval_metric": "logloss",
    }


def _objective(trial: optuna.Trial, train_df: pd.DataFrame, valid_df: pd.DataFrame, model_type: ModelType) -> float:
    params = _suggest_params(trial, model_type)
    # Uncalibrated: calibration doesn't change which base-model hyperparameters
    # are best, and doing a 5-fold calibration refit on every one of ~50 trials
    # would multiply the search's runtime for no benefit to the search itself.
    pipeline = build_pipeline(model_type=model_type, params=params, calibrate=False)
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[LABEL_COLUMN])
    proba = pipeline.predict_proba(valid_df[FEATURE_COLUMNS])
    home_col = list(pipeline.classes_).index(1)
    return log_loss(valid_df[LABEL_COLUMN], proba[:, home_col])


def tune(
    features: pd.DataFrame, *, valid_season: int, model_type: ModelType = "lightgbm", n_trials: int = 50
) -> dict:
    """Search hyperparameters, minimizing log loss on a single held-out season."""
    if model_type == "ensemble":
        raise ValueError(
            "The ensemble has no hyperparameters of its own -- tune 'lightgbm' and "
            "'xgboost' separately; the ensemble picks both tuned sets up automatically."
        )
    train_df = features[features["season"] < valid_season]
    valid_df = features[features["season"] == valid_season]
    if train_df.empty or valid_df.empty:
        raise ValueError(f"Not enough data to tune against valid_season={valid_season}")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: _objective(t, train_df, valid_df, model_type), n_trials=n_trials)
    return study.best_params
