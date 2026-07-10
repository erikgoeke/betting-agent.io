"""Run-line (home covers -1.5) and totals (over/under total runs) models.

Same feature set as the moneyline model (features.py's FEATURE_COLUMNS) and
same LightGBM/XGBoost choice, just different labels:
  - run-line is a binary classifier: did the home team win by 2+ runs (the
    standard MLB run line is almost always +/-1.5, so a 2-run win covers it).
  - totals is a *regressor* (expected total runs), not a classifier -- there's
    no historical totals-line data to grade an over/under call against (the
    free Odds API tier only carries moneyline, see odds.py/README), so a
    calibrated point estimate is the honest thing to produce here.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from sba.config import MODELS_DIR
from sba.features import FEATURE_COLUMNS, build_features
from sba.model import DEFAULT_PARAMS, ModelType

RUNLINE_LABEL_COLUMN = "home_covers_runline"
TOTAL_LABEL_COLUMN = "total_runs"
RUN_LINE_MARGIN = 2  # a 2+ run win covers the standard -1.5 MLB run line

RUNLINE_MODEL_PATH = MODELS_DIR / "mlb_runline.joblib"
TOTALS_MODEL_PATH = MODELS_DIR / "mlb_totals.joblib"

# The moneyline/run-line feature set is almost entirely home-minus-away *diffs*,
# which is exactly right for "who wins" but nearly blind to "how many total runs"
# -- a good-offense/good-offense matchup and a bad-offense/bad-offense matchup
# both diff to ~0 despite very different expected totals. These add the
# home-plus-away *sum* of the same underlying rolling stats as extra signal
# specifically for the totals regressor.
TOTALS_EXTRA_COLUMNS = ["combined_run_diff", "combined_off_ops", "combined_starter_fip", "combined_hard_hit_rate"]
TOTALS_FEATURE_COLUMNS = FEATURE_COLUMNS + TOTALS_EXTRA_COLUMNS


def build_runline_totals_features(games: pd.DataFrame) -> pd.DataFrame:
    """Same feature table as the moneyline model, plus the run-line/totals labels
    and totals-specific combined (sum, not diff) features."""
    features = build_features(games)
    features[RUNLINE_LABEL_COLUMN] = ((features["home_runs"] - features["away_runs"]) >= RUN_LINE_MARGIN).astype(int)
    features[TOTAL_LABEL_COLUMN] = features["home_runs"] + features["away_runs"]

    features["combined_run_diff"] = features["home_rolling_run_diff"] + features["away_rolling_run_diff"]
    features["combined_off_ops"] = features["home_off_ops"] + features["away_off_ops"]
    features["combined_starter_fip"] = features["home_starter_fip"] + features["away_starter_fip"]
    features["combined_hard_hit_rate"] = features["home_hard_hit_rate"] + features["away_hard_hit_rate"]
    return features


def _regressor_params(model_type: ModelType, params: dict | None) -> dict:
    merged = {**DEFAULT_PARAMS[model_type], **(params or {})}
    merged.pop("eval_metric", None)  # classifier-only kwarg
    return merged


def build_runline_pipeline(model_type: ModelType = "lightgbm", params: dict | None = None) -> Pipeline:
    merged = {**DEFAULT_PARAMS[model_type], **(params or {})}
    clf = lgb.LGBMClassifier(**merged) if model_type == "lightgbm" else xgb.XGBClassifier(**merged)
    return Pipeline([("clf", clf)])


def build_totals_pipeline(model_type: ModelType = "lightgbm", params: dict | None = None) -> Pipeline:
    merged = _regressor_params(model_type, params)
    reg = lgb.LGBMRegressor(**merged) if model_type == "lightgbm" else xgb.XGBRegressor(**merged)
    return Pipeline([("reg", reg)])


def train_runline(features: pd.DataFrame, *, model_type: ModelType = "lightgbm", params: dict | None = None) -> Pipeline:
    pipeline = build_runline_pipeline(model_type=model_type, params=params)
    pipeline.fit(features[FEATURE_COLUMNS], features[RUNLINE_LABEL_COLUMN])
    return pipeline


def train_totals(features: pd.DataFrame, *, model_type: ModelType = "lightgbm", params: dict | None = None) -> Pipeline:
    pipeline = build_totals_pipeline(model_type=model_type, params=params)
    pipeline.fit(features[TOTALS_FEATURE_COLUMNS], features[TOTAL_LABEL_COLUMN])
    return pipeline


def predict_runline_proba(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """P(home team covers the run line)."""
    proba = pipeline.predict_proba(features[FEATURE_COLUMNS])
    cover_col = list(pipeline.classes_).index(1)
    return pd.Series(proba[:, cover_col], index=features.index)


def predict_total_runs(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Expected total runs scored by both teams combined."""
    return pd.Series(pipeline.predict(features[TOTALS_FEATURE_COLUMNS]), index=features.index)


def save_runline(pipeline: Pipeline, path=RUNLINE_MODEL_PATH) -> None:
    joblib.dump(pipeline, path)


def load_runline(path=RUNLINE_MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def save_totals(pipeline: Pipeline, path=TOTALS_MODEL_PATH) -> None:
    joblib.dump(pipeline, path)


def load_totals(path=TOTALS_MODEL_PATH) -> Pipeline:
    return joblib.load(path)


@dataclass
class RunlineBacktestResult:
    train_seasons: list[int]
    test_season: int
    n_train: int
    n_test: int
    accuracy: float
    log_loss: float
    brier_score: float


@dataclass
class TotalsBacktestResult:
    train_seasons: list[int]
    test_season: int
    n_train: int
    n_test: int
    mae: float
    rmse: float


def run_runline_backtest(
    games: pd.DataFrame, test_season: int, *, model_type: ModelType = "lightgbm", params: dict | None = None
) -> RunlineBacktestResult:
    features = build_runline_totals_features(games)
    train_df = features[features["season"] < test_season]
    test_df = features[features["season"] == test_season]
    if train_df.empty or test_df.empty:
        raise ValueError(f"Not enough data for a {test_season} run-line backtest.")

    pipeline = train_runline(train_df, model_type=model_type, params=params)
    y_prob = predict_runline_proba(pipeline, test_df)
    y_true = test_df[RUNLINE_LABEL_COLUMN]

    return RunlineBacktestResult(
        train_seasons=sorted(train_df["season"].unique().tolist()),
        test_season=test_season,
        n_train=len(train_df),
        n_test=len(test_df),
        accuracy=accuracy_score(y_true, y_prob >= 0.5),
        log_loss=log_loss(y_true, y_prob),
        brier_score=brier_score_loss(y_true, y_prob),
    )


def run_totals_backtest(
    games: pd.DataFrame, test_season: int, *, model_type: ModelType = "lightgbm", params: dict | None = None
) -> TotalsBacktestResult:
    features = build_runline_totals_features(games)
    train_df = features[features["season"] < test_season]
    test_df = features[features["season"] == test_season]
    if train_df.empty or test_df.empty:
        raise ValueError(f"Not enough data for a {test_season} totals backtest.")

    pipeline = train_totals(train_df, model_type=model_type, params=params)
    y_pred = predict_total_runs(pipeline, test_df)
    y_true = test_df[TOTAL_LABEL_COLUMN]

    return TotalsBacktestResult(
        train_seasons=sorted(train_df["season"].unique().tolist()),
        test_season=test_season,
        n_train=len(train_df),
        n_test=len(test_df),
        mae=mean_absolute_error(y_true, y_pred),
        rmse=mean_squared_error(y_true, y_pred) ** 0.5,
    )
