"""Season-holdout evaluation of the win-probability model.

This validates the model's accuracy and calibration against real outcomes.
It does NOT measure betting ROI, closing-line value, or use public betting
percentages -- all three need data this project doesn't have and can't get
for free: ROI/CLV need historical odds *history* (the free Odds API tier is a
single live snapshot with no history, see odds.py/README), and public betting
% is a paid product (e.g. Action Network). Log loss, Brier score, and the
calibration table below are the honest substitutes: a model can be profitable
with well-calibrated probabilities even if its raw accuracy isn't much above
the market's, so calibration matters more here than accuracy alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from sba.features import FEATURE_COLUMNS, LABEL_COLUMN, build_features
from sba.model import ModelType, predict_proba, train


@dataclass
class BacktestResult:
    train_seasons: list[int]
    test_season: int
    n_train: int
    n_test: int
    accuracy: float
    log_loss: float
    brier_score: float
    calibration: pd.DataFrame


def calibration_table(y_true: pd.Series, y_prob: pd.Series, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob.values})
    df["bucket"] = pd.cut(df["y_prob"], bins=bins, include_lowest=True)
    summary = df.groupby("bucket", observed=True).agg(
        n=("y_true", "size"), predicted_mean=("y_prob", "mean"), actual_rate=("y_true", "mean")
    )
    return summary.reset_index()


def run_backtest(
    games: pd.DataFrame,
    test_season: int,
    *,
    model_type: ModelType = "lightgbm",
    params: dict | None = None,
    calibrate: bool = True,
) -> BacktestResult:
    features = build_features(games)
    train_df = features[features["season"] < test_season]
    test_df = features[features["season"] == test_season]

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Not enough data for a {test_season} holdout backtest "
            f"(train rows={len(train_df)}, test rows={len(test_df)})"
        )

    pipeline = train(train_df, model_type=model_type, params=params, calibrate=calibrate)
    y_prob = predict_proba(pipeline, test_df)
    y_true = test_df[LABEL_COLUMN]

    return BacktestResult(
        train_seasons=sorted(train_df["season"].unique().tolist()),
        test_season=test_season,
        n_train=len(train_df),
        n_test=len(test_df),
        accuracy=accuracy_score(y_true, y_prob >= 0.5),
        log_loss=log_loss(y_true, y_prob),
        brier_score=brier_score_loss(y_true, y_prob),
        calibration=calibration_table(y_true, y_prob),
    )
