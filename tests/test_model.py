import numpy as np
import pandas as pd
import pytest

from sba.features import FEATURE_COLUMNS, LABEL_COLUMN
from sba.model import build_pipeline, load, predict_proba, save, train


def _make_features(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    # Label correlated with the features so the model has something real to learn.
    signal = df[FEATURE_COLUMNS].sum(axis=1)
    df[LABEL_COLUMN] = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


@pytest.mark.parametrize("model_type", ["lightgbm", "xgboost"])
def test_train_predict_proba_returns_valid_probabilities(model_type):
    features = _make_features()
    pipeline = train(features, model_type=model_type)
    proba = predict_proba(pipeline, features)

    assert len(proba) == len(features)
    assert proba.between(0, 1).all()


def test_build_pipeline_rejects_unknown_model_type():
    with pytest.raises(ValueError):
        build_pipeline(model_type="not-a-real-model")


def test_save_and_load_round_trip(tmp_path):
    features = _make_features()
    pipeline = train(features, model_type="lightgbm")
    path = tmp_path / "model.joblib"

    save(pipeline, path=path)
    loaded = load(path=path)

    pd.testing.assert_series_equal(predict_proba(pipeline, features), predict_proba(loaded, features))
