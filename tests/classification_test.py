import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import namedtuple

from slickml.classification import XGBoostCVClassifier

ModelWithData = namedtuple("ModelWithData", ["model", "matrix", "features", "target"])


def get_params():
    """ Model Parameters """
    params = {
        "eval_metric": "auc",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "subsample": 0.9,
        "max_delta_step": 1,
        "verbosity": 0,
        "nthread": 4,
        "scale_pos_weight": 1,
    }
    return params


@pytest.fixture(scope="session")
def xgb_classifier():
    data = pd.read_csv("data/clf_data.csv")
    X = data.iloc[:, :-1]
    y = data.CLASS.values

    dtrain = xgb.DMatrix(data=X, label=y)

    # cross-validation
    cvr = xgb.cv(
        params=get_params(),
        dtrain=dtrain,
        num_boost_round=500,
        nfold=4,
        stratified=True,
        metrics="logloss",
        early_stopping_rounds=20,
        seed=1367,
        shuffle=True,
        callbacks=None,
    )

    model = xgb.train(
        params=get_params(),
        dtrain=dtrain,
        num_boost_round=len(cvr) - 1,
    )
    return ModelWithData(model=model, matrix=dtrain, features=X, target=y)


@pytest.mark.model
def test_model(xgb_classifier) -> None:
    """ Test Classification Model"""

    # xgb model
    model = xgb_classifier.model
    dtrain = xgb_classifier.matrix
    X = xgb_classifier.features
    y = xgb_classifier.target

    # test slickml -> xgb classifier
    xgb_sml = XGBoostCVClassifier(
        num_boost_round=500, n_splits=4, metrics=("logloss"), params=get_params()
    )
    xgb_sml.fit(X, y)

    # Test predictions
    np.testing.assert_array_equal(
        np.where(xgb_sml.predict_proba(X, y) >= 0.5, 1, 0),
        np.where(model.predict(dtrain) >= 0.5, 1, 0),
        err_msg="Predictions are not equal",
        verbose=True,
    )
