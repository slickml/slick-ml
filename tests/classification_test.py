import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import namedtuple
import sklearn.datasets as datasets

from slickml.utilities import df_to_csr
from slickml.classification import XGBoostCVClassifier

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


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
        "silent": True,
        "nthread": 4,
        "scale_pos_weight": 1,
    }
    return params


@pytest.fixture(scope="session")
def xgb_classifier():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data,columns=iris.feature_names)
    y = pd.DataFrame(iris.target,columns=['CLASS'])
    dtrain = xgb.DMatrix(
        data=df_to_csr(X, fillna=0.0, verbose=False),
        label=y,
        feature_names=X.columns.tolist(),
    )

    cvr = xgb.cv(
        params=get_params(),
        dtrain=dtrain,
        num_boost_round=500,
        nfold=4,
        stratified=True,
        metrics="logloss",
        early_stopping_rounds=None,
        seed=None,
        shuffle=True,
        callbacks=False,
    )

    model = xgb.train(
        params=get_params(),
        dtrain=dtrain,
        num_boost_round=len(cvr) - 1,
    )
    return ModelWithData(model=model, inference_data=X, target_data=y)


@pytest.mark.model
def test_model(xgb_classifier) -> None:
    """ Test Classification Model"""

    # xgb model
    model = xgb_classifier.model
    X,y = model.inference_data,model.target_data

    # test slickml -> xgb
    xgb_sml = XGBoostCVClassifier(
        num_boost_round=500, n_splits=4, metrics=("logloss"), params=get_params()
    )
    xgb_sml.fit(X, y)

    # Test predictions
    np.testing.assert_array_equal(
        model.predict(model.inference_data),
        xgb_sml.predict(xgb_model.inference_data),
    )
