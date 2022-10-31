from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

from slickml.base import BaseXGBoostEstimator
from tests.conftest import _ids


@dataclass
class XGBoostClassifier(BaseXGBoostEstimator):
    num_boost_round: Optional[int] = 200
    sparse_matrix: Optional[bool] = False
    scale_mean: Optional[bool] = False
    scale_std: Optional[bool] = False
    importance_type: Optional[str] = "total_gain"
    params: Optional[Dict[str, Union[str, float, int]]] = None

    def __post_init__(self) -> None:
        super().__post_init__()

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        self._check_X_y(
            X=X_train,
            y=y_train,
        )
        self.dtrain_ = self._dtrain(
            X_train=X_train,
            y_train=y_train,
        )
        return None

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    ) -> None:
        self.dtest_ = self._dtest(
            X_test=X_test,
            y_test=y_test,
        )
        return None


class TestBaseXGBoostEstimator:
    """Validates the intationation of any class inherited from `BaseXGBoostEstimator`."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"num_boost_round": "100"},
            {"sparse_matrix": 1},
            {"scale_mean": 1},
            {"scale_std": 1},
            {"sparse_matrix": True, "scale_mean": True},
            {"importance_type": "total_weight"},
        ],
        ids=_ids,
    )
    def test_basexgboosestimator_instantiation__fails__with_invalid_inputs(self, kwargs) -> None:
        """Validates `XGBoostClassifier` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostClassifier(**kwargs)

    @pytest.mark.parametrize(
        ("clf_train_test_x_y"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["clf_train_test_x_y"],
        ids=_ids,
    )
    def test_basexgboosestimator__passes__with_defaults(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
    ) -> None:
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = XGBoostClassifier(
            num_boost_round=10,
            sparse_matrix=True,
            scale_mean=False,
            scale_std=True,
            importance_type="gain",
            params={
                "foo": "bar",
            },
        )
        clf.fit(X_train, y_train)
        clf.predict(X_test, y_test)

        assert_that(clf).is_instance_of(XGBoostClassifier)
        assert_that(clf.num_boost_round).is_instance_of(int)
        assert_that(clf.num_boost_round).is_equal_to(10)
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_true()
        assert_that(clf.scale_mean).is_instance_of(bool)
        assert_that(clf.scale_mean).is_false()
        assert_that(clf.scale_std).is_instance_of(bool)
        assert_that(clf.scale_std).is_true()
        assert_that(clf.importance_type).is_instance_of(str)
        assert_that(clf.importance_type).is_equal_to("gain")
        assert_that(clf.params).is_instance_of(dict)
        assert_that(clf.params).is_equal_to(
            {
                "foo": "bar",
            },
        )

    @pytest.mark.parametrize(
        ("clf_train_test_x_y", "kwargs"),
        [
            ("dataframe", {"num_boost_round": 300}),
            ("dataframe", {"sparse_matrix": True}),
            ("dataframe", {"sparse_matrix": True, "scale_std": True}),
            ("dataframe", {"scale_mean": True}),
            ("dataframe", {"scale_std": True}),
            ("dataframe", {"scale_std": True, "scale_mean": True}),
            ("dataframe", {"importance_type": "cover"}),
            ("dataframe", {"params": {"max_depth": 4, "min_child_weight": 5}}),
        ],
        indirect=["clf_train_test_x_y"],
        ids=_ids,
    )
    def test_basexgboosestimator__passes__with_valid_inputs(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
        kwargs: Optional[Dict[str, Any]],
    ) -> None:
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = XGBoostClassifier(**kwargs)
        clf.fit(X_train, y_train)
        clf.predict(X_test, y_test)

        clf_without_y_test = XGBoostClassifier(**kwargs)
        clf_without_y_test.fit(X_train, y_train)
        clf_without_y_test.predict(X_test)

        assert_that(clf).is_instance_of(XGBoostClassifier)
        assert_that(clf.num_boost_round).is_instance_of(int)
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.scale_mean).is_instance_of(bool)
        assert_that(clf.scale_std).is_instance_of(bool)
        assert_that(clf.importance_type).is_instance_of(str)

        assert_that(clf_without_y_test).is_instance_of(XGBoostClassifier)
        assert_that(clf_without_y_test.num_boost_round).is_instance_of(int)
        assert_that(clf_without_y_test.sparse_matrix).is_instance_of(bool)
        assert_that(clf_without_y_test.scale_mean).is_instance_of(bool)
        assert_that(clf_without_y_test.scale_std).is_instance_of(bool)
        assert_that(clf_without_y_test.importance_type).is_instance_of(str)
