from typing import Any, Dict, List, Tuple, Union

import hyperopt
import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

from slickml.optimization import XGBoostHyperOptimizer
from tests.conftest import _ids


class TestXGBoostHyperOptimizer:
    """Validates `XGBoostHyperOptimizer` instantiation."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"n_iter": "10"},
            {"n_init_iter": "5"},
            {"metrics": "roc_auc"},
            {"objective": "binary"},
            {"sparse_matrix": 1},
            {"scale_mean": 1},
            {"scale_std": 1},
            {"sparse_matrix": True, "scale_mean": True},
            {"objective": "binary:logistic", "metrics": "mae"},
            {"objective": "reg:squarederror", "metrics": "auc"},
            {"importance_type": "total_weight"},
            {"n_splits": "4"},
            {"early_stopping_rounds": "20"},
            {"random_state": "42"},
            {"stratified": 1},
            {"shuffle": 0},
            {"verbose": 0},
        ],
        ids=_ids,
    )
    def test_xgboosthyperoptimizer_instantiation__fails__with_invalid_inputs(
        self,
        kwargs: List[Dict[str, Any]],
    ) -> None:
        """Validates `XGBoostHyperOptimizer` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostHyperOptimizer(**kwargs)

    @pytest.mark.parametrize(
        ("clf_x_y"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["clf_x_y"],
        ids=_ids,
    )
    def test_clf_xgboosthyperoptimizer__passes__with_defaults(
        self,
        clf_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostHyperOptimizer` instanation passes with default inputs for classification."""
        X, y = clf_x_y
        xho = XGBoostHyperOptimizer()
        xho.fit(X, y)
        best_params = xho.get_best_params()
        params_bounds = xho.get_params_bounds()
        results = xho.get_results()
        trials = xho.get_trials()

        assert_that(xho).is_instance_of(XGBoostHyperOptimizer)
        assert_that(xho.n_iter).is_instance_of(int)
        assert_that(xho.n_iter).is_equal_to(100)
        assert_that(xho.num_boost_round).is_instance_of(int)
        assert_that(xho.num_boost_round).is_equal_to(200)
        assert_that(xho.n_splits).is_instance_of(int)
        assert_that(xho.n_splits).is_equal_to(4)
        assert_that(xho.early_stopping_rounds).is_instance_of(int)
        assert_that(xho.early_stopping_rounds).is_equal_to(20)
        assert_that(xho.random_state).is_instance_of(int)
        assert_that(xho.random_state).is_equal_to(1367)
        assert_that(xho.metrics).is_instance_of(str)
        assert_that(xho.metrics).is_equal_to("auc")
        assert_that(xho.sparse_matrix).is_instance_of(bool)
        assert_that(xho.objective).is_instance_of(str)
        assert_that(xho.objective).is_equal_to("binary:logistic")
        assert_that(xho.sparse_matrix).is_false()
        assert_that(xho.stratified).is_instance_of(bool)
        assert_that(xho.stratified).is_true()
        assert_that(xho.shuffle).is_instance_of(bool)
        assert_that(xho.shuffle).is_true()
        assert_that(xho.verbose).is_instance_of(bool)
        assert_that(xho.verbose).is_true()
        assert_that(xho.scale_mean).is_instance_of(bool)
        assert_that(xho.scale_mean).is_false()
        assert_that(xho.scale_std).is_instance_of(bool)
        assert_that(xho.scale_std).is_false()
        assert_that(xho.importance_type).is_instance_of(str)
        assert_that(xho.importance_type).is_equal_to("total_gain")
        assert_that(xho.params_bounds).is_instance_of(dict)
        assert_that(xho.results_).is_instance_of(list)
        assert_that(best_params).is_instance_of(dict)
        assert_that(params_bounds).is_instance_of(dict)
        assert_that(results).is_instance_of(list)
        assert_that(trials).is_instance_of(hyperopt.Trials)

    @pytest.mark.parametrize(
        ("reg_x_y"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["reg_x_y"],
        ids=_ids,
    )
    def test_reg_xgboosthyperoptimizer__passes__with_defaults(
        self,
        reg_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostHyperOptimizer` instanation passes with default inputs for regression."""
        X, y = reg_x_y
        xho = XGBoostHyperOptimizer(
            metrics="rmse",
            objective="reg:squarederror",
        )
        xho.fit(X, y)
        best_params = xho.get_best_params()
        params_bounds = xho.get_params_bounds()
        results = xho.get_results()
        trials = xho.get_trials()

        assert_that(xho).is_instance_of(XGBoostHyperOptimizer)
        assert_that(xho.n_iter).is_instance_of(int)
        assert_that(xho.n_iter).is_equal_to(100)
        assert_that(xho.num_boost_round).is_instance_of(int)
        assert_that(xho.num_boost_round).is_equal_to(200)
        assert_that(xho.n_splits).is_instance_of(int)
        assert_that(xho.n_splits).is_equal_to(4)
        assert_that(xho.early_stopping_rounds).is_instance_of(int)
        assert_that(xho.early_stopping_rounds).is_equal_to(20)
        assert_that(xho.random_state).is_instance_of(int)
        assert_that(xho.random_state).is_equal_to(1367)
        assert_that(xho.metrics).is_instance_of(str)
        assert_that(xho.metrics).is_equal_to("rmse")
        assert_that(xho.objective).is_instance_of(str)
        assert_that(xho.objective).is_equal_to("reg:squarederror")
        assert_that(xho.sparse_matrix).is_instance_of(bool)
        assert_that(xho.sparse_matrix).is_false()
        assert_that(xho.shuffle).is_instance_of(bool)
        assert_that(xho.shuffle).is_true()
        assert_that(xho.verbose).is_instance_of(bool)
        assert_that(xho.verbose).is_true()
        assert_that(xho.scale_mean).is_instance_of(bool)
        assert_that(xho.scale_mean).is_false()
        assert_that(xho.scale_std).is_instance_of(bool)
        assert_that(xho.scale_std).is_false()
        assert_that(xho.importance_type).is_instance_of(str)
        assert_that(xho.importance_type).is_equal_to("total_gain")
        assert_that(xho.params_bounds).is_instance_of(dict)
        assert_that(xho.results_).is_instance_of(list)
        assert_that(best_params).is_instance_of(dict)
        assert_that(params_bounds).is_instance_of(dict)
        assert_that(results).is_instance_of(list)
        assert_that(results).is_instance_of(list)
        assert_that(trials).is_instance_of(hyperopt.Trials)

    @pytest.mark.parametrize(
        ("clf_x_y"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["clf_x_y"],
        ids=_ids,
    )
    def test_clf_xgboosthyperoptimizer__passes__with_custom_params_bounds(
        self,
        clf_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostHyperOptimizer` instanation passes with custom inputs for classification."""
        X, y = clf_x_y
        xho = XGBoostHyperOptimizer(
            params_bounds={
                "max_depth": hyperopt.hp.choice("max_depth", range(5, 10)),
            },
        )
        xho.fit(X, y)
        best_params = xho.get_best_params()
        params_bounds = xho.get_params_bounds()
        results = xho.get_results()
        trials = xho.get_trials()

        assert_that(xho).is_instance_of(XGBoostHyperOptimizer)
        assert_that(xho.n_iter).is_instance_of(int)
        assert_that(xho.n_iter).is_equal_to(100)
        assert_that(xho.num_boost_round).is_instance_of(int)
        assert_that(xho.num_boost_round).is_equal_to(200)
        assert_that(xho.n_splits).is_instance_of(int)
        assert_that(xho.n_splits).is_equal_to(4)
        assert_that(xho.early_stopping_rounds).is_instance_of(int)
        assert_that(xho.early_stopping_rounds).is_equal_to(20)
        assert_that(xho.random_state).is_instance_of(int)
        assert_that(xho.random_state).is_equal_to(1367)
        assert_that(xho.metrics).is_instance_of(str)
        assert_that(xho.metrics).is_equal_to("auc")
        assert_that(xho.objective).is_instance_of(str)
        assert_that(xho.objective).is_equal_to("binary:logistic")
        assert_that(xho.sparse_matrix).is_instance_of(bool)
        assert_that(xho.sparse_matrix).is_false()
        assert_that(xho.stratified).is_instance_of(bool)
        assert_that(xho.stratified).is_true()
        assert_that(xho.shuffle).is_instance_of(bool)
        assert_that(xho.shuffle).is_true()
        assert_that(xho.verbose).is_instance_of(bool)
        assert_that(xho.verbose).is_true()
        assert_that(xho.scale_mean).is_instance_of(bool)
        assert_that(xho.scale_mean).is_false()
        assert_that(xho.scale_std).is_instance_of(bool)
        assert_that(xho.scale_std).is_false()
        assert_that(xho.importance_type).is_instance_of(str)
        assert_that(xho.importance_type).is_equal_to("total_gain")
        assert_that(xho.params_bounds).is_instance_of(dict)
        assert_that(xho.results_).is_instance_of(list)
        assert_that(best_params).is_instance_of(dict)
        assert_that(params_bounds).is_instance_of(dict)
        assert_that(results).is_instance_of(list)
        assert_that(results).is_instance_of(list)
        assert_that(trials).is_instance_of(hyperopt.Trials)
