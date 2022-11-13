from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that
from bayes_opt import BayesianOptimization

from slickml.optimization import XGBoostBayesianOptimizer
from tests.conftest import _ids


class TestXGBoostBayesianOptimizer:
    """Validates `XGBoostBayesianOptimizer` instantiation."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"n_iter": "10"},
            {"n_init_iter": "5"},
            {"metrics": "roc_auc"},
            {"objective": "binary"},
            {"acquisition_criterion": "e"},
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
    def test_xgboostbayesianoptimizer_instantiation__fails__with_invalid_inputs(
        self,
        kwargs: List[Dict[str, Any]],
    ) -> None:
        """Validates `XGBoostBayesianOptimizer` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostBayesianOptimizer(**kwargs)

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
    def test_clf_xgboostbayesianoptimizer__passes__with_defaults(
        self,
        clf_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoosBayesianOptimizer` instanation passes with default inputs for classification."""
        X, y = clf_x_y
        xbo = XGBoostBayesianOptimizer()
        xbo.fit(X, y)
        best_params = xbo.get_best_params()
        params_bounds = xbo.get_params_bounds()
        results = xbo.get_results()
        best_results = xbo.get_best_results()
        optimizer = xbo.get_optimizer()

        assert_that(xbo).is_instance_of(XGBoostBayesianOptimizer)
        assert_that(xbo.n_iter).is_instance_of(int)
        assert_that(xbo.n_iter).is_equal_to(10)
        assert_that(xbo.n_init_iter).is_instance_of(int)
        assert_that(xbo.n_init_iter).is_equal_to(5)
        assert_that(xbo.num_boost_round).is_instance_of(int)
        assert_that(xbo.num_boost_round).is_equal_to(200)
        assert_that(xbo.n_splits).is_instance_of(int)
        assert_that(xbo.n_splits).is_equal_to(4)
        assert_that(xbo.early_stopping_rounds).is_instance_of(int)
        assert_that(xbo.early_stopping_rounds).is_equal_to(20)
        assert_that(xbo.random_state).is_instance_of(int)
        assert_that(xbo.random_state).is_equal_to(1367)
        assert_that(xbo.metrics).is_instance_of(str)
        assert_that(xbo.metrics).is_equal_to("auc")
        assert_that(xbo.sparse_matrix).is_instance_of(bool)
        assert_that(xbo.objective).is_instance_of(str)
        assert_that(xbo.objective).is_equal_to("binary:logistic")
        assert_that(xbo.sparse_matrix).is_false()
        assert_that(xbo.stratified).is_instance_of(bool)
        assert_that(xbo.stratified).is_true()
        assert_that(xbo.shuffle).is_instance_of(bool)
        assert_that(xbo.shuffle).is_true()
        assert_that(xbo.verbose).is_instance_of(int)
        assert_that(xbo.verbose).is_equal_to(2)
        assert_that(xbo.scale_mean).is_instance_of(bool)
        assert_that(xbo.scale_mean).is_false()
        assert_that(xbo.scale_std).is_instance_of(bool)
        assert_that(xbo.scale_std).is_false()
        assert_that(xbo.importance_type).is_instance_of(str)
        assert_that(xbo.importance_type).is_equal_to("total_gain")
        assert_that(xbo.params_bounds).is_instance_of(dict)
        assert_that(xbo.params_bounds).is_equal_to(
            {
                "max_depth": (2, 7),
                "learning_rate": (0.0, 1.0),
                "min_child_weight": (1.0, 20.0),
                "colsample_bytree": (0.1, 1.0),
                "subsample": (0.1, 1.0),
                "gamma": (0.0, 1.0),
                "reg_alpha": (0.0, 1.0),
                "reg_lambda": (0.0, 1.0),
            },
        )
        assert_that(xbo.results_).is_instance_of(pd.DataFrame)
        assert_that(xbo.optimizer_).is_instance_of(BayesianOptimization)
        assert_that(best_params).is_instance_of(dict)
        assert_that(params_bounds).is_instance_of(dict)
        assert_that(results).is_instance_of(pd.DataFrame)
        assert_that(best_results).is_instance_of(pd.DataFrame)
        assert_that(optimizer).is_instance_of(BayesianOptimization)

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
    def test_reg_xgboostbayesianoptimizer__passes__with_defaults(
        self,
        reg_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoosBayesianOptimizer` instanation passes with default inputs for regression."""
        X, y = reg_x_y
        xbo = XGBoostBayesianOptimizer(
            metrics="rmse",
            objective="reg:squarederror",
        )
        xbo.fit(X, y)
        best_params = xbo.get_best_params()
        params_bounds = xbo.get_params_bounds()
        results = xbo.get_results()
        best_results = xbo.get_best_results()
        optimizer = xbo.get_optimizer()

        assert_that(xbo).is_instance_of(XGBoostBayesianOptimizer)
        assert_that(xbo.n_iter).is_instance_of(int)
        assert_that(xbo.n_iter).is_equal_to(10)
        assert_that(xbo.n_init_iter).is_instance_of(int)
        assert_that(xbo.n_init_iter).is_equal_to(5)
        assert_that(xbo.num_boost_round).is_instance_of(int)
        assert_that(xbo.num_boost_round).is_equal_to(200)
        assert_that(xbo.n_splits).is_instance_of(int)
        assert_that(xbo.n_splits).is_equal_to(4)
        assert_that(xbo.early_stopping_rounds).is_instance_of(int)
        assert_that(xbo.early_stopping_rounds).is_equal_to(20)
        assert_that(xbo.random_state).is_instance_of(int)
        assert_that(xbo.random_state).is_equal_to(1367)
        assert_that(xbo.metrics).is_instance_of(str)
        assert_that(xbo.metrics).is_equal_to("rmse")
        assert_that(xbo.objective).is_instance_of(str)
        assert_that(xbo.objective).is_equal_to("reg:squarederror")
        assert_that(xbo.sparse_matrix).is_instance_of(bool)
        assert_that(xbo.sparse_matrix).is_false()
        assert_that(xbo.shuffle).is_instance_of(bool)
        assert_that(xbo.shuffle).is_true()
        assert_that(xbo.verbose).is_instance_of(int)
        assert_that(xbo.verbose).is_equal_to(2)
        assert_that(xbo.scale_mean).is_instance_of(bool)
        assert_that(xbo.scale_mean).is_false()
        assert_that(xbo.scale_std).is_instance_of(bool)
        assert_that(xbo.scale_std).is_false()
        assert_that(xbo.importance_type).is_instance_of(str)
        assert_that(xbo.importance_type).is_equal_to("total_gain")
        assert_that(xbo.params_bounds).is_instance_of(dict)
        assert_that(xbo.params_bounds).is_equal_to(
            {
                "max_depth": (2, 7),
                "learning_rate": (0.0, 1.0),
                "min_child_weight": (1.0, 20.0),
                "colsample_bytree": (0.1, 1.0),
                "subsample": (0.1, 1.0),
                "gamma": (0.0, 1.0),
                "reg_alpha": (0.0, 1.0),
                "reg_lambda": (0.0, 1.0),
            },
        )
        assert_that(xbo.results_).is_instance_of(pd.DataFrame)
        assert_that(xbo.optimizer_).is_instance_of(BayesianOptimization)
        assert_that(best_params).is_instance_of(dict)
        assert_that(params_bounds).is_instance_of(dict)
        assert_that(results).is_instance_of(pd.DataFrame)
        assert_that(best_results).is_instance_of(pd.DataFrame)
        assert_that(optimizer).is_instance_of(BayesianOptimization)

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
    def test_clf_xgboostbayesianoptimizer__passes__with_custom_params_bounds(
        self,
        clf_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoosBayesianOptimizer` instanation passes with default inputs for classification."""
        X, y = clf_x_y
        xbo = XGBoostBayesianOptimizer(
            params_bounds={
                "max_depth": (5, 10),
            },
        )
        xbo.fit(X, y)
        best_params = xbo.get_best_params()
        params_bounds = xbo.get_params_bounds()
        results = xbo.get_results()
        best_results = xbo.get_best_results()
        optimizer = xbo.get_optimizer()

        assert_that(xbo).is_instance_of(XGBoostBayesianOptimizer)
        assert_that(xbo.n_iter).is_instance_of(int)
        assert_that(xbo.n_iter).is_equal_to(10)
        assert_that(xbo.n_init_iter).is_instance_of(int)
        assert_that(xbo.n_init_iter).is_equal_to(5)
        assert_that(xbo.num_boost_round).is_instance_of(int)
        assert_that(xbo.num_boost_round).is_equal_to(200)
        assert_that(xbo.n_splits).is_instance_of(int)
        assert_that(xbo.n_splits).is_equal_to(4)
        assert_that(xbo.early_stopping_rounds).is_instance_of(int)
        assert_that(xbo.early_stopping_rounds).is_equal_to(20)
        assert_that(xbo.random_state).is_instance_of(int)
        assert_that(xbo.random_state).is_equal_to(1367)
        assert_that(xbo.metrics).is_instance_of(str)
        assert_that(xbo.metrics).is_equal_to("auc")
        assert_that(xbo.sparse_matrix).is_instance_of(bool)
        assert_that(xbo.objective).is_instance_of(str)
        assert_that(xbo.objective).is_equal_to("binary:logistic")
        assert_that(xbo.sparse_matrix).is_false()
        assert_that(xbo.stratified).is_instance_of(bool)
        assert_that(xbo.stratified).is_true()
        assert_that(xbo.shuffle).is_instance_of(bool)
        assert_that(xbo.shuffle).is_true()
        assert_that(xbo.verbose).is_instance_of(int)
        assert_that(xbo.verbose).is_equal_to(2)
        assert_that(xbo.scale_mean).is_instance_of(bool)
        assert_that(xbo.scale_mean).is_false()
        assert_that(xbo.scale_std).is_instance_of(bool)
        assert_that(xbo.scale_std).is_false()
        assert_that(xbo.importance_type).is_instance_of(str)
        assert_that(xbo.importance_type).is_equal_to("total_gain")
        assert_that(xbo.params_bounds).is_instance_of(dict)
        assert_that(xbo.params_bounds).is_equal_to(
            {
                "max_depth": (5, 10),
                "learning_rate": (0.0, 1.0),
                "min_child_weight": (1.0, 20.0),
                "colsample_bytree": (0.1, 1.0),
                "subsample": (0.1, 1.0),
                "gamma": (0.0, 1.0),
                "reg_alpha": (0.0, 1.0),
                "reg_lambda": (0.0, 1.0),
            },
        )
        assert_that(xbo.results_).is_instance_of(pd.DataFrame)
        assert_that(xbo.optimizer_).is_instance_of(BayesianOptimization)
        assert_that(best_params).is_instance_of(dict)
        assert_that(params_bounds).is_instance_of(dict)
        assert_that(results).is_instance_of(pd.DataFrame)
        assert_that(best_results).is_instance_of(pd.DataFrame)
        assert_that(optimizer).is_instance_of(BayesianOptimization)
