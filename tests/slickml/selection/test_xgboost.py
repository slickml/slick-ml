from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that
from matplotlib.figure import Figure
from sklearn.preprocessing import StandardScaler

from slickml.selection import XGBoostFeatureSelector
from tests.conftest import _ids


class TestXGBoostFeatureSelector:
    """Validates `XGBoostFeatureSelector` instantiation."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"n_iter": "100"},
            {"metrics": "roc_auc"},
            {"sparse_matrix": 1},
            {"scale_mean": 1},
            {"scale_std": 1},
            {"sparse_matrix": True, "scale_mean": True},
            {"params": ["auc"]},
            {"importance_type": "total_weight"},
            {"n_splits": "4"},
            {"early_stopping_rounds": "20"},
            {"random_state": "42"},
            {"stratified": 1},
            {"shuffle": 0},
            {"verbose": 0},
            {"callbacks": 1},
            {"nth_noise_threshold": "10"},
        ],
        ids=_ids,
    )
    def test_clf_xgboostfeatureselector_instantiation__fails__with_invalid_inputs(
        self,
        kwargs: List[Dict[str, Any]],
    ) -> None:
        """Validates `XGBoostFeatureSelector` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostFeatureSelector(**kwargs)

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
    def test_clf_xgboostfeatureselector__passes__with_defaults(
        self,
        clf_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostFeatureSelector` instanation passes with default inputs for classification."""
        X, y = clf_x_y
        xfs = XGBoostFeatureSelector()
        xfs.fit(X, y)
        params = xfs.get_params()
        default_params = xfs.get_default_params()
        feature_importance = xfs.get_feature_importance()
        cv_results = xfs.get_cv_results()
        feature_frequency = xfs.get_feature_frequency()
        cv_results_fig = xfs.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_frequency_fig = xfs.plot_frequency(
            display_plot=False,
            return_fig=True,
        )

        assert_that(xfs).is_instance_of(XGBoostFeatureSelector)
        assert_that(xfs.num_boost_round).is_instance_of(int)
        assert_that(xfs.num_boost_round).is_equal_to(200)
        assert_that(xfs.n_iter).is_instance_of(int)
        assert_that(xfs.n_iter).is_equal_to(3)
        assert_that(xfs.n_splits).is_instance_of(int)
        assert_that(xfs.n_splits).is_equal_to(4)
        assert_that(xfs.nth_noise_threshold).is_instance_of(int)
        assert_that(xfs.nth_noise_threshold).is_equal_to(1)
        assert_that(xfs.early_stopping_rounds).is_instance_of(int)
        assert_that(xfs.early_stopping_rounds).is_equal_to(20)
        assert_that(xfs.random_state).is_instance_of(int)
        assert_that(xfs.random_state).is_equal_to(1367)
        assert_that(xfs.metrics).is_instance_of(str)
        assert_that(xfs.metrics).is_equal_to("auc")
        assert_that(xfs.sparse_matrix).is_instance_of(bool)
        assert_that(xfs.sparse_matrix).is_false()
        assert_that(xfs.stratified).is_instance_of(bool)
        assert_that(xfs.stratified).is_true()
        assert_that(xfs.shuffle).is_instance_of(bool)
        assert_that(xfs.shuffle).is_true()
        assert_that(xfs.verbose_eval).is_instance_of(bool)
        assert_that(xfs.verbose_eval).is_false()
        assert_that(xfs.callbacks).is_none()
        assert_that(xfs.scale_mean).is_instance_of(bool)
        assert_that(xfs.scale_mean).is_false()
        assert_that(xfs.scale_std).is_instance_of(bool)
        assert_that(xfs.scale_std).is_false()
        assert_that(xfs.importance_type).is_instance_of(str)
        assert_that(xfs.importance_type).is_equal_to("total_gain")
        assert_that(xfs.params).is_instance_of(dict)
        assert_that(xfs.params).is_equal_to(
            {
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
            },
        )
        assert_that(xfs.cv_results_).is_instance_of(dict)
        assert_that(xfs.feature_importance_).is_instance_of(dict)
        assert_that(xfs.feature_frequency_).is_instance_of(pd.DataFrame)
        assert_that(xfs._scaler).is_none()
        assert_that(xfs.X).is_instance_of(pd.DataFrame)
        assert_that(xfs.y).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(dict)
        assert_that(feature_importance["model_iter1_fold1"]).is_instance_of(pd.DataFrame)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(feature_frequency).is_instance_of(pd.DataFrame)
        assert_that(cv_results_fig).is_instance_of(Figure)
        assert_that(feature_frequency_fig).is_instance_of(Figure)

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
    def test_reg_xgboostfeatureselector__passes__with_defaults(
        self,
        reg_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostFeatureSelector` instanation passes with default inputs for regression."""
        X, y = reg_x_y
        xfs = XGBoostFeatureSelector(metrics="rmse")
        xfs.fit(X, y)
        params = xfs.get_params()
        default_params = xfs.get_default_params()
        feature_importance = xfs.get_feature_importance()
        cv_results = xfs.get_cv_results()
        cv_results_fig = xfs.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_frequency_fig = xfs.plot_frequency(
            display_plot=False,
            return_fig=True,
        )

        assert_that(xfs).is_instance_of(XGBoostFeatureSelector)
        assert_that(xfs.num_boost_round).is_instance_of(int)
        assert_that(xfs.num_boost_round).is_equal_to(200)
        assert_that(xfs.n_iter).is_instance_of(int)
        assert_that(xfs.n_iter).is_equal_to(3)
        assert_that(xfs.n_splits).is_instance_of(int)
        assert_that(xfs.n_splits).is_equal_to(4)
        assert_that(xfs.nth_noise_threshold).is_instance_of(int)
        assert_that(xfs.nth_noise_threshold).is_equal_to(1)
        assert_that(xfs.early_stopping_rounds).is_instance_of(int)
        assert_that(xfs.early_stopping_rounds).is_equal_to(20)
        assert_that(xfs.random_state).is_instance_of(int)
        assert_that(xfs.random_state).is_equal_to(1367)
        assert_that(xfs.metrics).is_instance_of(str)
        assert_that(xfs.metrics).is_equal_to("rmse")
        assert_that(xfs.sparse_matrix).is_instance_of(bool)
        assert_that(xfs.sparse_matrix).is_false()
        assert_that(xfs.shuffle).is_instance_of(bool)
        assert_that(xfs.shuffle).is_true()
        assert_that(xfs.verbose_eval).is_instance_of(bool)
        assert_that(xfs.verbose_eval).is_false()
        assert_that(xfs.callbacks).is_none()
        assert_that(xfs.scale_mean).is_instance_of(bool)
        assert_that(xfs.scale_mean).is_false()
        assert_that(xfs.scale_std).is_instance_of(bool)
        assert_that(xfs.scale_std).is_false()
        assert_that(xfs.importance_type).is_instance_of(str)
        assert_that(xfs.importance_type).is_equal_to("total_gain")
        assert_that(xfs.params).is_instance_of(dict)
        assert_that(xfs.params).is_equal_to(
            {
                "eval_metric": "rmse",
                "tree_method": "hist",
                "objective": "reg:squarederror",
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
            },
        )
        assert_that(xfs.cv_results_).is_instance_of(dict)
        assert_that(xfs.feature_importance_).is_instance_of(dict)
        assert_that(xfs.feature_frequency_).is_instance_of(pd.DataFrame)
        assert_that(xfs._scaler).is_none()
        assert_that(xfs.X).is_instance_of(pd.DataFrame)
        assert_that(xfs.y).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(dict)
        assert_that(feature_importance["model_iter1_fold1"]).is_instance_of(pd.DataFrame)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(cv_results_fig).is_instance_of(Figure)
        assert_that(feature_frequency_fig).is_instance_of(Figure)

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
    def test_clf_xgboostfeatureselector__passes__with_valid_inputs(
        self,
        clf_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostFeatureSelector` instanation passes with valid inputs for classification."""
        X, y = clf_x_y
        xfs = XGBoostFeatureSelector(
            n_iter=1,
            sparse_matrix=True,
            scale_std=True,
            params={"eval_metric": "logloss"},
            early_stopping_rounds=1000,
            callbacks=True,
        )
        xfs.fit(X, y)
        params = xfs.get_params()
        default_params = xfs.get_default_params()
        feature_importance = xfs.get_feature_importance()
        cv_results = xfs.get_cv_results()
        feature_frequency = xfs.get_feature_frequency()
        cv_results_fig = xfs.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_frequency_fig = xfs.plot_frequency(
            display_plot=False,
            return_fig=True,
        )

        assert_that(xfs).is_instance_of(XGBoostFeatureSelector)
        assert_that(xfs.num_boost_round).is_instance_of(int)
        assert_that(xfs.num_boost_round).is_equal_to(200)
        assert_that(xfs.n_iter).is_instance_of(int)
        assert_that(xfs.n_iter).is_equal_to(1)
        assert_that(xfs.n_splits).is_instance_of(int)
        assert_that(xfs.n_splits).is_equal_to(4)
        assert_that(xfs.nth_noise_threshold).is_instance_of(int)
        assert_that(xfs.nth_noise_threshold).is_equal_to(1)
        assert_that(xfs.early_stopping_rounds).is_instance_of(int)
        assert_that(xfs.early_stopping_rounds).is_equal_to(1000)
        assert_that(xfs.random_state).is_instance_of(int)
        assert_that(xfs.random_state).is_equal_to(1367)
        assert_that(xfs.metrics).is_instance_of(str)
        assert_that(xfs.metrics).is_equal_to("auc")
        assert_that(xfs.sparse_matrix).is_instance_of(bool)
        assert_that(xfs.sparse_matrix).is_true()
        assert_that(xfs.stratified).is_instance_of(bool)
        assert_that(xfs.stratified).is_true()
        assert_that(xfs.shuffle).is_instance_of(bool)
        assert_that(xfs.shuffle).is_true()
        assert_that(xfs.verbose_eval).is_instance_of(bool)
        assert_that(xfs.verbose_eval).is_false()
        assert_that(xfs.callbacks).is_not_none()
        assert_that(xfs.scale_mean).is_instance_of(bool)
        assert_that(xfs.scale_mean).is_false()
        assert_that(xfs.scale_std).is_instance_of(bool)
        assert_that(xfs.scale_std).is_true()
        assert_that(xfs.importance_type).is_instance_of(str)
        assert_that(xfs.importance_type).is_equal_to("total_gain")
        assert_that(xfs.params).is_instance_of(dict)
        assert_that(xfs.params).is_equal_to(
            {
                "eval_metric": "logloss",
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
            },
        )
        assert_that(xfs.cv_results_).is_instance_of(dict)
        assert_that(xfs.feature_importance_).is_instance_of(dict)
        assert_that(xfs.feature_frequency_).is_instance_of(pd.DataFrame)
        assert_that(xfs._scaler).is_instance_of(StandardScaler)
        assert_that(xfs.X).is_instance_of(pd.DataFrame)
        assert_that(xfs.y).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(dict)
        assert_that(feature_importance["model_iter1_fold1"]).is_instance_of(pd.DataFrame)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(feature_frequency).is_instance_of(pd.DataFrame)
        assert_that(cv_results_fig).is_instance_of(Figure)
        assert_that(feature_frequency_fig).is_instance_of(Figure)

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
    def test_reg_xgboostfeatureselector__passes__with_valid_inputs(
        self,
        reg_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostFeatureSelector` instanation passes with valid inputs for regression."""
        X, y = reg_x_y
        xfs = XGBoostFeatureSelector(
            n_iter=1,
            metrics="rmse",
            sparse_matrix=True,
            scale_std=True,
            params={"eval_metric": "mae"},
            callbacks=True,
            early_stopping_rounds=1000,
        )
        xfs.fit(X, y)
        params = xfs.get_params()
        default_params = xfs.get_default_params()
        feature_importance = xfs.get_feature_importance()
        feature_frequency = xfs.get_feature_frequency()
        cv_results = xfs.get_cv_results()
        cv_results_fig = xfs.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_frequency_fig = xfs.plot_frequency(
            display_plot=False,
            return_fig=True,
        )

        assert_that(xfs).is_instance_of(XGBoostFeatureSelector)
        assert_that(xfs.num_boost_round).is_instance_of(int)
        assert_that(xfs.num_boost_round).is_equal_to(200)
        assert_that(xfs.n_iter).is_instance_of(int)
        assert_that(xfs.n_iter).is_equal_to(1)
        assert_that(xfs.n_splits).is_instance_of(int)
        assert_that(xfs.n_splits).is_equal_to(4)
        assert_that(xfs.nth_noise_threshold).is_instance_of(int)
        assert_that(xfs.nth_noise_threshold).is_equal_to(1)
        assert_that(xfs.early_stopping_rounds).is_instance_of(int)
        assert_that(xfs.early_stopping_rounds).is_equal_to(1000)
        assert_that(xfs.random_state).is_instance_of(int)
        assert_that(xfs.random_state).is_equal_to(1367)
        assert_that(xfs.metrics).is_instance_of(str)
        assert_that(xfs.metrics).is_equal_to("rmse")
        assert_that(xfs.sparse_matrix).is_instance_of(bool)
        assert_that(xfs.sparse_matrix).is_true()
        assert_that(xfs.shuffle).is_instance_of(bool)
        assert_that(xfs.shuffle).is_true()
        assert_that(xfs.verbose_eval).is_instance_of(bool)
        assert_that(xfs.verbose_eval).is_false()
        assert_that(xfs.callbacks).is_not_none()
        assert_that(xfs.scale_mean).is_instance_of(bool)
        assert_that(xfs.scale_mean).is_false()
        assert_that(xfs.scale_std).is_instance_of(bool)
        assert_that(xfs.scale_std).is_true()
        assert_that(xfs.importance_type).is_instance_of(str)
        assert_that(xfs.importance_type).is_equal_to("total_gain")
        assert_that(xfs.params).is_instance_of(dict)
        assert_that(xfs.params).is_equal_to(
            {
                "eval_metric": "mae",
                "tree_method": "hist",
                "objective": "reg:squarederror",
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
            },
        )
        assert_that(xfs.cv_results_).is_instance_of(dict)
        assert_that(xfs.feature_importance_).is_instance_of(dict)
        assert_that(xfs.feature_frequency_).is_instance_of(pd.DataFrame)
        assert_that(xfs._scaler).is_instance_of(StandardScaler)
        assert_that(xfs.X).is_instance_of(pd.DataFrame)
        assert_that(xfs.y).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(dict)
        assert_that(feature_importance["model_iter1_fold1"]).is_instance_of(pd.DataFrame)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(feature_frequency).is_instance_of(pd.DataFrame)
        assert_that(cv_results_fig).is_instance_of(Figure)
        assert_that(feature_frequency_fig).is_instance_of(Figure)
