from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import shap
import xgboost as xgb
from assertpy import assert_that
from matplotlib.figure import Figure

from slickml.classification import XGBoostCVClassifier
from tests.conftest import _ids


# TODO(amir): Currently `SHAP` raises a lot of warnings. Please figure out a way to dump these warnings
class TestXGBoostCVClassifier:
    """Validates `XGBoostClassifierCV` instantiation."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"num_boost_round": "100"},
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
        ],
        ids=_ids,
    )
    def test_xgboostcvclassifier_instantiation__fails__with_invalid_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates `XGBoostCVClassifier` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostCVClassifier(**kwargs)

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
    def test_xgboostcvclassifier__passes__with_defaults(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List[float]],
            Union[np.ndarray, List[float]],
        ],
    ) -> None:
        """Validates `XGBoostClassifier` instanation passes with default inputs."""
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = XGBoostCVClassifier()
        clf.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict_proba()` and `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred_proba = clf.predict_proba(X_test, y_test)
        y_pred = clf.predict(X_test, y_test)
        params = clf.get_params()
        default_params = clf.get_default_params()
        feature_importance = clf.get_feature_importance()
        cv_results = clf.get_cv_results()
        shap_explainer = clf.get_shap_explainer()
        cv_results_fig = clf.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_importance_fig = clf.plot_feature_importance(
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_test_fig = clf.plot_shap_waterfall(
            validation=True,
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_train_fig = clf.plot_shap_waterfall(
            validation=False,
            display_plot=False,
            return_fig=True,
        )
        # TODO(amir): figure out `return-fig` flag and test the figure object
        clf.plot_shap_summary(
            validation=True,
            display_plot=False,
        )
        clf.plot_shap_summary(
            validation=False,
            display_plot=False,
        )

        assert_that(clf).is_instance_of(XGBoostCVClassifier)
        assert_that(clf.num_boost_round).is_instance_of(int)
        assert_that(clf.num_boost_round).is_equal_to(200)
        assert_that(clf.n_splits).is_instance_of(int)
        assert_that(clf.n_splits).is_equal_to(4)
        assert_that(clf.early_stopping_rounds).is_instance_of(int)
        assert_that(clf.early_stopping_rounds).is_equal_to(20)
        assert_that(clf.random_state).is_instance_of(int)
        assert_that(clf.random_state).is_equal_to(1367)
        assert_that(clf.metrics).is_instance_of(str)
        assert_that(clf.metrics).is_equal_to("auc")
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_false()
        assert_that(clf.stratified).is_instance_of(bool)
        assert_that(clf.stratified).is_true()
        assert_that(clf.shuffle).is_instance_of(bool)
        assert_that(clf.shuffle).is_true()
        assert_that(clf.verbose).is_instance_of(bool)
        assert_that(clf.verbose).is_true()
        assert_that(clf.callbacks).is_none()
        assert_that(clf.scale_mean).is_instance_of(bool)
        assert_that(clf.scale_mean).is_false()
        assert_that(clf.scale_std).is_instance_of(bool)
        assert_that(clf.scale_std).is_false()
        assert_that(clf.importance_type).is_instance_of(str)
        assert_that(clf.importance_type).is_equal_to("total_gain")
        assert_that(clf.params).is_instance_of(dict)
        assert_that(clf.params).is_equal_to(
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
        assert_that(clf.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(clf.cv_results_.shape[1]).is_equal_to(4)
        assert_that(clf.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(clf.feature_importance_.shape[0]).is_equal_to(X_train.shape[1])
        assert_that(clf.feature_importance_.columns.tolist()).contains("total_gain")
        assert_that(clf.scaler_).is_none()
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_train_).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test_).is_instance_of(pd.DataFrame)
        assert_that(clf.shap_explainer_).is_instance_of(shap.TreeExplainer)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_instance_of(np.ndarray)
        pdt.assert_frame_equal(clf.X_train_, clf.X_train, check_dtype=True)
        npt.assert_array_equal(clf.y_train, y_train)
        pdt.assert_frame_equal(clf.X_test_, clf.X_test_, check_dtype=True)
        npt.assert_array_equal(clf.y_test, y_test)
        assert_that(clf.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(clf.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred_proba), 0.80043, decimal=5)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.93589, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(shap_explainer).is_instance_of(shap.TreeExplainer)
        assert_that(cv_results_fig).is_instance_of(Figure)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(clf.shap_values_test_), 0.02838, decimal=5)
        npt.assert_almost_equal(np.mean(clf.shap_values_train_), 0.03783, decimal=5)

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
    def test_xgboostcvclassifier__passes__with_defaults_and_no_test_targets(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List[float]],
            Union[np.ndarray, List[float]],
        ],
    ) -> None:
        """Validates `XGBoostClassifier` instanation passes with default inputs."""
        X_train, X_test, y_train, _ = clf_train_test_x_y
        clf = XGBoostCVClassifier()
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        params = clf.get_params()
        default_params = clf.get_default_params()
        shap_explainer = clf.get_shap_explainer()
        feature_importance = clf.get_feature_importance()
        cv_results = clf.get_cv_results()
        cv_results_fig = clf.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_importance_fig = clf.plot_feature_importance(
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_test_fig = clf.plot_shap_waterfall(
            validation=True,
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_train_fig = clf.plot_shap_waterfall(
            validation=False,
            display_plot=False,
            return_fig=True,
        )
        # TODO(amir): figure out `return-fig` flag and test the figure object
        clf.plot_shap_summary(
            validation=True,
            display_plot=False,
        )
        clf.plot_shap_summary(
            validation=False,
            display_plot=False,
        )

        assert_that(clf).is_instance_of(XGBoostCVClassifier)
        assert_that(clf.num_boost_round).is_instance_of(int)
        assert_that(clf.num_boost_round).is_equal_to(200)
        assert_that(clf.n_splits).is_instance_of(int)
        assert_that(clf.n_splits).is_equal_to(4)
        assert_that(clf.early_stopping_rounds).is_instance_of(int)
        assert_that(clf.early_stopping_rounds).is_equal_to(20)
        assert_that(clf.random_state).is_instance_of(int)
        assert_that(clf.random_state).is_equal_to(1367)
        assert_that(clf.metrics).is_instance_of(str)
        assert_that(clf.metrics).is_equal_to("auc")
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_false()
        assert_that(clf.stratified).is_instance_of(bool)
        assert_that(clf.stratified).is_true()
        assert_that(clf.shuffle).is_instance_of(bool)
        assert_that(clf.shuffle).is_true()
        assert_that(clf.verbose).is_instance_of(bool)
        assert_that(clf.verbose).is_true()
        assert_that(clf.callbacks).is_none()
        assert_that(clf.scale_mean).is_instance_of(bool)
        assert_that(clf.scale_mean).is_false()
        assert_that(clf.scale_std).is_instance_of(bool)
        assert_that(clf.scale_std).is_false()
        assert_that(clf.importance_type).is_instance_of(str)
        assert_that(clf.importance_type).is_equal_to("total_gain")
        assert_that(clf.params).is_instance_of(dict)
        assert_that(clf.params).is_equal_to(
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
        assert_that(clf.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(clf.cv_results_.shape[1]).is_equal_to(4)
        assert_that(clf.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(clf.feature_importance_.shape[0]).is_equal_to(X_train.shape[1])
        assert_that(clf.feature_importance_.columns.tolist()).contains("total_gain")
        assert_that(clf.scaler_).is_none()
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_train_).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test_).is_instance_of(pd.DataFrame)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_none()
        pdt.assert_frame_equal(clf.X_train_, clf.X_train, check_dtype=True)
        npt.assert_array_equal(clf.y_train, y_train)
        pdt.assert_frame_equal(clf.X_test_, clf.X_test_, check_dtype=True)
        assert_that(clf.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(clf.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(clf.shap_explainer_).is_instance_of(shap.TreeExplainer)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred_proba), 0.80043, decimal=5)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.93589, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.TreeExplainer)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(cv_results_fig).is_instance_of(Figure)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(clf.shap_values_test_), 0.02838, decimal=5)
        npt.assert_almost_equal(np.mean(clf.shap_values_train_), 0.03783, decimal=5)

    @pytest.mark.parametrize(
        ("clf_train_test_x_y", "kwargs"),
        [
            ("dataframe", {"n_splits": 10}),
            ("dataframe", {"early_stopping_rounds": 100}),
            ("dataframe", {"num_boost_round": 300}),
            ("dataframe", {"metrics": "aucpr"}),
            ("dataframe", {"shuffle": False}),
            ("dataframe", {"verbose": False}),
            ("dataframe", {"callbacks": True}),
            ("dataframe", {"stratified": False}),
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
    def test_xgboostcvclassifier__passes__with_valid_inputs(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List[float]],
            Union[np.ndarray, List[float]],
        ],
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates `XGBoostCVClassifier` instanation passes with valid inputs."""
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = XGBoostCVClassifier(**kwargs)
        clf.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict_proba()` and `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred_proba = clf.predict_proba(X_test, y_test)
        y_pred = clf.predict(X_test, y_test)
        params = clf.get_params()
        default_params = clf.get_default_params()
        shap_explainer = clf.get_shap_explainer()
        feature_importance = clf.get_feature_importance()
        cv_results = clf.get_cv_results()
        cv_results_fig = clf.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        feature_importance_fig = clf.plot_feature_importance(
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_test_fig = clf.plot_shap_waterfall(
            validation=True,
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_train_fig = clf.plot_shap_waterfall(
            validation=False,
            display_plot=False,
            return_fig=True,
        )
        # TODO(amir): figure out `return-fig` flag and test the figure object
        clf.plot_shap_summary(
            validation=True,
            display_plot=False,
        )
        clf.plot_shap_summary(
            validation=False,
            display_plot=False,
        )

        assert_that(clf).is_instance_of(XGBoostCVClassifier)
        assert_that(clf.num_boost_round).is_instance_of(int)
        assert_that(clf.metrics).is_instance_of(str)
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.scale_mean).is_instance_of(bool)
        assert_that(clf.scale_std).is_instance_of(bool)
        assert_that(clf.importance_type).is_instance_of(str)
        assert_that(clf.params).is_instance_of(dict)
        assert_that(clf.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(clf.feature_importance_.shape[0]).is_equal_to(X_train.shape[1])
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_train_).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test_).is_instance_of(pd.DataFrame)
        assert_that(clf.shap_explainer_).is_instance_of(shap.TreeExplainer)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_instance_of(np.ndarray)
        npt.assert_array_equal(clf.y_train, y_train)
        npt.assert_array_equal(clf.y_test, y_test)
        assert_that(clf.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(clf.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(clf.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        assert_that(y_pred).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(shap_explainer).is_instance_of(shap.TreeExplainer)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        assert_that(cv_results_fig).is_instance_of(Figure)

    @pytest.mark.parametrize(
        (
            "clf_train_test_x_y",
            "waterfall_kwargs",
            "summary_kwargs",
        ),
        [
            (
                "dataframe",
                {
                    "display_plot": False,
                    "return_fig": True,
                    "title": "Foo",
                    "figsize": (20, 20),
                    "bar_color": "navy",
                    "bar_thickness": 4.2,
                    "line_color": "red",
                    "marker": "s",
                    "markersize": 4.2,
                    "markeredgecolor": "green",
                    "markerfacecolor": "blue",
                    "markeredgewidth": 4.2,
                    "max_display": 2,
                    "fontsize": 20,
                },
                {
                    "display_plot": False,
                    "plot_type": "bar",
                    "figsize": (20, 20),
                    "color": "red",
                    "title": "foo",
                    "feature_names": ["a", "b", "c", "d", "e", "f"],
                    "class_names": ["yes", "no"],
                },
            ),
            (
                "dataframe",
                {
                    "display_plot": False,
                    "return_fig": True,
                },
                {
                    "display_plot": False,
                    "color": None,
                    "plot_type": "bar",
                },
            ),
        ],
        indirect=["clf_train_test_x_y"],
        ids=_ids,
    )
    def test_xgboostcvclassifier_shap_plots__passes__with_valid_inputs(
        self,
        clf_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        waterfall_kwargs: Dict[str, Any],
        summary_kwargs: Dict[str, Any],
    ) -> None:
        """Validates `XGBoostCVClassifier` Shap plots passes with valid inputs."""
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = XGBoostCVClassifier()
        clf.fit(X_train, y_train)
        _ = clf.predict_proba(X_test, y_test)
        shap_waterfall_fig = clf.plot_shap_waterfall(**waterfall_kwargs)
        # TODO(amir): how can we test the figure object ?
        clf.plot_shap_summary(**summary_kwargs)

        assert_that(shap_waterfall_fig).is_instance_of(Figure)


# TODO(amir): we have to test the callbacks + verbose logs
