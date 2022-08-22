from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import xgboost as xgb
from assertpy import assert_that
from matplotlib.figure import Figure
from pytest import FixtureRequest
from sklearn.model_selection import train_test_split

from slickml.classification import XGBoostClassifier
from tests.utils import _ids, _load_test_data_from_csv


# TODO(amir): Currently `SHAP` raises a lot of warnings. Please figure out a way to dump these warnings
class TestXGBoostClassifier:
    """Validates `XGBoostClassifier` instantiation."""

    @staticmethod
    @pytest.fixture(scope="module")
    def clf_x_y_data(
        request: FixtureRequest,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[np.ndarray, List],
        Union[np.ndarray, List],
    ]:
        """Returns stratified train/test sets."""
        df = _load_test_data_from_csv(
            filename="clf_test_data.csv",
        )
        y = df["CLASS"].values
        X = df.drop(
            ["CLASS"],
            axis=1,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
            stratify=y,
            random_state=1367,
        )
        if request.param == "dataframe":
            return (X_train, X_test, y_train, y_test)
        elif request.param == "array":
            return (X_train.values, X_test.values, y_train, y_test)
        elif request.param == "list":
            return (X_train, X_test, y_train.tolist(), y_test.tolist())
        else:
            return None

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {
                "num_boost_round": "100",
            },
            {
                "metrics": "roc_auc",
            },
            {
                "sparse_matrix": 1,
            },
            {
                "scale_mean": 1,
            },
            {
                "scale_std": 1,
            },
            {
                "sparse_matrix": True,
                "scale_mean": True,
            },
            {
                "params": ["auc"],
            },
            {
                "importance_type": "total_weight",
            },
        ],
        ids=_ids,
    )
    def test_instantiation__fails__with_invalid_inputs(self, kwargs) -> None:
        """Validates `XGBoostClassifier` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostClassifier(**kwargs)

    @pytest.mark.parametrize(
        ("clf_x_y_data"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["clf_x_y_data"],
        ids=_ids,
    )
    def test_xgboostclassifier__passes__with_defaults(
        self,
        clf_x_y_data: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostClassifier` instanation passes with default inputs."""
        X_train, X_test, y_train, y_test = clf_x_y_data
        clf = XGBoostClassifier()
        clf.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict_proba()` and `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred_proba = clf.predict_proba(X_test, y_test)
        y_pred = clf.predict(X_test, y_test)
        params = clf.get_params()
        default_params = clf.get_default_params()
        feature_importance = clf.get_feature_importance()
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

        assert_that(clf).is_instance_of(XGBoostClassifier)
        assert_that(clf.num_boost_round).is_instance_of(int)
        assert_that(clf.num_boost_round).is_equal_to(200)
        assert_that(clf.metrics).is_instance_of(str)
        assert_that(clf.metrics).is_equal_to("auc")
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_false()
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
        assert_that(clf.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(clf.feature_importance_.shape[0]).is_equal_to(X_train.shape[1])
        assert_that(clf.feature_importance_.columns.tolist()).contains("total_gain")
        assert_that(clf.scaler_).is_none()
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_train_).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test_).is_instance_of(pd.DataFrame)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_instance_of(np.ndarray)
        pdt.assert_frame_equal(clf.X_train_, clf.X_train, check_dtype=True)
        npt.assert_array_equal(clf.y_train, y_train)
        pdt.assert_frame_equal(clf.X_test_, clf.X_test_, check_dtype=True)
        npt.assert_array_equal(clf.y_test, y_test)
        assert_that(clf.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(clf.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred_proba), 0.800523, decimal=5)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.88461, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(clf.shap_values_test_), 0.09268, decimal=5)
        npt.assert_almost_equal(np.mean(clf.shap_values_train_), 0.11710, decimal=5)

    @pytest.mark.parametrize(
        ("clf_x_y_data", "kwargs"),
        [
            (
                "dataframe",
                {
                    "num_boost_round": 300,
                },
            ),
            (
                "dataframe",
                {
                    "metrics": "aucpr",
                },
            ),
            (
                "dataframe",
                {
                    "sparse_matrix": True,
                },
            ),
            (
                "dataframe",
                {
                    "sparse_matrix": True,
                    "scale_std": True,
                },
            ),
            (
                "dataframe",
                {
                    "scale_mean": True,
                },
            ),
            (
                "dataframe",
                {
                    "scale_std": True,
                },
            ),
            (
                "dataframe",
                {
                    "scale_std": True,
                    "scale_mean": True,
                },
            ),
            (
                "dataframe",
                {
                    "importance_type": "cover",
                },
            ),
            (
                "dataframe",
                {
                    "params": {
                        "max_depth": 4,
                        "min_child_weight": 5,
                    },
                },
            ),
        ],
        indirect=["clf_x_y_data"],
        ids=_ids,
    )
    def test_xgboostclassifier__passes__with_valid_inputs(
        self,
        clf_x_y_data: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
        kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Validates `XGBoostClassifier` instanation passes with valid inputs."""
        X_train, X_test, y_train, y_test = clf_x_y_data
        clf = XGBoostClassifier(**kwargs)
        clf.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict_proba()` and `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred_proba = clf.predict_proba(X_test, y_test)
        y_pred = clf.predict(X_test, y_test)
        params = clf.get_params()
        default_params = clf.get_default_params()
        feature_importance = clf.get_feature_importance()
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

        assert_that(clf).is_instance_of(XGBoostClassifier)
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
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_instance_of(np.ndarray)
        npt.assert_array_equal(clf.y_train, y_train)
        npt.assert_array_equal(clf.y_test, y_test)
        assert_that(clf.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(clf.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        assert_that(y_pred).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)

    @pytest.mark.parametrize(
        (
            "clf_x_y_data",
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
        indirect=["clf_x_y_data"],
        ids=_ids,
    )
    def test_xgboostclassifier_shap_plots__passes__with_valid_inputs(
        self,
        clf_x_y_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        waterfall_kwargs: Dict[str, Any],
        summary_kwargs: Dict[str, Any],
    ) -> None:
        """Validates `XGBoostClassifier` Shap plots passes with valid inputs."""
        X_train, X_test, y_train, y_test = clf_x_y_data
        clf = XGBoostClassifier()
        clf.fit(X_train, y_train)
        _ = clf.predict_proba(X_test, y_test)
        shap_waterfall_fig = clf.plot_shap_waterfall(**waterfall_kwargs)
        # TODO(amir): how can we test the figure object ?
        clf.plot_shap_summary(**summary_kwargs)

        assert_that(shap_waterfall_fig).is_instance_of(Figure)