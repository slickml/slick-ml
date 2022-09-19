from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import shap
import xgboost as xgb
from assertpy import assert_that
from matplotlib.figure import Figure
from pytest import FixtureRequest
from sklearn.model_selection import train_test_split

from slickml.regression import XGBoostRegressor
from tests.utils import _ids, _load_test_data_from_csv


# TODO(amir): Currently `SHAP` raises a lot of warnings. Please figure out a way to dump these warnings
# TODO(amir): test out regression with multiple outputs (y1, y2)
class TestXGBoostRegressor:
    """Validates `XGBoostRegressor` instantiation."""

    @staticmethod
    @pytest.fixture(scope="module")
    def reg_x_y_data(
        request: FixtureRequest,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[np.ndarray, List],
        Union[np.ndarray, List],
    ]:
        """Returns train/test sets."""
        df = _load_test_data_from_csv(
            filename="reg_test_data.csv",
        )
        # TODO(amir): try to pull-out multi target regression as well here
        y = df["TARGET1"].values
        X = df.drop(
            ["TARGET1", "TARGET2"],
            axis=1,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
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
            {"num_boost_round": "100"},
            {"metrics": "rms"},
            {"sparse_matrix": 1},
            {"scale_mean": 1},
            {"scale_std": 1},
            {"sparse_matrix": True, "scale_mean": True},
            {"params": ["reg:logistic"]},
            {"importance_type": "total_weight"},
        ],
        ids=_ids,
    )
    def test_xgboostregressor_instantiation__fails__with_invalid_inputs(self, kwargs) -> None:
        """Validates `XGBoostRegressor` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            XGBoostRegressor(**kwargs)

    @pytest.mark.parametrize(
        ("reg_x_y_data"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["reg_x_y_data"],
        ids=_ids,
    )
    def test_xgboostregressor__passes__with_defaults(
        self,
        reg_x_y_data: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostRegressor` instanation passes with default inputs."""
        X_train, X_test, y_train, y_test = reg_x_y_data
        reg = XGBoostRegressor()
        reg.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred = reg.predict(X_test, y_test)
        params = reg.get_params()
        default_params = reg.get_default_params()
        shap_explainer = reg.get_shap_explainer()
        feature_importance = reg.get_feature_importance()
        feature_importance_fig = reg.plot_feature_importance(
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_test_fig = reg.plot_shap_waterfall(
            validation=True,
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_train_fig = reg.plot_shap_waterfall(
            validation=False,
            display_plot=False,
            return_fig=True,
        )
        # TODO(amir): figure out `return-fig` flag and test the figure object
        reg.plot_shap_summary(
            validation=True,
            display_plot=False,
        )
        reg.plot_shap_summary(
            validation=False,
            display_plot=False,
        )

        assert_that(reg).is_instance_of(XGBoostRegressor)
        assert_that(reg.num_boost_round).is_instance_of(int)
        assert_that(reg.num_boost_round).is_equal_to(200)
        assert_that(reg.metrics).is_instance_of(str)
        assert_that(reg.metrics).is_equal_to("rmse")
        assert_that(reg.sparse_matrix).is_instance_of(bool)
        assert_that(reg.sparse_matrix).is_false()
        assert_that(reg.scale_mean).is_instance_of(bool)
        assert_that(reg.scale_mean).is_false()
        assert_that(reg.scale_std).is_instance_of(bool)
        assert_that(reg.scale_std).is_false()
        assert_that(reg.importance_type).is_instance_of(str)
        assert_that(reg.importance_type).is_equal_to("total_gain")
        assert_that(reg.params).is_instance_of(dict)
        assert_that(reg.params).is_equal_to(
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
        assert_that(reg.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(reg.feature_importance_.shape[0]).is_equal_to(11)
        assert_that(reg.feature_importance_.columns.tolist()).contains("total_gain")
        assert_that(reg.scaler_).is_none()
        assert_that(reg.shap_explainer_).is_instance_of(shap.TreeExplainer)
        assert_that(reg.X_train).is_instance_of(pd.DataFrame)
        assert_that(reg.X_train_).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test_).is_instance_of(pd.DataFrame)
        assert_that(reg.y_train).is_instance_of(np.ndarray)
        assert_that(reg.y_test).is_instance_of(np.ndarray)
        pdt.assert_frame_equal(reg.X_train_, reg.X_train, check_dtype=True)
        npt.assert_array_equal(reg.y_train, y_train)
        pdt.assert_frame_equal(reg.X_test_, reg.X_test_, check_dtype=True)
        npt.assert_array_equal(reg.y_test, y_test)
        assert_that(reg.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(reg.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.97498, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.TreeExplainer)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(reg.shap_values_test_), 9.43848e-07, decimal=5)
        npt.assert_almost_equal(np.mean(reg.shap_values_train_), -1.98011e-08, decimal=5)

    @pytest.mark.parametrize(
        ("reg_x_y_data"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["reg_x_y_data"],
        ids=_ids,
    )
    def test_xgboostregressor__passes__with_defaults_and_no_test_targets(
        self,
        reg_x_y_data: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `XGBoostRegressor` instanation passes with default inputs."""
        X_train, X_test, y_train, _ = reg_x_y_data
        reg = XGBoostRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        params = reg.get_params()
        default_params = reg.get_default_params()
        shap_explainer = reg.get_shap_explainer()
        feature_importance = reg.get_feature_importance()
        feature_importance_fig = reg.plot_feature_importance(
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_test_fig = reg.plot_shap_waterfall(
            validation=True,
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_train_fig = reg.plot_shap_waterfall(
            validation=False,
            display_plot=False,
            return_fig=True,
        )
        # TODO(amir): figure out `return-fig` flag and test the figure object
        reg.plot_shap_summary(
            validation=True,
            display_plot=False,
        )
        reg.plot_shap_summary(
            validation=False,
            display_plot=False,
        )

        assert_that(reg).is_instance_of(XGBoostRegressor)
        assert_that(reg.num_boost_round).is_instance_of(int)
        assert_that(reg.num_boost_round).is_equal_to(200)
        assert_that(reg.metrics).is_instance_of(str)
        assert_that(reg.metrics).is_equal_to("rmse")
        assert_that(reg.sparse_matrix).is_instance_of(bool)
        assert_that(reg.sparse_matrix).is_false()
        assert_that(reg.scale_mean).is_instance_of(bool)
        assert_that(reg.scale_mean).is_false()
        assert_that(reg.scale_std).is_instance_of(bool)
        assert_that(reg.scale_std).is_false()
        assert_that(reg.importance_type).is_instance_of(str)
        assert_that(reg.importance_type).is_equal_to("total_gain")
        assert_that(reg.params).is_instance_of(dict)
        assert_that(reg.params).is_equal_to(
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
        assert_that(reg.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(reg.feature_importance_.shape[0]).is_equal_to(11)
        assert_that(reg.feature_importance_.columns.tolist()).contains("total_gain")
        assert_that(reg.scaler_).is_none()
        assert_that(reg.shap_explainer_).is_instance_of(shap.TreeExplainer)
        assert_that(reg.X_train).is_instance_of(pd.DataFrame)
        assert_that(reg.X_train_).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test_).is_instance_of(pd.DataFrame)
        assert_that(reg.y_train).is_instance_of(np.ndarray)
        assert_that(reg.y_test).is_none()
        pdt.assert_frame_equal(reg.X_train_, reg.X_train, check_dtype=True)
        npt.assert_array_equal(reg.y_train, y_train)
        pdt.assert_frame_equal(reg.X_test_, reg.X_test_, check_dtype=True)
        assert_that(reg.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(reg.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.97498, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.TreeExplainer)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(reg.shap_values_test_), 9.43848e-07, decimal=5)
        npt.assert_almost_equal(np.mean(reg.shap_values_train_), -1.98011e-08, decimal=5)

    @pytest.mark.parametrize(
        ("reg_x_y_data", "kwargs"),
        [
            ("dataframe", {"num_boost_round": 300}),
            ("dataframe", {"metrics": "mae"}),
            ("dataframe", {"sparse_matrix": True}),
            ("dataframe", {"sparse_matrix": True, "scale_std": True}),
            ("dataframe", {"scale_mean": True}),
            ("dataframe", {"scale_std": True}),
            ("dataframe", {"scale_std": True, "scale_mean": True}),
            ("dataframe", {"importance_type": "cover"}),
            ("dataframe", {"params": {"max_depth": 4, "min_child_weight": 5}}),
        ],
        indirect=["reg_x_y_data"],
        ids=_ids,
    )
    def test_xgboostregressor__passes__with_valid_inputs(
        self,
        reg_x_y_data: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
        kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Validates `XGBoostRegressor` instanation passes with valid inputs."""
        X_train, X_test, y_train, y_test = reg_x_y_data
        reg = XGBoostRegressor(**kwargs)
        reg.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict()` functions would be able to do that where their
        # `y_test=None` by default
        y_pred = reg.predict(X_test, y_test)
        params = reg.get_params()
        default_params = reg.get_default_params()
        shap_explainer = reg.get_shap_explainer()
        feature_importance = reg.get_feature_importance()
        feature_importance_fig = reg.plot_feature_importance(
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_test_fig = reg.plot_shap_waterfall(
            validation=True,
            display_plot=False,
            return_fig=True,
        )
        shap_waterfall_train_fig = reg.plot_shap_waterfall(
            validation=False,
            display_plot=False,
            return_fig=True,
        )
        # TODO(amir): figure out `return-fig` flag and test the figure object
        reg.plot_shap_summary(
            validation=True,
            display_plot=False,
        )
        reg.plot_shap_summary(
            validation=False,
            display_plot=False,
        )

        assert_that(reg).is_instance_of(XGBoostRegressor)
        assert_that(reg.num_boost_round).is_instance_of(int)
        assert_that(reg.metrics).is_instance_of(str)
        assert_that(reg.sparse_matrix).is_instance_of(bool)
        assert_that(reg.scale_mean).is_instance_of(bool)
        assert_that(reg.scale_std).is_instance_of(bool)
        assert_that(reg.importance_type).is_instance_of(str)
        assert_that(reg.params).is_instance_of(dict)
        assert_that(reg.feature_importance_).is_instance_of(pd.DataFrame)
        assert_that(reg.shap_explainer_).is_instance_of(shap.TreeExplainer)
        assert_that(reg.X_train).is_instance_of(pd.DataFrame)
        assert_that(reg.X_train_).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test_).is_instance_of(pd.DataFrame)
        assert_that(reg.y_train).is_instance_of(np.ndarray)
        assert_that(reg.y_test).is_instance_of(np.ndarray)
        npt.assert_array_equal(reg.y_train, y_train)
        npt.assert_array_equal(reg.y_test, y_test)
        assert_that(reg.dtrain_).is_instance_of(xgb.DMatrix)
        assert_that(reg.dtest_).is_instance_of(xgb.DMatrix)
        assert_that(y_pred).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(default_params).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.TreeExplainer)
        assert_that(feature_importance).is_instance_of(pd.DataFrame)
        assert_that(feature_importance_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)

    @pytest.mark.parametrize(
        (
            "reg_x_y_data",
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
                    "feature_names": [f"x_{i}" for i in range(16)],
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
        indirect=["reg_x_y_data"],
        ids=_ids,
    )
    def test_xgboostregressor_shap_plots__passes__with_valid_inputs(
        self,
        reg_x_y_data: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        waterfall_kwargs: Dict[str, Any],
        summary_kwargs: Dict[str, Any],
    ) -> None:
        """Validates `XGBoostRegressor` Shap plots passes with valid inputs."""
        X_train, X_test, y_train, y_test = reg_x_y_data
        reg = XGBoostRegressor()
        reg.fit(X_train, y_train)
        _ = reg.predict(X_test, y_test)
        shap_waterfall_fig = reg.plot_shap_waterfall(**waterfall_kwargs)
        # TODO(amir): how can we test the figure object ?
        reg.plot_shap_summary(**summary_kwargs)

        assert_that(shap_waterfall_fig).is_instance_of(Figure)
