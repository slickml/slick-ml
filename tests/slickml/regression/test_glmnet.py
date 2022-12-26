from pathlib import Path  # noqa
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import shap
from assertpy import assert_that
from matplotlib.figure import Figure

from slickml.regression import GLMNetCVRegressor
from tests.conftest import _ids, _validate_figure_type_and_size


# TODO(amir): add lolipop plot for coeff + unit-test
# TODO(amir): add test for lambda-path param
# TODO(amir): test out regression with multiple outputs (y1, y2)
class TestGLMNetCVRegressor:
    """Validates `GLMNetCVRegressor` instantiation."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"alpha": "0.1"},
            {"n_lambda": "100"},
            {"n_splits": "4"},
            {"metric": "auc"},
            {"scale": 1},
            {"sparse_matrix": 1},
            {"sparse_matrix": True, "scale": True},
            {"fit_intercept": 1},
            {"random_state": "42"},
        ],
        ids=_ids,
    )
    def test_glmnetcvregressor_instantiation__fails__with_invalid_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates `GLMNetCVRegressor` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            GLMNetCVRegressor(**kwargs)

    @pytest.mark.parametrize(
        ("reg_train_test_x_y"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["reg_train_test_x_y"],
        ids=_ids,
    )
    def test_glmnetcvregressor__passes__with_defaults_and_no_test_targets(
        self,
        reg_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List[float]],
            Union[np.ndarray, List[float]],
        ],
    ) -> None:
        """Validates `GLMNetCVRegressor` instanation passes with default inputs."""
        X_train, X_test, y_train, _ = reg_train_test_x_y
        reg = GLMNetCVRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        params = reg.get_params()
        cv_results = reg.get_cv_results()
        results = reg.get_results()
        shap_explainer = reg.get_shap_explainer()
        cv_result_fig = reg.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        coeff_path_fig = reg.plot_coeff_path(
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

        assert_that(reg).is_instance_of(GLMNetCVRegressor)
        assert_that(reg.alpha).is_instance_of(float)
        assert_that(reg.alpha).is_equal_to(0.5)
        assert_that(reg.n_lambda).is_instance_of(int)
        assert_that(reg.n_lambda).is_equal_to(100)
        assert_that(reg.n_splits).is_instance_of(int)
        assert_that(reg.n_splits).is_equal_to(3)
        assert_that(reg.metric).is_instance_of(str)
        assert_that(reg.metric).is_equal_to("r2")
        assert_that(reg.scale).is_instance_of(bool)
        assert_that(reg.scale).is_true()
        assert_that(reg.sparse_matrix).is_instance_of(bool)
        assert_that(reg.sparse_matrix).is_false()
        assert_that(reg.fit_intercept).is_instance_of(bool)
        assert_that(reg.fit_intercept).is_true()
        assert_that(reg.cut_point).is_instance_of(float)
        assert_that(reg.cut_point).is_equal_to(1.0)
        assert_that(reg.min_lambda_ratio).is_instance_of(float)
        assert_that(reg.min_lambda_ratio).is_equal_to(1e-4)
        assert_that(reg.tolerance).is_instance_of(float)
        assert_that(reg.tolerance).is_equal_to(1e-7)
        assert_that(reg.max_iter).is_instance_of(int)
        assert_that(reg.max_iter).is_equal_to(100000)
        assert_that(reg.random_state).is_instance_of(int)
        assert_that(reg.random_state).is_equal_to(1367)
        assert_that(reg.lambda_path).is_none()
        assert_that(reg.max_features).is_none()
        assert_that(reg.X_train).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test).is_instance_of(pd.DataFrame)
        assert_that(reg.y_train).is_instance_of(np.ndarray)
        assert_that(reg.y_test).is_none()
        assert_that(reg.results_).is_instance_of(dict)
        assert_that(reg.shap_explainer_).is_instance_of(shap.LinearExplainer)
        assert_that(reg.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(reg.shap_values_train_).is_instance_of(np.ndarray)
        assert_that(reg.shap_values_test_).is_instance_of(np.ndarray)
        assert_that(y_pred).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(shap_explainer).is_instance_of(shap.LinearExplainer)
        assert_that(results).is_instance_of(dict)
        assert_that(cv_result_fig).is_instance_of(Figure)
        assert_that(coeff_path_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(y_pred), 0.97489, decimal=5)
        npt.assert_almost_equal(np.mean(reg.shap_values_test_), 2.32e-6, decimal=5)
        npt.assert_almost_equal(np.mean(reg.shap_values_train_), 8.13e-06, decimal=5)

    @pytest.mark.parametrize(
        ("reg_train_test_x_y"),
        [
            ("array"),
            ("dataframe"),
            ("list"),
        ],
        indirect=["reg_train_test_x_y"],
        ids=_ids,
    )
    def test_glmnetcvregressor__passes__with_defaults(
        self,
        reg_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List[float]],
            Union[np.ndarray, List[float]],
        ],
    ) -> None:
        """Validates `GLMNetCVRegressor` instanation passes with default inputs."""
        X_train, X_test, y_train, y_test = reg_train_test_x_y
        reg = GLMNetCVRegressor()
        reg.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth `predict()` function would be able
        # to do that where their `y_test=None` by default
        y_pred = reg.predict(X_test, y_test)
        params = reg.get_params()
        cv_results = reg.get_cv_results()
        results = reg.get_results()
        shap_explainer = reg.get_shap_explainer()
        cv_result_fig = reg.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        coeff_path_fig = reg.plot_coeff_path(
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

        assert_that(reg).is_instance_of(GLMNetCVRegressor)
        assert_that(reg.alpha).is_instance_of(float)
        assert_that(reg.alpha).is_equal_to(0.5)
        assert_that(reg.n_lambda).is_instance_of(int)
        assert_that(reg.n_lambda).is_equal_to(100)
        assert_that(reg.n_splits).is_instance_of(int)
        assert_that(reg.n_splits).is_equal_to(3)
        assert_that(reg.metric).is_instance_of(str)
        assert_that(reg.metric).is_equal_to("r2")
        assert_that(reg.scale).is_instance_of(bool)
        assert_that(reg.scale).is_true()
        assert_that(reg.sparse_matrix).is_instance_of(bool)
        assert_that(reg.sparse_matrix).is_false()
        assert_that(reg.fit_intercept).is_instance_of(bool)
        assert_that(reg.fit_intercept).is_true()
        assert_that(reg.cut_point).is_instance_of(float)
        assert_that(reg.cut_point).is_equal_to(1.0)
        assert_that(reg.min_lambda_ratio).is_instance_of(float)
        assert_that(reg.min_lambda_ratio).is_equal_to(1e-4)
        assert_that(reg.tolerance).is_instance_of(float)
        assert_that(reg.tolerance).is_equal_to(1e-7)
        assert_that(reg.max_iter).is_instance_of(int)
        assert_that(reg.max_iter).is_equal_to(100000)
        assert_that(reg.random_state).is_instance_of(int)
        assert_that(reg.random_state).is_equal_to(1367)
        assert_that(reg.lambda_path).is_none()
        assert_that(reg.max_features).is_none()
        assert_that(reg.X_train).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test).is_instance_of(pd.DataFrame)
        assert_that(reg.y_train).is_instance_of(np.ndarray)
        assert_that(reg.y_test).is_instance_of(np.ndarray)
        assert_that(reg.results_).is_instance_of(dict)
        assert_that(reg.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(reg.shap_explainer_).is_instance_of(shap.LinearExplainer)
        assert_that(reg.shap_values_train_).is_instance_of(np.ndarray)
        assert_that(reg.shap_values_test_).is_instance_of(np.ndarray)
        assert_that(y_pred).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(results).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.LinearExplainer)
        assert_that(cv_result_fig).is_instance_of(Figure)
        assert_that(coeff_path_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(y_pred), 0.97489, decimal=5)
        npt.assert_almost_equal(np.mean(reg.shap_values_test_), 2.32e-6, decimal=5)
        npt.assert_almost_equal(np.mean(reg.shap_values_train_), 8.13e-06, decimal=5)

    # TODO(amir): add a test for `lambda_path` parameter
    @pytest.mark.parametrize(
        ("reg_train_test_x_y", "kwargs"),
        [
            ("dataframe", {"alpha": 0.1}),
            ("dataframe", {"n_lambda": 50}),
            ("dataframe", {"n_splits": 5}),
            ("dataframe", {"metric": "mean_squared_error"}),
            ("dataframe", {"scale": False, "sparse_matrix": True}),
            ("dataframe", {"fit_intercept": False}),
            ("dataframe", {"cut_point": 2.0}),
            ("dataframe", {"random_state": 42}),
            ("dataframe", {"max_features": 10}),
        ],
        indirect=["reg_train_test_x_y"],
        ids=_ids,
    )
    def test_glmnetcvregressor__passes__with_valid_inputs(
        self,
        reg_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List[float]],
            Union[np.ndarray, List[float]],
        ],
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates `GLMNetCVRegressor` instanation passes with valid inputs."""
        X_train, X_test, y_train, y_test = reg_train_test_x_y
        reg = GLMNetCVRegressor(**kwargs)
        reg.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and `predict()` function would be able
        # to do that where their `y_test=None` by default
        y_pred = reg.predict(X_test, y_test)
        params = reg.get_params()
        cv_results = reg.get_cv_results()
        results = reg.get_results()
        coeff_df = reg.get_coeffs(output="dataframe")
        coeff_dict = reg.get_coeffs(output="dict")
        intercept = reg.get_intercept()
        shap_explainer = reg.get_shap_explainer()
        cv_result_fig = reg.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        coeff_path_fig = reg.plot_coeff_path(
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

        assert_that(reg).is_instance_of(GLMNetCVRegressor)
        assert_that(reg.alpha).is_instance_of(float)
        assert_that(reg.n_lambda).is_instance_of(int)
        assert_that(reg.n_splits).is_instance_of(int)
        assert_that(reg.metric).is_instance_of(str)
        assert_that(reg.scale).is_instance_of(bool)
        assert_that(reg.sparse_matrix).is_instance_of(bool)
        assert_that(reg.fit_intercept).is_instance_of(bool)
        assert_that(reg.cut_point).is_instance_of(float)
        assert_that(reg.min_lambda_ratio).is_instance_of(float)
        assert_that(reg.tolerance).is_instance_of(float)
        assert_that(reg.max_iter).is_instance_of(int)
        assert_that(reg.random_state).is_instance_of(int)
        assert_that(reg.lambda_path).is_none()
        assert_that(reg.X_train).is_instance_of(pd.DataFrame)
        assert_that(reg.X_test).is_instance_of(pd.DataFrame)
        assert_that(reg.y_train).is_instance_of(np.ndarray)
        assert_that(reg.y_test).is_instance_of(np.ndarray)
        assert_that(reg.results_).is_instance_of(dict)
        assert_that(reg.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(reg.intercept_).is_instance_of(float)
        assert_that(reg.coeff_).is_instance_of(pd.DataFrame)
        assert_that(reg.shap_explainer_).is_instance_of(shap.LinearExplainer)
        assert_that(y_pred).is_instance_of(np.ndarray)
        assert_that(params).is_instance_of(dict)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(results).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.LinearExplainer)
        assert_that(coeff_df).is_instance_of(pd.DataFrame)
        assert_that(coeff_dict).is_instance_of(dict)
        assert_that(intercept).is_instance_of(float)
        assert_that(cv_result_fig).is_instance_of(Figure)
        assert_that(coeff_path_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)

    @pytest.mark.parametrize(
        (
            "reg_train_test_x_y",
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
        indirect=["reg_train_test_x_y"],
        ids=_ids,
    )
    def test_glmnetcvregressor_shap_plots__passes__with_valid_inputs(
        self,
        reg_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        waterfall_kwargs: Dict[str, Any],
        summary_kwargs: Dict[str, Any],
    ) -> None:
        """Validates `GLMNetCVRegressor` Shap plots passes with valid inputs."""
        X_train, X_test, y_train, y_test = reg_train_test_x_y
        reg = GLMNetCVRegressor()
        reg.fit(X_train, y_train)
        _ = reg.predict(X_test, y_test)
        shap_waterfall_fig = reg.plot_shap_waterfall(**waterfall_kwargs)
        # TODO(amir): how can we test the figure object ?
        reg.plot_shap_summary(**summary_kwargs)

        assert_that(shap_waterfall_fig).is_instance_of(Figure)

    @pytest.mark.parametrize(
        ("reg_train_test_x_y"),
        [
            ("dataframe"),
        ],
        indirect=["reg_train_test_x_y"],
        ids=_ids,
    )
    def test_glmnetcvregressor_plots__passes__with_valid_save_paths(
        self,
        reg_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        figure_path: Path,
    ) -> None:
        """Validates `GLMNetCVRegressor` saving plots passes with valid paths."""
        X_train, X_test, y_train, y_test = reg_train_test_x_y
        reg = GLMNetCVRegressor()
        reg.fit(X_train, y_train)
        _ = reg.predict(X_test, y_test)
        cv_results_fig_path = figure_path / "cv_results_fig.png"  # type: ignore
        coeff_path_fig_path = figure_path / "coeff_path_fig.png"  # type: ignore
        shap_waterfall_fig_path = figure_path / "shap_waterfall_fig.png"  # type: ignore
        shap_summary_fig_path = figure_path / "shap_summary_fig.png"  # type: ignore

        reg.plot_cv_results(
            save_path=str(cv_results_fig_path),
            display_plot=False,
            return_fig=False,
        )
        reg.plot_coeff_path(
            save_path=str(coeff_path_fig_path),
            display_plot=False,
            return_fig=False,
        )
        reg.plot_shap_waterfall(
            save_path=str(shap_waterfall_fig_path),
            display_plot=False,
            return_fig=False,
        )
        reg.plot_shap_summary(
            save_path=str(shap_summary_fig_path),
            display_plot=False,
        )

        assert_that(cv_results_fig_path.parts[-1]).is_equal_to("cv_results_fig.png")
        _validate_figure_type_and_size(
            path=cv_results_fig_path,
            expected_size=(1371, 936),
        )
        assert_that(coeff_path_fig_path.parts[-1]).is_equal_to("coeff_path_fig.png")
        _validate_figure_type_and_size(
            path=coeff_path_fig_path,
            expected_size=(1604, 936),
        )
        assert_that(shap_waterfall_fig_path.parts[-1]).is_equal_to("shap_waterfall_fig.png")
        _validate_figure_type_and_size(
            path=shap_waterfall_fig_path,
            expected_size=(1394, 974),
        )
        assert_that(shap_summary_fig_path.parts[-1]).is_equal_to("shap_summary_fig.png")
        _validate_figure_type_and_size(
            path=shap_summary_fig_path,
            expected_size=(1487, 1560),
        )
