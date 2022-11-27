from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import shap
from assertpy import assert_that
from matplotlib.figure import Figure

from slickml.classification import GLMNetCVClassifier
from tests.conftest import _ids


# TODO(amir): add lolipop plot for coeff + unit-test
# TODO(amir): add test for lambda-path param
class TestGLMNetCVClassifier:
    """Validates `GLMNetCVClassifier` instantiation."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {"alpha": "0.1"},
            {"n_lambda": "100"},
            {"n_splits": "4"},
            {"metric": "auc_pr"},
            {"scale": 1},
            {"sparse_matrix": 1},
            {"sparse_matrix": True, "scale": True},
            {"fit_intercept": 1},
            {"random_state": "42"},
        ],
        ids=_ids,
    )
    def test_glmnetcvclassifier_instantiation__fails__with_invalid_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates `GLMNetCVClassifier` cannot be instantiated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            GLMNetCVClassifier(**kwargs)

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
    def test_glmnetcvclassifier__passes__with_defaults_and_no_test_targets(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `GLMNetCVClassifier` instanation passes with default inputs."""
        X_train, X_test, y_train, _ = clf_train_test_x_y
        clf = GLMNetCVClassifier()
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        params = clf.get_params()
        cv_results = clf.get_cv_results()
        results = clf.get_results()
        shap_explainer = clf.get_shap_explainer()
        cv_result_fig = clf.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        coeff_path_fig = clf.plot_coeff_path(
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

        assert_that(clf).is_instance_of(GLMNetCVClassifier)
        assert_that(clf.alpha).is_instance_of(float)
        assert_that(clf.alpha).is_equal_to(0.5)
        assert_that(clf.n_lambda).is_instance_of(int)
        assert_that(clf.n_lambda).is_equal_to(100)
        assert_that(clf.n_splits).is_instance_of(int)
        assert_that(clf.n_splits).is_equal_to(3)
        assert_that(clf.metric).is_instance_of(str)
        assert_that(clf.metric).is_equal_to("roc_auc")
        assert_that(clf.scale).is_instance_of(bool)
        assert_that(clf.scale).is_true()
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_false()
        assert_that(clf.fit_intercept).is_instance_of(bool)
        assert_that(clf.fit_intercept).is_true()
        assert_that(clf.cut_point).is_instance_of(float)
        assert_that(clf.cut_point).is_equal_to(1.0)
        assert_that(clf.min_lambda_ratio).is_instance_of(float)
        assert_that(clf.min_lambda_ratio).is_equal_to(1e-4)
        assert_that(clf.tolerance).is_instance_of(float)
        assert_that(clf.tolerance).is_equal_to(1e-7)
        assert_that(clf.max_iter).is_instance_of(int)
        assert_that(clf.max_iter).is_equal_to(100000)
        assert_that(clf.random_state).is_instance_of(int)
        assert_that(clf.random_state).is_equal_to(1367)
        assert_that(clf.lambda_path).is_none()
        assert_that(clf.max_features).is_none()
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_none()
        assert_that(clf.results_).is_instance_of(dict)
        assert_that(clf.shap_explainer_).is_instance_of(shap.LinearExplainer)
        assert_that(clf.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(clf.shap_values_train_).is_instance_of(np.ndarray)
        assert_that(clf.shap_values_test_).is_instance_of(np.ndarray)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred_proba), 0.81188, decimal=5)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.96153, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(shap_explainer).is_instance_of(shap.LinearExplainer)
        assert_that(results).is_instance_of(dict)
        assert_that(cv_result_fig).is_instance_of(Figure)
        assert_that(coeff_path_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(clf.shap_values_test_), 0.00529, decimal=5)
        npt.assert_almost_equal(np.mean(clf.shap_values_train_), 0.01112, decimal=5)

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
    def test_glmnetcvclassifier__passes__with_defaults(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
    ) -> None:
        """Validates `GLMNetCVClassifier` instanation passes with default inputs."""
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = GLMNetCVClassifier()
        clf.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict_proba()` and `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred_proba = clf.predict_proba(X_test, y_test)
        y_pred = clf.predict(X_test, y_test)
        params = clf.get_params()
        cv_results = clf.get_cv_results()
        results = clf.get_results()
        shap_explainer = clf.get_shap_explainer()
        cv_result_fig = clf.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        coeff_path_fig = clf.plot_coeff_path(
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

        assert_that(clf).is_instance_of(GLMNetCVClassifier)
        assert_that(clf.alpha).is_instance_of(float)
        assert_that(clf.alpha).is_equal_to(0.5)
        assert_that(clf.n_lambda).is_instance_of(int)
        assert_that(clf.n_lambda).is_equal_to(100)
        assert_that(clf.n_splits).is_instance_of(int)
        assert_that(clf.n_splits).is_equal_to(3)
        assert_that(clf.metric).is_instance_of(str)
        assert_that(clf.metric).is_equal_to("roc_auc")
        assert_that(clf.scale).is_instance_of(bool)
        assert_that(clf.scale).is_true()
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_false()
        assert_that(clf.fit_intercept).is_instance_of(bool)
        assert_that(clf.fit_intercept).is_true()
        assert_that(clf.cut_point).is_instance_of(float)
        assert_that(clf.cut_point).is_equal_to(1.0)
        assert_that(clf.min_lambda_ratio).is_instance_of(float)
        assert_that(clf.min_lambda_ratio).is_equal_to(1e-4)
        assert_that(clf.tolerance).is_instance_of(float)
        assert_that(clf.tolerance).is_equal_to(1e-7)
        assert_that(clf.max_iter).is_instance_of(int)
        assert_that(clf.max_iter).is_equal_to(100000)
        assert_that(clf.random_state).is_instance_of(int)
        assert_that(clf.random_state).is_equal_to(1367)
        assert_that(clf.lambda_path).is_none()
        assert_that(clf.max_features).is_none()
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_instance_of(np.ndarray)
        assert_that(clf.results_).is_instance_of(dict)
        assert_that(clf.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(clf.shap_explainer_).is_instance_of(shap.LinearExplainer)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
        assert_that(clf.shap_values_train_).is_instance_of(np.ndarray)
        assert_that(clf.shap_values_test_).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred_proba), 0.81188, decimal=5)
        assert_that(y_pred).is_instance_of(np.ndarray)
        npt.assert_almost_equal(np.mean(y_pred), 0.96153, decimal=5)
        assert_that(params).is_instance_of(dict)
        assert_that(cv_results).is_instance_of(pd.DataFrame)
        assert_that(results).is_instance_of(dict)
        assert_that(shap_explainer).is_instance_of(shap.LinearExplainer)
        assert_that(cv_result_fig).is_instance_of(Figure)
        assert_that(coeff_path_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_test_fig).is_instance_of(Figure)
        assert_that(shap_waterfall_train_fig).is_instance_of(Figure)
        npt.assert_almost_equal(np.mean(clf.shap_values_test_), 0.00529, decimal=5)
        npt.assert_almost_equal(np.mean(clf.shap_values_train_), 0.01112, decimal=5)

    # TODO(amir): add a test for `lambda_path` parameter
    @pytest.mark.parametrize(
        ("clf_train_test_x_y", "kwargs"),
        [
            ("dataframe", {"alpha": 0.9}),
            ("dataframe", {"n_lambda": 200}),
            ("dataframe", {"n_splits": 10}),
            ("dataframe", {"metric": "average_precision"}),
            ("dataframe", {"scale": False, "sparse_matrix": True}),
            ("dataframe", {"fit_intercept": False}),
            ("dataframe", {"cut_point": 2.0}),
            ("dataframe", {"random_state": 42}),
            ("dataframe", {"max_features": 10}),
        ],
        indirect=["clf_train_test_x_y"],
        ids=_ids,
    )
    def test_glmnetcvclassifier__passes__with_valid_inputs(
        self,
        clf_train_test_x_y: Tuple[
            Union[pd.DataFrame, np.ndarray],
            Union[pd.DataFrame, np.ndarray],
            Union[np.ndarray, List],
            Union[np.ndarray, List],
        ],
        kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Validates `GLMNetCVClassifier` instanation passes with valid inputs."""
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = GLMNetCVClassifier(**kwargs)
        clf.fit(X_train, y_train)
        # Note: we pass `y_test` for the sake of testing while in inference we might night have
        # access to ground truth and both `predict_proba()` and `predict()` functions would be able
        # to do that where their `y_test=None` by default
        y_pred_proba = clf.predict_proba(X_test, y_test)
        y_pred = clf.predict(X_test, y_test)
        params = clf.get_params()
        cv_results = clf.get_cv_results()
        results = clf.get_results()
        coeff_df = clf.get_coeffs(output="dataframe")
        coeff_dict = clf.get_coeffs(output="dict")
        intercept = clf.get_intercept()
        shap_explainer = clf.get_shap_explainer()
        cv_result_fig = clf.plot_cv_results(
            display_plot=False,
            return_fig=True,
        )
        coeff_path_fig = clf.plot_coeff_path(
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

        assert_that(clf).is_instance_of(GLMNetCVClassifier)
        assert_that(clf.alpha).is_instance_of(float)
        assert_that(clf.n_lambda).is_instance_of(int)
        assert_that(clf.n_splits).is_instance_of(int)
        assert_that(clf.metric).is_instance_of(str)
        assert_that(clf.scale).is_instance_of(bool)
        assert_that(clf.sparse_matrix).is_instance_of(bool)
        assert_that(clf.fit_intercept).is_instance_of(bool)
        assert_that(clf.cut_point).is_instance_of(float)
        assert_that(clf.min_lambda_ratio).is_instance_of(float)
        assert_that(clf.tolerance).is_instance_of(float)
        assert_that(clf.max_iter).is_instance_of(int)
        assert_that(clf.random_state).is_instance_of(int)
        assert_that(clf.lambda_path).is_none()
        assert_that(clf.X_train).is_instance_of(pd.DataFrame)
        assert_that(clf.X_test).is_instance_of(pd.DataFrame)
        assert_that(clf.y_train).is_instance_of(np.ndarray)
        assert_that(clf.y_test).is_instance_of(np.ndarray)
        assert_that(clf.results_).is_instance_of(dict)
        assert_that(clf.cv_results_).is_instance_of(pd.DataFrame)
        assert_that(clf.intercept_).is_instance_of(float)
        assert_that(clf.coeff_).is_instance_of(pd.DataFrame)
        assert_that(clf.shap_explainer_).is_instance_of(shap.LinearExplainer)
        assert_that(y_pred_proba).is_instance_of(np.ndarray)
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
    def test_glmnetcvclassifier_shap_plots__passes__with_valid_inputs(
        self,
        clf_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
        waterfall_kwargs: Dict[str, Any],
        summary_kwargs: Dict[str, Any],
    ) -> None:
        """Validates `GLMNetCVClassifier` Shap plots passes with valid inputs."""
        X_train, X_test, y_train, y_test = clf_train_test_x_y
        clf = GLMNetCVClassifier()
        clf.fit(X_train, y_train)
        _ = clf.predict_proba(X_test, y_test)
        shap_waterfall_fig = clf.plot_shap_waterfall(**waterfall_kwargs)
        # TODO(amir): how can we test the figure object ?
        clf.plot_shap_summary(**summary_kwargs)

        assert_that(shap_waterfall_fig).is_instance_of(Figure)
