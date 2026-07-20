"""Targeted tests for remaining branch / line coverage gaps."""

from typing import Any, Dict, Tuple
from unittest.mock import patch

import hyperopt
import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

from slickml.classification import GLMNetCVClassifier, XGBoostCVClassifier
from slickml.metrics import BinaryClassificationMetrics, RegressionMetrics
from slickml.optimization import XGBoostHyperOptimizer
from slickml.regression import GLMNetCVRegressor, XGBoostCVRegressor
from slickml.selection import XGBoostFeatureSelector
from slickml.visualization._glmnet import _ylabel
from tests.conftest import _ids


@pytest.mark.parametrize(
    ("clf_train_test_x_y", "kwargs"),
    [
        ("dataframe", {"lambda_path": [0.1, 0.05, 0.01]}),
        ("dataframe", {"lambda_path": pd.Series([0.1, 0.05, 0.01])}),
        ("dataframe", {"lambda_path": np.array([0.1, 0.05, 0.01])}),
        ("dataframe", {"max_features": 5, "lambda_path": [0.2, 0.1]}),
    ],
    indirect=["clf_train_test_x_y"],
    ids=_ids,
)
def test_glmnetcvclassifier__passes__with_lambda_path(
    clf_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    kwargs: Dict[str, Any],
) -> None:
    """Validates classifier `lambda_path` / `max_features` validation paths."""
    X_train, X_test, y_train, y_test = clf_train_test_x_y
    clf = GLMNetCVClassifier(**kwargs)
    clf.fit(X_train, y_train)
    assert_that(clf.lambda_path).is_instance_of(np.ndarray)
    _ = clf.predict_proba(X_test, y_test)


@pytest.mark.parametrize(
    ("reg_train_test_x_y", "kwargs"),
    [
        ("dataframe", {"lambda_path": [0.1, 0.05, 0.01]}),
        ("dataframe", {"lambda_path": pd.Series([0.1, 0.05, 0.01])}),
        ("dataframe", {"lambda_path": np.array([0.1, 0.05, 0.01])}),
        ("dataframe", {"max_features": 5, "lambda_path": [0.2, 0.1]}),
    ],
    indirect=["reg_train_test_x_y"],
    ids=_ids,
)
def test_glmnetcvregressor__passes__with_lambda_path(
    reg_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    kwargs: Dict[str, Any],
) -> None:
    """Validates regressor `lambda_path` / `max_features` validation paths."""
    X_train, X_test, y_train, y_test = reg_train_test_x_y
    reg = GLMNetCVRegressor(**kwargs)
    reg.fit(X_train, y_train)
    assert_that(reg.lambda_path).is_instance_of(np.ndarray)
    _ = reg.predict(X_test, y_test)


@pytest.mark.parametrize(
    ("clf_x_y"),
    [("dataframe")],
    indirect=["clf_x_y"],
    ids=_ids,
)
def test_xgboosthyperoptimizer__handles__fmin_exception(
    clf_x_y: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Validates hyperopt STATUS_FAIL path when `fmin` raises."""
    X, y = clf_x_y
    xho = XGBoostHyperOptimizer(n_iter=2)
    with patch(
        "slickml.optimization._hyperopt.hyperopt.fmin",
        side_effect=RuntimeError("boom"),
    ):
        xho.fit(X, y)
    best = xho.get_best_params()
    assert_that(best["status"]).is_equal_to(hyperopt.STATUS_FAIL)
    assert_that(best["exception"]).contains("boom")


def test_binary_classification_metrics__skips_display__when_display_df_false() -> None:
    """Validates `display_df=False` skips IPython display."""
    with patch("slickml.metrics._classification.display") as mock_display:
        m = BinaryClassificationMetrics(
            y_true=[0, 0, 1, 1],
            y_pred_proba=[0.1, 0.4, 0.35, 0.8],
            display_df=False,
        )
        mock_display.assert_not_called()
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)


def test_regression_metrics__skips_display__when_display_df_false() -> None:
    """Validates `display_df=False` skips IPython display."""
    with patch("slickml.metrics._regression.display") as mock_display:
        m = RegressionMetrics(
            y_true=[3, 0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
            display_df=False,
        )
        mock_display.assert_not_called()
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)


def test_ylabel__passes__when_scoring_is_none() -> None:
    """Validates `_ylabel` default titles when scoring is unset."""
    assert_that(
        _ylabel(
            **{"params": {"scoring": None, "n_splits": 4}, "module": "glmnet.logistic"},  # type: ignore[arg-type]
        ),
    ).contains("ACCURACY")
    assert_that(
        _ylabel(
            **{"params": {"scoring": None, "n_splits": 5}, "module": "glmnet.linear"},  # type: ignore[arg-type]
        ),
    ).contains("R^2")
    assert_that(
        _ylabel(
            **{"params": {"scoring": "roc_auc", "n_splits": 4}, "module": "glmnet.logistic"},  # type: ignore[arg-type]
        ),
    ).contains("ROC AUC")


def test_xgboostcvclassifier_verbose_log__noop__when_metrics_is_none() -> None:
    """Validates `_verbose_log` early-exit when metrics is None."""
    clf = XGBoostCVClassifier()
    clf.metrics = None
    clf._verbose_log()


def test_xgboostcvregressor_verbose_log__noop__when_metrics_is_none() -> None:
    """Validates `_verbose_log` early-exit when metrics is None."""
    reg = XGBoostCVRegressor()
    reg.metrics = None
    reg._verbose_log()


@pytest.mark.parametrize(
    ("clf_train_test_x_y"),
    [("dataframe")],
    indirect=["clf_train_test_x_y"],
    ids=_ids,
)
def test_glmnet_plots__cover_legend_false_and_display_plot(
    clf_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Validates GLMNet plot branches for legend=False and display_plot=True."""
    X_train, X_test, y_train, y_test = clf_train_test_x_y
    clf = GLMNetCVClassifier()
    clf.fit(X_train, y_train)
    _ = clf.predict_proba(X_test, y_test)

    with patch("slickml.visualization._glmnet.plt.show") as mock_show:
        clf.plot_cv_results(legend=False, display_plot=True, return_fig=False)
        clf.plot_coeff_path(legend=False, display_plot=True, return_fig=False)
        assert_that(mock_show.call_count).is_equal_to(2)

    with patch("slickml.visualization._shap.plt.show") as mock_show:
        clf.plot_shap_summary(display_plot=True, class_inds=[0])
        clf.plot_shap_waterfall(display_plot=True, return_fig=False)
        assert_that(mock_show.call_count).is_greater_than_or_equal_to(1)


@pytest.mark.parametrize(
    ("reg_train_test_x_y"),
    [("dataframe")],
    indirect=["reg_train_test_x_y"],
    ids=_ids,
)
def test_glmnet_reg_plots__cover_display_plot(
    reg_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Validates GLMNet regressor plot display_plot=True path."""
    X_train, X_test, y_train, y_test = reg_train_test_x_y
    reg = GLMNetCVRegressor()
    reg.fit(X_train, y_train)
    _ = reg.predict(X_test, y_test)
    with patch("slickml.visualization._glmnet.plt.show") as mock_show:
        reg.plot_cv_results(legend=False, display_plot=True, return_fig=False)
        assert_that(mock_show.call_count).is_equal_to(1)


def test_metrics_plots__cover_display_plot() -> None:
    """Validates metrics plotters call plt.show when display_plot=True."""
    with patch("slickml.visualization._metrics.plt.show") as mock_show:
        BinaryClassificationMetrics(
            y_true=[0, 0, 1, 1],
            y_pred_proba=[0.1, 0.4, 0.35, 0.8],
            display_df=False,
        ).plot(display_plot=True, return_fig=False)
        RegressionMetrics(
            y_true=[3, 0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
            display_df=False,
        ).plot(display_plot=True, return_fig=False)
        assert_that(mock_show.call_count).is_equal_to(2)


@pytest.mark.parametrize(
    ("clf_train_test_x_y"),
    [("dataframe")],
    indirect=["clf_train_test_x_y"],
    ids=_ids,
)
def test_xgboostcv_plots__cover_display_plot(
    clf_train_test_x_y: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    """Validates XGBoost CV plot display_plot=True path."""
    X_train, X_test, y_train, y_test = clf_train_test_x_y
    clf = XGBoostCVClassifier(n_splits=2, num_boost_round=10, early_stopping_rounds=5)
    clf.fit(X_train, y_train)
    _ = clf.predict_proba(X_test, y_test)
    with patch("slickml.visualization._xgboost.plt.show") as mock_show:
        clf.plot_cv_results(display_plot=True, return_fig=False)
        clf.plot_feature_importance(display_plot=True, return_fig=False)
        assert_that(mock_show.call_count).is_equal_to(2)


@pytest.mark.parametrize(
    ("clf_x_y"),
    [("dataframe")],
    indirect=["clf_x_y"],
    ids=_ids,
)
def test_xgboostfeatureselector__covers_false_branch_paths(
    clf_x_y: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Validates selection false-branch paths for eval_metric / metrics / _freq."""
    X, y = clf_x_y
    xfs = XGBoostFeatureSelector(
        n_iter=1,
        n_splits=2,
        early_stopping_rounds=1000,
    )
    # Non-str eval_metric skips external CV logging / plotting_cv population
    assert_that(xfs.params).is_not_none()
    xfs.params["eval_metric"] = ["logloss"]  # type: ignore[index,assignment]
    xfs.fit(X, y)
    assert_that(xfs._plotting_cv()).is_equal_to({})

    xfs.metrics = None
    assert_that(xfs._plotting_cv()).is_equal_to({})

    xfs.n_splits = None
    freq = xfs._freq()
    assert_that(freq).is_instance_of(pd.DataFrame)
    assert_that("Frequency (%)" in freq.columns).is_false()

    # Internal metrics print false-branch during fit
    xfs2 = XGBoostFeatureSelector(
        n_iter=1,
        n_splits=2,
        early_stopping_rounds=1000,
    )
    xfs2.metrics = None
    xfs2.fit(X, y)


@pytest.mark.parametrize(
    ("clf_x_y"),
    [("dataframe")],
    indirect=["clf_x_y"],
    ids=_ids,
)
def test_xgboostfeatureselector_plots__cover_display_plot(
    clf_x_y: Tuple[pd.DataFrame, np.ndarray],
) -> None:
    """Validates feature-selector plot display_plot=True path."""
    X, y = clf_x_y
    xfs = XGBoostFeatureSelector(
        n_iter=1,
        n_splits=2,
        early_stopping_rounds=1000,
    )
    xfs.fit(X, y)
    with patch("slickml.visualization._selection.plt.show") as mock_show:
        xfs.plot_frequency(display_plot=True, return_fig=False)
        xfs.plot_cv_results(display_plot=True, return_fig=False)
        assert_that(mock_show.call_count).is_equal_to(2)


def test_binary_classification_metrics__covers_prec_rec_annotate_else() -> None:
    """Validates precision-recall annotate else-branch when threshold > 0.5."""
    with patch("slickml.visualization._metrics.plt.show"):
        m = BinaryClassificationMetrics(
            y_true=[0, 0, 1, 1],
            y_pred_proba=[0.1, 0.4, 0.35, 0.8],
            display_df=False,
        )
        # Force threshold into the > 0.5 annotate branch
        plotting = m._plotting_dict()
        plotting["prec_rec_threshold"] = 0.75
        from slickml.visualization._metrics import plot_binary_classification_metrics

        plot_binary_classification_metrics(**plotting, display_plot=True, return_fig=False)
