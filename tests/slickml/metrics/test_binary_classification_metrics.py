from pathlib import Path  # noqa
from typing import Any, Dict

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from assertpy import assert_that
from matplotlib.figure import Figure

from slickml.metrics import BinaryClassificationMetrics
from tests.conftest import _ids, _validate_figure_type_and_size


# TODO(amir): the case for `average_method = None` which is in `__post_init__`
# has not been tested at all; I dont even recall why we had to do this.
class TestBinaryClassificationMetrics:
    """Validates `BinaryClassificationMetrics` instantiation."""

    # TODO(amir): update this test once `y_true` and `y_pred` types are fixed with `check_var()`
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [0, 1],
                "y_pred_proba": [0.5, 0.9],
                "threshold": "0.5",
            },
            {
                "y_true": [0, 1],
                "y_pred_proba": [0.5, 0.9],
                "threshold": -0.1,
            },
            {
                "y_true": [0, 1],
                "y_pred_proba": [0.5, 0.9],
                "threshold": 1.1,
            },
            {
                "y_true": [0, 1],
                "y_pred_proba": [0, 1],
                "average_method": None,
            },
            {
                "y_true": [0, 1],
                "y_pred_proba": [0, 1],
                "precision_digits": "3",
            },
            {
                "y_true": [0, 1],
                "y_pred_proba": [0, 1],
                "display_df": 1,
            },
        ],
        ids=_ids,
    )
    def test_binary_classification_metrics__fails__with_invalid_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates that binary classification metrics cannot be calculated with invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            BinaryClassificationMetrics(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [0, 0, 1, 1],
                "y_pred_proba": [0.1, 0.4, 0.35, 0.8],
                "average_method": "binary",
            },
            {
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred_proba": np.array([0.1, 0.4, 0.35, 0.8]),
                "average_method": "binary",
            },
            {
                "y_true": pd.Series([0, 0, 1, 1]),
                "y_pred_proba": pd.Series([0.1, 0.4, 0.35, 0.8]),
                "average_method": "binary",
            },
        ],
        ids=_ids,
    )
    def test_binary_classification__passes__with_binary_average_method(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates binary classification metrics calculation with "binary" average method."""

        m = BinaryClassificationMetrics(**kwargs)
        f = m.plot(
            display_plot=False,
            return_fig=True,
        )

        assert_that(m.y_true).is_instance_of(np.ndarray)
        assert_that(m.y_pred_proba).is_instance_of(np.ndarray)
        assert_that(m.threshold).is_instance_of(float)
        assert_that(m.threshold).is_equal_to(0.5)
        assert_that(m.average_method).is_instance_of(str)
        assert_that(m.average_method).is_equal_to("binary")
        assert_that(m.precision_digits).is_instance_of(int)
        assert_that(m.precision_digits).is_equal_to(3)
        assert_that(m.display_df).is_instance_of(bool)
        assert_that(m.display_df).is_true()
        assert_that(m.y_pred_).is_instance_of(np.ndarray)
        assert_that(m.accuracy_).is_instance_of(float)
        assert_that(m.balanced_accuracy_).is_instance_of(float)
        assert_that(m.fpr_list_).is_instance_of(np.ndarray)
        assert_that(m.tpr_list_).is_instance_of(np.ndarray)
        assert_that(m.roc_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_roc_).is_instance_of(float)
        assert_that(m.precision_list_).is_instance_of(np.ndarray)
        assert_that(m.recall_list_).is_instance_of(np.ndarray)
        assert_that(m.pr_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_pr_).is_instance_of(float)
        assert_that(m.precision_).is_instance_of(float)
        assert_that(m.recall_).is_instance_of(float)
        assert_that(m.f1_).is_instance_of(float)
        assert_that(m.f2_).is_instance_of(float)
        assert_that(m.f05_).is_instance_of(float)
        assert_that(m.average_precision_).is_instance_of(float)
        assert_that(m.tn_).is_instance_of(np.int64)
        assert_that(m.fp_).is_instance_of(np.int64)
        assert_that(m.fn_).is_instance_of(np.int64)
        assert_that(m.tp_).is_instance_of(np.int64)
        assert_that(m.threat_score_).is_instance_of(float)
        assert_that(m.youden_index_).is_instance_of(np.int64)
        assert_that(m.youden_threshold_).is_instance_of(float)
        assert_that(m.sens_spec_threshold_).is_instance_of(float)
        assert_that(m.thresholds_dict_).is_instance_of(dict)
        assert_that(m.metrics_dict_).is_instance_of(dict)
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)
        assert_that(m.average_methods_).is_instance_of(list)
        assert_that(m.plotting_dict_).is_instance_of(dict)
        assert_that(m.get_metrics(dtype="dataframe")).is_instance_of(pd.DataFrame)
        assert_that(m.get_metrics(dtype="dict")).is_instance_of(dict)
        npt.assert_almost_equal(np.mean(m.y_pred_), 0.25, decimal=5)
        npt.assert_almost_equal(m.accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(m.balanced_accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.fpr_list_), 0.4, decimal=5)
        npt.assert_almost_equal(np.mean(m.tpr_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.roc_thresholds_), np.inf, decimal=5)
        npt.assert_almost_equal(m.auc_roc_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.precision_list_), 0.73333, decimal=5)
        npt.assert_almost_equal(np.mean(m.recall_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.pr_thresholds_), 0.4125, decimal=5)
        npt.assert_almost_equal(m.auc_pr_, 0.791666, decimal=5)
        npt.assert_almost_equal(m.precision_, 1.0, decimal=5)
        npt.assert_almost_equal(m.recall_, 0.5, decimal=5)
        npt.assert_almost_equal(m.f1_, 0.66666, decimal=5)
        npt.assert_almost_equal(m.f2_, 0.55555, decimal=5)
        npt.assert_almost_equal(m.f05_, 0.83333, decimal=5)
        npt.assert_almost_equal(m.average_precision_, 0.83333, decimal=5)
        npt.assert_almost_equal(m.threat_score_, 0.5, decimal=5)
        npt.assert_almost_equal(m.youden_threshold_, 0.8, decimal=5)
        npt.assert_almost_equal(m.sens_spec_threshold_, 0.4, decimal=5)
        npt.assert_almost_equal(m.prec_rec_threshold_, 0.4, decimal=5)
        assert_that(m.tn_).is_equal_to(2)
        assert_that(m.fp_).is_equal_to(0)
        assert_that(m.fn_).is_equal_to(1)
        assert_that(m.tp_).is_equal_to(1)
        assert_that(m.youden_index_).is_equal_to(1)
        assert_that(m.average_methods_).is_subset_of(
            [
                "binary",
                "weighted",
                "macro",
                "micro",
            ],
        )

        # TODO(amir): figure out a better way to test out figures
        assert_that(f).is_instance_of(Figure)
        assert_that(f.__dict__["_original_dpi"]).is_equal_to(100)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [0, 0, 1, 1],
                "y_pred_proba": [0.1, 0.4, 0.35, 0.8],
                "average_method": "weighted",
            },
            {
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred_proba": np.array([0.1, 0.4, 0.35, 0.8]),
                "average_method": "weighted",
            },
            {
                "y_true": pd.Series([0, 0, 1, 1]),
                "y_pred_proba": pd.Series([0.1, 0.4, 0.35, 0.8]),
                "average_method": "weighted",
            },
        ],
        ids=_ids,
    )
    def test_binary_classification__passes__with_weighted_average_method(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates binary classification metrics calculation with "weighted" average method."""
        m = BinaryClassificationMetrics(**kwargs)
        f = m.plot(
            display_plot=False,
            return_fig=True,
        )

        assert_that(m.y_true).is_instance_of(np.ndarray)
        assert_that(m.y_pred_proba).is_instance_of(np.ndarray)
        assert_that(m.threshold).is_instance_of(float)
        assert_that(m.threshold).is_equal_to(0.5)
        assert_that(m.average_method).is_instance_of(str)
        assert_that(m.average_method).is_equal_to("weighted")
        assert_that(m.precision_digits).is_instance_of(int)
        assert_that(m.precision_digits).is_equal_to(3)
        assert_that(m.display_df).is_instance_of(bool)
        assert_that(m.display_df).is_true()
        assert_that(m.y_pred_).is_instance_of(np.ndarray)
        assert_that(m.accuracy_).is_instance_of(float)
        assert_that(m.balanced_accuracy_).is_instance_of(float)
        assert_that(m.fpr_list_).is_instance_of(np.ndarray)
        assert_that(m.tpr_list_).is_instance_of(np.ndarray)
        assert_that(m.roc_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_roc_).is_instance_of(float)
        assert_that(m.precision_list_).is_instance_of(np.ndarray)
        assert_that(m.recall_list_).is_instance_of(np.ndarray)
        assert_that(m.pr_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_pr_).is_instance_of(float)
        assert_that(m.precision_).is_instance_of(float)
        assert_that(m.recall_).is_instance_of(float)
        assert_that(m.f1_).is_instance_of(float)
        assert_that(m.f2_).is_instance_of(float)
        assert_that(m.f05_).is_instance_of(float)
        assert_that(m.average_precision_).is_instance_of(float)
        assert_that(m.tn_).is_instance_of(np.int64)
        assert_that(m.fp_).is_instance_of(np.int64)
        assert_that(m.fn_).is_instance_of(np.int64)
        assert_that(m.tp_).is_instance_of(np.int64)
        assert_that(m.threat_score_).is_instance_of(float)
        assert_that(m.youden_index_).is_instance_of(np.int64)
        assert_that(m.youden_threshold_).is_instance_of(float)
        assert_that(m.sens_spec_threshold_).is_instance_of(float)
        assert_that(m.thresholds_dict_).is_instance_of(dict)
        assert_that(m.metrics_dict_).is_instance_of(dict)
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)
        assert_that(m.average_methods_).is_instance_of(list)
        assert_that(m.plotting_dict_).is_instance_of(dict)
        assert_that(m.get_metrics(dtype="dataframe")).is_instance_of(pd.DataFrame)
        assert_that(m.get_metrics(dtype="dict")).is_instance_of(dict)
        npt.assert_almost_equal(np.mean(m.y_pred_), 0.25, decimal=5)
        npt.assert_almost_equal(m.accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(m.balanced_accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.fpr_list_), 0.4, decimal=5)
        npt.assert_almost_equal(np.mean(m.tpr_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.roc_thresholds_), np.inf, decimal=5)
        npt.assert_almost_equal(m.auc_roc_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.precision_list_), 0.73333, decimal=5)
        npt.assert_almost_equal(np.mean(m.recall_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.pr_thresholds_), 0.4125, decimal=5)
        npt.assert_almost_equal(m.auc_pr_, 0.79166, decimal=5)
        npt.assert_almost_equal(m.precision_, 0.833333, decimal=5)
        npt.assert_almost_equal(m.recall_, 0.75, decimal=5)
        npt.assert_almost_equal(m.f1_, 0.73333, decimal=5)
        npt.assert_almost_equal(m.f2_, 0.73232, decimal=5)
        npt.assert_almost_equal(m.f05_, 0.77380, decimal=5)
        npt.assert_almost_equal(m.average_precision_, 0.83333, decimal=5)
        npt.assert_almost_equal(m.threat_score_, 0.611111, decimal=5)
        npt.assert_almost_equal(m.youden_threshold_, 0.8, decimal=5)
        npt.assert_almost_equal(m.sens_spec_threshold_, 0.4, decimal=5)
        npt.assert_almost_equal(m.prec_rec_threshold_, 0.4, decimal=5)
        assert_that(m.tn_).is_equal_to(2)
        assert_that(m.fp_).is_equal_to(0)
        assert_that(m.fn_).is_equal_to(1)
        assert_that(m.tp_).is_equal_to(1)
        assert_that(m.youden_index_).is_equal_to(1)
        assert_that(m.average_methods_).is_subset_of(
            [
                "binary",
                "weighted",
                "macro",
                "micro",
            ],
        )

        # # TODO(amir): figure out a better way to test out figures
        assert_that(f).is_instance_of(Figure)
        assert_that(f.__dict__["_original_dpi"]).is_equal_to(100)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [0, 0, 1, 1],
                "y_pred_proba": [0.1, 0.4, 0.35, 0.8],
                "average_method": "macro",
            },
            {
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred_proba": np.array([0.1, 0.4, 0.35, 0.8]),
                "average_method": "macro",
            },
            {
                "y_true": pd.Series([0, 0, 1, 1]),
                "y_pred_proba": pd.Series([0.1, 0.4, 0.35, 0.8]),
                "average_method": "macro",
            },
        ],
        ids=_ids,
    )
    def test_binary_classification__passes__with_macro_average_method(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates binary classification metrics calculation with "macro" average method."""
        m = BinaryClassificationMetrics(**kwargs)
        f = m.plot(
            display_plot=False,
            return_fig=True,
        )

        assert_that(m.y_true).is_instance_of(np.ndarray)
        assert_that(m.y_pred_proba).is_instance_of(np.ndarray)
        assert_that(m.threshold).is_instance_of(float)
        assert_that(m.threshold).is_equal_to(0.5)
        assert_that(m.average_method).is_instance_of(str)
        assert_that(m.average_method).is_equal_to("macro")
        assert_that(m.precision_digits).is_instance_of(int)
        assert_that(m.precision_digits).is_equal_to(3)
        assert_that(m.display_df).is_instance_of(bool)
        assert_that(m.display_df).is_true()
        assert_that(m.y_pred_).is_instance_of(np.ndarray)
        assert_that(m.accuracy_).is_instance_of(float)
        assert_that(m.balanced_accuracy_).is_instance_of(float)
        assert_that(m.fpr_list_).is_instance_of(np.ndarray)
        assert_that(m.tpr_list_).is_instance_of(np.ndarray)
        assert_that(m.roc_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_roc_).is_instance_of(float)
        assert_that(m.precision_list_).is_instance_of(np.ndarray)
        assert_that(m.recall_list_).is_instance_of(np.ndarray)
        assert_that(m.pr_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_pr_).is_instance_of(float)
        assert_that(m.precision_).is_instance_of(float)
        assert_that(m.recall_).is_instance_of(float)
        assert_that(m.f1_).is_instance_of(float)
        assert_that(m.f2_).is_instance_of(float)
        assert_that(m.f05_).is_instance_of(float)
        assert_that(m.average_precision_).is_instance_of(float)
        assert_that(m.tn_).is_instance_of(np.int64)
        assert_that(m.fp_).is_instance_of(np.int64)
        assert_that(m.fn_).is_instance_of(np.int64)
        assert_that(m.tp_).is_instance_of(np.int64)
        assert_that(m.threat_score_).is_instance_of(float)
        assert_that(m.youden_index_).is_instance_of(np.int64)
        assert_that(m.youden_threshold_).is_instance_of(float)
        assert_that(m.sens_spec_threshold_).is_instance_of(float)
        assert_that(m.thresholds_dict_).is_instance_of(dict)
        assert_that(m.metrics_dict_).is_instance_of(dict)
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)
        assert_that(m.average_methods_).is_instance_of(list)
        assert_that(m.plotting_dict_).is_instance_of(dict)
        assert_that(m.get_metrics(dtype="dataframe")).is_instance_of(pd.DataFrame)
        assert_that(m.get_metrics(dtype="dict")).is_instance_of(dict)
        npt.assert_almost_equal(np.mean(m.y_pred_), 0.25, decimal=5)
        npt.assert_almost_equal(m.accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(m.balanced_accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.fpr_list_), 0.4, decimal=5)
        npt.assert_almost_equal(np.mean(m.tpr_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.roc_thresholds_), np.inf, decimal=5)
        npt.assert_almost_equal(m.auc_roc_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.precision_list_), 0.73333, decimal=5)
        npt.assert_almost_equal(np.mean(m.recall_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.pr_thresholds_), 0.4125, decimal=5)
        npt.assert_almost_equal(m.auc_pr_, 0.79166, decimal=5)
        npt.assert_almost_equal(m.precision_, 0.83333, decimal=5)
        npt.assert_almost_equal(m.recall_, 0.75, decimal=5)
        npt.assert_almost_equal(m.f1_, 0.73333, decimal=5)
        npt.assert_almost_equal(m.f2_, 0.73232, decimal=5)
        npt.assert_almost_equal(m.f05_, 0.77380, decimal=5)
        npt.assert_almost_equal(m.average_precision_, 0.83333, decimal=5)
        npt.assert_almost_equal(m.threat_score_, 0.58333, decimal=5)
        npt.assert_almost_equal(m.youden_threshold_, 0.8, decimal=5)
        npt.assert_almost_equal(m.sens_spec_threshold_, 0.4, decimal=5)
        npt.assert_almost_equal(m.prec_rec_threshold_, 0.4, decimal=5)
        assert_that(m.tn_).is_equal_to(2)
        assert_that(m.fp_).is_equal_to(0)
        assert_that(m.fn_).is_equal_to(1)
        assert_that(m.tp_).is_equal_to(1)
        assert_that(m.youden_index_).is_equal_to(1)
        assert_that(m.average_methods_).is_subset_of(
            [
                "binary",
                "weighted",
                "macro",
                "micro",
            ],
        )

        # # TODO(amir): figure out a better way to test out figures
        assert_that(f).is_instance_of(Figure)
        assert_that(f.__dict__["_original_dpi"]).is_equal_to(100)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [0, 0, 1, 1],
                "y_pred_proba": [0.1, 0.4, 0.35, 0.8],
                "average_method": "micro",
            },
            {
                "y_true": np.array([0, 0, 1, 1]),
                "y_pred_proba": np.array([0.1, 0.4, 0.35, 0.8]),
                "average_method": "micro",
            },
            {
                "y_true": pd.Series([0, 0, 1, 1]),
                "y_pred_proba": pd.Series([0.1, 0.4, 0.35, 0.8]),
                "average_method": "micro",
            },
        ],
        ids=_ids,
    )
    def test_binary_classification__passes__with_micro_average_method(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates binary classification metrics calculation with "micro" average method."""
        m = BinaryClassificationMetrics(**kwargs)
        f = m.plot(
            display_plot=False,
            return_fig=True,
        )

        assert_that(m.y_true).is_instance_of(np.ndarray)
        assert_that(m.y_pred_proba).is_instance_of(np.ndarray)
        assert_that(m.threshold).is_instance_of(float)
        assert_that(m.threshold).is_equal_to(0.5)
        assert_that(m.average_method).is_instance_of(str)
        assert_that(m.average_method).is_equal_to("micro")
        assert_that(m.precision_digits).is_instance_of(int)
        assert_that(m.precision_digits).is_equal_to(3)
        assert_that(m.display_df).is_instance_of(bool)
        assert_that(m.display_df).is_true()
        assert_that(m.y_pred_).is_instance_of(np.ndarray)
        assert_that(m.accuracy_).is_instance_of(float)
        assert_that(m.balanced_accuracy_).is_instance_of(float)
        assert_that(m.fpr_list_).is_instance_of(np.ndarray)
        assert_that(m.tpr_list_).is_instance_of(np.ndarray)
        assert_that(m.roc_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_roc_).is_instance_of(float)
        assert_that(m.precision_list_).is_instance_of(np.ndarray)
        assert_that(m.recall_list_).is_instance_of(np.ndarray)
        assert_that(m.pr_thresholds_).is_instance_of(np.ndarray)
        assert_that(m.auc_pr_).is_instance_of(float)
        assert_that(m.precision_).is_instance_of(float)
        assert_that(m.recall_).is_instance_of(float)
        assert_that(m.f1_).is_instance_of(float)
        assert_that(m.f2_).is_instance_of(float)
        assert_that(m.f05_).is_instance_of(float)
        assert_that(m.average_precision_).is_instance_of(float)
        assert_that(m.tn_).is_instance_of(np.int64)
        assert_that(m.fp_).is_instance_of(np.int64)
        assert_that(m.fn_).is_instance_of(np.int64)
        assert_that(m.tp_).is_instance_of(np.int64)
        assert_that(m.threat_score_).is_instance_of(float)
        assert_that(m.youden_index_).is_instance_of(np.int64)
        assert_that(m.youden_threshold_).is_instance_of(float)
        assert_that(m.sens_spec_threshold_).is_instance_of(float)
        assert_that(m.thresholds_dict_).is_instance_of(dict)
        assert_that(m.metrics_dict_).is_instance_of(dict)
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)
        assert_that(m.average_methods_).is_instance_of(list)
        assert_that(m.plotting_dict_).is_instance_of(dict)
        assert_that(m.get_metrics(dtype="dataframe")).is_instance_of(pd.DataFrame)
        assert_that(m.get_metrics(dtype="dict")).is_instance_of(dict)
        npt.assert_almost_equal(np.mean(m.y_pred_), 0.25, decimal=5)
        npt.assert_almost_equal(m.accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(m.balanced_accuracy_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.fpr_list_), 0.4, decimal=5)
        npt.assert_almost_equal(np.mean(m.tpr_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.roc_thresholds_), np.inf, decimal=5)
        npt.assert_almost_equal(m.auc_roc_, 0.75, decimal=5)
        npt.assert_almost_equal(np.mean(m.precision_list_), 0.73333, decimal=5)
        npt.assert_almost_equal(np.mean(m.recall_list_), 0.6, decimal=5)
        npt.assert_almost_equal(np.mean(m.pr_thresholds_), 0.4125, decimal=5)
        npt.assert_almost_equal(m.auc_pr_, 0.79166, decimal=5)
        npt.assert_almost_equal(m.precision_, 0.75, decimal=5)
        npt.assert_almost_equal(m.recall_, 0.75, decimal=5)
        npt.assert_almost_equal(m.f1_, 0.75, decimal=5)
        npt.assert_almost_equal(m.f2_, 0.75, decimal=5)
        npt.assert_almost_equal(m.f05_, 0.75, decimal=5)
        npt.assert_almost_equal(m.average_precision_, 0.83333, decimal=5)
        npt.assert_almost_equal(m.threat_score_, 0.5, decimal=5)
        npt.assert_almost_equal(m.youden_threshold_, 0.8, decimal=5)
        npt.assert_almost_equal(m.sens_spec_threshold_, 0.4, decimal=5)
        npt.assert_almost_equal(m.prec_rec_threshold_, 0.4, decimal=5)
        assert_that(m.tn_).is_equal_to(2)
        assert_that(m.fp_).is_equal_to(0)
        assert_that(m.fn_).is_equal_to(1)
        assert_that(m.tp_).is_equal_to(1)
        assert_that(m.youden_index_).is_equal_to(1)
        assert_that(m.average_methods_).is_subset_of(
            [
                "binary",
                "weighted",
                "macro",
                "micro",
            ],
        )

        # # TODO(amir): figure out a better way to test out figures
        assert_that(f).is_instance_of(Figure)
        assert_that(f.__dict__["_original_dpi"]).is_equal_to(100)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "dtype": "list",
            },
            {
                "dtype": "tuple",
            },
            {
                "dtype": "array",
            },
            {
                "dtype": pd.DataFrame,
            },
            {
                "dtype": dict,
            },
            {
                "dtype": None,
            },
        ],
        ids=_ids,
    )
    def test_get_metrics__fails__with_invalid_dtypes(
        self,
        kwargs: Dict[str, str],
    ) -> None:
        """Validates `get_metrics()` only works with "dataframe" and "dict" as input."""
        m = BinaryClassificationMetrics(
            y_true=[1, 0, 1, 0],
            y_pred_proba=[0.5, 0.0, 0.2, 0.8],
        )

        with pytest.raises((ValueError, TypeError)):
            _ = m.get_metrics(**kwargs)

    # TODO(amir): currently this test is wrong; figure out what is wrong
    # my hunch is it gets called in other average methods; so it will be tested implicitly
    # which is not good here
    def test_average_method__is_none__when_binary(self) -> None:
        """Validates `average_method` set to be None if "binary" passed."""
        m = BinaryClassificationMetrics(
            y_true=[1, 0, 1, 0],
            y_pred_proba=[0.5, 0.0, 0.2, 0.8],
        )

        assert_that(m.average_method).is_equal_to("binary")
        # TODO(amir): the average method should be None?
        # assert_that(m.average_method).is_none()

    def test_binary_classification_metrics_plots__passes__with_valid_save_paths(
        self,
        figure_path: Path,
    ) -> None:
        """Validates `BinaryClassificationMetrics` saving plots passes with valid paths."""
        m = BinaryClassificationMetrics(
            y_true=[1, 0, 1, 0],
            y_pred_proba=[0.5, 0.0, 0.2, 0.8],
        )
        clf_metrics_fig_path = figure_path / "clf_metrics.png"  # type: ignore
        m.plot(
            save_path=str(clf_metrics_fig_path),
            return_fig=False,
            display_plot=False,
        )

        assert_that(clf_metrics_fig_path.parts[-1]).is_equal_to("clf_metrics.png")
        _validate_figure_type_and_size(
            path=clf_metrics_fig_path,
            expected_size=(2297, 2016),
        )
