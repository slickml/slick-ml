from pathlib import Path  # noqa
from typing import Any, Dict

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from assertpy import assert_that
from matplotlib.figure import Figure

from slickml.metrics import RegressionMetrics
from tests.conftest import _ids, _validate_figure_type_and_size


# TODO(amir): tests for multi-outputs + "variance_weighted" and "raw_values" methods are still missing
class TestRegressionMetrics:
    """Validates `RegressionMetrics` instantiation."""

    # TODO(amir): update this test once `y_true` and `y_pred` types are fixed with `check_var()`
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "multioutput": True,
            },
            {
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "precision_digits": 3.0,
            },
            {
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "display_df": 1,
            },
        ],
        ids=_ids,
    )
    def test_regression_metrics__fails__with_invalid_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates that regression metrics cannot be calculated with invalid inputs."""
        with pytest.raises(TypeError):
            RegressionMetrics(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "y_true": [3, -0.5, 2, 7],
                "y_pred": [2.5, 0.0, 2, 8],
            },
            {
                "y_true": np.array([3, -0.5, 2, 7]),
                "y_pred": np.array([2.5, 0.0, 2, 8]),
            },
            {
                "y_true": pd.Series([3, -0.5, 2, 7]),
                "y_pred": pd.Series([2.5, 0.0, 2, 8]),
            },
        ],
        ids=_ids,
    )
    def test_regression_metrics__passes__with_default_inputs(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        """Validates regression metrics calculation with default inputs.

        Notes
        -----
        The input `y_true` contains negative values. Therefore, mean-squared-log-error will be None.
        `_msle()` is being tested separately.
        """
        m = RegressionMetrics(**kwargs)
        f = m.plot(
            display_plot=False,
            return_fig=True,
        )

        assert_that(m.y_true).is_instance_of(np.ndarray)
        assert_that(m.y_pred).is_instance_of(np.ndarray)
        assert_that(m.multioutput).is_instance_of(str)
        assert_that(m.multioutput).is_equal_to("uniform_average")
        assert_that(m.precision_digits).is_instance_of(int)
        assert_that(m.precision_digits).is_equal_to(3)
        assert_that(m.display_df).is_instance_of(bool)
        assert_that(m.display_df).is_true()

        assert_that(m.y_residual_).is_instance_of(np.ndarray)
        assert_that(m.y_residual_normsq_).is_instance_of(np.ndarray)
        assert_that(m.r2_).is_instance_of(float)
        assert_that(m.mae_).is_instance_of(float)
        assert_that(m.mse_).is_instance_of(float)
        assert_that(m.msle_).is_none()
        assert_that(m.mape_).is_instance_of(float)
        assert_that(m.auc_rec_).is_instance_of(float)
        assert_that(m.deviation_).is_instance_of(np.ndarray)
        assert_that(m.accuracy_).is_instance_of(np.ndarray)
        assert_that(m.y_ratio_).is_instance_of(np.ndarray)
        assert_that(m.mean_y_ratio_).is_instance_of(float)
        assert_that(m.std_y_ratio_).is_instance_of(float)
        assert_that(m.cv_y_ratio_).is_instance_of(float)
        assert_that(m.metrics_dict_).is_instance_of(dict)
        assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)
        assert_that(m.metrics_dict_).is_instance_of(dict)
        assert_that(m.get_metrics(dtype="dataframe")).is_instance_of(pd.DataFrame)
        assert_that(m.get_metrics(dtype="dict")).is_instance_of(dict)

        npt.assert_almost_equal(m.r2_, 0.94860, decimal=5)
        npt.assert_almost_equal(m.mae_, 0.5, decimal=5)
        npt.assert_almost_equal(m.mse_, 0.375, decimal=5)
        npt.assert_almost_equal(m.mape_, 0.32738, decimal=5)
        npt.assert_almost_equal(m.auc_rec_, 0.68666, decimal=5)
        npt.assert_almost_equal(m.mean_y_ratio_, 0.74404, decimal=5)
        npt.assert_almost_equal(m.std_y_ratio_, 0.44332, decimal=5)
        npt.assert_almost_equal(m.cv_y_ratio_, 0.59582, decimal=5)
        npt.assert_almost_equal(np.mean(m.y_residual_), -0.25, decimal=5)
        npt.assert_almost_equal(np.mean(m.y_residual_normsq_), 0.60355, decimal=5)
        npt.assert_almost_equal(np.mean(m.deviation_), 0.49499, decimal=5)
        npt.assert_almost_equal(np.mean(m.accuracy_), 0.69, decimal=5)
        npt.assert_almost_equal(np.mean(m.y_ratio_), 0.74404, decimal=5)

        # TODO(amir): figure out a better way to test out figures
        assert_that(f).is_instance_of(Figure)
        assert_that(f.__dict__["_original_dpi"]).is_equal_to(100)

    def test_msle__passes__with_non_negative_inputs(self) -> None:
        """Validates the calculation of `_msle()` with non-negative inputs."""
        m = RegressionMetrics(
            y_true=[3, 0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )

        assert_that(m.msle_).is_not_none()
        assert_that(m.msle_).is_instance_of(float)
        npt.assert_almost_equal(m.msle_, 0.04902, decimal=5)

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
        m = RegressionMetrics(
            y_true=[3, 0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )

        with pytest.raises((ValueError, TypeError)):
            _ = m.get_metrics(**kwargs)

    def test_regression_metrics_plots__passes__with_valid_save_paths(
        self,
        figure_path: Path,
    ) -> None:
        """Validates `RegressionMetrics` saving plots passes with valid paths."""
        m = RegressionMetrics(
            y_true=[3, 0.5, 2, 7],
            y_pred=[2.5, 0.0, 2, 8],
        )
        reg_metrics_fig_path = figure_path / "reg_metrics.png"  # type: ignore
        m.plot(
            save_path=str(reg_metrics_fig_path),
            return_fig=False,
            display_plot=False,
        )

        assert_that(reg_metrics_fig_path.parts[-1]).is_equal_to("reg_metrics.png")
        _validate_figure_type_and_size(
            path=reg_metrics_fig_path,
            expected_size=(2037, 2632),
        )
