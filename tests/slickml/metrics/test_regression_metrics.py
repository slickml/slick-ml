from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that
from matplotlib.figure import Figure

from slickml.metrics import RegressionMetrics
from tests.utils import _ids

_FLOATING_POINT_THRESHOLD = 1e-5


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
def test_regression_metrics__fails__with_invalid_inputs(kwargs: Dict[str, Any]) -> None:
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
    kwargs: Dict[str, Union[List[float], np.ndarray, pd.Series]],
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
    assert_that(m.accuracy_).is_instance_of(list)
    assert_that(m.y_ratio_).is_instance_of(np.ndarray)
    assert_that(m.mean_y_ratio_).is_instance_of(float)
    assert_that(m.std_y_ratio_).is_instance_of(float)
    assert_that(m.cv_y_ratio_).is_instance_of(float)
    assert_that(m.metrics_dict_).is_instance_of(dict)
    assert_that(m.metrics_df_).is_instance_of(pd.DataFrame)
    assert_that(m.metrics_dict_).is_instance_of(dict)

    assert_that(
        np.abs(
            m.r2_ - 0.9486081370449679,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.mae_ - 0.5,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.mse_ - 0.375,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.mape_ - 0.3273809523809524,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.auc_rec_ - 0.6864583333333334,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.mean_y_ratio_ - 0.7440476190476191,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.std_y_ratio_ - 0.4433225281277483,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            m.cv_y_ratio_ - 0.5958254778036937,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            np.mean(m.y_residual_) - (-0.25),
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            np.mean(m.y_residual_normsq_) - 0.6035533905932737,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            np.mean(m.deviation_) - 0.49499999999999994,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            np.mean(m.accuracy_) - 0.69,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
    assert_that(
        np.abs(
            np.mean(m.y_ratio_) - 0.7440476190476191,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)

    # TODO(amir): figure out a better way to test out figures
    assert_that(f).is_instance_of(Figure)
    assert_that(f.__dict__["_original_dpi"]).is_equal_to(100)


def test_msle__passes__with_non_negative_inputs() -> None:
    """Validates the calculation of `_msle()` with non-negative inputs."""
    m = RegressionMetrics(
        y_true=[3, 0.5, 2, 7],
        y_pred=[2.5, 0.0, 2, 8],
    )

    assert_that(m.msle_).is_not_none()
    assert_that(m.msle_).is_instance_of(float)
    assert_that(
        np.abs(
            m.msle_ - 0.0490263575494607,
        ),
    ).is_less_than(_FLOATING_POINT_THRESHOLD)
