import re
from typing import Any, Dict

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from assertpy import assert_that
from pytest import CaptureFixture
from scipy.sparse import csr_matrix

from slickml.utils import add_noisy_features, array_to_df, df_to_csr, memory_use_csr
from tests.conftest import _captured_log, _ids


def test_df_to_csr__passes__with_default_inputs(
    datafarame_for_testing: pd.DataFrame,
) -> None:
    """Validates conversion of a pandas DataFrame into CSR matrix with default inputs."""
    df = datafarame_for_testing
    csr = df_to_csr(df)

    assert_that(csr).is_instance_of(csr_matrix)
    assert_that(csr.shape).is_equal_to(df.shape)
    assert_that(csr.data.shape[0]).is_equal_to(np.count_nonzero(df))
    assert_that(all(csr.data == df.values.flatten()[np.flatnonzero(df.values)])).is_true()


def test_df_to_csr__passes__when_verbose_is_true(
    capsys: CaptureFixture,
    datafarame_for_testing: pd.DataFrame,
) -> None:
    """Validates if the logged memory usage in standard output is accurate."""
    df = datafarame_for_testing
    csr = df_to_csr(df, verbose=True)
    output, error = _captured_log(capsys)

    assert_that(error).is_empty()
    assert_that(output).is_not_empty()
    npt.assert_almost_equal(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-2,
        ),
        memory_use_csr(csr) / 2**20,
        decimal=5,
    )
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-2,
        ),
    ).is_instance_of(float)
    npt.assert_almost_equal(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-3,
        ),
        memory_use_csr(csr),
        decimal=5,
    )
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-3,
        ),
    ).is_instance_of(float)
    npt.assert_almost_equal(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-4,
        ),
        df.memory_usage().sum() / 2**10,
        decimal=1,
    )
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-4,
        ),
    ).is_instance_of(float)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"df": [42], "fillna": 0.0, "verbose": False},
        {"df": pd.DataFrame(), "fillna": 0, "verbose": False},
        {"df": pd.DataFrame(), "fillna": 0.0, "verbose": 1},
    ],
    ids=_ids,
)
def test_df_to_csr__fails__with_invalid_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that conversion cannot be done with invalid inputs."""
    with pytest.raises(TypeError):
        df_to_csr(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"csr": [42]},
        {"csr": pd.DataFrame()},
        {"csr": np.arange(10)},
    ],
    ids=_ids,
)
def test_memory_use_csr__fails__with_invalid_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that getting the memory usage info cannot be done with invalid inputs."""
    with pytest.raises(TypeError):
        memory_use_csr(**kwargs)


def test_memory_use_csr__passes__with_default_inputs(
    sparse_matrix_for_testing: csr_matrix,
) -> None:
    """Validates that memory usage info with default inputs is accurate."""
    mem = memory_use_csr(
        csr=sparse_matrix_for_testing,
    )

    assert_that(mem).is_instance_of(int)
    assert_that(mem).is_equal_to(88)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"X": pd.DataFrame({"foo": [1, 2, 3, 4, 5]})},
        {"X": np.array([[1, 2, 3, 4, 5]])},
    ],
    ids=_ids,
)
def test_add_noisy_features__passes__with_default_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that noisy features are getting augmented successfully."""
    df_noisy = add_noisy_features(**kwargs)

    assert_that(df_noisy).is_instance_of(pd.DataFrame)
    assert_that(df_noisy).is_not_empty()
    assert_that(df_noisy.shape[0]).is_equal_to(kwargs["X"].shape[0])
    assert_that(df_noisy.shape[1]).is_equal_to(kwargs["X"].shape[1] * 2)
    assert_that(df_noisy.sum().sum()).is_equal_to(kwargs["X"].sum().sum() * 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"X": [42]},
        {"X": pd.DataFrame({"foo": [42]}), "random_state": "one"},
        {"X": pd.DataFrame({"foo": [42]}), "prefix": 42},
    ],
    ids=_ids,
)
def test_add_noisy_features__fails__with_invalid_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that noisy features cannot be added with invalid inputs."""
    with pytest.raises(TypeError):
        add_noisy_features(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"X": [42]},
        {"X": pd.DataFrame()},
        {"X": {"X": 42}},
        {"X": np.array(([42])), "prefix": 123},
        {"X": np.array(([42])), "prefix": "123", "delimiter": 123},
    ],
    ids=_ids,
)
def test_array_to_df__fails__with_invalid_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that conversion cannot be done with invalid inputs."""
    with pytest.raises(TypeError):
        array_to_df(**kwargs)


def test_array_to_df__passes__with_1d_array_and_default_inputs() -> None:
    """Validates conversion of a 1D numpy array into a pandas DataFrame with default inputs."""
    X = np.array([1, 2, 3])
    df = array_to_df(X=X)

    assert_that(X.ndim).is_equal_to(1)
    assert_that(df).is_instance_of(pd.DataFrame)
    assert_that(df.shape).is_equal_to((1, 3))
    assert_that(df.columns.tolist()).is_equal_to(["F_0", "F_1", "F_2"])
    assert_that(df.sum().sum()).is_equal_to(X.sum().sum())


def test_array_to_df__passes__with_2d_array_and_default_inputs() -> None:
    """Validates conversion of a 2D numpy array into a pandas DataFrame with default inputs."""
    X = np.array([[1, 2, 3]])
    df = array_to_df(X)

    assert_that(X.ndim).is_equal_to(2)
    assert_that(df).is_instance_of(pd.DataFrame)
    assert_that(df.shape).is_equal_to((1, 3))
    assert_that(df.columns.tolist()).is_equal_to(["F_0", "F_1", "F_2"])
    assert_that(df.sum().sum()).is_equal_to(X.sum().sum())


def test_array_to_df__passes__with_1d_array_and_custom_inputs() -> None:
    """Validates conversion of a 1D numpy array into a pandas DataFrame with custom inputs."""
    X = np.array([1, 2, 3])
    prefix = "x"
    delimiter = "<>"
    df = array_to_df(
        X=X,
        prefix=prefix,
        delimiter=delimiter,
    )

    assert_that(X.ndim).is_equal_to(1)
    assert_that(df).is_instance_of(pd.DataFrame)
    assert_that(df.shape).is_equal_to((1, 3))
    assert_that(df.columns.tolist()).is_equal_to(["x<>0", "x<>1", "x<>2"])
    assert_that(df.sum().sum()).is_equal_to(X.sum().sum())


def test_array_to_df__passes__with_2d_array_and_custom_inputs() -> None:
    """Validates conversion of a 2D numpy array into a pandas DataFrame with custom inputs."""
    X = np.array([[1, 2, 3]])
    prefix = "x"
    delimiter = "<>"
    df = array_to_df(
        X=X,
        prefix=prefix,
        delimiter=delimiter,
    )

    assert_that(X.ndim).is_equal_to(2)
    assert_that(df).is_instance_of(pd.DataFrame)
    assert_that(df.shape).is_equal_to((1, 3))
    assert_that(df.columns.tolist()).is_equal_to(["x<>0", "x<>1", "x<>2"])
    assert_that(df.sum().sum()).is_equal_to(X.sum().sum())


def _captured_memory_use_from_stdout(
    captured_output: str,
    index: int,
) -> float:
    """Helper function that cleans up the captured standard output and returns memory usage.

    Here is an example of the captured log in string format:
    output = "<class 'pandas.core.frame.DataFrame'>\n
    RangeIndex: 100 entries, 0 to 99\n
    Data columns (total 1 columns):\n
    #   Column  Non-Null Count  Dtype  \n
    ---  ------  --------------  -----  \n
    0   foo     100 non-null    float64\n
    dtypes: float64(1)\n
    memory usage: 928.0 bytes\n
    CSR memory usage: 1604.0 bytes\n
    CSR memory usage: 0.00153 MB
    \n"
    So, the return values based on each passed index are:
    index=-2 --> 0.00153
    index=-3 --> 1604.0
    index=-4 --> 928.0

    Parameters
    ----------
    captured_output : str
        Captured output string from standard output

    index : int
        Index of list of captured output splitted by `"\n"`

    Returns
    -------
    float
        Extracted memory usage from captured log
    """
    output_list = captured_output.split("\n")
    return float(
        re.findall(
            r"\d+\.\d+",
            output_list[index],
        )[0],
    )
