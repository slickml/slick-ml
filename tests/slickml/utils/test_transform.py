import re
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that
from scipy.sparse import csr_matrix

from slickml.utils import df_to_csr, memory_use_csr
from tests.utils import (
    _captured_log,
    _dummy_pandas_dataframe,
    _dummy_sparse_matrix,
    _ids,
)


def test_df_to_csr__passes__with_default_inputs() -> None:
    """Validates conversion of a pandas DataFrame into CSR matrix with default inputs."""
    df = _dummy_pandas_dataframe()
    csr = df_to_csr(df)

    assert_that(csr).is_instance_of(csr_matrix)
    assert_that(csr.shape).is_equal_to(df.shape)
    assert_that(csr.data.shape).is_equal_to(df["foo"].shape)
    assert_that(all(csr.data == df["foo"].values)).is_true()


def test_df_to_csr__when__verbose_is_true(capsys) -> None:
    """Validates if the logged memory usage in standard output is accurate."""
    df = _dummy_pandas_dataframe()
    csr = df_to_csr(df, verbose=True)
    output, error = _captured_log(capsys)

    assert_that(error).is_empty()
    assert_that(output).is_not_empty()
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-2,
        )
        - np.round(memory_use_csr(csr) / 2**20, 5),
    ).is_less_than(0.000001)
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-2,
        ),
    ).is_instance_of(float)
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-3,
        ),
    ).is_equal_to(memory_use_csr(csr))
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-3,
        ),
    ).is_instance_of(float)
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-4,
        ),
    ).is_equal_to(df.memory_usage().sum())
    assert_that(
        _captured_memory_use_from_stdout(
            captured_output=output,
            index=-4,
        ),
    ).is_instance_of(float)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "df": [42],
            "fillna": 0.0,
            "verbose": False,
        },
        {
            "df": pd.DataFrame(),
            "fillna": 0,
            "verbose": False,
        },
        {
            "df": pd.DataFrame(),
            "fillna": 0.0,
            "verbose": 1,
        },
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
        {
            "csr": [42],
        },
        {
            "csr": pd.DataFrame(),
        },
        {
            "csr": np.arange(10),
        },
    ],
    ids=_ids,
)
def test_memory_use_csr__fails_with_invalid_inputs(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        memory_use_csr(**kwargs)


def test_memory_use_csr__passes_with_default_inputs() -> None:
    mem = memory_use_csr(
        csr=_dummy_sparse_matrix(),
    )

    assert_that(mem).is_instance_of(int)
    assert_that(mem).is_equal_to(88)


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
