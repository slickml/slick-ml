from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from assertpy import assert_that

from slickml.utils import ExtendedEnum, check_var
from tests.utils import _ids


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "var": "42",
            "var_name": "foo_str",
            "dtypes": int,
            "values": None,
        },
        {
            "var": 42.0,
            "var_name": "foo_float",
            "dtypes": int,
            "values": None,
        },
        {
            "var": pd.DataFrame({"foo": [1, 2, 3]}),
            "var_name": "foo_dataframe",
            "dtypes": np.ndarray,
            "values": None,
        },
        {
            "var": np.array([[1, 2, 3]]),
            "var_name": "foo_array",
            "dtypes": pd.DataFrame,
            "values": None,
        },
        {
            "var": [1, 2, 3],
            "var_name": "foo_list",
            "dtypes": tuple,
            "values": None,
        },
        {
            "var": {"x": 42},
            "var_name": "foo_dict",
            "dtypes": list,
            "values": None,
        },
        {
            "var": (1, 2, 3),
            "var_name": "foo_tuple",
            "dtypes": list,
            "values": None,
        },
        {
            "var": 42,
            "var_name": "foo_str",
            "dtypes": (str),
            "values": None,
        },
        {
            "var": "42",
            "var_name": "foo_int_or_float",
            "dtypes": (int, float),
            "values": None,
        },
        {
            "var": np.array([1, 2, 3]),
            "var_name": "foo_array",
            "dtypes": np.array,
            "values": None,
        },
    ],
    ids=_ids,
)
def test_check_var__fails__with_invalid_types_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that check variable fails with inputs with invalid types."""
    with pytest.raises(TypeError):
        check_var(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "var": "42",
            "var_name": "foo_str",
            "dtypes": str,
            "values": "43",
        },
        {
            "var": "42",
            "var_name": "foo_str",
            "dtypes": str,
            "values": ("43", "44"),
        },
        {
            "var": 42.0,
            "var_name": "foo_float",
            "dtypes": float,
            "values": 42.000000001,
        },
        {
            "var": pd.DataFrame({"foo": [1, 2, 3]}),
            "var_name": "foo_dataframe",
            "dtypes": pd.DataFrame,
            "values": pd.DataFrame({"fooo": [1, 2, 3]}),
        },
        {
            "var": np.array([[1, 2, 3]]),
            "var_name": "foo_array",
            "dtypes": np.ndarray,
            "values": np.array([[1, 2, 3.0000000001]]),
        },
        {
            "var": [1, 2, 3],
            "var_name": "foo_list",
            "dtypes": list,
            "values": [1, 2, 3.0000001],
        },
        {
            "var": {"x": 42},
            "var_name": "foo_dict",
            "dtypes": dict,
            "values": {"x": 41},
        },
        {
            "var": (1, 2, 3),
            "var_name": "foo_tuple",
            "dtypes": tuple,
            "values": (1, 2),
        },
    ],
    ids=_ids,
)
def test_check_var__fails__with_invalid_values_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that check variable fails with inputs with valid types but invalid values."""
    with pytest.raises(ValueError):
        check_var(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "var": "42",
            "var_name": "foo_str",
            "dtypes": str,
            "values": None,
        },
        {
            "var": "42",
            "var_name": "foo_str",
            "dtypes": str,
            "values": "42",
        },
        {
            "var": "42",
            "var_name": "foo_str",
            "dtypes": str,
            "values": ("42", "43"),
        },
        {
            "var": 42.0,
            "var_name": "foo_float",
            "dtypes": float,
            "values": None,
        },
        {
            "var": 42.0,
            "var_name": "foo_float",
            "dtypes": float,
            "values": (40, 42),
        },
    ],
    ids=_ids,
)
def test_check_var__passes__with_valid_inputs(kwargs: Dict[str, Any]) -> None:
    """Validates that check variables are successful."""
    assert_that(check_var(**kwargs)).is_none()


class FooBarBazQux(ExtendedEnum):
    """Dummy ExtendedEnum class for sake of unit-tests."""

    FOO = "foo"
    BAR = "bar"
    BAZ = "baz"
    QUX = "qux"


class TestExtendedEnum:
    """Validates ExtendedEnum instantiation."""

    def test_extended_enum_instantiation__passes__with_default_values(self) -> None:
        """Validates the test enum class FooBarBazQux instantiation and properties."""

        assert_that(FooBarBazQux.FOO).is_instance_of(Enum)
        assert_that(FooBarBazQux.BAR).is_instance_of(Enum)
        assert_that(FooBarBazQux.BAZ).is_instance_of(Enum)
        assert_that(FooBarBazQux.QUX).is_instance_of(Enum)
        assert_that(FooBarBazQux.FOO.value).is_instance_of(str)
        assert_that(FooBarBazQux.BAR.value).is_instance_of(str)
        assert_that(FooBarBazQux.BAZ.value).is_instance_of(str)
        assert_that(FooBarBazQux.QUX.value).is_instance_of(str)
        assert_that(FooBarBazQux.FOO.name).is_instance_of(str)
        assert_that(FooBarBazQux.BAR.name).is_instance_of(str)
        assert_that(FooBarBazQux.BAZ.name).is_instance_of(str)
        assert_that(FooBarBazQux.QUX.name).is_instance_of(str)
        assert_that(str(FooBarBazQux.FOO)).is_instance_of(str)
        assert_that(str(FooBarBazQux.BAR)).is_instance_of(str)
        assert_that(str(FooBarBazQux.BAZ)).is_instance_of(str)
        assert_that(str(FooBarBazQux.QUX)).is_instance_of(str)
        assert_that(FooBarBazQux.names()).is_instance_of(list)
        assert_that(FooBarBazQux.names()).is_iterable()
        assert_that(FooBarBazQux.values()).is_instance_of(list)
        assert_that(FooBarBazQux.values()).is_iterable()
        assert_that(FooBarBazQux.to_dict()).is_instance_of(dict)
        assert_that(FooBarBazQux.to_dict()).is_iterable()
        assert_that(sorted(FooBarBazQux.names())).is_equal_to(
            sorted(
                [
                    "FOO",
                    "BAR",
                    "BAZ",
                    "QUX",
                ],
            ),
        )
        assert_that(sorted(FooBarBazQux.values())).is_equal_to(
            sorted(
                [
                    "foo",
                    "bar",
                    "baz",
                    "qux",
                ],
            ),
        )
        assert_that(FooBarBazQux.to_dict()).is_equal_to(
            {
                "FOO": "foo",
                "BAR": "bar",
                "BAZ": "baz",
                "QUX": "qux",
            },
        )
