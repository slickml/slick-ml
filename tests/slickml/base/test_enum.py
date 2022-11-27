from enum import Enum

from assertpy import assert_that

from slickml.base import ExtendedEnum


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

        # TODO(amir): mypy still does not support types of enum subclasses
        # https://github.com/python/mypy/issues/6037
        # https://github.com/microsoft/pyright/issues/1751
        assert_that(FooBarBazQux.FOO.value).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.BAR.value).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.BAZ.value).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.QUX.value).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.FOO.name).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.BAR.name).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.BAZ.name).is_instance_of(str)  # type: ignore
        assert_that(FooBarBazQux.QUX.name).is_instance_of(str)  # type: ignore
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
