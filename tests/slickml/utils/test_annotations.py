from dataclasses import dataclass

import pytest
from assertpy import assert_that

from slickml.utils import deprecated

_X = 1367
_ALTERNATIVE = "slickml.best()"
_SINCE = "version 1.2.3"


def test_deprecated_function__passes__with_defaults() -> None:
    """Validates annotating a function with ``deprecated`` decorator and default arguments."""

    @deprecated()
    def foo(x: int) -> int:
        """Docstring for function ``foo``."""
        return x

    with pytest.warns(FutureWarning) as record:
        r = foo(x=_X)

    assert_that(r).is_instance_of(int)
    assert_that(r).is_equal_to(_X)
    assert_that(foo.__doc__).contains(
        _default_deprecation_warning(),
    )
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _default_deprecation_warning(),
    )


def test_deprecated_function__passes__with_custom_inputs() -> None:
    """Validates annotating a function with ``deprecated`` decorator and custom arguments."""

    @deprecated(
        alternative=_ALTERNATIVE,
        since=_SINCE,
    )
    def foo(x: int) -> int:
        """Docstring for function ``foo``."""
        return x

    with pytest.warns(FutureWarning) as record:
        r = foo(x=_X)

    assert_that(r).is_instance_of(int)
    assert_that(r).is_equal_to(_X)
    assert_that(foo.__doc__).contains("Docstring for function ``foo``.")
    assert_that(foo.__doc__).contains(
        _custom_deprecation_warning(),
    )
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _custom_deprecation_warning(),
    )


def test_deprecated_function__passes__with_custom_inputs_and_no_docstring() -> None:
    """Validates annotating a function with ``deprecated`` decorator and custom arguments."""

    @deprecated(
        alternative=_ALTERNATIVE,
        since=_SINCE,
    )
    def foo(x: int) -> int:
        return x

    with pytest.warns(FutureWarning) as record:
        r = foo(x=_X)

    assert_that(r).is_instance_of(int)
    assert_that(r).is_equal_to(_X)
    assert_that(foo.__doc__).contains(
        _custom_deprecation_warning(),
    )
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _custom_deprecation_warning(),
    )


def test_deprecated_class__passes__with_defaults() -> None:
    """Validates annotating a class with ``deprecated`` decorator and default arguments."""

    @deprecated()
    class Bar:
        """Docstring for class ``Bar``."""

        def __init__(self, x: int) -> None:
            self.x = x

    with pytest.warns(FutureWarning) as record:
        b = Bar(x=_X)

    assert_that(b.x).is_instance_of(int)
    assert_that(b.x).is_equal_to(_X)
    assert_that(b.__doc__).contains("Docstring for class ``Bar``.")
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _default_deprecation_warning(),
    )


def test_deprecated_class__passes__with_custom_inputs() -> None:
    """Validates annotating a class with ``deprecated`` decorator and custom arguments."""

    @deprecated(
        alternative=_ALTERNATIVE,
        since=_SINCE,
    )
    class Bar:
        """Docstring for class ``Bar``."""

        def __init__(self, x: int) -> None:
            self.x = x

    with pytest.warns(FutureWarning) as record:
        b = Bar(x=_X)

    assert_that(b.x).is_instance_of(int)
    assert_that(b.x).is_equal_to(_X)
    assert_that(b.__doc__).contains("Docstring for class ``Bar``.")
    # `class.__doc__` is currently get wrapped. Uncomment below once the decorator is fixed
    # assert_that(b.__doc__).contains(
    #     "is deprecated since version 1.2.3 and will be removed in a future release. Use ``Baz`` instead.",
    # )
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _custom_deprecation_warning(),
    )


def test_deprecated_dataclass__passes__with_defaults() -> None:
    """Validates annotating a dataclass with ``deprecated`` decorator and default arguments."""

    @deprecated()
    @dataclass
    class Baz:
        """Docstring for class ``Baz``."""

        x: int

    b = Baz(x=_X)

    assert_that(b.x).is_instance_of(int)
    assert_that(b.x).is_equal_to(_X)
    assert_that(b.__doc__).contains("Docstring for class ``Baz``.")


def test_deprecated_dataclass__passes__with_custom_inputs() -> None:
    """Validates annotating a dataclass with ``deprecated`` decorator and default arguments."""

    @deprecated(
        alternative=_ALTERNATIVE,
        since=_SINCE,
    )
    @dataclass
    class Baz:
        """Docstring for class ``Baz``."""

        x: int

    with pytest.warns(FutureWarning) as record:
        b = Baz(x=_X)

    assert_that(b.x).is_instance_of(int)
    assert_that(b.x).is_equal_to(_X)
    assert_that(b.__doc__).contains("Docstring for class ``Baz``.")
    # `class.__doc__` is currently get wrapped. Uncomment below once the decorator is fixed
    # assert_that(b.__doc__).contains(
    #     "is deprecated since version 1.2.3 and will be removed in a future release. Use ``Qux`` instead.",
    # )
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _custom_deprecation_warning(),
    )


def test_deprecated_method_of_dataclass__passes__with_custom_inputs() -> None:
    """Validates annotating a dataclass with ``deprecated`` decorator and default arguments."""

    @dataclass
    class Qux:
        """Docstring for class ``Qux``."""

        @deprecated(
            alternative=_ALTERNATIVE,
            since=_SINCE,
        )
        def qux(self, x: int) -> int:
            """Docstring for method ``qux``."""
            return x

    with pytest.warns(FutureWarning) as record:
        q = Qux()
        r = q.qux(x=_X)

    assert_that(r).is_instance_of(int)
    assert_that(r).is_equal_to(_X)
    assert_that(q.__doc__).contains("Docstring for class ``Qux``")
    # `class.__doc__` is currently get wrapped. Uncomment below once the decorator is fixed
    # assert_that(q.__doc__).contains(
    #     "is deprecated since version 1.2.3 and will be removed in a future release. Use ``qux_v2`` instead.",
    # )
    assert_that(record).is_length(1)
    assert_that(record[0].message.args[0]).contains(
        _custom_deprecation_warning(),
    )


def _default_deprecation_warning() -> str:
    """Returns default deprecation warning message.

    Returns
    -------
    str
    """
    return "is deprecated and will be removed in a future release."


def _custom_deprecation_warning(
    alternative: str = _ALTERNATIVE,
    since: str = _SINCE,
) -> str:
    """Returns default deprecation warning message.

    alternative : str
        Alternative method

    since : str
        Version that deprecation begins

    Returns
    -------
    str
    """
    return f"is deprecated since {since} and will be removed in a future release. Use ``{alternative}`` instead."
