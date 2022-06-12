import pytest
from assertpy import assert_that

from slickml._dummy import C

CASES = [
    {
        "a": 1,
        "b": "foo",
    },
]


@pytest.mark.parametrize("kwargs", CASES)
class TestC:
    def test_c(self, kwargs):
        c = C(**kwargs)
        assert_that(c.a).is_equal_to(kwargs["a"])
        assert_that(c.b).is_equal_to(kwargs["b"])
        assert_that(c.a).is_instance_of(int)
        assert_that(c.b).is_instance_of(str)
