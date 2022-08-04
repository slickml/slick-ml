from enum import Enum

from assertpy import assert_that

from slickml.utils import Colors


class TestColors:
    """Validates Colors instantiation."""

    def test_colors_instantiation__passes__with_default_values(self) -> None:
        """Validates the test enum class Colors instantiation and properties."""

        assert_that(Colors.BLUE).is_instance_of(Enum)
        assert_that(Colors.RED).is_instance_of(Enum)
        assert_that(Colors.BOLD).is_instance_of(Enum)
        assert_that(Colors.UNDERLINE).is_instance_of(Enum)
        assert_that(Colors.END).is_instance_of(Enum)
        assert_that(Colors.BLUE.value).is_instance_of(str)
        assert_that(Colors.RED.value).is_instance_of(str)
        assert_that(Colors.BOLD.value).is_instance_of(str)
        assert_that(Colors.UNDERLINE.value).is_instance_of(str)
        assert_that(Colors.END.value).is_instance_of(str)
        assert_that(Colors.BLUE.name).is_instance_of(str)
        assert_that(Colors.RED.name).is_instance_of(str)
        assert_that(Colors.BOLD.name).is_instance_of(str)
        assert_that(Colors.UNDERLINE.name).is_instance_of(str)
        assert_that(Colors.END.name).is_instance_of(str)
        assert_that(str(Colors.BLUE)).is_instance_of(str)
        assert_that(str(Colors.RED)).is_instance_of(str)
        assert_that(str(Colors.BOLD)).is_instance_of(str)
        assert_that(str(Colors.UNDERLINE)).is_instance_of(str)
        assert_that(str(Colors.END)).is_instance_of(str)
        assert_that(Colors.names()).is_instance_of(list)
        assert_that(Colors.names()).is_iterable()
        assert_that(Colors.values()).is_instance_of(list)
        assert_that(Colors.values()).is_iterable()
        assert_that(Colors.pairs()).is_instance_of(dict)
        assert_that(Colors.pairs()).is_iterable()
        assert_that(Colors.names()).contains(
            "PURPLE",
            "END",
        )
        assert_that(Colors.values()).contains(
            "\x1b[39m",
            "\x1b[90m",
        )
        assert_that(Colors.pairs()).is_equal_to(
            {
                "PURPLE": "\033[95m",
                "END": "\033[0m",
            },
            include=(
                "PURPLE",
                "END",
            ),
        )
