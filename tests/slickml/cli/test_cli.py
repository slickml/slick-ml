import pytest
from assertpy import assert_that
from click.testing import CliRunner

from slickml import __version__
from slickml.cli import cli
from tests.conftest import _ids


def test_cli__passes__without_args() -> None:
    """Validates the `slickml` command."""

    runner = CliRunner()
    result = runner.invoke(cli)

    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).is_instance_of(str)


@pytest.mark.parametrize(
    ("args"),
    ["--help", "-h"],
    ids=_ids,
)
def test_cli__passes__with_help_option(args: str) -> None:
    """Validates the `slickml --help | -h` command."""

    runner = CliRunner()
    result = runner.invoke(
        cli,
        args=args,
    )

    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).is_instance_of(str)
    assert_that(result.output).contains("♡♡♡ Welcome to SlickML CLI ♡♡♡")


@pytest.mark.parametrize(
    ("args"),
    ["--version", "-v"],
    ids=_ids,
)
def test_cli__passes__with_version_option(args: str) -> None:
    """Validates the `slickml --version | -v` command."""

    runner = CliRunner()
    result = runner.invoke(
        cli,
        args=args,
    )

    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).is_instance_of(str)
    assert_that(result.output).is_equal_to(f"cli, version {__version__}\n")


def test_cli__passes__with_version_command() -> None:
    """Validates the `slickml version` command."""

    runner = CliRunner()
    result = runner.invoke(
        cli,
        args="version",
    )

    assert_that(result.exit_code).is_equal_to(0)
    assert_that(result.output).is_instance_of(str)
    assert_that(result.output).contains(f"SlickML Version: {__version__}")
