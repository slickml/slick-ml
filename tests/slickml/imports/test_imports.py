import re
from importlib.metadata import distribution
from typing import List

import pytest
from assertpy import assert_that


def _get_valid_dependencies(dist_name: str = "slickml") -> List[str]:
    """Helper function to retrieve the package dependencies names from package metadata."""
    # map the libraries that their installation name differs from imported name (i.e. plugins)
    mapping = {
        "bayesian-optimization": "bayes_opt",
        "scikit-learn": "sklearn",
        "python-glmnet": "glmnet",
    }
    # TODO(amir): mypy: https://stackoverflow.com/questions/61712618/why-does-mypy-not-consider-a-class-as-iterable-if-it-has-len-and-getitem
    libs: List[str] = []
    for dep in distribution(dist_name).requires or []:  # type: ignore[union-attr]
        # Skip optional extras (dev/docs); only validate runtime Requires-Dist entries
        if "extra ==" in dep:
            continue
        name = re.split(r"[><=!]", dep, maxsplit=1)[0].strip()
        libs.append(mapping.get(name, name))
    return list(set(libs))


def _get_invalid_dependencies() -> List[str]:
    """Helper function to return not-required / not-availabe dependencies cannot be imported."""
    return [
        "cookiecutter",
        "pyarrow",
        "plotly",
        "foo",
        "bar",
    ]


class TestImports:
    """Validates required and not-required dependencies imports."""

    @pytest.mark.parametrize(
        "package",
        _get_valid_dependencies(),
    )
    def test_imports__passes__with_valid_dependencies(self, package: str) -> None:
        """Validates all the required dependencies are imported successfully."""
        assert_that(__import__(package)).is_true()
        assert_that(__import__(package)).is_not_none()

    @pytest.mark.parametrize(
        "package",
        _get_invalid_dependencies(),
    )
    def test_imports__fails__with_invalid_dependencies(self, package: str) -> None:
        """Validates that not-required / not-availabe dependencies cannot be imported."""
        with pytest.raises(ModuleNotFoundError):
            __import__(package)
