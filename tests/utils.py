import importlib.resources as pkg_resources
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pytest import CaptureFixture
from scipy.sparse import csr_matrix

from tests import resources


# TODO(amir): what if values is list ?
def _ids(values: Any) -> str:
    """Returns a user-friendly test case ID from the parametrized values.

    Parameters
    ----------
    values : Any
        Test resource values

    Returns
    -------
    str
    """
    if isinstance(values, dict):
        return ", ".join(f"{k} : {v}" for (k, v) in values.items())
    else:
        return str(values)


def _load_test_scenarios_from_json(filename: str) -> Dict[str, Any]:
    """Returns a json file contains valid and invalid test cases that can be used for `pytest.fixtures`.

    Parameters
    ----------
    filename : str
        Json filename

    Returns
    -------
    Dict[str, Any]
    """
    return json.loads(
        pkg_resources.read_text(
            resources,
            filename,
        ),
    )


def _load_test_data_from_csv(filename: str) -> pd.DataFrame:
    """Returns a `pandas.DataFrame` data loaded from a csv file that can be used for `pytest.fixtures`.

    Parameters
    ----------
    filename : str
        Data filename

    Returns
    -------
    pd.DataFrame
    """
    with pkg_resources.path(resources, filename) as path:
        return pd.read_csv(path)


def _captured_log(capsys: CaptureFixture) -> Tuple[str, str]:
    """Returns the captured standard output/error via `pytest.capsys` [1]_.

    References
    ----------
    .. [1] https://docs.pytest.org/en/7.1.x/how-to/capture-stdout-stderr.html

    Returns
    -------
    Tuple[str]
        Captured output and caputred error
    """
    captured = capsys.readouterr()
    return (captured.out, captured.err)


def _dummy_pandas_dataframe(
    size: Optional[int] = 100,
    random_state: Optional[int] = 1367,
) -> pd.DataFrame:
    """Returns a dummy pandas DataFrame that can be used for `pytest.fixtures`.

    The DataFrame shape is (size, 4), two features ("feature_1", "feature_2"), and two targets
    ("binary_target", "multi_target").

    Parameters
    ----------
    size : Optional[int], optional
        Number of samples, by default 100

    random_state : Optional[int], optional
        Random seed, by default 1367

    Returns
    -------
    pd.DataFrame
    """
    np.random.seed(
        seed=random_state,
    )
    return pd.DataFrame(
        {
            "feature_1": np.random.random_sample(
                size=size,
            ),
            "feature_2": np.random.random_sample(
                size=size,
            ),
            "binary_target": np.random.randint(
                low=0,
                high=2,
                size=size,
                dtype=int,
            ),
            "multi_target": np.random.randint(
                low=0,
                high=3,
                size=size,
                dtype=int,
            ),
        },
    )


def _dummy_sparse_matrix() -> csr_matrix:
    """Returns a sparse matrix in CSR format with a shape of (3,3) with float entries.

    The numpy representation `_dummy_sparse_matrix().toarray()` is as follows:
    array([[1., 0., 2.],
           [0., 0., 3.],
           [4., 5., 6.]])

    Returns
    -------
    csr_matrix
    """
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    return csr_matrix(
        (data, (row, col)),
        shape=(3, 3),
        dtype=np.float64,
    )
