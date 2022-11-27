import importlib.resources as pkg_resources
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from pytest import CaptureFixture, FixtureRequest
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from tests import resources


@pytest.fixture(scope="session")
def clf_train_test_x_y(
    request: FixtureRequest,
) -> Optional[
    Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[np.ndarray, List[float]],
        Union[np.ndarray, List[float]],
    ]
]:
    """Returns stratified train/test features/targets sets as a `pytest.fixture` for binary classification problems.

    Parameters
    ----------
    request : FixtureRequest
        Fixture request for params

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray],
          Union[pd.DataFrame, np.ndarray],
          Union[np.ndarray, List],
          Union[np.ndarray, List],
         ]
    """
    df = _load_test_data_from_csv(
        filename="clf_test_data.csv",
    )
    y = df["CLASS"].values
    X = df.drop(
        ["CLASS"],
        axis=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=1367,
    )
    if request.param == "dataframe":
        return (X_train, X_test, y_train, y_test)
    elif request.param == "array":
        return (X_train.values, X_test.values, y_train, y_test)
    elif request.param == "list":
        return (X_train, X_test, y_train.tolist(), y_test.tolist())
    else:
        return None


@pytest.fixture(scope="session")
def clf_x_y(
    request: FixtureRequest,
) -> Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[np.ndarray, List[float]]]]:
    """Returns features/targets sets a `pytest.fixture` for binary classification problems.

    Parameters
    ----------
    request : FixtureRequest
        Fixture request for params

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[np.ndarray, List]]
    """
    df = _load_test_data_from_csv(
        filename="clf_test_data.csv",
    )
    y = df["CLASS"].values
    X = df.drop(
        ["CLASS"],
        axis=1,
    )
    if request.param == "dataframe":
        return (X, y)
    elif request.param == "array":
        return (X.values, y)
    elif request.param == "list":
        return (X, y.tolist())
    else:
        return None


@pytest.fixture(scope="session")
def reg_train_test_x_y(
    request: FixtureRequest,
) -> Optional[
    Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        Union[np.ndarray, List[float]],
        Union[np.ndarray, List[float]],
    ]
]:
    """Returns train/test features/targets sets as a `pytest.fixture` for regression problems.

    Parameters
    ----------
    request : FixtureRequest
        Fixture request for params

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray],
          Union[pd.DataFrame, np.ndarray],
          Union[np.ndarray, List],
          Union[np.ndarray, List],
         ]
    """
    df = _load_test_data_from_csv(
        filename="reg_test_data.csv",
    )
    # TODO(amir): try to pull-out multi target regression as well here
    y = df["TARGET1"].values
    X = df.drop(
        ["TARGET1", "TARGET2"],
        axis=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=1367,
    )
    if request.param == "dataframe":
        return (X_train, X_test, y_train, y_test)
    elif request.param == "array":
        return (X_train.values, X_test.values, y_train, y_test)
    elif request.param == "list":
        return (X_train, X_test, y_train.tolist(), y_test.tolist())
    else:
        return None


@pytest.fixture(scope="session")
def reg_x_y(
    request: FixtureRequest,
) -> Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[np.ndarray, List[float]]]]:
    """Returns features/targets sets a `pytest.fixture` for regression problems.

    Parameters
    ----------
    request : FixtureRequest
        Fixture request for params

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[np.ndarray, List]]
    """
    df = _load_test_data_from_csv(
        filename="reg_test_data.csv",
    )
    # TODO(amir): try to pull-out multi target regression as well here
    y = df["TARGET1"].values
    X = df.drop(
        ["TARGET1", "TARGET2"],
        axis=1,
    )
    if request.param == "dataframe":
        return (X, y)
    elif request.param == "array":
        return (X.values, y)
    elif request.param == "list":
        return (X, y.tolist())
    else:
        return None


@pytest.fixture(scope="session")
def datafarame_for_testing() -> pd.DataFrame:
    """Returns a `pandas.DataFrame` as `pytest.fixture`.

    Returns
    -------
    pd.DataFrame
    """
    return _dummy_pandas_dataframe(
        size=100,
        random_state=1367,
    )


@pytest.fixture(scope="session")
def sparse_matrix_for_testing() -> csr_matrix:
    """Returns a `scipy.csr_matrix` as `pytest.fixture`.

    Returns
    -------
    csr_matrix
    """
    return _dummy_sparse_matrix()


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

    Parameters
    ----------
    capsys : CaptureFixture
        Pytest capture fixture to read output and error

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

    Notes
    -----
    The DataFrame shape is (size, 4), two features ("feature_1", "feature_2"), and two targets
    ("binary_target", "multi_target").

    Parameters
    ----------
    size : int, optional
        Number of samples, by default 100

    random_state : int, optional
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

    Notes
    -----
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
