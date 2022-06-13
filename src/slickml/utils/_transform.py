from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def memory_use_csr(csr: csr_matrix) -> int:
    """Memory use of a Compressed Sparse Row (CSR) matrix in bytes.

    Parameters
    ----------
    csr : csr_matrix
        Compressed sparse row matrix

    Returns
    -------
    int
        Memory use in bytes

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> from slickml.utils import memory_use_csr
    >>> csr = csr_matrix((3, 4), dtype=np.int8)
    >>> mem = memory_use_csr(csr=csr)
    """
    if not isinstance(csr, csr_matrix):
        raise TypeError("The input csr must have scipy.sparse.csr_matrix dtype.")

    return csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes


# TODO(amir): currently `pydantic.validate_arguments` is in beta version and they dont supoort
# pandas.core: https://pydantic-docs.helpmanual.io/usage/validation_decorator/
# we eventually should be able to use it to decorate our functions and remove the validations pieces
def df_to_csr(
    df: pd.DataFrame,
    *,
    fillna: Optional[float] = 0.0,
    verbose: Optional[bool] = False,
) -> csr_matrix:
    """Transforms a pandas DataFrame into a Compressed Sparse Row (CSR) matrix [1]_.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    fillna : Optional[float], optional
        Value to fill nulls, by default 0.0

    verbose : Optional[bool], optional
        Whether to show the memory usage comparison of csr matrix and pandas DataFrame, by default False

    Returns
    -------
    csr_matrix
        Transformed pandas DataFrame in CSR matrix format

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

    Examples
    --------
    >>> import pandas as pd
    >>> from slickml.utils import df_to_csr
    >>> csr = df_to_csr(
    ...     df=pd.DataFrame({"foo": [0, 1, 0, 1]}),
    ...     fillna=0.0,
    ...     verbose=True,
    ... )
    """
    # TODO(amir): currently `pydantic.validate_arguments` is in beta version and they dont support
    # pandas.core: https://pydantic-docs.helpmanual.io/usage/validation_decorator/
    # we eventually should be able to use it to decorate our functions and remove the validations pieces
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input df must have pandas.DataFrame dtype.")
    if not isinstance(fillna, float):
        raise TypeError("The input fillna must have float dtype.")
    if not isinstance(verbose, bool):
        raise TypeError("The inpit verbose must have bool dtype.")

    # TODO(amir): figure out how to ditch `.copy()` across package
    df_ = df.copy()
    csr = (
        df_.astype(
            pd.SparseDtype(
                dtype="float",
                fill_value=fillna,
            ),
        )
        .sparse.to_coo()
        .tocsr()
    )
    if verbose:
        df_.info(
            verbose=True,
            memory_usage="deep",
        )
        print(f"CSR memory usage: {memory_use_csr(csr):.1f} bytes")
        print(f"CSR memory usage: {memory_use_csr(csr)/2**20:.5f} MB")

    return csr


def add_noisy_features(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    random_state: Optional[int] = 1367,
    prefix: Optional[str] = "noisy",
) -> pd.DataFrame:
    """Creates a new feature matrix augmented with noisy features via permutation.

    The main goal of this algorithm to augment permutated records as noisy features to explore the
    stability of any trained models. In principle, we are permutating the target classes. The input
    data with a shape of (n, m) would be transformed into an output data with a shape of (n, 2m).

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input features

    random_state : Optional[int], optional
        Random seed for randomizing the permutations and reproducibility, by default 1367

    prefix : Optional[str], optional
        Prefix string that will be added to the noisy features' names, by default "noisy"

    Returns
    -------
    pd.DataFrame
        Transformed feature matrix with noisy features and shape of (n, 2m)

    Examples
    --------
    >>> import pandas as pd
    >>> from slickml.utils import add_noisy_features
    >>> csr = add_noisy_features(
    ...     df=pd.DataFrame({"foo": [1, 2, 3, 4, 5]}),
    ...     random_state=1367,
    ...     prefix="noisy",
    ... )
    """
    if not isinstance(random_state, int):
        raise TypeError("The input random_state must have int dtype.")
    if not isinstance(prefix, str):
        raise TypeError("The input prefix must have str dtype.")
    # TODO(amir): figure out how to ditch `.copy()` across package
    X_ = X.copy()
    if isinstance(X_, np.ndarray):
        df_ = pd.DataFrame(
            data=X_,
            columns=[f"F_{i}" for i in range(X_.shape[1])],
        )
    elif isinstance(X_, pd.DataFrame):
        df_ = X_
    else:
        raise TypeError("The input X must have pd.DataFrame or np.ndarray dtype.")

    np.random.seed(
        seed=random_state,
    )
    df = df_.copy().reset_index(
        drop=True,
    )
    noisy_df = df_.copy()
    noisy_cols = {col: f"{prefix}_{col}" for col in noisy_df.columns.tolist()}
    noisy_df.rename(
        columns=noisy_cols,
        inplace=True,
    )
    noisy_df = noisy_df.reindex(
        np.random.permutation(
            noisy_df.index,
        ),
    )
    return pd.concat(
        [
            df,
            noisy_df.reset_index(
                drop=True,
            ),
        ],
        axis=1,
    )
