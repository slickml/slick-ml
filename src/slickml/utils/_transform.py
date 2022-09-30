from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from slickml.utils._validation import check_var


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
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from slickml.utils import memory_use_csr
    >>> csr = csr_matrix((3, 4), dtype=np.int8)
    >>> mem = memory_use_csr(csr=csr)
    """
    check_var(
        csr,
        var_name="csr",
        dtypes=csr_matrix,
    )

    return csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes


def df_to_csr(
    df: pd.DataFrame,
    *,
    fillna: Optional[float] = 0.0,
    verbose: Optional[bool] = False,
) -> csr_matrix:
    """Transforms a pandas DataFrame into a Compressed Sparse Row (CSR) matrix [csr-api]_.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    fillna : float, optional
        Value to fill nulls, by default 0.0

    verbose : bool, optional
        Whether to show the memory usage comparison of csr matrix and pandas DataFrame, by default False

    Returns
    -------
    csr_matrix
        Transformed pandas DataFrame in CSR matrix format

    Notes
    -----
    This utility function is being used across API when the ``sparse_matrix=True`` for all
    classifiers and regressors. In practice, when we are dealing with sparse matrices, it does make
    sense to employ this functionality. It should be noted that using sparse matrices when the input
    matrix is dense would actually end up using more memory. This can be checked by passing
    ``verbose=True`` option or using ``memory_use_csr()`` function directly on top of your csr
    matrix. Additionally, you can compare the memory usage of the csr matrix with the input
    ``pandas.DataFrame`` via ``df.memory_usage().sum()``.

    References
    ----------
    .. [csr-api] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

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
    check_var(
        df,
        var_name="df",
        dtypes=pd.DataFrame,
    )
    check_var(
        fillna,
        var_name="fillna",
        dtypes=float,
    )
    check_var(
        verbose,
        var_name="verbose",
        dtypes=bool,
    )

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


def array_to_df(
    X: np.ndarray,
    *,
    prefix: Optional[str] = "F",
    delimiter: Optional[str] = "_",
) -> pd.DataFrame:
    """Transforms a numpy array into a pandas DataFrame.

    The ``prefix`` and ``delimiter`` along with the index of each column (0-based index) of the
    array are used to create the columnnames of the DataFrame.

    Parameters
    ----------
    X : np.ndarray
        Input array

    prefix : str, optional
        Prefix string for each column name, by default "F"

    delimiter : str, optional
        Delimiter to separate prefix and index number, by default "_"

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> import numpy as np
    >>> from slickml.utils import array_to_df
    >>> df = array_to_df(
    ...     X=np.array([1, 2, 3]),
    ...     prefix="F",
    ...     delimiter="_",
    ... )

    """
    check_var(
        X,
        var_name="X",
        dtypes=np.ndarray,
    )
    check_var(
        prefix,
        var_name="prefix",
        dtypes=str,
    )
    check_var(
        delimiter,
        var_name="delimiter",
        dtypes=str,
    )

    X_ = X
    if X_.ndim == 1:
        X_ = X_.reshape(1, -1)

    return pd.DataFrame(
        data=X_,
        columns=[f"{prefix}{delimiter}{i}" for i in range(X_.shape[1])],
    )


# TODO(amir): add functionality for List[List[float]] as the input data as well
def add_noisy_features(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    random_state: Optional[int] = 1367,
    prefix: Optional[str] = "noisy",
) -> pd.DataFrame:
    """Creates a new feature matrix augmented with noisy features via permutation.

    The main goal of this algorithm to augment permutated records as noisy features to explore the
    stability of any trained models. In principle, we are permutating the target classes. The input
    data with a shape of ``(n, m)`` would be transformed into an output data with a shape of
    ``(n, 2m)``.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input features

    random_state : int, optional
        Random seed for randomizing the permutations and reproducibility, by default 1367

    prefix : str, optional
        Prefix string that will be added to the noisy features' names, by default "noisy"

    Returns
    -------
    pd.DataFrame
        Transformed feature matrix with noisy features and shape of (n, 2m)

    Examples
    --------
    >>> import pandas as pd
    >>> from slickml.utils import add_noisy_features
    >>> df_noisy = add_noisy_features(
    ...     df=pd.DataFrame({"foo": [1, 2, 3, 4, 5]}),
    ...     random_state=1367,
    ...     prefix="noisy",
    ... )
    """
    check_var(
        X,
        var_name="X",
        dtypes=(pd.DataFrame, np.ndarray),
    )
    check_var(
        random_state,
        var_name="random_state",
        dtypes=int,
    )
    check_var(
        prefix,
        var_name="prefix",
        dtypes=str,
    )

    df_ = X
    if isinstance(df_, np.ndarray):
        df_ = array_to_df(
            df_,
            prefix="F",
            delimiter="_",
        )

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
