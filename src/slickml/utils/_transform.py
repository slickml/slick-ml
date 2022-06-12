from typing import Optional

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
    ...     df=pd.DataFrame(
    ...         {
    ...             "foo": [42],
    ...         },
    ...     )
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
