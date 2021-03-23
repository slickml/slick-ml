import numpy as np
import pandas as pd


def join_dictionaries(dict1, dict2):
    """Join two dictionaries.
    Function to join two input dictionaries. For the pairs
    with the same keys, the set of values will be stored in a list.

    Parameters
    ----------
    dict1: dict() or key-value pairs

    dict2: dict() or key-value pairs
    """
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        raise TypeError("The Type for dict1 and dict2 should be dict!")

    dictionary = {}
    d1Keys = list(dict1.keys())
    d2Keys = list(dict2.keys())
    combinedKeys = list(set(d1Keys + d2Keys))

    for key in combinedKeys:
        d1Vals = []
        d2Vals = []
        if key in d1Keys:
            d1Vals = dict1[key]
            if isinstance(d1Vals, (int, float, str)):
                d1Vals = [d1Vals]

        if key in d2Keys:
            d2Vals = dict2[key]
            if isinstance(d2Vals, (int, float, str)):
                d2Vals = [d2Vals]

        dictionary[key] = list(set(d1Vals + d2Vals))

    return dictionary


def memory_use_csr(csr):
    """Memory use in bytes by sparse matrix in csr format.

    Parameters
    ----------
    csr: sparse matric in csr format
    """
    return csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes


def df_to_csr(df, fillna=0.0, verbose=False):
    """Convert pandas.DataFrame to a sparse csr matrix.

    Parameters
    ----------
    df: pandas.DataFrame
        Input feautres

    fillna: Value to fill null values, (default=0.0)
        Note: csr matrices assume the values have float dtype.

    verbose: Flag to show the memory usage of csr matrix, (default=False)
    """
    df_ = df.copy()
    csr = df_.astype(pd.SparseDtype("float", fillna)).sparse.to_coo().tocsr()
    if verbose:
        df_.info(memory_usage="deep")
        print(f"CSR Memory Usage: {memory_use_csr(csr)/2**20:.3} MB")

    return csr


def pd_explode(df, column):
    """Function to explodes a column into columnar format.

    Parameters
    ----------
    df: panads.DataFrame
        Input features

    column: str
        Name of the column wanting to explode
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must have Pandas DataFrame dtype")
    if not isinstance(column, str):
        raise TypeError("Input column name must have str dtype")

    df_ = df.copy()
    vals = df_[column].values.tolist()
    rs = [len(r) for r in vals]
    a = np.repeat(
        df_[[col for col in df_.columns.tolist() if col != column]].values, rs, axis=0
    )
    return pd.DataFrame(np.column_stack((a, np.concatenate(vals))), columns=df_.columns)


def pd_struct_explode(df, column):

    """Function to explodes a column into columnar format.
    This is useful function when you load data from SQL or Spark
    with "struct" dtype into pandas.DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data

    column: str
        Name of the column wanting to explode
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must have Pandas DataFrame dtype")
    if not isinstance(column, str):
        raise TypeError("Input column name must have str dtype")

    df_ = df.copy()
    dicts = [val.asDict() for val in df_[column].values.tolist()]
    keys = dicts[0].keys()
    column_names = (
        str(list(df_.columns))
        .replace(f"{column}", f"{str(list(keys)).replace('[','').replace(']','')}")
        .replace("''", "'")
    )
    rs = [[d[key] for key in keys] for d in dicts]
    return pd.concat(
        [pd.DataFrame(rs, columns=keys), df_.drop([column], axis=1)], axis=1
    ).loc[:, eval(column_names)]
