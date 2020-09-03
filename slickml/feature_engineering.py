import numpy as np
import pandas as pd


def noisy_features(X, random_state=None, prefix=None):
    """Funtion to use the permuted rows as noisy features.
    This is function to permute the rows of a array or a dataframe
    and add them back to the as noisy features to explore stability
    of the models. In better words, we are permuting the target class.
    The input data with shape of (n, m) would be transformed
    to output data with shape of(n, 2m).
    Parameters
    ----------
    X: numpy.array() or Pandas DataFrame() or array in form of list of list
        Input data (features)
    random_state : integer, optional (default=1367)
        Random seed for randomizing the permutations
    prefix= str, optional (default="noisy")
        Prefix string that will be added to the noisy features.
    """
    X_ = X.copy()
    if isinstance(X_, np.ndarray):
        df_ = pd.DataFrame(X_, columns=[f"F_{i}" for i in range(X_.shape[1])])
    elif isinstance(X_, pd.DataFrame):
        df_ = X_
    elif isinstance(X_, list):
        df_ = pd.DataFrame(X_, columns=[f"F_{i}" for i in range(len(X_[0]))])
    else:
        raise TypeError("Only numpy arrays, lists, and Pandas DataFrames are allowed")

    if random_state is None:
        random_state = 1367

    if prefix is None:
        prefix = "noisy"

    df = df_.copy().reset_index(drop=True)

    noisy_df = df_.copy()
    noisy_cols = {col: f"{prefix}_{col}" for col in noisy_df.columns.tolist()}
    noisy_df.rename(columns=noisy_cols, inplace=True)
    np.random.seed(seed=random_state)
    noisy_df = noisy_df.reindex(np.random.permutation(noisy_df.index))

    merged_df = pd.concat([df, noisy_df.reset_index(drop=True)], axis=1)

    return merged_df

def memory_use_csr(csr):
    """Memory use in bytes by sparse matrix in csr format.
    Parameters
    ----------
    csr: sparse matric in csr format 
    """
    return csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes

def df_to_csr(df, fillna = 0., verbose = False):
    """Convert Pandas DataFrame to a sparse csr matrix.
    Parameters
    ----------
    df: Pandas DataFrame
    fillna: Value to fill null values, (default=0.0)
        Note: csr matrices assume the values have float dtype.
    verbose: Flag to show the memory usage of csr matrix, (default=False)
    """
    df_ = df.copy()
    csr = (
        df_.astype("float")\
           .fillna(fillna)\
           .to_sparse(fill_value=fillna)\
           .to_coo()\
           .tocsr()
    )
    if verbose:
        df_.info(memory_usage="deep")
        print(F"CSR Memory Usage: {memory_use_csr(csr)/2**20:.3} MB")
        
    return csr





