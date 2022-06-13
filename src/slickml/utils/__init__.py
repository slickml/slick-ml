"""

```df_to_csr```

Transforms a pandas DataFrame into a Compressed Sparse Row (CSR) matrix [1]_.

Notes
-----
This utility function is being used across API when the `sparse_matrix=True` for all classifiers
and regressors. In practice, when we are dealing with sparse matrices, it does make sense to employ
this functionality. It should be noted that using sparse matrices when the input matrix is dense
would actually end up using more momory. This can be checked by passing `verbose=True` or using
`memory_use_csr()` function directly on top of your csr matrix. In addition, you can compare the
memory usage of the csr matrix with the input pandas DataFrame via `df.memory_usage().sum()`.

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


```memory_use_csr```

Calculates the memory usage of a Compressed Sparse Row (CSR) matrix in bytes.

Examples
--------
>>> from scipy.sparse import csr_matrix
>>> from slickml.utils import memory_use_csr
>>> csr = csr_matrix((3, 4), dtype=np.int8)
>>> mem = memory_use_csr(csr=csr)


```add_noisy_features`

Creates a new feature matrix augmented with noisy features via permutation. The main goal of this
algorithm is to augment permutated records as noisy features to explore the stability of any trained
model. In principle, we are permutating the target classes. The input data with a shape of (n, m)
would be transformed into an output data with a shape of (n, 2m).

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

from slickml.utils._transform import add_noisy_features, df_to_csr, memory_use_csr

__all__ = [
    "memory_use_csr",
    "df_to_csr",
    "add_noisy_features",
]
