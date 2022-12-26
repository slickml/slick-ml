from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from slickml.utils._transform import array_to_df, check_var, df_to_csr


@dataclass
class BaseXGBoostEstimator(ABC, BaseEstimator):
    """Base Estimator for XGBoost.

    Notes
    -----
    This is an `abstractbaseclass` using XGBoost [xgboost-api]_ that can be used for any estimator using
    XGBoost as the base estimator such as ``XGBoostCVClassifier``, ``XGBoostRegressor``,
    ``XGBoostFeatureSelector``, ``XGBoostBayesianOptimizer``, and so on. This base estimator comes
    with the base validation utilities that can reduce the amount of copy/paste codes in the
    downstream classes.

    Parameters
    ----------
    num_boost_round : int
        Number of boosting rounds to fit a model

    sparse_matrix : bool
        Whether to convert the input features to sparse matrix with csr format or not. This would
        increase the speed of feature selection for relatively large/sparse datasets. Consequently,
        this would actually act like an un-optimize solution for dense feature matrix. Additionally,
        this parameter cannot be used along with ``scale_mean=True`` standardizing the feature matrix
        to have a mean value of zeros would turn the feature matrix into a dense matrix. Therefore,
        by default our API banned this feature

    scale_mean : bool
        Whether to standarize the feauture matrix to have a mean value of zero per feature (center
        the features before scaling). As laid out in ``sparse_matrix``, ``scale_mean=False`` when
        using ``sparse_matrix=True``, since centering the feature matrix would decrease the sparsity
        and in practice it does not make any sense to use sparse matrix method and it would make
        it worse. The ``StandardScaler`` object can be accessed via ``cls.scaler_`` if ``scale_mean`` or
        ``scale_strd`` is used unless it is ``None``

    scale_std : bool
        Whether to scale the feauture matrix to have unit variance (or equivalently, unit standard
        deviation) per feature. The ``StandardScaler`` object can be accessed via ``cls.scaler_``
        if ``scale_mean`` or ``scale_strd`` is used unless it is ``None``

    importance_type : str
        Importance type of ``xgboost.train()`` with possible values ``"weight"``, ``"gain"``,
        ``"total_gain"``, ``"cover"``, ``"total_cover"``

    params : Dict[str, Union[str, float, int]], optional
        Set of parameters required for fitting a Booster

    Methods
    -------
    fit(X, y)
        Abstract method to fit a model to the features/target depend on the task

    References
    ----------
    .. [xgboost-api] https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """

    num_boost_round: Optional[int]
    sparse_matrix: Optional[bool]
    scale_mean: Optional[bool]
    scale_std: Optional[bool]
    importance_type: Optional[str]
    params: Optional[Dict[str, Union[str, float, int]]] = None

    def __post_init__(self) -> None:
        """Post instantiation validations and assignments."""
        check_var(
            self.num_boost_round,
            var_name="num_boost_round",
            dtypes=int,
        )
        check_var(
            self.sparse_matrix,
            var_name="sparse_matrix",
            dtypes=bool,
        )
        check_var(
            self.scale_mean,
            var_name="scale_mean",
            dtypes=bool,
        )
        check_var(
            self.scale_std,
            var_name="scale_std",
            dtypes=bool,
        )
        check_var(
            self.importance_type,
            var_name="importance_type",
            dtypes=str,
            values=(
                "weight",
                "gain",
                "total_gain",
                "cover",
                "total_cover",
            ),
        )
        # The `StandardScaler` with `mean=True` would turn a sparse matrix into a dense matrix
        if self.sparse_matrix and self.scale_mean:
            raise ValueError(
                "The scale_mean should be False in conjuction of using sparse_matrix=True.",
            )

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """`Abstractmethod` to fit a model to the features/targets depends on the task.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input data for training (features)

        y : Union[List[float], np.ndarray, pd.Series]
            Input ground truth for training (targets)

        Returns
        -------
        None
        """
        ...  # pragma: no cover

    # TODO(amir): check the `y_train` type; maybe we need to have `list_to_array()` in utils?
    def _dtrain(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[List[float], np.ndarray, pd.Series],
    ) -> xgb.DMatrix:
        """Returns a proper dtrain matrix compatible with sparse/standardized matrices.

        Parameters
        ----------
        X_train : Union[pd.DataFrame, np.ndarray]
            Input data for training (features)

        y_train : Union[List[float], np.ndarray, pd.Series]
            Input ground truth for training (targets)

        See Also
        --------
        :meth:`_dtest()`

        Returns
        -------
        xgb.DMatrix
        """
        check_var(
            X_train,
            var_name="X_train",
            dtypes=(
                pd.DataFrame,
                np.ndarray,
            ),
        )
        check_var(
            y_train,
            var_name="y_train",
            dtypes=(
                list,
                np.ndarray,
                pd.Series,
            ),
        )

        if isinstance(X_train, np.ndarray):
            self.X_train = array_to_df(
                X=X_train,
                prefix="F",
                delimiter="_",
            )
        else:
            self.X_train = X_train

        if not isinstance(y_train, np.ndarray):
            self.y_train = np.array(y_train)
        else:
            self.y_train = y_train

        # TODO(amir): move `StandardScaler` to utils
        if self.scale_mean or self.scale_std:
            self.scaler_ = StandardScaler(
                copy=True,
                with_mean=self.scale_mean,
                with_std=self.scale_std,
            )
            self.X_train_ = pd.DataFrame(
                self.scaler_.fit_transform(self.X_train),
                columns=self.X_train.columns.tolist(),
            )
        else:
            self.scaler_ = None
            self.X_train_ = self.X_train

        if not self.sparse_matrix:
            dtrain = xgb.DMatrix(
                data=self.X_train_,
                label=self.y_train,
            )
        else:
            dtrain = xgb.DMatrix(
                data=df_to_csr(
                    self.X_train_,
                    fillna=0.0,
                    verbose=False,
                ),
                label=self.y_train,
                feature_names=self.X_train_.columns.tolist(),
            )

        return dtrain

    def _dtest(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    ) -> xgb.DMatrix:
        """Returns a proper dtest matrix compatible with sparse/standardized matrices.

        If ``scale_mean=True`` or ``scale_std=True``, the ``StandardScaler`` object ``(scaler_)``
        which is being fitted on ``X_train`` will be used to **only** transform ``X_test`` to make
        sure there is no data leak in the transformation. Additionally, ``y_test`` is optional since
        it might not be available while validating the model (inference).

        Parameters
        ----------
        X_test : Union[pd.DataFrame, np.ndarray]
            Input data for testing (features)

        y_test : Union[List[float], np.ndarray, pd.Series], optional
            Input ground truth for testing (targets)

        See Also
        --------
        :meth:`_dtrain()`

        Returns
        -------
        xgb.DMatrix
        """
        check_var(
            X_test,
            var_name="X_test",
            dtypes=(
                pd.DataFrame,
                np.ndarray,
            ),
        )
        if y_test is not None:
            check_var(
                y_test,
                var_name="y_test",
                dtypes=(
                    list,
                    np.ndarray,
                    pd.Series,
                ),
            )
            if not isinstance(y_test, np.ndarray):
                self.y_test = np.array(y_test)
            else:
                self.y_test = y_test
        else:
            self.y_test = y_test

        if isinstance(X_test, np.ndarray):
            self.X_test = array_to_df(
                X=X_test,
                prefix="F",
                delimiter="_",
            )
        else:
            self.X_test = X_test

        if self.scale_mean or self.scale_std:
            self.X_test_ = pd.DataFrame(
                self.scaler_.transform(self.X_test),
                columns=self.X_test.columns.tolist(),
            )
        else:
            self.X_test_ = self.X_test

        if not self.sparse_matrix:
            dtest = xgb.DMatrix(
                data=self.X_test_,
                label=self.y_test,
            )
        else:
            dtest = xgb.DMatrix(
                data=df_to_csr(
                    self.X_test_,
                    fillna=0.0,
                    verbose=False,
                ),
                label=self.y_test,
                feature_names=self.X_test_.columns.tolist(),
            )

        return dtest

    def _check_X_y(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Validates/pre-processes the input matrices (features/targets).

        Returns
        -------
        None
        """
        check_var(
            X,
            var_name="X",
            dtypes=(
                pd.DataFrame,
                np.ndarray,
            ),
        )
        check_var(
            y,
            var_name="y",
            dtypes=(
                list,
                np.ndarray,
                pd.Series,
            ),
        )

        if isinstance(X, np.ndarray):
            self.X = array_to_df(
                X=X,
                prefix="F",
                delimiter="_",
            )
        else:
            self.X = X

        if not isinstance(y, np.ndarray):
            self.y = np.array(y)
        else:
            self.y = y

        return None
