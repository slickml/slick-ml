from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization

from slickml.base import BaseXGBoostEstimator
from slickml.utils import check_var


# TODO(amir): add multi-class objective for multi-lable classification
# TODO(amir): currently the `params` dont do anything. we need to make the process of combination of
# `params` and `params_pbounds` dynamic with `_xgb_eval()`
@dataclass
class XGBoostBayesianOptimizer(BaseXGBoostEstimator):
    """XGBoost Hyper-Parameters Tuner using Bayesian Optimization.

    This is wrapper using Bayesian Optimization algorithm [bayesian-optimization]_ to tune the
    hyper-parameter of XGBoost [xgboost-api]_ using ``xgboost.cv()`` functionality with n-folds
    cross-validation iteratively. This feature can be used to find the set of optimized set of
    hyper-parameters for both classification and regression tasks.

    Notes
    -----
    The optimizier objective is always to maximize the target values. Therefore, in case of using a
    metric such as ``logloss``, ``error``, ``mae``, ``rmse``, or ``rmsle``, the negative value of the
    metric will be maximized. One of the big pitfall of the current implementation is the way we are
    sampling hyper-parameters from the ``params_bounds`` where we are looking for an integer which
    is not possible. Therefore, for some of cases i.e. ``max_depth`` we must cast the sampled value
    which is mathematically wrong (i.e. ``f(1.1) != f(1)``).

    Parameters
    ----------
    n_iter : int, optional
        Number of iteration rounds for hyper-parameters tuning after initialization, by default 10

    n_init_iter : int, optional
        Number of initial iterations to initialize the optimizer, by default 5

    n_splits : int, optional
        Number of folds for cross-validation, by default 4

    metrics : str, optional
        Metrics to be tracked at cross-validation fitting time depends on the task
        (classification vs regression) with possible values of "auc", "aucpr", "error", "logloss",
        "rmse", "rmsle", "mae". Note this is different than `eval_metric` that needs to be passed to
        `params` dict, by default "auc"

    objective : str, optional
        Objective function depending on the task whether it is regression or classification. Possible
        objectives for classification ``"binary:logistic"`` and for regression ``"reg:logistic"``,
        ``"reg:squarederror"``, and ``"reg:squaredlogerror"``, by default "binary:logistic"

    acquisition_criterion : str, optional
        Acquisition criterion method with possible options of ``"ei"`` (Expected Improvement),
        ``"ucb"`` (Upper Confidence Bounds), and ``"poi"`` (Probability Of Improvement), by default "ei"

    params_bounds : Dict[str, Tuple[Union[int, float], Union[int, float]]], optional
        Set of hyper-parameters boundaries for Bayesian Optimization where all fields are required,
        by default {"max_depth" : (2, 7), "learning_rate" : (0, 1), "min_child_weight" : (1, 20),
        "colsample_bytree": (0.1, 1.0), "subsample" : (0.1, 1), "gamma" : (0, 1),
        "reg_alpha" : (0, 1), "reg_lambda" : (0, 1)}

    num_boost_round : int, optional
        Number of boosting rounds to fit a model, by default 200

    early_stopping_rounds : int, optional
        The criterion to early abort the ``xgboost.cv()`` phase if the test metric is not improved,
        by default 20

    random_state : int, optional
        Random seed number, by default 1367

    stratified : bool, optional
        Whether to use stratificaiton of the targets (only available for classification tasks) to run
        ``xgboost.cv()`` to find the best number of boosting round at each fold of each iteration,
        by default True

    shuffle : bool, optional
        Whether to shuffle data to have the ability of building stratified folds in ``xgboost.cv()``,
        by default True

    sparse_matrix : bool, optional
        Whether to convert the input features to sparse matrix with csr format or not. This would
        increase the speed of feature selection for relatively large/sparse datasets. Consequently,
        this would actually act like an un-optimize solution for dense feature matrix. Additionally,
        this parameter cannot be used along with ``scale_mean=True`` standardizing the feature matrix
        to have a mean value of zeros would turn the feature matrix into a dense matrix. Therefore,
        by default our API banned this feature, by default False

    scale_mean : bool, optional
        Whether to standarize the feauture matrix to have a mean value of zero per feature (center
        the features before scaling). As laid out in ``sparse_matrix``, ``scale_mean=False`` when
        using ``sparse_matrix=True``, since centering the feature matrix would decrease the sparsity
        and in practice it does not make any sense to use sparse matrix method and it would make
        it worse. The ``StandardScaler`` object can be accessed via ``cls.scaler_`` if ``scale_mean`` or
        ``scale_strd`` is used unless it is ``None``, by default False

    scale_std : bool, optional
        Whether to scale the feauture matrix to have unit variance (or equivalently, unit standard
        deviation) per feature. The ``StandardScaler`` object can be accessed via ``cls.scaler_``
        if ``scale_mean`` or ``scale_strd`` is used unless it is ``None``, by default False

    importance_type : str, optional
        Importance type of ``xgboost.train()`` with possible values ``"weight"``, ``"gain"``,
        ``"total_gain"``, ``"cover"``, ``"total_cover"``, by default "total_gain"

    verbose : bool, optional
        Whether to show the Bayesian Optimization progress at each iteration, by default True

    Methods
    -------
    fit(X, y)
        Fits the Bayesian optimization algorithm to tune the hyper-parameters

    get_optimizer()
        Returns the fitted Bayesian Optimiziation object

    get_results()
        Returns all the optimization results including target and params

    get_best_results()
        Return the results based on the best (tuned) hyper-parameters

    get_best_params()
        Returns the tuned hyper-parameters as a dictionary

    get_params_bounds()
        Returns the parameters boundaries

    Attributes
    ----------
    optimizer_ :
        Returns the fitted Bayesian Optimiziation object

    results_
        Returns all the optimization results including target and params

    best_params_
        Returns the tuned hyper-parameters as a dictionary

    best_results_
        Return the results based on the best (tuned) hyper-parameters

    References
    ----------
    .. [xgboost-api] https://xgboost.readthedocs.io/en/latest/python/python_api.html
    .. [bayesian-optimization] https://github.com/fmfn/BayesianOptimization
    """

    n_iter: Optional[int] = 10
    n_init_iter: Optional[int] = 5
    n_splits: Optional[int] = 4
    metrics: Optional[str] = "auc"
    objective: Optional[str] = "binary:logistic"
    acquisition_criterion: Optional[str] = "ei"
    params_bounds: Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]] = None
    num_boost_round: Optional[int] = 200
    early_stopping_rounds: Optional[int] = 20
    sparse_matrix: Optional[bool] = False
    scale_mean: Optional[bool] = False
    scale_std: Optional[bool] = False
    importance_type: Optional[str] = "total_gain"
    stratified: Optional[bool] = True
    shuffle: Optional[bool] = True
    random_state: Optional[int] = 1367
    verbose: Optional[bool] = True

    def __post_init__(self) -> None:
        """Post instantiation validations and assignments."""
        super().__post_init__()
        check_var(
            self.n_iter,
            var_name="n_iter",
            dtypes=int,
        )
        check_var(
            self.n_init_iter,
            var_name="n_iter",
            dtypes=int,
        )
        check_var(
            self.n_splits,
            var_name="n_splits",
            dtypes=int,
        )
        check_var(
            self.metrics,
            var_name="metrics",
            dtypes=str,
            values=(
                "auc",
                "aucpr",
                "error",
                "logloss",
                "rmse",
                "rmsle",
                "mae",
            ),
        )
        check_var(
            self.early_stopping_rounds,
            var_name="early_stopping_rounds",
            dtypes=int,
        )
        check_var(
            self.random_state,
            var_name="random_state",
            dtypes=int,
        )
        check_var(
            self.stratified,
            var_name="stratified",
            dtypes=bool,
        )
        check_var(
            self.shuffle,
            var_name="shuffle",
            dtypes=bool,
        )
        check_var(
            self.verbose,
            var_name="verbose",
            dtypes=bool,
        )
        # TODO(amir): use type overload
        self.verbose = self._verbose()  # type: ignore
        check_var(
            self.objective,
            var_name="objective",
            dtypes=str,
            values=(
                "binary:logistic",
                "reg:squarederror",
                "reg:squaredlogerror",
                "reg:logistic",
            ),
        )
        check_var(
            self.acquisition_criterion,
            var_name="acquisition_criterion",
            dtypes=str,
            values=(
                "ei",
                "ucb",
                "poi",
            ),
        )

        # The default set of params bounds can be updated based on the given params bounds by user
        _default_params_bounds = self._default_params_bounds()
        if self.params_bounds is not None:
            check_var(
                self.params_bounds,
                var_name="params_bounds",
                dtypes=dict,
            )
            _default_params_bounds.update(self.params_bounds)
            self.params_bounds = _default_params_bounds
        else:
            self.params_bounds = self._default_params_bounds()

        # classification/regression metrics and objectives should be aligned
        self._metrics_and_objectives_should_be_aligned()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Fits the main hyper-parameter tuning algorithm.

        Notes
        -----
        At each iteration, one set of parameters gets passed from the `params_bounds` and the
        evaluation occurs based on the cross-validation results. Bayesian optimizier always
        maximizes the objectives. Therefore, based on the `metrics` we should be careful
        when using `self.metrics` that are supposed to get minimized i.e. `error`. For those,
        we can maximize `(-1) * metric`. One of the big pitfall of the current implementation
        is the way we are sampling hyper-parameters from the `params_bounds` where we are looking
        for an integer which is not possible. Therefore, for some of cases i.e. `max_depth` we
        must cast the sampled value which is mathematically wrong (i.e. f(1.1) != f(1)).

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

        def _xgb_eval(
            max_depth: float,
            subsample: float,
            colsample_bytree: float,
            min_child_weight: float,
            learning_rate: float,
            gamma: float,
            reg_alpha: float,
            reg_lambda: float,
        ) -> float:
            """Inner hyper-parameter evaluation.

            Returns
            -------
            float
            """
            params = self._inner_params(
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                learning_rate=learning_rate,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
            )
            if self.metrics in self._clf_metrics():
                _cvr = xgb.cv(
                    params=params,
                    dtrain=self.dtrain_,
                    num_boost_round=self.num_boost_round,
                    nfold=self.n_splits,
                    stratified=self.stratified,
                    metrics=self.metrics,
                    early_stopping_rounds=self.early_stopping_rounds,
                    seed=self.random_state,
                    shuffle=self.shuffle,
                )
            else:
                _cvr = xgb.cv(
                    params=params,
                    dtrain=self.dtrain_,
                    num_boost_round=self.num_boost_round,
                    nfold=self.n_splits,
                    metrics=self.metrics,
                    early_stopping_rounds=self.early_stopping_rounds,
                    seed=self.random_state,
                    shuffle=self.shuffle,
                )

            if self.metrics in self._metrics_should_be_minimized():
                return (-1) * _cvr.iloc[-1][2]
            else:
                return _cvr.iloc[-1][2]

        self.dtrain_ = self._dtrain(
            X_train=X,
            y_train=y,
        )
        self.optimizer_ = BayesianOptimization(
            f=_xgb_eval,
            pbounds=self.params_bounds,
            random_state=self.random_state,
            verbose=self.verbose,
            constraint=None,
            bounds_transformer=None,
        )
        self.optimizer_.maximize(
            init_points=self.n_init_iter,
            n_iter=self.n_iter,
            acq=self.acquisition_criterion,
            kappa=2.576,
            kappa_decay=1,
            kappa_decay_delay=0,
            xi=0.0,
        )
        self.results_ = self.get_results()
        self.best_params_ = self.get_best_params()
        self.best_results_ = self.get_best_results()

        return None

    def get_params_bounds(self) -> Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]]:
        """Returns the hyper-parameters boundaries for the tuning process.

        Returns
        -------
        Dict[str, Tuple[Union[int, float], Union[int, float]]]
        """
        return self.params_bounds

    def get_optimizer(self) -> BayesianOptimization:
        """Return the Bayesian Optimization object.

        Returns
        -------
        BayesianOptimization
        """
        return self.optimizer_

    def get_results(self) -> pd.DataFrame:
        """Returns the hyper-parameter optimization results.

        Returns
        -------
        pd.DataFrame
        """
        frames = []
        for idx, res in enumerate(self.optimizer_.res):
            data = res["params"]
            data[self.metrics] = res["target"]
            frames.append(
                pd.DataFrame(
                    data=data,
                    index=[idx],
                ),
            )

        df_results = pd.concat(
            frames,
            axis=0,
        )
        df_results["max_depth"] = df_results["max_depth"].astype(int)

        return df_results

    def get_best_params(self) -> Dict[str, Union[str, float, int]]:
        """Returns the tuned results of the optimization as the best set of hyper-parameters.

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        targets = []
        for _, rs in enumerate(self.optimizer_.res):
            targets.append(rs["target"])
        best_params = self.optimizer_.res[targets.index(max(targets))]["params"]
        best_params["max_depth"] = int(best_params["max_depth"])

        return best_params

    def get_best_results(self) -> pd.DataFrame:
        """Returns the performance of the best (tuned) set of hyper-parameters.

        Returns
        -------
        pd.DataFrame
        """
        cond = self.results_[self.metrics] == self.results_[self.metrics].max()
        return self.results_.loc[cond, :].reset_index(drop=True)

    def _default_params_bounds(self) -> Dict[str, Tuple[Union[int, float], Union[int, float]]]:
        """Default set of parameters when the class is being instantiated with ``params_bounds=None``.

        Notes
        -----
        The default set of parameters would be a little bit different depends on the type of selection
        whether a classification or regression `metric` is being used.

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        return {
            "max_depth": (2, 7),
            "learning_rate": (0.0, 1.0),
            "min_child_weight": (1.0, 20.0),
            "colsample_bytree": (0.1, 1.0),
            "subsample": (0.1, 1.0),
            "gamma": (0.0, 1.0),
            "reg_alpha": (0.0, 1.0),
            "reg_lambda": (0.0, 1.0),
        }

    def _verbose(self) -> int:
        """Returns verbosity level based on `verbose`.

        Returns
        -------
        int
        """
        return 2 if self.verbose else 0

    def _inner_params(
        self,
        max_depth: float,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
        learning_rate: float,
        gamma: float,
        reg_alpha: float,
        reg_lambda: float,
    ) -> Dict[str, Union[str, float, int, None]]:
        """Default set of parameters passed in inner evaluation.

        Notes
        -----
        The default set of inners parameters would be a little bit different depends on the type of
        task whether a classification or regression `metric` is being used.

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        _params = {
            "eval_metric": self.metrics,
            "objective": self.objective,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "learning_rate": learning_rate,
            "max_depth": int(max_depth),
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "hist",
            "nthread": 4,
            "max_delta_step": 1,
            "verbosity": 0,
        }
        # TODO(amir): this way prolly breaks for imbalanced classification
        if self.metrics in self._clf_metrics():
            _params["scale_pos_weight"] = 1

        return _params

    def _metrics_should_be_minimized(self) -> Set[str]:
        """Returns the default metrics that should be minimized.

        Returns
        -------
        Set[str]
        """
        return {
            "error",
            "logloss",
            "rmse",
            "rmsle",
            "mae",
        }

    def _clf_metrics(self) -> Set[str]:
        """Returns the default classification metrics.

        Returns
        -------
        Set[str]
        """
        return {
            "auc",
            "aucpr",
            "error",
            "logloss",
        }

    def _clf_objectives(self) -> Set[str]:
        """Returns the default classification objectives.

        Returns
        -------
        Set[str]
        """
        return {
            "binary:logistic",
        }

    def _metrics_and_objectives_should_be_aligned(self) -> None:
        """Predicate to validate the given metric and objective are aligned.

        Raises
        ------
        ValueError

        Returns
        -------
        None
        """
        if self.metrics in self._clf_metrics() and self.objective not in self._clf_objectives():
            raise ValueError("Classification metrics cannot be used with regression objectives.")

        if self.metrics not in self._clf_metrics() and self.objective in self._clf_objectives():
            raise ValueError("Regression metrics cannot be used with classification objectives.")

        return None
