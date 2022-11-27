from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

import hyperopt
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt.pyll import stochastic

from slickml.base import BaseXGBoostEstimator
from slickml.utils import check_var


@dataclass
class XGBoostHyperOptimizer(BaseXGBoostEstimator):
    """XGBoost Hyper-Parameters Tuner using HyperOpt Optimization.

    This is wrapper using HyperOpt [hyperopt]_ a Python library for serial and parallel optimization
    over search spaces, which may include real-valued, discrete, and conditional dimensions to tune the
    hyper-parameter of XGBoost [xgboost-api]_ using ``xgboost.cv()`` functionality with n-folds
    cross-validation iteratively. This feature can be used to find the set of optimized set of
    hyper-parameters for both classification and regression tasks.

    Notes
    -----
    The optimizier objective is always to minimize the target values. Therefore, in case of using a
    metric such as ``auc``, or ``aucpr`` the negative value of the metric will be minimized.

    Parameters
    ----------
    n_iter : int, optional
        Maximum number of iteration rounds for hyper-parameters tuning before convergance, by default 100

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

    params_bounds : Dict[str, Any], optional
        Set of hyper-parameters boundaries for HyperOpt using``hyperopt.hp`` and `hyperopt.pyll_utils`,
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
        Whether to show the HyperOpt Optimization progress at each iteration, by default True

    Methods
    -------
    fit(X, y)
        Fits the HyperOpt optimization algorithm to tune the hyper-parameters

    get_best_params()
        Returns the tuned hyper-parameters as a dictionary

    get_results()
        Returns all the optimization trials

    get_trials()
        Return the trials object

    get_params_bounds()
        Returns the parameters boundaries

    Attributes
    ----------
    best_params_
        Returns the tuned hyper-parameters as a dictionary

    results_
        Returns all the optimization trials as results

    References
    ----------
    .. [xgboost-api] https://xgboost.readthedocs.io/en/latest/python/python_api.html
    .. [hyperopt] https://github.com/hyperopt/hyperopt
    """

    n_iter: Optional[int] = 100
    n_splits: Optional[int] = 4
    metrics: Optional[str] = "auc"
    num_boost_round: Optional[int] = 200
    objective: Optional[str] = "binary:logistic"
    params_bounds: Optional[Dict[str, Any]] = None
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
            var_name="n_inter",
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
        # The default set of params bounds can be updated based on the given params bounds by user
        _default_params_bounds = self._default_params_bounds()
        if self.params_bounds is not None:
            check_var(
                self.params_bounds,
                var_name="params_bounds",
                dtypes=dict,
            )
            # TODO(amir): here we update the defaults while the user only wants to test out a small
            # search space; we can also just let the user decide; this is doable since the inner
            # scope only needs `space`
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
        evaluation occurs based on the cross-validation results. Hyper optimizier always
        minimizes the objectives. Therefore, based on the `metrics` we should be careful
        when using `self.metrics` that are supposed to get maximized i.e. `auc`. For those,
        we can maximize `(-1) * metric`.

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

        def _xgb_eval(space: Dict[str, Any]) -> Dict[str, Union[float, str]]:
            """Inner hyper-parameter evaluation.

            Returns
            -------
            Dict[str, Union[float, str]]
            """
            params = stochastic.sample(space)
            params.update(
                self._inner_params(),
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
                loss = _cvr.iloc[-1][2]
            else:
                loss = (-1) * _cvr.iloc[-1][2]

            return {
                "loss": loss,
                "status": hyperopt.STATUS_OK,
            }

        self.dtrain_ = self._dtrain(
            X_train=X,
            y_train=y,
        )
        self.trials_ = hyperopt.Trials()

        try:
            self.best_params_ = hyperopt.fmin(
                fn=_xgb_eval,
                space=self.params_bounds,
                algo=hyperopt.tpe.suggest,
                max_evals=self.n_iter,
                trials=self.trials_,
                verbose=self.verbose,
                rstate=None,
                allow_trials_fmin=True,
                catch_eval_exceptions=False,
                return_argmin=True,
                max_queue_len=1,
                timeout=None,
                loss_threshold=None,
                pass_expr_memo_ctrl=None,
                points_to_evaluate=None,
                show_progressbar=True,
                early_stop_fn=None,
                trials_save_file="",
            )
            self.results_ = self.trials_.trials
        # TODO(amir): log error
        except Exception as e:
            self.best_params_ = {
                "status": hyperopt.STATUS_FAIL,
                "exception": str(e),
            }

        return None

    def get_best_params(self) -> Dict[str, Union[str, float, int]]:
        """Returns the tuned results of the optimization as the best set of hyper-parameters.

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        return self.best_params_

    def get_results(self) -> List[Dict[str, Any]]:
        """Return all trials results.

        Returns
        -------
        List[Dict[str, Any]]
        """
        return self.results_

    def get_trials(self) -> hyperopt.Trials:
        """Returns the `Trials` object passed to the optimizer.

        Returns
        -------
        hyperopt.Trials
        """
        return self.trials_

    def get_params_bounds(self) -> Optional[Dict[str, Any]]:
        """Returns the hyper-parameters boundaries for the tuning process.

        Returns
        -------
        Dict[str, Any]
        """
        return self.params_bounds

    # TODO(amir): check the type checker for return type
    def _default_params_bounds(self) -> Dict[str, Any]:
        """Default set of parameters when the class is being instantiated with ``params_bounds=None``.

        Notes
        -----
        The default set of parameters would be a little bit different depends on the type of selection
        whether a classification or regression `metric` is being used.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "max_depth": hyperopt.hp.choice("max_depth", range(2, 7)),
            "learning_rate": hyperopt.hp.quniform("learning_rate", 0.01, 1.0, 0.01),
            "min_child_weight": hyperopt.hp.quniform("min_child_weight", 1.0, 20.0, 1),
            "colsample_bytree": hyperopt.hp.quniform("colsample_bytree", 0.1, 1.0, 0.01),
            "subsample": hyperopt.hp.quniform("subsample", 0.1, 1, 0.01),
            "gamma": hyperopt.hp.quniform("gamma", 0.0, 1.0, 0.01),
            "reg_alpha": hyperopt.hp.quniform("reg_alpha", 0.0, 1.0, 0.01),
            "reg_lambda": hyperopt.hp.quniform("reg_lambda", 0.0, 1.0, 0.01),
        }

    def _inner_params(self) -> Dict[str, Union[str, float, int]]:
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
            "tree_method": "hist",
            "nthread": 4,
            "max_delta_step": 1,
            "verbosity": 0,
        }
        # TODO(amir): this way prolly breaks for imbalanced classification
        if self.metrics in self._clf_metrics():
            _params["scale_pos_weight"] = 1

        return _params  # type: ignore

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
