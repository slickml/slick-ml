import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll.stochastic import sample

from slickml.classification import XGBoostClassifier
from slickml.regression import XGBoostRegressor


class XGBoostClassifierHyperOpt(XGBoostClassifier):
    """XGBoost Classifier - Hyperparameter Optimization.
    This class uses HyperOpt, a Python library for serial and parallel
    optimization over search spaces, which may include real-valued, discrete,
    and conditional dimensions to train a XGBoost model.Main reference is
    HyperOpt GitHub: (https://github.com/hyperopt/hyperopt)

    Parameters
    ----------
    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("auc"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "auc", "aucpr", "error", "logloss"

    early_stopping_rounds: int, optional (default=20)
        The criterion to early abort the xgboost.cv() phase
        if the test metric is not improved

    random_state: int, optional (default=1367)
        Random seed

    stratified: bool, optional (default=True)
        Flag to stratificaiton of the targets to run xgboost.cv() to
        find the best number of boosting round at each fold of
        each iteration

    shuffle: bool, optional (default=True)
        Flag to shuffle data to have the ability of building
        stratified folds in xgboost.cv()

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    func_name: str
        Function name for performing optimization

    space: dict()
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp

    trials: HyperOpt Object
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    algo: HyperOpt Object
        Provides logic for sequential search of the hyperparameter space

    max_evals: int
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    verbose: str, optional (default=False)
        Print evaluation results from model

    Attributes
    ----------
    fit(X_train, y_train): instance method
        Returns None and applies the tuning process using
        the (X_train, y_train) and the given set of hyperparameters

    xgb_cv(space): instance method
        Optimization function for XGBoost utilizing cross-validation
        based on space params

    get_optimization_results(): instance method
        Returns pd.DataFrame for best parameters from all runs

    get_optimization_trials(): instance method
        Returns dict of best parameters for each individual trial run
    """

    def __init__(
        self,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        early_stopping_rounds=None,
        random_state=None,
        stratified=True,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        func_name=None,
        space=None,
        max_evals=None,
        verbose=False,
    ):
        super().__init__(
            num_boost_round,
            metrics,
            sparse_matrix,
            scale_mean,
            scale_std,
        )

        if n_splits is None:
            self.n_splits = 4
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must have integer dtype.")
            else:
                self.n_splits = n_splits

        if early_stopping_rounds is None:
            self.early_stopping_rounds = 20
        else:
            if not isinstance(early_stopping_rounds, int):
                raise TypeError("The input early_stopping_rounds must have integer dtype.")
            else:
                self.early_stopping_rounds = early_stopping_rounds

        if random_state is None:
            self.random_state = 1367
        else:
            if not isinstance(random_state, int):
                raise TypeError("The input random_state must have integer dtype.")
            else:
                self.random_state = random_state

        if not isinstance(stratified, bool):
            raise TypeError("The input stratified must have bool dtype.")
        else:
            self.stratified = stratified

        if not isinstance(shuffle, bool):
            raise TypeError("The input shuffle must have bool dtype.")
        else:
            self.shuffle = shuffle

        if isinstance(func_name, str):
            self.fn = getattr(self, func_name)
        else:
            raise TypeError("The input must be a valid function name: 'xgb_cv'.")

        if isinstance(space, dict):
            self.space = space
        else:
            raise TypeError("The input space must have dict dtype.")

        if max_evals is None:
            self.max_evals = 100
        else:
            if not isinstance(max_evals, int):
                raise TypeError("The input max_evals must have integer dtype.")
            else:
                self.max_evals = max_evals

        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError("The input verbose must be a boolean")

    def fit(self, X_train, y_train):
        """
        Fit model for a given a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # define train set
        self.dtrain_ = self._dtrain(X_train, y_train)

        # define algo
        self.algo = tpe.suggest

        # define Trials()
        self.trials = Trials()

        try:
            self.optimization_results_ = fmin(
                fn=self.fn,
                space=self.space,
                algo=self.algo,
                max_evals=self.max_evals,
                trials=self.trials,
            )
        except Exception as e:
            self.optimization_results_ = {"status": STATUS_FAIL, "exception": str(e)}

    def xgb_cv(self, space):
        """
        Function to perform XGBoost Cross-Validation with stochastic parameters
        for hyperparameter optimization.

        Parameters
        ----------
        space: dict()
            The set of possible arguments to `fn` is the set of objects
            that could be created with non-zero probability by drawing randomly
            from this stochastic program involving involving hp
        """

        # cvr
        cvr = xgb.cv(
            params=sample(self.space),
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round,
            nfold=self.n_splits,
            stratified=self.stratified,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.random_state,
            shuffle=self.shuffle,
            verbose_eval=self.verbose,
        )

        # loss
        loss = cvr.iloc[-1:, 0]

        return {"loss": loss, "status": STATUS_OK}

    def get_optimization_results(self):
        """
        Function to return pd.DataFrame with best results
        """

        return pd.DataFrame(self.optimization_results_, index=[0])

    def get_optimization_trials(self):
        """
        Function to return dict results from all 'n' trials
        """

        return self.trials


class XGBoostRegressorHyperOpt(XGBoostRegressor):
    """XGBoost Regressor - Hyperparameter Optimization.
    This class uses HyperOpt, a Python library for serial and parallel
    optimization over search spaces, which may include real-valued, discrete,
    and conditional dimensions to train a XGBoost model.Main reference is
    HyperOpt GitHub: (https://github.com/hyperopt/hyperopt)

    Parameters
    ----------
    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("rmse"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "rmse", "rmsle", "mae"

    early_stopping_rounds: int, optional (default=20)
        The criterion to early abort the xgboost.cv() phase
        if the test metric is not improved

    random_state: int, optional (default=1367)
        Random seed

    shuffle: bool, optional (default=True)
        Flag to shuffle data to have the ability of building
        stratified folds in xgboost.cv()

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    func_name: str
        Function name for performing optimization

    space: dict()
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp

    trials: HyperOpt Object
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    algo: HyperOpt Object
        Provides logic for sequential search of the hyperparameter space

    max_evals: int
        Storage for completed, ongoing, and scheduled evaluation points.
        If None, then a temporary `base.Trials` instance will be created.
        If a trials object, then that trials object will be affected by
        side-effect of this call

    verbose: str, optional (default=False)
        Print evaluation results from model

    Attributes
    ----------
    fit(X_train, y_train): instance method
        Returns None and applies the tuning process using
        the (X_train, y_train) and the given set of hyperparameters

    xgb_cv(space): instance method
        Optimization function for XGBoost utilizing cross-validation
        based on space params

    get_optimization_results(): instance method
        Returns pd.DataFrame for best parameters from all runs

    get_optimization_trials(): instance method
        Returns dict of best parameters for each individual trial run
    """

    def __init__(
        self,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        early_stopping_rounds=None,
        random_state=None,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        func_name=None,
        space=None,
        max_evals=None,
        verbose=False,
    ):
        super().__init__(
            num_boost_round,
            metrics,
            sparse_matrix,
            scale_mean,
            scale_std,
        )

        if n_splits is None:
            self.n_splits = 4
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must have integer dtype.")
            else:
                self.n_splits = n_splits

        if early_stopping_rounds is None:
            self.early_stopping_rounds = 20
        else:
            if not isinstance(early_stopping_rounds, int):
                raise TypeError("The input early_stopping_rounds must have integer dtype.")
            else:
                self.early_stopping_rounds = early_stopping_rounds

        if random_state is None:
            self.random_state = 1367
        else:
            if not isinstance(random_state, int):
                raise TypeError("The input random_state must have integer dtype.")
            else:
                self.random_state = random_state

        if not isinstance(shuffle, bool):
            raise TypeError("The input shuffle must have bool dtype.")
        else:
            self.shuffle = shuffle

        if isinstance(func_name, str):
            self.fn = getattr(self, func_name)
        else:
            raise TypeError("The input must be a valid function name: 'xgb_cv'.")

        if isinstance(space, dict):
            self.space = space
        else:
            raise TypeError("The input space must have dict dtype.")

        if max_evals is None:
            self.max_evals = 100
        else:
            if not isinstance(max_evals, int):
                raise TypeError("The input max_evals must have integer dtype.")
            else:
                self.max_evals = max_evals

        if isinstance(verbose, bool):
            self.verbose = verbose
        else:
            raise TypeError("The input verbose must be a boolean")

    def fit(self, X_train, y_train):
        """
        Fit model for a given a hyperparameter space
        according to a given algorithm, allowing up to a certain number of
        function evaluations.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # define train set
        self.dtrain_ = self._dtrain(X_train, y_train)

        # define algo
        self.algo = tpe.suggest

        # define Trials()
        self.trials = Trials()

        try:
            self.optimization_results_ = fmin(
                fn=self.fn,
                space=self.space,
                algo=self.algo,
                max_evals=self.max_evals,
                trials=self.trials,
            )
        except Exception as e:
            self.optimization_results_ = {"status": STATUS_FAIL, "exception": str(e)}

    def xgb_cv(self, space):
        """
        Function to perform XGBoost Cross-Validation with stochastic parameters
        for hyperparameter optimization.

        Parameters
        ----------
        space: dict()
            The set of possible arguments to `fn` is the set of objects
            that could be created with non-zero probability by drawing randomly
            from this stochastic program involving involving hp
        """

        # cvr
        cvr = xgb.cv(
            params=sample(self.space),
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round,
            nfold=self.n_splits,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.random_state,
            shuffle=self.shuffle,
            verbose_eval=self.verbose,
        )

        # loss
        loss = cvr.iloc[-1:, 0]

        return {"loss": loss, "status": STATUS_OK}

    def get_optimization_results(self):
        """
        Function to return pd.DataFrame with best results
        """

        return pd.DataFrame(self.optimization_results_, index=[0])

    def get_optimization_trials(self):
        """
        Function to return dict results from all 'n' trials
        """

        return self.trials
