import gc
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.figure import Figure
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from slickml.base import BaseXGBoostEstimator
from slickml.utils import Colors, add_noisy_features, check_var, df_to_csr
from slickml.visualization import plot_xfs_cv_results, plot_xfs_feature_frequency


# TODO(amir): ditch print with logging
# TODO(amir): expose `groups` in `cv.split()`
# TODO(amir): define an `abstractclass` under `base/selection` and inherit from it
# that should ease the process and reduce the amount of amount we need to copy for each algo
# TODO(amir): currently we have all the `feature_importance` calculated at each fold of each iteration
# we can apply some stats (mean/median to be simple) on top of them and plot the `feature_importance`
# as well
@dataclass
class XGBoostFeatureSelector(BaseXGBoostEstimator):
    """XGBoost Feature Selector.

    Notes
    -----
    This is a wrapper using XGBoost [xgboost-api]_ to perform a frequency-based feature selection
    algorithm with n-folds cross-validation on top of an augmented data with noisy features iteratively.
    At each n-fold of cross-validation of each iteration, the best number of boostin rounds will be found
    to over-come the possibility of over-fitting, and the feature-importance of the best trained model
    will be used to select the features. Finally, the frequency of the features that showed up at each
    feature importance phase of each cross-validation fold of each iteration will the benchmark of
    feature selection. In principle, the maximum frequency of each feature can be `n_iter` times `n_splits`.

    Parameters
    ----------
    n_iter : int, optional
        Number of iteration to repeat the feature selection algorithm, by default 3

    num_boost_round : int, optional
        Number of boosting rounds to fit a model, by default 200

    n_splits : int, optional
        Number of folds for cross-validation, by default 4

    metrics : str, optional
        Metrics to be tracked at cross-validation fitting time depends on the task
        (classification vs regression) with possible values of "auc", "aucpr", "error", "logloss",
        "rmse", "rmsle", "mae". Note this is different than `eval_metric` that needs to be passed to
        `params` dict, by default "auc"

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

    nth_noise_threshold : int, optional
        The threshold to keep all the features up to the `n-th` noisy feature at each fold of each
        iteration. For example, for a feature selection with 4 iterations and 5-folds cv, the maximum
        number of noisy features would be 4*5=20, by default 1.

    importance_type : str, optional
        Importance type of ``xgboost.train()`` with possible values ``"weight"``, ``"gain"``,
        ``"total_gain"``, ``"cover"``, ``"total_cover"``, by default "total_gain"

    params : Dict[str, Union[str, float, int]], optional
        Set of parameters required for fitting a Booster, by default for a classification task
        {"eval_metric": "auc", "tree_method": "hist", "objective": "binary:logistic", "learning_rate": 0.05,
        "max_depth": 2, "min_child_weight": 1, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "subsample": 0.9, "max_delta_step": 1, "verbosity": 0, "nthread": 4, "scale_pos_weight": 1}
        and by default for any regression task {"eval_metric": "rmse", "tree_method": "hist",
        "objective": "reg:squarederror", "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 1,
        "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 0.9, "max_delta_step": 1,
        "verbosity": 0, "nthread": 4} Other options for objective: ``"reg:logistic"``, ``"reg:squaredlogerror"``

    verbose_eval : bool, optional
        Whether to show the results of `xgboost.train()` on train/test sets using `eval_metric`, by
        default False

    callbacks : bool, optional
        Whether to logging standard deviation of metrics on train data and track the early stopping
        criterion, by default False

    Methods
    -------
    fit(X, y)
        Fits the main feature selection algorithm

    get_feature_frequency()
        Returns the total feature frequency of the best model

    get_feature_importance()
        Returns feature importance based on `importance_type` at each fold of each iteration of the
        selection process as a dict of dataframes

    get_cv_results()
        Returns the total internal/external cross-validation results

    plot_frequency()
        Visualizes the selected features frequency as a bar chart

    plot_cv_results()
        Visualizies the cross-validation results

    get_params()
        Returns the final set of train parameters

    get_default_params()
        Returns the default set of train parameters

    Attributes
    ----------
    feature_importance_
        Returns a dict of all feature importance dataframes based on `importance_type` at each fold
        of each iteration during selection process

    feature_frequency_
        Returns a pandas.DataFrame cosists of total frequency of each feature during the selection process

    cv_results_
        Return a dict of the total internal/external cross-validation results

    plotting_cv_
        Returns the required elements to visualize the histograms of total internal/external
        cross-validation results

    References
    ----------
    .. [xgboost-api] https://xgboost.readthedocs.io/en/latest/python/python_api.html
    .. [markers-api] https://matplotlib.org/stable/api/markers_api.html
    .. [seaborn-distplot-deprecation] https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    """

    n_iter: Optional[int] = 3
    n_splits: Optional[int] = 4
    metrics: Optional[str] = "auc"
    num_boost_round: Optional[int] = 200
    sparse_matrix: Optional[bool] = False
    scale_mean: Optional[bool] = False
    scale_std: Optional[bool] = False
    early_stopping_rounds: Optional[int] = 20
    nth_noise_threshold: Optional[int] = 1
    random_state: Optional[int] = 1367
    importance_type: Optional[str] = "total_gain"
    stratified: Optional[bool] = True
    shuffle: Optional[bool] = True
    verbose_eval: Optional[bool] = False
    params: Optional[Dict[str, Union[str, float, int]]] = None
    callbacks: Optional[bool] = False

    def __post_init__(self) -> None:
        """Post instantiation validations and assignments."""
        super().__post_init__()
        check_var(
            self.n_iter,
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
            self.nth_noise_threshold,
            var_name="nth_noise_threshold",
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
            self.verbose_eval,
            var_name="verbose_eval",
            dtypes=bool,
        )
        check_var(
            self.callbacks,
            var_name="callbacks",
            dtypes=bool,
        )
        self._callbacks()

        # The default set of params can be updated based on the given params by user
        _default_params = self._default_params()
        if self.params is not None:
            check_var(
                self.params,
                var_name="params",
                dtypes=dict,
            )
            _default_params.update(self.params)
            self.params = _default_params
        else:
            self.params = _default_params

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Fits the main feature selection algorithm.

        Parameters
        ----------
        X_train : Union[pd.DataFrame, np.ndarray]
            Input data for training (features)

        y_train : Union[List[float], np.ndarray, pd.Series]
            Input ground truth for training (targets)

        Returns
        -------
        None
        """
        self._check_X_y(
            X=X,
            y=y,
        )
        self.cv_results_ = defaultdict(list)  # type: ignore
        self.feature_importance_ = {}
        self.selected_features = []

        # main algorithm loop
        for iteration in range(self.n_iter):  # type: ignore
            print(
                str(Colors.BOLD)
                + "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* "
                + str(Colors.B_Green)
                + f"Iteration {iteration + 1}"
                + str(Colors.END)
                + str(Colors.BOLD)
                + " *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*",
            )

            # internal/external cross-validation results at each iteration
            int_cv_train2 = []
            int_cv_test2 = []
            ext_cv_train2 = []
            ext_cv_test2 = []

            # update random state
            self._random_state = self.random_state + iteration  # type: ignore

            # add noisy featuers by permutation based on targets
            _X_permuted = add_noisy_features(
                X=self.X,
                random_state=self._random_state,
                prefix="noisy",
            )
            _columns, _X_permuted_values = _X_permuted.columns.tolist(), _X_permuted.values

            # k-folds cross-validation (stratified only for classifications)
            if self.metrics in self._clf_metrics():
                cv = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self._random_state,
                )
            else:
                cv = KFold(
                    n_splits=self.n_splits,
                    shuffle=self.shuffle,
                    random_state=self._random_state,
                )

            # set a counter for nfolds cv
            ijk = 1
            for train_index, test_index in cv.split(_X_permuted_values, self.y):
                _X_train, _X_test = pd.DataFrame(
                    data=_X_permuted_values[train_index],
                    columns=_columns,
                ), pd.DataFrame(
                    data=_X_permuted_values[test_index],
                    columns=_columns,
                )
                _y_train, _y_test = self.y[train_index], self.y[test_index]

                # _dtrain / _dtest goes here
                self.dtrain_, self.dtest_ = self._dtrain(
                    X_train=_X_train,
                    y_train=_y_train,
                ), self._dtest(
                    X_test=_X_test,
                    y_test=_y_test,
                )

                # watchlist during final training
                self._watchlist = [
                    (self.dtrain_, "train"),
                    (self.dtest_, "eval"),
                ]

                # store training results
                self._evals_result: Dict[str, Any] = {}

                # call xgb cv
                self._cvr = self._cv()

                # append cv results
                self.cv_results_["int_cv_train"] += [self._cvr.iloc[-1][0]]
                self.cv_results_["int_cv_test"] += [self._cvr.iloc[-1][2]]

                # append temp cv results
                int_cv_train2.append(self._cvr.iloc[-1][0])
                int_cv_test2.append(self._cvr.iloc[-1][2])

                # store best trained model
                self._best_model = self._model()

                # store feature gain
                _feature_gain = self._xgb_imp_to_df()
                self.feature_importance_[f"model_iter{iteration+1}_fold{ijk}"] = _feature_gain

                # check wheather noisy feature is being selected
                if _feature_gain["feature"].str.contains("noisy").sum() != 0:
                    _gain_threshold = _feature_gain.loc[
                        _feature_gain["feature"].str.contains("noisy"),
                        self.importance_type,
                    ].values.tolist()[
                        self.nth_noise_threshold - 1  # type: ignore
                    ]
                else:
                    _gain_threshold = 0.0

                # select features where their gain > _gain_threshold
                self.selected_features.extend(
                    _feature_gain.loc[
                        _feature_gain[self.importance_type] > _gain_threshold,
                        "feature",
                    ].values.tolist(),
                )

                # final eval results for train/test external cross-validation
                if self.params is not None and isinstance(self.params["eval_metric"], str):
                    self.cv_results_["ext_cv_train"] += [
                        self._evals_result["train"][self.params["eval_metric"]][-1],
                    ]
                    self.cv_results_["ext_cv_test"] += [
                        self._evals_result["eval"][self.params["eval_metric"]][-1],
                    ]
                    ext_cv_train2.append(
                        self._evals_result["train"][self.params["eval_metric"]][-1],
                    )
                    ext_cv_test2.append(
                        self._evals_result["eval"][self.params["eval_metric"]][-1],
                    )

                    # TODO(amir): ditch print with logging
                    print(
                        str(Colors.BOLD)
                        + "*-*-*-*-*-*-*-*-*-*-*-* "
                        + str(Colors.F_Green)
                        + f"Fold = {ijk}/{self.n_splits}"
                        + str(Colors.F_Black)
                        + " -- "
                        + str(Colors.F_Red)
                        + f"Train {self.params['eval_metric'].upper()}"
                        + " = "
                        + f"{self._evals_result['train'][self.params['eval_metric']][-1]:.3f}"
                        + str(Colors.F_Black)
                        + " -- "
                        + str(Colors.F_Blue)
                        + f"Test {self.params['eval_metric'].upper()}"
                        + " = "
                        + f"{self._evals_result['eval'][self.params['eval_metric']][-1]:.3f}"
                        + str(Colors.END)
                        + str(Colors.BOLD)
                        + " *-*-*-*-*-*-*-*-*-*-*-*",
                    )
                # free memory here at each fold
                del (
                    self._best_model,
                    self._watchlist,
                    self.dtrain_,
                    self.dtest_,
                    self._cvr,
                    self._evals_result,
                    _feature_gain,
                    _X_train,
                    _y_train,
                    _X_test,
                    _y_test,
                )

                ijk += 1
                gc.collect()

            # print internal metrics results
            # TODO(amir): replace print with logging
            if self.metrics is not None and self.n_splits is not None:
                print(
                    str(Colors.BOLD)
                    + "*-*-* "
                    + str(Colors.GREEN)
                    + f"Internal {self.n_splits}-Folds CV:"
                    + str(Colors.END)
                    + str(Colors.BOLD)
                    + " -*-*- "
                    + str(Colors.F_Red)
                    + f"Train {self.metrics.upper()}"
                    + " = "
                    + f"{np.mean(int_cv_train2):.3f}"
                    + " +/- "
                    + f"{np.std(int_cv_train2):.3f}"
                    + str(Colors.END)
                    + str(Colors.BOLD)
                    + " -*-*- "
                    + str(Colors.F_Blue)
                    + f"Test {self.metrics.upper()}"
                    + " = "
                    + f"{np.mean(int_cv_test2):.3f}"
                    + " +/- "
                    + f"{np.std(int_cv_test2):.3f}"
                    + str(Colors.END)
                    + str(Colors.BOLD)
                    + " *-*-*",
                )

            #  print external eval_metric results
            # TODO(amir): replace print with logging
            if self.params is not None and isinstance(self.params["eval_metric"], str):
                print(
                    str(Colors.BOLD)
                    + "*-*-* "
                    + str(Colors.GREEN)
                    + f"External {self.n_splits}-Folds CV:"
                    + str(Colors.END)
                    + str(Colors.BOLD)
                    + " -*-*- "
                    + str(Colors.F_Red)
                    + f"Train {self.params['eval_metric'].upper()}"
                    + " = "
                    + f"{np.mean(ext_cv_train2):.3f}"
                    + " +/- "
                    + f"{np.std(ext_cv_train2):.3f}"
                    + str(Colors.END)
                    + str(Colors.BOLD)
                    + " -*-*- "
                    + str(Colors.F_Blue)
                    + f"Test {self.params['eval_metric'].upper()}"
                    + " = "
                    + f"{np.mean(ext_cv_test2):.3f}"
                    + " +/- "
                    + f"{np.std(ext_cv_test2):.3f}"
                    + str(Colors.END)
                    + str(Colors.BOLD)
                    + " *-*-*\n",
                )

            # free memory here at iteration
            del (
                int_cv_train2,
                int_cv_test2,
                ext_cv_train2,
                ext_cv_test2,
                _X_permuted,
                _X_permuted_values,
                _columns,
            )
            gc.collect()

        self.plotting_cv_ = self._plotting_cv()
        self.feature_frequency_ = self._freq()

        return None

    def plot_frequency(
        self,
        *,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (8, 4),
        show_freq_pct: Optional[bool] = True,
        color: Optional[str] = "#87CEEB",
        marker: Optional[str] = "o",
        markersize: Optional[Union[int, float]] = 10,
        markeredgecolor: Optional[str] = "#1F77B4",
        markerfacecolor: Optional[str] = "#1F77B4",
        markeredgewidth: Optional[Union[int, float]] = 1,
        fontsize: Optional[Union[int, float]] = 12,
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Visualizes the selected features frequency as a bar chart.

        Notes
        -----
        This plotting function can be used along with ``feature_frequency_`` attribute of any
        frequency-based feature selection algorithm such as ``XGBoostFeatureSelector``.

        Parameters
        ----------
        feature importance : pd.DataFrame
            Feature importance (``feature_frequency_`` attribute)

        figsize : tuple, optional
            Figure size, by default (8, 4)

        show_freq_pct : bool, optional
            Whether to show the features frequency in percent, by default True

        color : str, optional
            Color of the horizontal lines of lollipops, by default "#87CEEB"

        marker : str, optional
            Marker style of the lollipops. More valid marker styles can be found at [markers-api]_, by default "o"

        markersize : Union[int, float], optional
            Markersize, by default 10

        markeredgecolor : str, optional
            Marker edge color, by default "#1F77B4"

        markerfacecolor : str, optional
            Marker face color, by defualt "#1F77B4"

        markeredgewidth : Union[int, float], optional
            Marker edge width, by default 1

        fontsize : Union[int, float], optional
            Fontsize for xlabel and ylabel, and ticks parameters, by default 12

        save_path : str, optional
            The full or relative path to save the plot including the image format such as
            "myplot.png" or "../../myplot.pdf", by default None

        display_plot : bool, optional
            Whether to show the plot, by default True

        return_fig : bool, optional
            Whether to return figure object, by default False

        Returns
        -------
        Figure, optional
        """
        return plot_xfs_feature_frequency(
            freq=self.feature_frequency_,
            figsize=figsize,
            show_freq_pct=show_freq_pct,
            color=color,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth,
            fontsize=fontsize,
            save_path=save_path,
            display_plot=display_plot,
            return_fig=return_fig,
        )

    def plot_cv_results(
        self,
        *,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (10, 8),
        internalcvcolor: Optional[str] = "#4169E1",
        externalcvcolor: Optional[str] = "#8A2BE2",
        sharex: Optional[bool] = False,
        sharey: Optional[bool] = False,
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Visualizies the cross-validation results.

        Notes
        -----
        It visualizes the internal and external cross-validiation performance during the selection
        process. The `internal` refers to the performance of the train/test folds during the ``xgboost.cv()``
        using ``metrics`` rounds to help the best number of boosting round while the `external` refers to
        the performance of ``xgboost.train()`` based on watchlist using ``eval_metric``. Additionally,
        `sns.distplot` previously was used which is now deprecated. More details in [seaborn-distplot-deprecation]_.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (10, 8)

        internalcvcolor : str, optional
            Color of the histograms for internal cv results, by default "#4169E1"

        externalcvcolor : str, optional
            Color of the histograms for external cv results, by default "#8A2BE2"

        sharex : bool, optional
            Whether to share "X" axis for each column of subplots, by default False

        sharey : bool, optional
            Whether to share "Y" axis for each row of subplots, by default False

        save_path : str, optional
            The full or relative path to save the plot including the image format such as
            "myplot.png" or "../../myplot.pdf", by default None

        display_plot : bool, optional
            Whether to show the plot, by default True

        return_fig : bool, optional
            Whether to return figure object, by default False

        kwargs : Dict[str, Any]
            Required plooting elements (``plotting_cv_`` attribute of ``XGBoostFeatureSelector``)

        Returns
        -------
        Figure, optional
        """
        return plot_xfs_cv_results(
            figsize=figsize,
            internalcvcolor=internalcvcolor,
            externalcvcolor=externalcvcolor,
            sharex=sharex,
            sharey=sharey,
            save_path=save_path,
            display_plot=display_plot,
            return_fig=return_fig,
            **self.plotting_cv_,
        )

    def get_params(self) -> Optional[Dict[str, Union[str, float, int]]]:
        """Returns the final set of train parameters.

        The default set of parameters will be updated with the new ones that passed to ``params``.

        See Also
        --------
        :meth:`get_default_params()`

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        return self.params

    def get_default_params(self) -> Dict[str, Union[str, float, int]]:
        """Returns the default set of train parameters.

        The default set of parameters will be used when ``params=None``.

        See Also
        --------
        :meth:`get_params()`

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        return self._default_params()

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Returns the feature importance of the trained booster based on the given ``importance_type``.

        Returns
        -------
        pd.DataFrame
        """
        return self.feature_importance_

    def get_feature_frequency(self) -> pd.DataFrame:
        """ReturnS the total feature frequency of the best model at each fold of each iteration.

        Returns
        -------
        pd.DataFrame
        """
        return self.feature_frequency_

    def get_cv_results(self) -> pd.DataFrame:
        """Returns internal and external cross-validation results.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(
            data=self.cv_results_,
        )

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
        if self.scale_mean or self.scale_std:
            self._scaler = StandardScaler(
                copy=True,
                with_mean=self.scale_mean,
                with_std=self.scale_std,
            )
            _X_train = pd.DataFrame(
                self._scaler.fit_transform(X_train),
                columns=X_train.columns.tolist(),
            )
        else:
            self._scaler = None
            _X_train = X_train

        if not self.sparse_matrix:
            dtrain = xgb.DMatrix(
                data=_X_train,
                label=y_train,
            )
        else:
            dtrain = xgb.DMatrix(
                data=df_to_csr(
                    _X_train,
                    fillna=0.0,
                    verbose=False,
                ),
                label=y_train,
                feature_names=_X_train.columns.tolist(),
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
        if self.scale_mean or self.scale_std:
            _X_test = pd.DataFrame(
                self._scaler.transform(X_test),
                columns=X_test.columns.tolist(),
            )
        else:
            _X_test = X_test

        if not self.sparse_matrix:
            dtest = xgb.DMatrix(
                data=_X_test,
                label=y_test,
            )
        else:
            dtest = xgb.DMatrix(
                data=df_to_csr(
                    _X_test,
                    fillna=0.0,
                    verbose=False,
                ),
                label=y_test,
                feature_names=_X_test.columns.tolist(),
            )

        return dtest

    def _xgb_imp_to_df(self) -> pd.DataFrame:
        """Returns the feature importance dict object into a pandas dataframe.

        Returns
        -------
        pd.DataFrame
        """
        data: Dict[str, List[float]] = {
            "feature": [],
            f"{self.importance_type}": [],
        }
        features_gain = self._best_model.get_score(importance_type=self.importance_type)
        for key, val in features_gain.items():
            data["feature"].append(key)
            data[f"{self.importance_type}"].append(val)

        return (
            pd.DataFrame(data)
            .sort_values(
                by=f"{self.importance_type}",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    def _cv(self) -> pd.DataFrame:
        """Returns XGBoost cross-validation results to find the best number of boosting rounds.

        Returns
        -------
        pd.DataFrame
        """
        if self.metrics in self._clf_metrics():
            return xgb.cv(
                params=self.params,
                dtrain=self.dtrain_,
                num_boost_round=self.num_boost_round,
                nfold=self.n_splits,
                stratified=self.stratified,
                metrics=self.metrics,
                early_stopping_rounds=self.early_stopping_rounds,
                seed=self._random_state,
                verbose_eval=self.verbose_eval,
                shuffle=self.shuffle,
                callbacks=self.callbacks,
                as_pandas=True,
            )
        else:
            return xgb.cv(
                params=self.params,
                dtrain=self.dtrain_,
                num_boost_round=self.num_boost_round,
                nfold=self.n_splits,
                metrics=self.metrics,
                early_stopping_rounds=self.early_stopping_rounds,
                seed=self._random_state,
                verbose_eval=self.verbose_eval,
                shuffle=self.shuffle,
                callbacks=self.callbacks,
                as_pandas=True,
            )

    def _model(self) -> xgb.Booster:
        """Returns the trained `xgb.Booster` model based on the best number of boosting round.

        Returns
        -------
        xgb.Booster
        """
        return xgb.train(
            params=self.params,
            dtrain=self.dtrain_,
            num_boost_round=len(self._cvr) - 1,
            evals=self._watchlist,
            evals_result=self._evals_result,
            verbose_eval=self.verbose_eval,
        )

    def _freq(self) -> pd.DataFrame:
        """Returns feature frequency of the selected features.

        Returns
        -------
        pd.DataFrame
        """
        unique_elements, counts_elements = np.unique(
            self.selected_features,
            return_counts=True,
        )
        feature_frequency = pd.DataFrame(
            data={
                "Feature": list(unique_elements),
                "Frequency": [float(i) for i in list(counts_elements)],
            },
        )
        if self.n_splits is not None and self.n_iter is not None:
            feature_frequency["Frequency (%)"] = round(
                (feature_frequency["Frequency"] / float(self.n_splits * self.n_iter) * 100),
                ndigits=2,
            )
        return feature_frequency.sort_values(
            by=["Frequency", "Frequency (%)"],
            ascending=[False, False],
        ).reset_index(drop=True)

    # TODO(amir): investigate more for other callback options ?
    def _callbacks(self) -> None:
        """Returns a list of callbacks.

        The implemented callbacks are including ``xgboost.callback.EvaluationMonitor`` and
        ``xgboost.callback.EarlyStopping`` [callback-api]_.

        Returns
        -------
        None
        """
        if self.callbacks:
            # TODO(amir): ditch print with logger
            print(
                "Warning: The `cv` will break if the `early_stopping_rounds` criterion was not satisfied.",
            )
            # TODO(amir): use type overload
            self.callbacks = [  # type: ignore
                xgb.callback.EvaluationMonitor(
                    rank=0,
                    period=1,
                    show_stdv=True,
                ),
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_rounds,
                ),
            ]
        else:
            self.callbacks = None

        return None

    def _default_params(self) -> Dict[str, Union[str, float, int]]:
        """Default set of parameters when the class is being instantiated with ``params=None``.

        Notes
        -----
        The default set of parameters would be a little bit different depends on the type of selection
        whether a classification or regression `metric` is being used.

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
        if self.metrics in self._clf_metrics():
            return {
                "eval_metric": "auc",
                "tree_method": "hist",
                "objective": "binary:logistic",
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "subsample": 0.9,
                "max_delta_step": 1,
                "verbosity": 0,
                "nthread": 4,
                "scale_pos_weight": 1,
            }
        else:
            return {
                "eval_metric": "rmse",
                "tree_method": "hist",
                "objective": "reg:squarederror",
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_child_weight": 1,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "subsample": 0.9,
                "max_delta_step": 1,
                "verbosity": 0,
                "nthread": 4,
            }

    def _plotting_cv(self) -> Dict[str, Any]:
        """Returns the required elements for plotting cross-validation results.

        Returns
        -------
        Dict[str, Any]
        """
        p: Dict[str, Any] = {}
        if (
            self.metrics is not None
            and self.params is not None
            and isinstance(self.params["eval_metric"], str)
        ):
            p["metric"] = self.metrics.upper()
            p["eval_metric"] = self.params["eval_metric"].upper()
            p["n_splits"] = self.n_splits
            p["int_cv_train"] = self.cv_results_["int_cv_train"]
            p["int_cv_test"] = self.cv_results_["int_cv_test"]
            p["ext_cv_train"] = self.cv_results_["ext_cv_train"]
            p["ext_cv_test"] = self.cv_results_["ext_cv_test"]

        return p

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
