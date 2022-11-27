from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import glmnet
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, RegressorMixin

from slickml.utils import array_to_df, check_var, df_to_csr
from slickml.visualization import (
    plot_glmnet_coeff_path,
    plot_glmnet_cv_results,
    plot_shap_summary,
    plot_shap_waterfall,
)


# TODO(amir): add lolli-pop plot for coeff
@dataclass
class GLMNetCVRegressor(BaseEstimator, RegressorMixin):
    """GLMNetCVRegressor.

    This is wrapper using GLM-Net [glmnet-api]_ to train a Regularized Linear Model via ElasticNet regression and
    find the optimal penalty values through N-Folds cross validation. In principle, GLMNet (also known
    as ElasticNet) can also be used for feature selection and dimensionality reduction using the LASSO
    (Least Absolute Shrinkage and Selection Operator) Regression part of the alogrithm while reaching
    a solid solution using the Ridge Regression part of the algorithm.

    Parameters
    ----------
    alpha : float, optional
        The stability parameter with a possible values of  ``0 <= alpha <= 1`` where ``alpha=0.0``
        and ``alpha=1.0`` will lead to classic Ridge and LASSO regression models, respectively, by
        default 0.5

    n_lambda : int, optional
        Maximum number of penalty values to compute, by default 100

    n_splits : int, optional
        Number of cross validation folds for computing performance metrics and determining
        ``lambda_best_`` and ``lambda_max_``. If non-zero, must beat least 3, by default 3

    metric : str, optional
        Metric used for model selection during cross validation. Valid options are ``"r2"``,
        ``"mean_squared_error"``, ``"mean_absolute_error"``, and ``"median_absolute_error"``.
        The metric affects the selection of ``lambda_best_`` and ``lambda_max_``. Thus, fitting the
        same data with different metric methods will result in the selection of different models, by
        default "r2"

    scale : bool, optional
        Whether to standardize the input features to have a mean value of 0.0 and standard deviation
        of 1 prior to fitting. The final coefficients will be on the scale of the original data regardless
        of this step. Therefore, there is no need to pre-process the data when using ``scale=True``,
        by default True

    sparse_matrix : bool, optional
        Whether to convert the input features to sparse matrix with csr format or not. This would increase
        the speed of feature selection for relatively large sparse datasets.  Additionally, this
        parameter cannot be used along with ``scale=True`` where standardizing the feature matrix
        to have a mean value of zero would turn the feature matrix into a dense matrix, by default False

    fit_intercept : bool, optional
        Include an intercept term in the model, by default True

    cut_point : float, optional
        The cut point to use for selecting ``lambda_best_``. Based on this value, the distance between
        ``lambda_max_`` and ``lambda_best_`` would be ``cut_point * standard_error(lambda_best_)``
        ``arg_max(lambda) for cv_score(lambda) >= cv_score(lambda_max_) - cut_point * standard_error(lambda_max_),
        by default 1.0

    min_lambda_ratio : float, optional
        In combination with ``n_lambda``, the ratio of the smallest and largest values of lambda
        computed ``(min_lambda/max_lambda >= min_lambda_ratio)``, by default 1e-4

    tolerance : float, optional
        Convergence criteria tolerance, by default 1e-7

    max_iter : int, optional
        Maximum passes over the data, by default 100000

    random_state : int, optional
        Seed for the random number generator. The glmnet solver is not
        deterministic, this seed is used for determining the cv folds.

    lambda_path : Union[List[float], np.ndarray, pd.Series], optional
        In place of supplying ``n_lambda``, provide an array of specific values to compute. The
        specified values must be in decreasing order. When None, the path of lambda values will be
        determined automatically. A maximum of ``n_lambda`` values will be computed, by default None

    max_features : int, optional
        Optional maximum number of features with nonzero coefficients after regularization. If not
        set, defaults to the number features (``X_train.shape[1]``) during fit. Note, this will be
        ignored if the user specifies ``lambda_path``, by default None

    Methods
    -------
    fit(X_train, y_train)
        Fits a ``glmnet.ElasticNet`` to input training data. Proper ``X_train`` matrix based on chosen
        options i.e. ``sparse_matrix``, and ``scale`` is being created based on the passed ``X_train``
        and ``y_train``

    predict(X_test, y_test)
        Returns the prediction target (response) values

    plot_coeff_path():
        Visualizes the coefficients' paths

    plot_cv_results()
        Visualizes the cross-validation results

    plot_shap_summary()
        Visualizes Shapley values summary plot

    plot_shap_waterfall()
        Visualizes Shapley values waterfall plot

    get_shap_explainer()
        Returns the fitted ``shap.LinearExplainer`` object

    get_params():
        Returns parameters

    get_intercept():
        Returns model's intercept

    get_coeffs():
        Returns non-zero coefficients

    get_cv_results():
        Returns cross-validation results

    get_results():
        Returns model's total results

    Attributes
    ----------
    X_train : pd.DataFrame
        Returns training data set

    X_test : pd.DataFrame
        Returns transformed testing data set

    y_train : np.ndarray
        Returns the list of training ground truth for training (targets)

    y_test : np.ndarray
        Returns the list of testing ground truth for training (targets)

    coeff_ : pd.DataFrame
        Return the model's non-zero coefficients

    intercept_ : float
        Return the model's intercept

    cv_results_ : pd.DataFrame
        Returns the cross-validation results

    results_ : Dict[str, Any]
        Returns the model's total results

    params_ : Dict[str, Any]
        Returns model's fitting parameters

    shap_values_train_ : np.ndarray
        Shapley values from ``LinearExplainer`` using ``X_train``

    shap_values_test_ : np.ndarray
        Shapley values from ``LinearExplainer`` using ``X_test``

    shap_explainer_ : shap.LinearExplainer
        Shap LinearExplainer with independent masker using ``X_Test``

    model_ : glmnet.ElasticNet
        Returns fitted ``glmnet.ElasticNet`` model

    References
    ----------
    .. [glmnet-api] https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
    .. [markers-api] https://matplotlib.org/stable/api/markers_api.html
    .. [yscale] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html
    .. [shap-api] https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
    """

    alpha: Optional[float] = 0.5
    n_lambda: Optional[int] = 100
    n_splits: Optional[int] = 3
    metric: Optional[str] = "r2"
    scale: Optional[bool] = True
    sparse_matrix: Optional[bool] = False
    fit_intercept: Optional[bool] = True
    cut_point: Optional[float] = 1.0
    min_lambda_ratio: Optional[float] = 1e-4
    tolerance: Optional[float] = 1e-7
    max_iter: Optional[int] = 100000
    random_state: Optional[int] = 1367
    lambda_path: Optional[Union[List[float], np.ndarray, pd.Series]] = None
    max_features: Optional[int] = None

    def __post_init__(self) -> None:
        """Post instantiation validations and assignments."""
        check_var(
            self.alpha,
            var_name="alpha",
            dtypes=float,
        )
        check_var(
            self.n_lambda,
            var_name="n_lambda",
            dtypes=int,
        )
        check_var(
            self.n_splits,
            var_name="n_splits",
            dtypes=int,
        )
        # TODO(amir): metric should be able to be a `CALLABLE` as well with signature "scorer(estimator, X, y)".
        check_var(
            self.metric,
            var_name="metric",
            dtypes=str,
            values=(
                "r2",
                "mean_squared_error",
                "mean_absolute_error",
                "median_absolute_error",
            ),
        )
        check_var(
            self.scale,
            var_name="scale",
            dtypes=bool,
        )
        check_var(
            self.sparse_matrix,
            var_name="sparse_matrix",
            dtypes=bool,
        )
        check_var(
            self.fit_intercept,
            var_name="fit_intercept",
            dtypes=bool,
        )
        check_var(
            self.cut_point,
            var_name="cut_point",
            dtypes=float,
        )
        check_var(
            self.min_lambda_ratio,
            var_name="min_lambda_ratio",
            dtypes=float,
        )
        check_var(
            self.tolerance,
            var_name="tolerance",
            dtypes=float,
        )
        check_var(
            self.max_iter,
            var_name="max_iter",
            dtypes=int,
        )
        check_var(
            self.random_state,
            var_name="random_state",
            dtypes=int,
        )
        if self.max_features:
            check_var(
                self.max_features,
                var_name="max_features",
                dtypes=int,
            )
        if self.lambda_path:
            check_var(
                self.lambda_path,
                var_name="lambda_path",
                dtypes=(
                    list,
                    np.ndarray,
                    pd.Series,
                ),
            )
            if not isinstance(self.lambda_path, np.ndarray):
                self.lambda_path = np.array(self.lambda_path)

        # The `scale=True` would turn a sparse matrix into a dense matrix
        if self.sparse_matrix and self.scale:
            raise ValueError(
                "The scale should be False in conjuction of using sparse_matrix=True.",
            )

    # TODO(amir): expose `groups` in args since glmnet supports it
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Fits a ``glmnet.ElasticNet`` to input training data.

        Notes
        -----
        For the cases that ``sparse_matrix=True``, a CSR format of the input will be used via
        ``df_to_csr()`` function.

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
        self._dtrain(
            X_train=X_train,
            y_train=y_train,
        )
        self.model_ = self._model()
        self.coeff_ = self._coeff_to_df()
        self.results_ = self._results()
        self.cv_results_ = self._cv_results()
        self.intercept_ = self.model_.intercept_
        self.params_ = self.model_.get_params()

        return None

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
        lamb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns the prediction response values (targets).

        Parameters
        ----------
        X_test : Union[pd.DataFrame, np.ndarray]
            Input data for testing (features)

        y_test : Union[List[float], np.ndarray, pd.Series], optional
            Input ground truth for testing (targets)

        lamb : np.ndarray, optional
            Values with shape ``(n_lambda,)`` of lambda from ``lambda_path_`` from which to make
            predictions. If no values are provided (None), the returned predictions will be those
            corresponding to ``lambda_best_``. The values of lamb must also be in the range of
            ``lambda_path_``, values greater than ``max(lambda_path_)`` or less than
            ``min(lambda_path_)`` will be clipped

        Returns
        -------
        np.ndarray
        """
        self._dtest(
            X_test=X_test,
            y_test=y_test,
        )
        if self.sparse_matrix:
            self.y_pred_ = self.model_.predict(
                X=df_to_csr(
                    self.X_test,
                    fillna=0.0,
                    verbose=False,
                ),
                lamb=lamb,
            )
        else:
            self.y_pred_ = self.model_.predict(
                X=self.X_test,
                lamb=lamb,
            )

        return self.y_pred_

    def plot_cv_results(
        self,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (8, 5),
        marker: Optional[str] = "o",
        markersize: Optional[Union[int, float]] = 5,
        color: Optional[str] = "red",
        errorbarcolor: Optional[str] = "black",
        maxlambdacolor: Optional[str] = "purple",
        bestlambdacolor: Optional[str] = "navy",
        linestyle: Optional[str] = "--",
        fontsize: Optional[Union[int, float]] = 12,
        grid: Optional[bool] = True,
        legend: Optional[bool] = True,
        legendloc: Optional[Union[int, str]] = "best",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Visualizes the GLMNet cross-validation results.

        Notes
        -----
        This plotting function can be used along with ``results_`` attribute of any of
        ``GLMNetCVClassifier``, or ``GLMNetCVRegressor`` classes as ``kwargs``.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (8, 5)

        marker : str, optional
            Marker style of the metric to distinguish the error bars. More valid marker styles can be
            found at [markers-api]_, by default "o"

        markersize : Union[int, float], optional
            Markersize, by default 5

        color : str, optional
            Line and marker color, by default "red"

        errorbarcolor : str, optional
            Error bar color, by default "black"

        maxlambdacolor : str, optional
            Color of vertical line for ``lambda_max_``, by default "purple"

        bestlambdacolor : str, optional
            Color of vertical line for ``lambda_best_``, by default "navy"

        linestyle : str, optional
            Linestyle of vertical lambda lines, by default "--"

        fontsize : Union[int, float], optional
            Fontsize of the title. The fontsizes of xlabel, ylabel, tick_params, and legend are resized
            with 0.85, 0.85, 0.75, and 0.85 fraction of title fontsize, respectively, by default 12

        grid : bool, optional
            Whether to show (x,y) grid on the plot or not, by default True

        legend : bool, optional
            Whether to show legend on the plot or not, by default True

        legendloc : Union[int, str], optional
            Location of legend, by default "best"

        xlabel : str, optional
            Xlabel of the plot, by default "-Log(Lambda)"

        ylabel : str, optional
            Ylabel of the plot, by default "{n_splits}-Folds CV Mean {metric}"

        title : str, optional
            Title of the plot, by default "Best {lambda_best} with {n} Features"

        save_path : str, optional
            The full or relative path to save the plot including the image format such as
            "myplot.png" or "../../myplot.pdf", by default None

        display_plot : bool, optional
            Whether to show the plot, by default True

        return_fig : bool, optional
            Whether to return figure object, by default False

        **kwargs : Dict[str, Any]
            Key-value pairs of results. ``results_`` attribute can be used

        See Also
        --------
        :class:`slickml.classification.GLMNetCVClassifier`
        :class:`slickml.regression.GLMNetCVRegressor`

        Returns
        -------
        Figure, optional
        """
        return plot_glmnet_cv_results(
            figsize=figsize,
            marker=marker,
            markersize=markersize,
            color=color,
            errorbarcolor=errorbarcolor,
            maxlambdacolor=maxlambdacolor,
            bestlambdacolor=bestlambdacolor,
            linestyle=linestyle,
            fontsize=fontsize,
            grid=grid,
            legend=legend,
            legendloc=legendloc,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            save_path=save_path,
            display_plot=display_plot,
            return_fig=return_fig,
            **self.results_,
        )

    def plot_coeff_path(
        self,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (8, 5),
        linestyle: Optional[str] = "-",
        fontsize: Optional[Union[int, float]] = 12,
        grid: Optional[bool] = True,
        legend: Optional[bool] = True,
        legendloc: Optional[Union[int, str]] = "center",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = "Coefficients",
        title: Optional[str] = None,
        bbox_to_anchor: Tuple[float, float] = (1.1, 0.5),
        yscale: Optional[str] = "linear",
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Visualizes the GLMNet coefficients' paths.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (8, 5)

        linestyle : str, optional
            Linestyle of paths, by default "-"

        fontsize : Union[int, float], optional
            Fontsize of the title. The fontsizes of xlabel, ylabel, tick_params, and legend are resized
            with 0.85, 0.85, 0.75, and 0.85 fraction of title fontsize, respectively, by default 12

        grid : bool, optional
            Whether to show (x,y) grid on the plot or not, by default True

        legend : bool, optional
            Whether to show legend on the plot or not, by default True

        legendloc : Union[int, str], optional
            Location of legend, by default "center"

        xlabel : str, optional
            Xlabel of the plot, by default "-Log(Lambda)"

        ylabel : str, optional
            Ylabel of the plot, by default "Coefficients"

        title : str, optional
            Title of the plot, by default "Best {lambda_best} with {n} Features"

        yscale : str, optiona
            Scale for y-axis (coefficients). Possible options are ``"linear"``, ``"log"``, ``"symlog"``,
            ``"logit"`` [yscale]_, by default "linear"

        bbox_to_anchor : Tuple[float, float], optional
            Relative coordinates for legend location outside of the plot, by default (1.1, 0.5)

        save_path : str, optional
            The full or relative path to save the plot including the image format such as
            "myplot.png" or "../../myplot.pdf", by default None

        display_plot : bool, optional
            Whether to show the plot, by default True

        return_fig : bool, optional
            Whether to return figure object, by default False

        **kwargs : Dict[str, Any]
            Key-value pairs of results. ``results_`` attribute can be used

        Returns
        -------
        Figure, optional
        """
        return plot_glmnet_coeff_path(
            figsize=figsize,
            linestyle=linestyle,
            fontsize=fontsize,
            grid=grid,
            legend=legend,
            legendloc=legendloc,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            bbox_to_anchor=bbox_to_anchor,
            yscale=yscale,
            save_path=save_path,
            display_plot=display_plot,
            return_fig=return_fig,
            **self.results_,
        )

    def plot_shap_summary(
        self,
        validation: Optional[bool] = True,
        plot_type: Optional[str] = "dot",
        figsize: Optional[Union[str, Tuple[float, float]]] = "auto",
        color: Optional[str] = None,
        cmap: Optional[LinearSegmentedColormap] = None,
        max_display: Optional[int] = 20,
        feature_names: Optional[List[str]] = None,
        layered_violin_max_num_bins: Optional[int] = 10,
        title: Optional[str] = None,
        sort: Optional[bool] = True,
        color_bar: Optional[bool] = True,
        class_names: Optional[List[str]] = None,
        class_inds: Optional[List[int]] = None,
        color_bar_label: Optional[str] = "Feature Value",
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
    ) -> None:
        """Visualizes shap beeswarm plot as summary of shapley values.

        Notes
        -----
        This is a helper function to plot the ``shap`` summary plot based on all types of
        ``shap.Explainer`` including ``shap.LinearExplainer`` for linear models, ``shap.TreeExplainer``
        for tree-based models, and ``shap.DeepExplainer`` deep neural network models. More on details
        are available at [shap-api]_. Note that this function should be ran after the ``predict_proba()``
        to make sure the ``X_test`` is being instansiated or set ``validation=False``.

        Parameters
        ----------
        validation : bool, optional
            Whether to calculate Shap values of using the validation data ``X_test`` or not. When
            ``validation=False``, Shap values are calculated using ``X_train``, be default True

        plot_type : str, optional
            The type of summary plot where possible options are "bar", "dot", "violin", "layered_violin",
            and "compact_dot". Recommendations are "dot" for single-output such as binary classifications,
            "bar" for multi-output problems, "compact_dot" for Shap interactions, by default "dot"

        figsize : tuple, optional
            Figure size where "auto" is auto-scaled figure size based on the number of features that are
            being displayed. Passing a single float will cause each row to be that many inches high.
            Passing a pair of floats will scale the plot by that number of inches. If None is passed
            then the size of the current figure will be left unchanged, by default "auto"

        color : str, optional
            Color of plots when ``plot_type="violin"`` and ``plot_type=layered_violin"`` are "RdBl"
            color-map while color of the horizontal lines when ``plot_type="bar"`` is "#D0AAF3", by
            default None

        cmap : LinearSegmentedColormap, optional
            Color map when ``plot_type="violin"`` and ``plot_type=layered_violin"``, by default "RdBl"

        max_display : int, optional
            Limit to show the number of features in the plot, by default 20

        feature_names : List[str], optional
            List of feature names to pass. It should follow the order of features, by default None

        layered_violin_max_num_bins : int, optional
            The number of bins for calculating the violin plots ranges and outliers, by default 10

        title : str, optional
            Title of the plot, by default None

        sort : bool, optional
            Flag to plot sorted shap vlues in descending order, by default True

        color_bar : bool, optional
            Flag to show a color bar when ``plot_type="dot"`` or ``plot_type="violin"``

        class_names : List[str], optional
            List of class names for multi-output problems, by default None

        class_inds : List[int], optional
            List of class indices for multi-output problems, by default None

        color_bar_label : str, optional
            Label for color bar, by default "Feature Value"

        save_path : str, optional
            The full or relative path to save the plot including the image format such as
            "myplot.png" or "../../myplot.pdf", by default None

        display_plot : bool, optional
            Whether to show the plot, by default True

        Returns
        -------
        None
        """
        self._explainer()

        if validation:
            shap_values = self.shap_values_test_
            features = self.X_test
        else:
            shap_values = self.shap_values_train_
            features = self.X_train

        return plot_shap_summary(
            shap_values=shap_values,
            features=features,
            plot_type=plot_type,
            figsize=figsize,
            color=color,
            cmap=cmap,
            max_display=max_display,
            feature_names=feature_names,
            layered_violin_max_num_bins=layered_violin_max_num_bins,
            title=title,
            sort=sort,
            color_bar=color_bar,
            class_names=class_names,
            class_inds=class_inds,
            color_bar_label=color_bar_label,
            save_path=save_path,
            display_plot=display_plot,
        )

    def plot_shap_waterfall(
        self,
        validation: Optional[bool] = True,
        figsize: Optional[Tuple[float, float]] = (8, 5),
        bar_color: Optional[str] = "#B3C3F3",
        bar_thickness: Optional[Union[float, int]] = 0.5,
        line_color: Optional[str] = "purple",
        marker: Optional[str] = "o",
        markersize: Optional[Union[int, float]] = 7,
        markeredgecolor: Optional[str] = "purple",
        markerfacecolor: Optional[str] = "purple",
        markeredgewidth: Optional[Union[int, float]] = 1,
        max_display: Optional[int] = 20,
        title: Optional[str] = None,
        fontsize: Optional[Union[int, float]] = 12,
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = True,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Visualizes the Shapley values as a waterfall plot.

        Notes
        -----
        Waterfall is defined as the cumulitative/composite ratios of shap values per feature.
        Therefore, it can be easily seen with each feature how much explainability we can achieve.
        Note that this function should be ran after the ``predict_proba()`` to make sure the
        ``X_test`` is being instansiated or set ``validation=False``.

        Parameters
        ----------
        validation : bool, optional
            Whether to calculate Shap values of using the validation data ``X_test`` or not. When
            ``validation=False``, Shap values are calculated using ``X_train``, be default True

        figsize : Tuple[float, float], optional
            Figure size, by default (8, 5)

        bar_color : str, optional
            Color of the horizontal bar lines, "#B3C3F3"

        bar_thickness : Union[float, int], optional
            Thickness (hight) of the horizontal bar lines, by default 0.5

        line_color : str, optional
            Color of the line plot, by default "purple"

        marker : str, optional
            Marker style of the lollipops. More valid marker styles can be found at [2]_, by default "o"

        markersize : Union[int, float], optional
            Markersize, by default 7

        markeredgecolor : str, optional
            Marker edge color, by default "purple"

        markerfacecolor: str, optional
            Marker face color, by default "purple"

        markeredgewidth : Union[int, float], optional
            Marker edge width, by default 1

        max_display : int, optional
            Limit to show the number of features in the plot, by default 20

        title : str, optional
            Title of the plot, by default None

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
        self._explainer()

        if validation:
            shap_values = self.shap_values_test_
            features = self.X_test
        else:
            shap_values = self.shap_values_train_
            features = self.X_train

        return plot_shap_waterfall(
            shap_values=shap_values,
            features=features,
            figsize=figsize,
            bar_color=bar_color,
            bar_thickness=bar_thickness,
            line_color=line_color,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth,
            max_display=max_display,
            title=title,
            fontsize=fontsize,
            save_path=save_path,
            display_plot=display_plot,
            return_fig=return_fig,
        )

    def get_intercept(self) -> float:
        """Returns the model's intercept.

        Returns
        -------
        float
        """
        return self.intercept_

    def get_coeffs(
        self,
        output: Optional[str] = "dataframe",
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """Returns model's coefficients in different format.

        Parameters
        ----------
        output : str, optional
            Output format with possible values of "dataframe" and "dict", by default "dataframe"

        Returns
        -------
        Union[Dict[str, float], pd.DataFrame]
        """
        check_var(
            output,
            var_name="output",
            dtypes=str,
            values=("dataframe", "dict"),
        )
        if output == "dataframe":
            return self._coeff_to_df()
        else:
            return self._coeff_to_dict()

    def get_params(self) -> Dict[str, Any]:
        """Returns model's parameters.

        Returns
        -------
        Dict[str, Any]
        """
        return self.params_

    def get_shap_explainer(self) -> shap.LinearExplainer:
        """Returns ``shap.LinearExplainer`` object.

        Returns
        -------
        shap.LinearExplainer
        """
        self._explainer()

        return self.shap_explainer_

    def get_cv_results(self) -> pd.DataFrame:
        """Returns model's cross-validation results.

        See Also
        --------
        :meth:`get_results()`

        Returns
        -------
        pd.DataFrame
        """
        return self.cv_results_

    def get_results(self) -> Dict[str, Any]:
        """Returns model's total results.

        See Also
        --------
        :meth:`get_cv_results()`

        Returns
        -------
        Dict[str, Any]
        """
        return self.results_

    def _dtrain(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Returns the features matrix and targets array.

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
        Tuple[pd.DataFrame, np.ndarray]
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

        return None

    def _dtest(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    ) -> None:
        """Returns the features matrix and targets array.

        Note that ``y_test`` is optional since it might not be available while validating the
        model (inference).

        Parameters
        ----------
        X_test : Union[pd.DataFrame, np.ndarray]
            Input data for training (features)

        y_test : Union[List[float], np.ndarray, pd.Series], optional
            Input ground truth for training (targets), by default None

        See Also
        --------
        :meth:`_dtrain()`

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
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

        return None

    def _model(self) -> glmnet.ElasticNet:
        """Fits a ``glmnet.ElasticNet`` model.

        Returns
        -------
        glmnet.ElasticNet
        """
        model = glmnet.ElasticNet(
            alpha=self.alpha,
            n_lambda=self.n_lambda,
            min_lambda_ratio=self.min_lambda_ratio,
            lambda_path=self.lambda_path,
            standardize=self.scale,
            fit_intercept=self.fit_intercept,
            cut_point=self.cut_point,
            n_splits=self.n_splits,
            scoring=self.metric,
            n_jobs=-1,
            tol=self.tolerance,
            max_iter=self.max_iter,
            random_state=self.random_state,
            max_features=self.max_features,
            verbose=False,
        )

        if self.sparse_matrix:
            # TODO(amir): expose groups: array, shape (n_samples,)
            # Group labels for the samples used while splitting the dataset into train/test set.
            # If the groups are specified, the groups will be passed to
            # ``sklearn.model_selection.GroupKFold```. If None, then data will be split randomly
            # for K-fold cross-validation via sklearn.model_selection.KFold.
            model.fit(
                X=df_to_csr(
                    self.X_train,
                    fillna=0.0,
                    verbose=False,
                ),
                y=self.y_train,
                sample_weight=None,
                relative_penalties=None,
                groups=None,
            )
        else:
            model.fit(
                X=self.X_train,
                y=self.y_train,
                sample_weight=None,
                relative_penalties=None,
                groups=None,
            )

        return model

    def _explainer(self) -> None:
        """Fits a ``shap.LinearExplainer`` using an independent masker.

        Returns
        -------
        None
        """
        # TODO(amir): currently, this does not make sense
        # https://shap.readthedocs.io/en/latest/generated/shap.explainers.Linear.html
        # https://stackoverflow.com/questions/66560839/what-do-maskers-really-do-in-shap-package-and-fit-them-to-train-or-test
        self.shap_explainer_ = shap.LinearExplainer(
            model=self.model_,
            masker=shap.maskers.Independent(
                data=self.X_train,
            ),
        )
        self.shap_values_train_ = self.shap_explainer_.shap_values(
            X=self.X_train,
        )
        self.shap_values_test_ = self.shap_explainer_.shap_values(
            X=self.X_test,
        )
        return None

    def _coeff_to_df(self) -> pd.DataFrame:
        """Returns the non-zero coeff for the ``lambda_best_``.

        See Also
        --------
        :meth:`_coeff_to_dict()`

        Returns
        -------
        pd.DataFrame
        """
        return (
            pd.DataFrame(
                data=self._coeff_to_dict().items(),
                columns=[
                    "feature",
                    "coeff",
                ],
            )
            .sort_values(
                by="coeff",
                ascending=False,
            )
            .reset_index(
                drop=True,
            )
        )

    def _coeff_to_dict(self) -> Dict[str, float]:
        """Returns the non-zero coeff for the ``lambda_best_``.

        See Also
        --------
        :meth:`_coeff_to_df()`

        Returns
        -------
        Dict[str, float]
        """
        idx = list(
            np.nonzero(
                np.reshape(
                    self.model_.coef_,
                    (1, -1),
                ),
            )[1],
        )
        # TODO(amir): why I have this here ? [self.model_.coef_[0][i] for i in idx],
        return dict(
            zip(
                [self.X_train.columns.tolist()[i] for i in idx],
                [self.model_.coef_.reshape(-1, self.model_.coef_.shape[-1])[0][i] for i in idx],
            ),
        )

    def _results(self) -> Dict[str, Any]:
        """Returns fitted ``glmnet.ElasticNet`` results as a nested dictionary.

        Returns
        -------
        Dict[str, Any]
        """
        results = {}
        results["coeff"] = self._coeff_to_dict()
        results["coeff_path"] = dict(
            zip(
                [f"{col}" for col in self.X_train.columns.tolist()],
                (self.model_.coef_path_.reshape(-1, self.model_.coef_path_.shape[-1])).tolist(),
            ),
        )
        results["cv_standard_error"] = self.model_.cv_standard_error_.tolist()
        results["cv_mean_score"] = self.model_.cv_mean_score_.tolist()
        results["lambda_path"] = self.model_.lambda_path_.tolist()
        results["lambda_best"] = self.model_.lambda_best_[0]
        results["lambda_max"] = self.model_.lambda_max_
        results["n_lambda"] = self.model_.n_lambda_
        results["intercept"] = self.model_.intercept_
        results["intercept_path"] = self.model_.intercept_path_.tolist()[0]
        results["params"] = self.model_.get_params()
        results["module"] = self.model_.__module__

        return results

    def _cv_results(self) -> pd.DataFrame:
        """Returns fitted ``glmnet.ElasticNet`` results.

        Results are including coeff. paths, intercept paths, lambda paths, and mean/standard-error
        of the metric through cross-validation.

        Returns
        -------
        pd.DataFrame
        """
        df = pd.DataFrame(
            (
                self.model_.coef_path_.reshape(
                    -1,
                    self.model_.coef_path_.shape[-1],
                )
            ).T,
            columns=[f"{col}_coeff_path" for col in self.X_train.columns.tolist()],
        )
        df["intercept_path"] = (
            self.model_.intercept_path_.reshape(
                -1,
                self.model_.intercept_path_.shape[-1],
            )
        ).T
        df["lambda_path"] = self.model_.lambda_path_
        df["cv_standard_error"] = self.model_.cv_standard_error_
        df["cv_mean_score"] = self.model_.cv_standard_error_

        return df
