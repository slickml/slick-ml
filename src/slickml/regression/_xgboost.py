from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.base import RegressorMixin

from slickml.base import BaseXGBoostEstimator
from slickml.utils import check_var
from slickml.visualization import (
    plot_shap_summary,
    plot_shap_waterfall,
    plot_xgb_feature_importance,
)


@dataclass
class XGBoostRegressor(BaseXGBoostEstimator, RegressorMixin):
    """XGBoost Regressor.

    This is wrapper using XGBoost regressor to train a XGBoost [xgboost-api]_ model using the number of
    boosting rounds from the inputs. This is also the base class for ``XGBoostCVRegressor``.

    Parameters
    ----------
    num_boost_round : int, optional
        Number of boosting rounds to fit a model, by default 200

    sparse_matrix : bool, optional
        Whether to convert the input features to sparse matrix with csr format or not. This would
        increase the speed of feature selection for relatively large/sparse datasets. Consequently,
        this would actually act like an un-optimize solution for dense feature matrix. Additionally,
        this feature cannot be used along with ``scale_mean=True`` standardizing the feature matrix
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

    params : Dict[str, Union[str, float, int]], optional
        Set of parameters required for fitting a Booster, by default {"eval_metric": "rmse",
        "tree_method": "hist", "objective": "reg:squarederror", "learning_rate": 0.05,
        "max_depth": 2, "min_child_weight": 1, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "subsample": 0.9, "max_delta_step": 1, "verbosity": 0, "nthread": 4}
        Other options for objective: ``"reg:logistic"``, ``"reg:squaredlogerror"``

    Methods
    -------
    fit(X_train, y_train)
        Fits a ``XGBoost.Booster`` to input training data. Proper ``dtrain_`` matrix based on chosen
        options i.e. ``sparse_matrix``, ``scale_mean``, ``scale_std`` is being created based on the
        passed ``X_train`` and ``y_train``

    predict(X_test, y_test)
        Returns prediction target values

    get_params()
        Returns final set of train parameters. The default set of parameters will be updated with
        the new ones that passed to ``params``

    get_default_params()
        Returns the default set of train parameters. The default set of parameters will be used when
        ``params=None``

    get_feature_importance()
        Returns the feature importance of the trained booster based on the given ``importance_type``

    get_shap_explainer()
        Returns the ``shap.TreeExplainer``

    plot_shap_summary()
        Visualizes Shapley values summary plot

    plot_shap_waterfall()
        Visualizes Shapley values waterfall plot

    Attributes
    ----------
    feature_importance_ : pd.DataFrame
        Features importance based on the given ``importance_type``

    scaler_ : StandardScaler, optional
        Standardization object when ``scale_mean=True`` or ``scale_std=True`` unless it is ``None``

    X_train_ : pd.DataFrame
        Fitted and Transformed features when ``scale_mean=True`` or ``scale_std=True``. In other case, it will
        be the same as the passed ``X_train`` features

    X_test_ : pd.DataFrame
        Transformed features when ``scale_mean=True`` or ``scale_std=True`` using `clf.scaler_` that
        has be fitted on ``X_train`` and ``y_train`` data. In other case, it will be the same as the
        passed ``X_train`` features

    dtrain_ : xgb.DMatrix
        Training data matrix via ``xgboost.DMatrix(clf.X_train_, clf.y_train)``

    dtest_ : xgb.DMatrix
        Testing data matrix via ``xgboost.DMatrix(clf.X_test_, clf.y_test)`` or
        ``xgboost.DMatrix(clf.X_test_, None)`` when ``y_test`` is not available in inference

    shap_values_train_ : np.ndarray
        Shapley values from ``TreeExplainer`` using ``X_train_``

    shap_values_test_ : np.ndarray
        Shapley values from ``TreeExplainer`` using ``X_test_``

    shap_explainer_ : shap.TreeExplainer
        Shap TreeExplainer object

    model_ : xgboost.Booster
        XGBoost Booster object

    References
    ----------
    .. [xgboost-api] https://xgboost.readthedocs.io/en/latest/python/python_api.html
    .. [markers-api] https://matplotlib.org/stable/api/markers_api.html
    .. [shap-api] https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html
    """

    num_boost_round: Optional[int] = 200
    sparse_matrix: Optional[bool] = False
    scale_mean: Optional[bool] = False
    scale_std: Optional[bool] = False
    importance_type: Optional[str] = "total_gain"
    params: Optional[Dict[str, Union[str, float, int]]] = None

    def __post_init__(self) -> None:
        """Post instantiation validations and assignments."""
        super().__post_init__()

        # The default set of params can be updated based on the given params by user
        _default_params = self._default_params()
        if self.params:
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
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[List[float], np.ndarray, pd.Series],
    ) -> None:
        """Fits a ``XGBoost.Booster`` to input training data.

        Notes
        -----
        Proper ``dtrain_`` matrix based on chosen options i.e. ``sparse_matrix``, ``scale_mean``,
        ``scale_std`` is being created based on the passed ``X_train`` and ``y_train``.

        Parameters
        ----------
        X_train : Union[pd.DataFrame, np.ndarray]
            Input data for training (features)

        y_train : Union[List[float], np.ndarray, pd.Series]
            Input ground truth for training (targets)

        See Also
        --------
        :meth:`xgboost.train()`

        Returns
        -------
        None
        """
        self.dtrain_ = self._dtrain(
            X_train=X_train,
            y_train=y_train,
        )
        self.model_ = self._model()
        self.feature_importance_ = self._imp_to_df()

        return None

    def predict(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Optional[Union[List[float], np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """Returns the prediction target (response) values.

        Parameters
        ----------
        X_test : Union[pd.DataFrame, np.ndarray]
            Input data for testing (features)

        y_test : Union[List[float], np.ndarray, pd.Series], optional
            Input ground truth for testing (targets)

        Returns
        -------
        np.ndarray
        """
        self.dtest_ = self._dtest(
            X_test=X_test,
            y_test=y_test,
        )
        self.y_pred_ = self.model_.predict(
            data=self.dtest_,
            output_margin=False,
        )

        return self.y_pred_

    def plot_feature_importance(
        self,
        figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (8, 5),
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
        """Visualizes the XGBoost feature importance as bar chart.

        Parameters
        ----------
        feature importance : pd.DataFrame
            Feature importance (``feature_importance_`` attribute)

        figsize : Tuple[Union[int, float], Union[int, float]], optional
            Figure size, by default (8, 5)

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
        return plot_xgb_feature_importance(
            feature_importance=self.feature_importance_,
            figsize=figsize,
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
        are available at [shap-api]_. Note that this function should be ran after the ``predict()``
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
            features = self.X_test_
        else:
            shap_values = self.shap_values_train_
            features = self.X_train_

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
        Note that this function should be ran after the ``predict()`` to make sure the
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
            features = self.X_test_
        else:
            shap_values = self.shap_values_train_
            features = self.X_train_

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

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns the feature importance of the trained booster based on the given ``importance_type``.

        Returns
        -------
        pd.DataFrame
        """
        return self.feature_importance_

    def get_shap_explainer(self) -> shap.TreeExplainer:
        """Returns the ``shap.TreeExplainer`` object.

        Returns
        -------
        shap.TreeExplainer
        """
        self._explainer()
        return self.shap_explainer_

    def _model(self) -> xgb.Booster:
        """Fits a ``XGBoost.Booster`` based on the given number of boosting round on ``dtrain_`` matrix.

        Returns
        -------
        xgb.Booster
        """
        return xgb.train(
            params=self.params,
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round,  # type: ignore
        )

    def _imp_to_df(self) -> pd.DataFrame:
        """Converts the feature importance object to ``pd.DataFrame``.

        Returns
        -------
        pd.DataFrame
        """
        data: Dict[str, List[float]] = {
            "feature": [],
            f"{self.importance_type}": [],
        }
        features_gain = self.model_.get_score(
            importance_type=self.importance_type,
        )
        for key, val in features_gain.items():
            data["feature"].append(key)
            data[f"{self.importance_type}"].append(val)

        return (
            pd.DataFrame(data)
            .sort_values(
                by=f"{self.importance_type}",
                ascending=False,
            )
            .reset_index(
                drop=True,
            )
        )

    def _explainer(self) -> None:
        """Fits a ``shap.TreeExplainer`` on the ``X_train_`` and ``X_test_`` data.

        Returns
        -------
        None
        """
        self.shap_explainer_ = shap.TreeExplainer(
            model=self.model_,
        )
        self.shap_values_test_ = self.shap_explainer_.shap_values(
            X=self.X_test_,
        )
        self.shap_values_train_ = self.shap_explainer_.shap_values(
            X=self.X_train_,
        )

        return None

    @staticmethod
    def _default_params() -> Dict[str, Union[str, float, int]]:
        """Default set of parameters when the class is being instantiated with ``params=None``.

        Returns
        -------
        Dict[str, Union[str, float, int]]
        """
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
