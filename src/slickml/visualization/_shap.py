from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from slickml.utils import check_var

# TODO(amir): this options should be set globally too
sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2


# TODO(amir): double check this for `multi-outputs` problems
# TODO(amir): implement return_fig: Optional[bool] = False,
def plot_shap_summary(
    shap_values: np.ndarray,
    features: Union[pd.DataFrame, np.ndarray],
    *,
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
    are available at [shap-api]_.

    Parameters
    ----------
    shap_values : np.ndarray
        Calculated SHAP values array. For single output explanations such as binary classification
        problems, this will be a matrix of SHAP values with a shape of ``(n_samples, n_features)``.
        Additionally, for multi-output explanations this would be a list of such matrices of
        SHAP values (``List[np.ndarray]``)

    features : Union[pd.DataFrame, np.ndarray]
        The feature matrix that was used to calculate the SHAP values. For the case of Numpy array
        it is recommened to pass the ``feature_names`` list as well for better visualization results

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

    References
    ----------
    .. [shap-api] https://shap-lrjball.readthedocs.io/en/latest/generated/shap.summary_plot.html

    Returns
    -------
    None
    """
    check_var(
        shap_values,
        var_name="shap_values",
        dtypes=np.ndarray,
    )
    check_var(
        features,
        var_name="features",
        dtypes=(np.ndarray, pd.DataFrame),
    )
    check_var(
        plot_type,
        var_name="plot_type",
        dtypes=str,
        values=(
            "bar",
            "dot",
            "violin",
            "layered_violin",
            "compact_dot",
        ),
    )
    check_var(
        figsize,
        var_name="figsize",
        dtypes=(str, tuple),
    )
    if color:
        check_var(
            color,
            var_name="color",
            dtypes=str,
        )
    if not color and plot_type == "bar":
        color = "#D0AAF3"
    if cmap:
        check_var(
            cmap,
            var_name="cmap",
            dtypes=LinearSegmentedColormap,
        )
    check_var(
        max_display,
        var_name="max_display",
        dtypes=int,
    )
    if feature_names:
        check_var(
            feature_names,
            var_name="feature_names",
            dtypes=list,
        )
    check_var(
        layered_violin_max_num_bins,
        var_name="layered_violin_max_num_bins",
        dtypes=int,
    )
    if title:
        check_var(
            title,
            var_name="title",
            dtypes=str,
        )
    check_var(
        sort,
        var_name="sort",
        dtypes=bool,
    )
    check_var(
        color_bar,
        var_name="color_bar",
        dtypes=bool,
    )
    if class_names:
        check_var(
            class_names,
            var_name="class_names",
            dtypes=list,
        )
    if class_inds:
        check_var(
            class_inds,
            var_name="class_inds",
            dtypes=list,
        )
    check_var(
        color_bar_label,
        var_name="color_bar_label",
        dtypes=str,
    )
    if save_path:
        check_var(
            save_path,
            var_name="save_path",
            dtypes=str,
        )
    check_var(
        display_plot,
        var_name="display_plot",
        dtypes=bool,
    )

    shap.summary_plot(
        shap_values,
        features,
        plot_type=plot_type,
        plot_size=figsize,
        color=color,
        cmap=cmap,
        max_display=max_display,
        feature_names=feature_names,
        title=title,
        show=display_plot,
        sort=sort,
        color_bar=color_bar,
        layered_violin_max_num_bins=layered_violin_max_num_bins,
        class_names=class_names,
        class_inds=class_inds,
        color_bar_label=color_bar_label,
    )

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=200,
        )

    if display_plot:
        plt.show()

    return None


def plot_shap_waterfall(
    shap_values: np.ndarray,
    features: Union[pd.DataFrame, np.ndarray],
    *,
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
    """Visualizes the Shapley values as a waterfall plot. pl
    This function is a helper function to plot the shap summary plot
    based on all types of shap explainers including tree, linear, and dnn.

    Parameters
    ----------
    shap_values : np.ndarray
        Calculated SHAP values array. For single output explanations such as binary classification
        problems, this will be a matrix of SHAP values with a shape of ``(n_samples, n_features)``.
        Additionally, for multi-output explanations this would be a list of such matrices of
        SHAP values (``List[np.ndarray]``)

    features : Union[pd.DataFrame, np.ndarray]
        The feature matrix that was used to calculate the SHAP values. For the case of Numpy array
        it is recommened to pass the ``feature_names`` list as well for better visualization results

    figsize : Tuple[float, float], optional
        Figure size, by default (8, 5)

    bar_color : str, optional
        Color of the horizontal bar lines, "#B3C3F3"

    bar_thickness : Union[float, int], optional
        Thickness (hight) of the horizontal bar lines, by default 0.5

    line_color : str, optional
        Color of the line plot, by default "purple"

    marker : str, optional
        Marker style of the lollipops. More valid marker styles can be found at [markers-api]_, by default "o"

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

    References
    ----------
    .. [markers-api] https://matplotlib.org/stable/api/markers_api.html

    Returns
    -------
    Figure, optional
    """
    check_var(
        shap_values,
        var_name="shap_values",
        dtypes=np.ndarray,
    )
    check_var(
        features,
        var_name="features",
        dtypes=(np.ndarray, pd.DataFrame),
    )
    check_var(
        figsize,
        var_name="figsize",
        dtypes=(str, tuple),
    )
    check_var(
        bar_color,
        var_name="bar_color",
        dtypes=str,
    )
    check_var(
        bar_thickness,
        var_name="bar_thickness",
        dtypes=(float, int),
    )
    check_var(
        line_color,
        var_name="line_color",
        dtypes=str,
    )
    check_var(
        marker,
        var_name="marker",
        dtypes=str,
    )
    check_var(
        markersize,
        var_name="markersize",
        dtypes=(int, float),
    )
    check_var(
        markeredgecolor,
        var_name="markeredgecolor",
        dtypes=str,
    )
    check_var(
        markerfacecolor,
        var_name="markerfacecolor",
        dtypes=str,
    )
    check_var(
        markeredgewidth,
        var_name="markeredgewidth",
        dtypes=(int, float),
    )
    check_var(
        max_display,
        var_name="max_display",
        dtypes=int,
    )
    if title:
        check_var(
            title,
            var_name="title",
            dtypes=str,
        )
    check_var(
        fontsize,
        var_name="font_size",
        dtypes=(int, float),
    )
    if save_path:
        check_var(
            save_path,
            var_name="save_path",
            dtypes=str,
        )
    check_var(
        display_plot,
        var_name="display_plot",
        dtypes=bool,
    )
    check_var(
        return_fig,
        var_name="return_fig",
        dtypes=bool,
    )

    # main calculation of cum/comp ratios
    feature_names = features.columns
    shap_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    feature_names = feature_names[np.argsort(shap_ratio)[::-1]]
    shap_ratio_order = np.sort(shap_ratio)[::-1]
    cum_sum = np.cumsum(shap_ratio_order)
    feature_names = feature_names[:max_display]
    shap_ratio_order = shap_ratio_order[:max_display]
    cum_sum = cum_sum[:max_display]

    fig, ax1 = plt.subplots(figsize=figsize)

    # subplot 1: cumsum shap line-marker plot
    ax1.plot(
        cum_sum[::-1],
        feature_names[::-1],
        color=line_color,
        marker=marker,
        markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor,
        markeredgewidth=markeredgewidth,
        markersize=markersize,
    )

    # subplot2: barplot
    ax2 = ax1.twiny()
    ax2.barh(
        feature_names[::-1],
        shap_ratio_order[::-1],
        height=bar_thickness,
        alpha=0.6,
        color=bar_color,
    )
    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(
        np.arange(
            0,
            round(cum_sum.max(), -1) + 1,
            10,
        ),
    )
    ax2.set_xticks(
        np.arange(
            0,
            round(shap_ratio_order.max(), -1) + 1,
            10,
        ),
    )
    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
    )
    ax1.set(
        ylim=[
            -1,
            len(feature_names),
        ],
        xlabel="Cumulative Ratio (%)",
        ylabel="Feature",
        title=title,
    )
    ax2.set(
        xlabel="Composition Ratio (%)",
    )

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=200,
        )

    if display_plot:
        plt.show()

    if return_fig:
        return fig

    return None
