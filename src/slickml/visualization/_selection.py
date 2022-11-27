from typing import Any, Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from slickml.utils import check_var

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2


# TODO(amir): take out the bar chart function into `_base.py` plotting
# technically, this is the same as `plot_xgb_feature_importance()`
# and call it for feature frequency and feature importance
def plot_xfs_feature_frequency(
    freq: pd.DataFrame,
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

    See Also
    --------
    :class:`slickml.selection.XGBoostFeatureSelector`

    References
    ----------
    .. [markers-api] https://matplotlib.org/stable/api/markers_api.html

    Returns
    -------
    Figure, optional
    """
    check_var(
        figsize,
        var_name="figsize",
        dtypes=tuple,
    )
    check_var(
        show_freq_pct,
        var_name="show_freq_pct",
        dtypes=bool,
    )
    check_var(
        color,
        var_name="color",
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
        dtypes=(float, int),
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
        fontsize,
        var_name="fontsize",
        dtypes=(int, float),
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

    # choose whether the feature frequency is being plotted by count or percentage
    if show_freq_pct:
        col = "Frequency (%)"
    else:
        col = "Frequency"

    # reindex freq
    freq = freq.reindex(
        index=range(len(freq) - 1, -1, -1),
    )

    fig, ax = plt.subplots(figsize=figsize)
    # TODO(amir): add vline option too ?
    ax.hlines(
        y=freq["Feature"],
        xmin=0,
        xmax=freq[col],
        color=color,
    )
    ax.plot(
        freq[col],
        freq["Feature"].values,
        marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor,
        markeredgewidth=markeredgewidth,
    )
    ax.set_xlabel(
        f"{col}",
        fontsize=fontsize,
    )
    ax.set_ylabel(
        "Feature",
        fontsize=fontsize,
    )
    ax.set_title(
        "Selected Features Frequency",
        fontsize=fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize,
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


def plot_xfs_cv_results(
    *,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (10, 8),
    internalcvcolor: Optional[str] = "#4169E1",
    externalcvcolor: Optional[str] = "#8A2BE2",
    sharex: Optional[bool] = False,
    sharey: Optional[bool] = False,
    save_path: Optional[str] = None,
    display_plot: Optional[bool] = True,
    return_fig: Optional[bool] = False,
    **kwargs: Dict[str, Any],
) -> Optional[Figure]:
    """Visualizies the cross-validation results of ``XGBoostFeatureSelector``.

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

    See Also
    --------
    :class:`slickml.selection.XGBoostFeatureSelector`

    Refereces
    ---------
    .. [seaborn-distplot-deprecation] https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751

    Returns
    -------
    Figure, optional
    """

    check_var(
        figsize,
        var_name="figsize",
        dtypes=tuple,
    )
    check_var(
        internalcvcolor,
        var_name="internalcvcolor",
        dtypes=str,
    )
    check_var(
        externalcvcolor,
        var_name="externalcvcolor",
        dtypes=str,
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    sns.histplot(
        data=kwargs["int_cv_train"],
        color=internalcvcolor,
        ax=ax1,
        stat="density",
        kde=True,
        kde_kws={
            "cut": 3,
        },
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
    )
    sns.histplot(
        data=kwargs["int_cv_test"],
        color=internalcvcolor,
        ax=ax2,
        stat="density",
        kde=True,
        kde_kws={
            "cut": 3,
        },
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
    )
    sns.histplot(
        data=kwargs["ext_cv_train"],
        color=externalcvcolor,
        ax=ax3,
        stat="density",
        kde=True,
        kde_kws={
            "cut": 3,
        },
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
    )
    sns.histplot(
        data=kwargs["ext_cv_test"],
        color=externalcvcolor,
        ax=ax4,
        stat="density",
        kde=True,
        kde_kws={
            "cut": 3,
        },
        alpha=0.4,
        edgecolor=(1, 1, 1, 0.4),
    )
    ax1.set(
        title=f"Internal {kwargs['n_splits']}-Folds CV {kwargs['metric']} - Train",
    )
    ax2.set(
        title=f"Internal {kwargs['n_splits']}-Folds CV {kwargs['metric']} - Test",
    )
    ax3.set(
        title=f"External {kwargs['n_splits']}-Folds CV {kwargs['eval_metric']} - Train",
    )
    ax4.set(
        title=f"External {kwargs['n_splits']}-Folds CV {kwargs['eval_metric']} - Test",
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
