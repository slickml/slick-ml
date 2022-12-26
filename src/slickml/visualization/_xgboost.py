from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from slickml.utils import check_var

# TODO(amir): this options should be set globally too
sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2


# TODO(amir): we can prolly take out the `bar_plot()` into a general pattern
# as part of base-viz functions and just call it; the same pattern gets repeated in glmnet
# or any time we wanna have a horizontal/vertical bar chart
# TODO(amir): for now we ship this; but we gotta come back to this when the main refactor is done
# TODO(amir): add the functionality for vertical plot as well
def plot_xgb_feature_importance(
    feature_importance: pd.DataFrame,
    *,
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
    """Visualizes the XGBoost feature importance as a bar chart.

    Notes
    -----
    This plotting function can be used along with ``feature_importance_`` attribute of
    any of ``XGBoostClassifier``, ``XGBoostCVClassifier``, ``XGBoostRegressor``, or
    ``XGBoostCVRegressor`` classes.

    Parameters
    ----------
    feature importance : pd.DataFrame
        Feature importance (``feature_importance_`` attribute)

    figsize : tuple, optional
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

    See Also
    --------
    :class:`slickml.classification.XGBoostClassifier`
    :class:`slickml.classification.XGBoostCVClassifier`
    :class:`slickml.regression.XGBoostRegressor`
    :class:`slickml.regression.XGBoostCVRegressor`

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
    if save_path:
        check_var(
            save_path,
            var_name="save_path",
            dtypes=str,
        )

    # TODO(amir): take this out into a utility functions
    # prep feature importance
    cols = feature_importance.columns.tolist()
    coly, colx = cols[0], cols[1]
    feature_importance = feature_importance.reindex(
        index=range(len(feature_importance) - 1, -1, -1),
    )

    fig, ax = plt.subplots(
        figsize=figsize,
    )
    ax.hlines(
        y=feature_importance[coly],
        xmin=0,
        xmax=feature_importance[colx],
        color=color,
    )
    ax.plot(
        feature_importance[colx],
        feature_importance[coly].values,
        marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor,
        markeredgewidth=markeredgewidth,
    )
    # find max value, and put importance values on the plot
    max_val = feature_importance[colx].max()
    for index, value in enumerate(feature_importance[colx]):
        ax.text(
            value + 0.05 * max_val,
            index * 1.01,
            f"{value:.2f}",
        )

    ax.set_xlabel(
        f"{' '.join(colx.split('_')).title()}",
        fontsize=fontsize,
    )
    ax.set_ylabel(
        f"{coly.title()}",
        fontsize=fontsize,
    )
    ax.set_title(
        "Feature Importance",
        fontsize=fontsize,
    )
    ax.set(
        xlim=[
            None,
            feature_importance[colx].max() * 1.13,
        ],
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


def plot_xgb_cv_results(
    cv_results: pd.DataFrame,
    *,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = (8, 5),
    linestyle: Optional[str] = "--",
    train_label: Optional[str] = "Train",
    test_label: Optional[str] = "Test",
    train_color: Optional[str] = "navy",
    train_std_color: Optional[str] = "#B3C3F3",
    test_color: Optional[str] = "purple",
    test_std_color: Optional[str] = "#D0AAF3",
    save_path: Optional[str] = None,
    display_plot: Optional[bool] = False,
    return_fig: Optional[bool] = False,
) -> Optional[Figure]:
    """Visualizes the cv_results of ``XGBoostCVClassifier``.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Cross-validation results

    figsize : Tuple[Union[int, float], Union[int, float]], optional
        Figure size, by default (8, 5)

    linestyle : str, optional
        Style of lines [linestyles-api]_, by default "--"

    train_label : str, optional
        Label in the figure legend for the train line, by default "Train"

    test_label : str, optional
        Label in the figure legend for the test line, by default "Test"

    train_color : str, optional
        Color of the training line, by default "navy"

    train_std_color : str, optional
        Color of the edge color of the training std bars, by default "#B3C3F3"

    test_color : str, optional
        Color of the testing line, by default "purple"

    test_std_color : str, optional
        Color of the edge color of the testing std bars, by default "#D0AAF3"

    save_path : str, optional
        The full or relative path to save the plot including the image format such as
        "myplot.png" or "../../myplot.pdf", by default None

    display_plot : bool, optional
        Whether to show the plot, by default False

    return_fig : bool, optional
        Whether to return figure object, by default False

    See Also
    --------
    :class:`slickml.classification.XGBoostCVClassifier`

    References
    ----------
    .. [linestyles-api] https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

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
        linestyle,
        var_name="linestyle",
        dtypes=str,
    )
    check_var(
        train_label,
        var_name="train_label",
        dtypes=str,
    )
    check_var(
        test_label,
        var_name="test_label",
        dtypes=str,
    )
    check_var(
        train_color,
        var_name="train_color",
        dtypes=str,
    )
    check_var(
        train_std_color,
        var_name="train_std_color",
        dtypes=str,
    )
    check_var(
        test_color,
        var_name="test_color",
        dtypes=str,
    )
    check_var(
        test_std_color,
        var_name="test_std_color",
        dtypes=str,
    )
    if save_path:
        check_var(
            save_path,
            var_name="save_path",
            dtypes=str,
        )

    # TODO(amir): optimize this part
    # update metrics capitalizations for title/labels
    metric = cv_results.columns.tolist()[0].split("-")[1]
    metrics = [
        "AUC",
        "AUCPR",
        "Error",
        "LogLoss",
        "MAE",
        "RMSE",
        "RMSLE",
    ]
    for m in metrics:
        if m.lower() == metric:
            metric = m

    fig, ax = plt.subplots(
        figsize=figsize,
    )
    ax.errorbar(
        x=range(cv_results.shape[0]),
        y=cv_results.iloc[:, 0],
        yerr=cv_results.iloc[:, 1],
        fmt=linestyle,
        ecolor=train_std_color,
        c=train_color,
        label=train_label,
    )
    ax.errorbar(
        x=range(cv_results.shape[0]),
        y=cv_results.iloc[:, 2],
        yerr=cv_results.iloc[:, 3],
        fmt=linestyle,
        ecolor=test_std_color,
        c=test_color,
        label=test_label,
    )
    ax.set_xlabel(
        "# of Boosting Rounds",
        fontsize=12,
    )
    ax.set_ylabel(
        f"""{metric}""",
        fontsize=12,
    )
    ax.set_title(
        f"""{metric} Evolution vs Boosting Rounds""",
        fontsize=12,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax.legend(
        loc=0,
        prop={
            "size": 12,
        },
        framealpha=0.0,
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
