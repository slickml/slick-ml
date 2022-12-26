from typing import Any, Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from slickml.utils import check_var

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2


def plot_glmnet_cv_results(
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
    **kwargs: Dict[str, Any],
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
        color,
        var_name="color",
        dtypes=str,
    )
    check_var(
        errorbarcolor,
        var_name="errorbarcolor",
        dtypes=str,
    )
    check_var(
        maxlambdacolor,
        var_name="maxlambdacolor",
        dtypes=str,
    )
    check_var(
        bestlambdacolor,
        var_name="bestlambdacolor",
        dtypes=str,
    )
    check_var(
        linestyle,
        var_name="linestyle",
        dtypes=str,
    )
    check_var(
        fontsize,
        var_name="fontsize",
        dtypes=(int, float),
    )
    check_var(
        grid,
        var_name="grid",
        dtypes=bool,
    )
    check_var(
        legend,
        var_name="legend",
        dtypes=bool,
    )
    check_var(
        legendloc,
        var_name="legendloc",
        dtypes=(int, str),
    )
    if xlabel:
        check_var(
            xlabel,
            var_name="xlabel",
            dtypes=str,
        )
    else:
        xlabel = _xlabel()
    if ylabel:
        check_var(
            ylabel,
            var_name="ylabel",
            dtypes=str,
        )
    else:
        ylabel = _ylabel(**kwargs)
    if title:
        check_var(
            title,
            var_name="title",
            dtypes=str,
        )
    else:
        title = _title(**kwargs)
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
    # TODO(amir): standardize the fontsize
    fig, ax = plt.subplots(
        figsize=figsize,
    )
    ax.errorbar(
        -np.log(kwargs["lambda_path"]),
        kwargs["cv_mean_score"],
        yerr=kwargs["cv_standard_error"],
        color=color,
        ecolor=errorbarcolor,
        marker=marker,
        markersize=markersize,
    )
    ax.vlines(
        -np.log(kwargs["lambda_max"]),
        ymin=min(kwargs["cv_mean_score"]) - 0.05,  # type: ignore
        ymax=max(kwargs["cv_mean_score"]) + 0.05,  # type: ignore
        linestyles=linestyle,
        color=maxlambdacolor,
        label=r"max $\lambda$",
    )
    ax.vlines(
        -np.log(kwargs["lambda_best"]),
        ymin=min(kwargs["cv_mean_score"]) - 0.05,  # type: ignore
        ymax=max(kwargs["cv_mean_score"]) + 0.05,  # type: ignore
        linestyles=linestyle,
        color=bestlambdacolor,
        label=r"best $\lambda$",
    )
    ax.set_ylim(
        [
            min(kwargs["cv_mean_score"]) - 0.05,  # type: ignore
            max(kwargs["cv_mean_score"]) + 0.05,  # type: ignore
        ],
    )
    ax.set_xlabel(
        xlabel,
        fontsize=fontsize * 0.85,  # type: ignore
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize * 0.85,  # type: ignore
    )
    ax.set_title(
        title,
        fontsize=fontsize,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize * 0.75,  # type: ignore
    )
    ax.grid(grid)

    if legend:
        ax.legend(
            loc=legendloc,
            prop={
                "size": fontsize * 0.85,  # type: ignore
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


def plot_glmnet_coeff_path(
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
    **kwargs: Dict[str, Any],
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

    yscale : str, optional
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

    References
    ----------
    .. [yscale] https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html

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
        fontsize,
        var_name="fontsize",
        dtypes=(int, float),
    )
    check_var(
        grid,
        var_name="grid",
        dtypes=bool,
    )
    check_var(
        legend,
        var_name="legend",
        dtypes=bool,
    )
    check_var(
        legendloc,
        var_name="legendloc",
        dtypes=(int, str),
    )
    if xlabel:
        check_var(
            xlabel,
            var_name="xlabel",
            dtypes=str,
        )
    else:
        xlabel = _xlabel()
    check_var(
        ylabel,
        var_name="ylabel",
        dtypes=str,
    )
    if title:
        check_var(
            title,
            var_name="title",
            dtypes=str,
        )
    else:
        title = _title(**kwargs)

    check_var(
        bbox_to_anchor,
        var_name="bbox_to_anchor",
        dtypes=tuple,
    )
    check_var(
        yscale,
        var_name="yscale",
        dtypes=str,
        values=("linear", "log", "symlog", "logit"),
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

    fig, ax = plt.subplots(figsize=figsize)

    for feature, coeff_path in kwargs["coeff_path"].items():
        if feature in kwargs["coeff"]:
            ax.plot(
                -np.log(kwargs["lambda_path"]),
                coeff_path,
                linestyle=linestyle,
                label=feature,
            )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=fontsize * 0.75,  # type: ignore
    )
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize * 0.85,  # type: ignore
    )
    ax.set_xlabel(
        xlabel,
        fontsize=fontsize * 0.85,  # type: ignore
    )
    ax.set_title(
        title,
        fontsize=fontsize,
    )
    ax.set_yscale(yscale)
    ax.grid(grid)

    if legend:
        ax.legend(
            loc=legendloc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
            prop={
                "size": fontsize * 0.75,  # type: ignore
            },
            framealpha=0.0,
            fancybox=True,
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


def _xlabel() -> str:
    """Returns xlabel.

    Returns
    -------
    str
    """
    return r"-$Log(\lambda)$"


def _ylabel(**kwargs: Dict[str, Any]) -> str:
    """Returns ylabel.

    Parameters
    ----------
    kwargs : Dict[str, Any]
        Key-value pairs of results (``results_``)

    Returns
    -------
    str
    """
    if kwargs["params"]["scoring"] is None:
        if kwargs["module"] == "glmnet.linear":
            ylabel = rf"""{kwargs["params"]["n_splits"]}-Folds CV Mean $R^2$"""
        elif kwargs["module"] == "glmnet.logistic":
            ylabel = f"""{kwargs["params"]["n_splits"]}-Folds CV Mean ACCURACY"""
    else:
        ylabel = f"""{kwargs["params"]["n_splits"]}-Folds CV Mean {' '.join((kwargs["params"]["scoring"]).split("_")).upper()}"""

    return ylabel


def _title(**kwargs: Dict[str, Any]) -> str:
    """Returns title.

    Parameters
    ----------
    kwargs : Dict[str, Any]
        Key-value pairs of results (``results_``)

    Returns
    -------
    str
    """
    return rf"""Best $\lambda$ = {kwargs["lambda_best"]:.3f} with {len(kwargs["coeff"])} Features"""
