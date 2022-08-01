import warnings
from typing import Any, Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import seaborn as sns
from matplotlib.figure import Figure

from slickml.utils import check_var

# TODO(amir): this options should be set globally too
sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2
warnings.filterwarnings("ignore")


# TODO(amir): standardize all `plotting` params and options
# currently, lots of options are duplicated; this can be done globally
# TODO(amir): we are exposing this function in public API, therefore we need to probably expose
# `kwargs` values; unless this function wont be useful; we can also make this private
# TODO(amir): I think it would make sense to return `fig` so it can be tested
# TODO(amir): there are warnings on seaborn regarding passing positional args instead of kwargs
def plot_regression_metrics(
    figsize: Optional[Tuple[float, float]] = (12, 16),
    save_path: Optional[str] = None,
    display_plot: Optional[bool] = True,
    **kwargs: Dict[str, Any],
) -> Figure:
    """Visualizes regression metrics using `plotting_dict_` attribute of `RegressionMetrics`.

    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Figure size, by default (12, 16)

    save_path : str, optional
        The full or relative path to save the plot including the image format such as
        "myplot.png" or "../../myplot.pdf", by default None

    display_plot : bool, optional
        Whether to show the plot, by default True

    **kwargs : Dict[str, Any]
        Key-value pairs of regression metrics plot

    Returns
    -------
    Figure
    """
    check_var(
        figsize,
        var_name="figsize",
        dtypes=tuple,
    )
    check_var(
        display_plot,
        var_name="display_plot",
        dtypes=bool,
    )
    # TODO(amir): double check this
    if save_path is not None:
        check_var(
            save_path,
            var_name="save_path",
            dtypes=str,
        )

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=figsize,
    )

    # subplot 1: actual vs predicted
    sns.regplot(
        kwargs["y_true"],
        kwargs["y_pred"],
        marker="o",
        scatter_kws={
            "edgecolors": "navy",
        },
        color="#B3C3F3",
        fit_reg=False,
        ax=ax1,
    )
    ax1.plot(
        [
            kwargs["y_true"].min(),
            kwargs["y_true"].max(),
        ],
        [
            kwargs["y_true"].min(),
            kwargs["y_true"].max(),
        ],
        "r--",
        lw=3,
    )
    ax1.set(
        xlabel="Actual Values",
        ylabel="Predicted Values",
        title="Actual-Predicted",
    )
    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax1.text(
        0.05,
        0.93,
        f"MAPE = {kwargs['mape']:.3f}",
        fontsize=12,
        transform=ax1.transAxes,
    )
    ax1.text(
        0.05,
        0.86,
        f"$R^2$ = {kwargs['r2']:.3f}",
        fontsize=12,
        transform=ax1.transAxes,
    )

    # -----------------------------------
    # subplot 2: Q-Q Normal Plot
    scp.stats.probplot(
        kwargs["y_residual"],
        fit=True,
        dist="norm",
        plot=ax2,
    )
    ax2.get_lines()[0].set_marker("o")
    ax2.get_lines()[0].set_markerfacecolor("#B3C3F3")
    ax2.get_lines()[0].set_markeredgecolor("navy")
    ax2.get_lines()[0].set_markersize(6.0)
    ax2.get_lines()[1].set_linewidth(3.0)
    ax2.get_lines()[1].set_linestyle("--")
    ax2.set(
        xlabel="Quantiles",
        ylabel="Residuals",
        title="Q-Q",
    )
    ax2.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )

    # -----------------------------------
    # subplot 3: Residuals vs Fitted
    sns.residplot(
        kwargs["y_pred"],
        kwargs["y_true"],
        lowess=True,
        order=1,
        line_kws={
            "color": "red",
            "lw": 3,
            "ls": "--",
            "alpha": 1,
        },
        scatter_kws={
            "edgecolors": "navy",
        },
        color="#B3C3F3",
        robust=True,
        ax=ax3,
    )
    ax3.set(
        xlabel="Predicted Values",
        ylabel="Residuals",
        title="Residuals-Predicted",
    )
    ax3.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )

    # -----------------------------------
    # subplot 4: Sqrt Standard Residuals vs Fitted
    sns.regplot(
        kwargs["y_pred"],
        kwargs["y_residual_normsq"],
        lowess=True,
        line_kws={
            "color": "red",
            "lw": 3,
            "ls": "--",
            "alpha": 1,
        },
        scatter_kws={"edgecolors": "navy"},
        color="#B3C3F3",
        ax=ax4,
    )
    ax4.set(
        xlabel="Predicted Values",
        ylabel="Standardized Residuals Norm",
        title="Scale-Location",
    )
    ax4.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )

    # -----------------------------------
    # subplot 5: Histogram of Coeff. of Variations
    freqs, _, _ = ax5.hist(
        kwargs["y_ratio"],
        histtype="bar",
        bins=np.arange(0.75, 1.25, 0.01),
        alpha=1.0,
        color="#B3C3F3",
        edgecolor="navy",
    )

    ax5.set_xticks(
        [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
        ],
    )
    ax5.set_xticklabels(
        [
            "Less",
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            "More",
        ],
        rotation=30,
    )
    ax5.set(
        ylabel="Frequency",
        title="Prediction Variation",
    )
    ax5.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax5_ylim = max(freqs)
    ax5.text(
        0.65,
        ax5_ylim,
        rf"""$\mu$ = {kwargs['mean_y_ratio']:.3f}""",
        fontsize=12,
    )
    ax5.text(
        0.65,
        0.93 * ax5_ylim,
        f"CV = {kwargs['cv_y_ratio']:.3f}",
        fontsize=12,
    )

    # -----------------------------------
    # subplot 6: REC
    ax6.plot(
        kwargs["deviation"],
        kwargs["accuracy"],
        color="red",
        label=f"AUC = {kwargs['auc_rec']:.3f}",
    )
    ax6.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="Deviation",
        ylabel="Accuracy",
        title="REC Curve",
    )
    ax6.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax6.legend(
        prop={"size": 12},
        loc=4,
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

    # TODO(amir): investigate the options to return axes as well if needed
    return fig
