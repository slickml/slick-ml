from typing import Any, Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score

from slickml.utils import check_var

# TODO(amir): this options should be set globally too
sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2


# TODO(amir): globally adjust options, fonts, ...
def plot_binary_classification_metrics(
    figsize: Optional[Tuple[float, float]] = (12, 12),
    save_path: Optional[str] = None,
    display_plot: Optional[bool] = False,
    return_fig: Optional[bool] = False,
    **kwargs: Dict[str, Any],
) -> Optional[Figure]:
    """Visualizes binary classification metrics using ``plotting_dict_`` attribute of ``BinaryClassificationMetrics``.

    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Figure size, by default (12, 12)

    save_path : str, optional
        The full or relative path to save the plot including the image format such as
        "myplot.png" or "../../myplot.pdf", by default None

    display_plot : bool, optional
        Whether to show the plot, by default False

    return_fig : bool, optional
        Whether to return figure object, by default False

    **kwargs : Dict[str, Any]
        Key-value pairs of regression metrics plot

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

    # TODO(amir): move this to a function ?
    # prepare thresholds for plotting
    thr_set1 = np.arange(
        min(kwargs["roc_thresholds"]),
        max(kwargs["roc_thresholds"]),
        0.01,
    )
    thr_set2 = np.arange(
        min(kwargs["pr_thresholds"]),
        max(kwargs["pr_thresholds"]),
        0.01,
    )
    f1_list = [
        2
        * (kwargs["precision_list"][i] * kwargs["recall_list"][i])  # type: ignore
        / (kwargs["precision_list"][i] + kwargs["recall_list"][i])  # type: ignore
        for i in range(len(kwargs["precision_list"]))
    ]
    queue_rate_list = [(kwargs["y_pred_proba"] >= thr).mean() for thr in thr_set2]
    accuracy_list = [
        accuracy_score(kwargs["y_true"], (kwargs["y_pred_proba"] >= thr).astype(int))
        for thr in thr_set1
    ]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
    )

    # -----------------------------------
    # subplot 1: roc curve
    ax1.plot(
        kwargs["fpr_list"],
        kwargs["tpr_list"],
        color="red",
        label=f"AUC = {kwargs['auc_roc']:.3f}",
    )
    ax1.plot(
        kwargs["fpr_list"][kwargs["youden_index"]],  # type: ignore
        kwargs["tpr_list"][kwargs["youden_index"]],  # type: ignore
        marker="o",
        color="navy",
        ms=10,
    )
    ax1.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        title="ROC Curve",
    )
    ax1.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax1.legend(
        prop={"size": 12},
        loc=0,
        framealpha=0.0,
    )
    ax1.annotate(
        f"Threshold = {kwargs['youden_threshold']:.3f}",
        xy=(
            kwargs["fpr_list"][kwargs["youden_index"]],  # type: ignore
            kwargs["tpr_list"][kwargs["youden_index"]],  # type: ignore
        ),
        xycoords="data",
        xytext=(
            kwargs["fpr_list"][kwargs["youden_index"]] + 0.4,  # type: ignore
            kwargs["tpr_list"][kwargs["youden_index"]] - 0.4,  # type: ignore
        ),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
        ),
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    # -----------------------------------
    # subplot 2: preferred scores vs thresholds
    ax2.plot(
        kwargs["roc_thresholds"],
        1 - kwargs["fpr_list"],  # type: ignore
        label="Specificity",
    )
    ax2.plot(
        kwargs["roc_thresholds"],
        kwargs["tpr_list"],
        label="Sensitivity",
    )
    ax2.plot(
        thr_set1,
        accuracy_list,
        label="Accuracy",
    )
    ax2.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="Threshold",
        ylabel="Score",
        title="Preferred Scores vs Thresholds",
    )
    ax2.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax2.legend(
        bbox_to_anchor=(1.2, 0.5),
        loc="center",
        ncol=1,
        framealpha=0.0,
    )
    ax2.axvline(
        kwargs["sens_spec_threshold"],
        color="k",
        ls="--",
    )
    if isinstance(kwargs["sens_spec_threshold"], float) and kwargs["sens_spec_threshold"] <= 0.5:
        ax2.annotate(
            f"Threshold = {kwargs['sens_spec_threshold']:.3f}",
            xy=(kwargs["sens_spec_threshold"], 0.05),
            xycoords="data",
            xytext=(kwargs["sens_spec_threshold"] + 0.1, 0.05),  # type: ignore
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    else:
        ax2.annotate(
            f"Threshold = {kwargs['sens_spec_threshold']:.3f}",
            xy=(kwargs["sens_spec_threshold"], 0.05),
            xycoords="data",
            xytext=(kwargs["sens_spec_threshold"] - 0.4, 0.05),  # type: ignore
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    # -----------------------------------
    # subplot 3: precision-recall curve
    ax3.plot(
        kwargs["recall_list"],
        kwargs["precision_list"],
        color="red",
        label=f"PR AUC ={kwargs['auc_pr']:.3f}",
    )
    ax3.plot(
        kwargs["recall_list"][kwargs["prec_rec_index"]],  # type: ignore
        kwargs["precision_list"][kwargs["prec_rec_index"]],  # type: ignore
        marker="o",
        color="navy",
        ms=10,
    )
    ax3.axvline(
        x=kwargs["recall_list"][kwargs["prec_rec_index"]],  # type: ignore
        ymin=kwargs["recall_list"][kwargs["prec_rec_index"]],  # type: ignore
        ymax=kwargs["precision_list"][kwargs["prec_rec_index"]],  # type: ignore
        color="navy",
        ls="--",
    )
    ax3.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve",
    )
    ax3.legend(
        prop={"size": 12},
        loc=0,
        framealpha=0.0,
    )
    ax3.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax3.annotate(
        f"Threshold = {kwargs['prec_rec_threshold']:.3f}",
        xy=(
            kwargs["recall_list"][kwargs["prec_rec_index"]],  # type: ignore
            kwargs["precision_list"][kwargs["prec_rec_index"]],  # type: ignore
        ),
        xycoords="data",
        xytext=(
            kwargs["recall_list"][kwargs["prec_rec_index"]] - 0.4,  # type: ignore
            kwargs["precision_list"][kwargs["prec_rec_index"]] - 0.4,  # type: ignore
        ),
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    # -----------------------------------
    # subplot 4: preferred Scores vs Thresholds
    ax4.plot(
        kwargs["pr_thresholds"],
        kwargs["precision_list"][1:],  # type: ignore
        label="Precision",
    )
    ax4.plot(
        kwargs["pr_thresholds"],
        kwargs["recall_list"][1:],  # type: ignore
        label="Recall",
    )
    ax4.plot(
        kwargs["pr_thresholds"],
        f1_list[1:],
        label="F1-Score",
    )
    ax4.plot(
        thr_set2,
        queue_rate_list,
        label="Queue Rate",
    )
    ax4.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="Threshold",
        ylabel="Score",
        title="Preferred Scores vs Thresholds",
    )
    ax4.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax4.legend(
        bbox_to_anchor=(1.2, 0.5),
        loc="center",
        ncol=1,
        framealpha=0.0,
    )
    ax4.axvline(
        kwargs["prec_rec_threshold"],
        color="k",
        ls="--",
    )

    if isinstance(kwargs["prec_rec_threshold"], float) and kwargs["prec_rec_threshold"] <= 0.5:
        ax4.annotate(
            f"Threshold = {kwargs['prec_rec_threshold']:.3f}",
            xy=(kwargs["prec_rec_threshold"], 0.03),
            xycoords="data",
            xytext=(kwargs["prec_rec_threshold"] + 0.1, 0.03),  # type: ignore
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    else:
        ax4.annotate(
            f"Threshold = {kwargs['prec_rec_threshold']:.3f}",
            xy=(kwargs["prec_rec_threshold"], 0.03),
            xycoords="data",
            xytext=(kwargs["prec_rec_threshold"] - 0.4, 0.03),  # type: ignore
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
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


# TODO(amir): standardize all `plotting` params and options
# currently, lots of options are duplicated; this can be done globally
# TODO(amir): we are exposing this function in public API, therefore we need to probably expose
# `kwargs` values; unless this function wont be useful; we can also make this private
# TODO(amir): I think it would make sense to return `fig` so it can be tested
# TODO(amir): there are warnings on seaborn regarding passing positional args instead of kwargs
def plot_regression_metrics(
    figsize: Optional[Tuple[float, float]] = (12, 16),
    save_path: Optional[str] = None,
    display_plot: Optional[bool] = False,
    return_fig: Optional[bool] = False,
    **kwargs: Dict[str, Any],
) -> Figure:
    """Visualizes regression metrics using ``plotting_dict_`` attribute of ``RegressionMetrics``.

    Parameters
    ----------
    figsize : Tuple[float, float], optional
        Figure size, by default (12, 16)

    save_path : str, optional
        The full or relative path to save the plot including the image format such as
        "myplot.png" or "../../myplot.pdf", by default None

    display_plot : bool, optional
        Whether to show the plot, by default False

    return_fig : bool, optional
        Whether to return figure object, by default False

    **kwargs : Dict[str, Any]
        Key-value pairs of regression metrics plot

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

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=figsize,
    )

    # subplot 1: actual vs predicted
    sns.regplot(
        x=kwargs["y_true"],
        y=kwargs["y_pred"],
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
            kwargs["y_true"].min(),  # type: ignore
            kwargs["y_true"].max(),  # type: ignore
        ],
        [
            kwargs["y_true"].min(),  # type: ignore
            kwargs["y_true"].max(),  # type: ignore
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
        x=kwargs["y_pred"],
        y=kwargs["y_true"],
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
        x=kwargs["y_pred"],
        y=kwargs["y_residual_normsq"],
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

    if return_fig:
        return fig

    # TODO(amir): investigate the options to return axes as well if needed
    # TODO(amir): investigate a better option for `plt.show()`; currently, no matter what the
    # figure is being shown when returning `fig`; what would be a global pattern here that can be
    # tested as well? Should we sacrifice the testing part here?
    return None
