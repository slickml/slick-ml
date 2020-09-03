import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

from IPython.core.display import display, HTML
import warnings

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))


def plot_binary_classification_metrics(figsize=None, **kwargs):
    """Function to plot binary classification metrics.
    This function is a helper function based on the plotting_dict
    attribute of the BinaryClassificationMetrics class.
    Parameters
    ----------
    figsize: tuple, optional, (default=(12, 12))
        Figure size
    """

    # initializing figsize
    if figsize is None:
        figsize = (12, 12)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize!")

    # prepare thresholds for plotting
    thr_set1 = np.arange(
        min(kwargs["roc_thresholds"]), max(kwargs["roc_thresholds"]), 0.01
    )
    thr_set2 = np.arange(
        min(kwargs["pr_thresholds"]), max(kwargs["pr_thresholds"]), 0.01
    )
    f1_list = [
        2
        * (kwargs["precision_list"][i] * kwargs["recall_list"][i])
        / (kwargs["precision_list"][i] + kwargs["recall_list"][i])
        for i in range(len(kwargs["precision_list"]))
    ]
    queue_rate_list = [(kwargs["y_pred_proba"] >= thr).mean() for thr in thr_set2]
    accuracy_list = [
        accuracy_score(kwargs["y_true"], (kwargs["y_pred_proba"] >= thr).astype(int))
        for thr in thr_set1
    ]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # subplot 1: roc curve
    ax1.plot(
        kwargs["fpr_list"],
        kwargs["tpr_list"],
        color="red",
        label=f"AUC = {kwargs['auc_roc']:.3f}",
    )
    ax1.plot(
        kwargs["fpr_list"][kwargs["youden_index"]],
        kwargs["tpr_list"][kwargs["youden_index"]],
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

    # TODO: adjust all font sizes

    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.legend(prop={"size": 12}, loc=0)
    ax1.annotate(
        f"Threshold = {kwargs['youden_threshold']:.3f}",
        xy=(
            kwargs["fpr_list"][kwargs["youden_index"]],
            kwargs["tpr_list"][kwargs["youden_index"]],
        ),
        xycoords="data",
        xytext=(
            kwargs["fpr_list"][kwargs["youden_index"]] + 0.4,
            kwargs["tpr_list"][kwargs["youden_index"]] - 0.4,
        ),
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    # subplot 2: preferred scores vs thresholds
    ax2.plot(kwargs["roc_thresholds"], 1 - kwargs["fpr_list"], label="Specificity")
    ax2.plot(kwargs["roc_thresholds"], kwargs["tpr_list"], label="Sensitivity")
    ax2.plot(thr_set1, accuracy_list, label="Accuracy")

    ax2.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="Threshold",
        ylabel="Score",
        title="Preferred Scores vs Thresholds",
    )

    ax2.tick_params(axis="both", which="major", labelsize=12)

    ax2.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1)
    ax2.axvline(kwargs["sens_spec_threshold"], color="k", ls="--")
    if kwargs["sens_spec_threshold"] <= 0.5:
        ax2.annotate(
            f"Threshold = {kwargs['sens_spec_threshold']:.3f}",
            xy=(kwargs["sens_spec_threshold"], 0.05),
            xycoords="data",
            xytext=(kwargs["sens_spec_threshold"] + 0.1, 0.05),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    else:
        ax2.annotate(
            f"Threshold = {kwargs['sens_spec_threshold']:.3f}",
            xy=(kwargs["sens_spec_threshold"], 0.05),
            xycoords="data",
            xytext=(kwargs["sens_spec_threshold"] - 0.4, 0.05),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    # subplot 3: precision-recall curve
    ax3.plot(
        kwargs["recall_list"],
        kwargs["precision_list"],
        color="red",
        label=f"PR AUC ={kwargs['auc_pr']:.3f}",
    )
    ax3.plot(
        kwargs["recall_list"][kwargs["prec_rec_index"]],
        kwargs["precision_list"][kwargs["prec_rec_index"]],
        marker="o",
        color="navy",
        ms=10,
    )
    ax3.axvline(
        x=kwargs["recall_list"][kwargs["prec_rec_index"]],
        ymin=kwargs["recall_list"][kwargs["prec_rec_index"]],
        ymax=kwargs["precision_list"][kwargs["prec_rec_index"]],
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

    ax3.legend(prop={"size": 12}, loc=0)
    ax3.tick_params(axis="both", which="major", labelsize=12)
    ax3.annotate(
        f"Threshold = {kwargs['prec_rec_threshold']:.3f}",
        xy=(
            kwargs["recall_list"][kwargs["prec_rec_index"]],
            kwargs["precision_list"][kwargs["prec_rec_index"]],
        ),
        xycoords="data",
        xytext=(
            kwargs["recall_list"][kwargs["prec_rec_index"]] - 0.4,
            kwargs["precision_list"][kwargs["prec_rec_index"]] - 0.4,
        ),
        arrowprops=dict(facecolor="black", shrink=0.05),
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    # subplot 4: preferred Scores vs Thresholds
    ax4.plot(kwargs["pr_thresholds"], kwargs["precision_list"][1:], label="Precision")
    ax4.plot(kwargs["pr_thresholds"], kwargs["recall_list"][1:], label="Recall")
    ax4.plot(kwargs["pr_thresholds"], f1_list[1:], label="F1-Score")
    ax4.plot(thr_set2, queue_rate_list, label="Queue Rate")

    ax4.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.01],
        xlabel="Threshold",
        ylabel="Score",
        title="Preferred Scores vs Thresholds",
    )

    ax4.tick_params(axis="both", which="major", labelsize=12)
    ax4.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1)
    ax4.axvline(kwargs["prec_rec_threshold"], color="k", ls="--")

    if kwargs["prec_rec_threshold"] <= 0.5:
        ax4.annotate(
            f"Threshold = {kwargs['prec_rec_threshold']:.3f}",
            xy=(kwargs["prec_rec_threshold"], 0.03),
            xycoords="data",
            xytext=(kwargs["prec_rec_threshold"] + 0.1, 0.03),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )
    else:
        ax4.annotate(
            f"Threshold = {kwargs['prec_rec_threshold']:.3f}",
            xy=(kwargs["prec_rec_threshold"], 0.03),
            xycoords="data",
            xytext=(kwargs["prec_rec_threshold"] - 0.4, 0.03),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    plt.show()
