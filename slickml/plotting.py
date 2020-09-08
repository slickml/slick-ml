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
        raise TypeError("Only tuple and list types are allowed for figsize.")

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


def plot_xfs_feature_frequency(
    freq,
    figsize=None,
    freq_pct=True,
    color=None,
    marker=None,
    markersize=None,
    markeredgecolor=None,
    markerfacecolor=None,
    markeredgewidth=None,
    fontsize=None,
):
    """Function to plot selected features frequency.
    This function is a helper function based on the features_frequency
    attribute of the XGBoostFeatureSelector class.
    Parameters
    ----------
    freq: Pandas DataFrame
        Feature frequency
    figsize: tuple, optional, (default=(8, 4))
        Figure size
    freq_pct: bool, optional, (default=True)
        Flag to show the features frequency in percent
    color: str, optional, (default="#87CEEB")
        Color of the vertical lines of lollipops
    marker: str, optional, (default="o")
        Market style of the lollipops. Complete valid
        marker styke can be found at:
        (https://matplotlib.org/2.1.1/api/markers_api.html#module-matplotlib.markers)
    markersize: int or float, optional, (default=10)
        Markersize
    markeredgecolor: str, optional, (default="1F77B4")
        Marker edge color
    markerfacecolor: str, optional, (default="1F77B4")
        Marker face color
    markeredgewidth: int or float, optional, (default=1)
        Marker edge width
    fontsize: int or float, optional, (default=12)
        Fontsize for xlabel and ylabel, and ticks parameters
    """

    # initializing figsize
    if figsize is None:
        figsize = (8, 4)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # # initializing column to plot
    if freq_pct:
        col = "Frequency (%)"
    else:
        col = "Frequency"

    # initializing color
    if color is None:
        color = "#87CEEB"
    elif isinstance(color, str):
        color = color
    else:
        raise TypeError("Only str type is allowed for color.")

    # initializing marker
    if marker is None:
        marker = "o"
    elif isinstance(marker, str):
        marker = marker
    else:
        raise TypeError("Only str type is allowed for marker.")

    # initializing markersize
    if markersize is None:
        markersize = 10
    elif isinstance(markersize, float) or isinstance(markersize, int):
        markersize = markersize
    else:
        raise TypeError("Only int and float types are allowed for markersize.")

    # initializing markeredgecolor
    if markeredgecolor is None:
        markeredgecolor = "#1F77B4"
    elif isinstance(markeredgecolor, str):
        markeredgecolor = markeredgecolor
    else:
        raise TypeError("Only str type is allowed for markeredgecolor.")

    # initializing markerfacecolor
    if markerfacecolor is None:
        markerfacecolor = "#1F77B4"
    elif isinstance(markerfacecolor, str):
        markerfacecolor = markerfacecolor
    else:
        raise TypeError("Only str type is allowed for markerfacecolor.")

    # initializing markeredgewidth
    if markeredgewidth is None:
        markeredgewidth = 1
    elif isinstance(markeredgewidth, int) or isinstance(markeredgewidth, float):
        markeredgecolor = markeredgecolor
    else:
        raise TypeError("Only int and float types are allowed for markeredgewidth.")

    # initializing fontsize
    if fontsize is None:
        fontsize = 12
    elif isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only int and float types are allowed for fontsize.")

    # reindex freq
    freq = freq.reindex(index=[idx for idx in range(len(freq) - 1, -1, -1)])

    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(y=freq["Feature"], xmin=0, xmax=freq[col], color=color)
    ax.plot(
        freq[col],
        freq["Feature"].values,
        marker,
        markersize=markersize,
        markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor,
        markeredgewidth=markeredgewidth,
    )

    ax.set_xlabel(f"{col}", fontsize=fontsize)
    ax.set_ylabel("Feature", fontsize=fontsize)
    ax.set_title("Important Features Frequency", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.show()


def plot_xfs_cv_results(
    figsize=None, int_color=None, ext_color=None, sharex=False, sharey=False, **kwargs
):
    """
    Function to plot the cross-validation results of
    XGBoostFeatureSelector. It visualizes the internal
    and external performance during the selection process.
    Internal refers to the performance of train/test folds
    during the xgboost.cv() using "metrics" rounds to help
    the best number of boosting round. External refers to
    the performance of xgboost.train() on watchlist using
    eval_metric.
    Parameters
    ----------
    figsize: tuple, optional, (default=(8, 4))
        Figure size
    int_color: str, optional, (default="#4169E1")
        Color of the histograms for internal cv results
    ext_color: str, optional, (default="#8A2BE2")
        Color of the histograms for external cv results
    sharex: bool, optional, (default=False)
        Flag to share "X" axis for each column of subplots
    sharey: bool, optional, (default=False)
        Flag to share "Y" axis for each row of subplots
    kwargs: dict
        Plotting object plotting_cv_
    """

    # initializing figsize
    if figsize is None:
        figsize = (10, 8)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing internal color
    if int_color is None:
        int_color = "#4169E1"
    elif isinstance(int_color, str):
        int_color = int_color
    else:
        raise TypeError("Only str type is allowed for int_color.")

    # initializing external color
    if ext_color is None:
        ext_color = "#8A2BE2"
    elif isinstance(ext_color, str):
        ext_color = ext_color
    else:
        raise TypeError("Only str type is allowed for ext_color.")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=figsize, sharex=sharex, sharey=sharey
    )
    sns.distplot(kwargs["int_cv_train"], color=int_color, ax=ax1, axlabel="")
    sns.distplot(kwargs["int_cv_test"], color=int_color, ax=ax2, axlabel="")
    sns.distplot(kwargs["ext_cv_train"], color=ext_color, ax=ax3, axlabel="")
    sns.distplot(kwargs["ext_cv_test"], color=ext_color, ax=ax4, axlabel="")
    ax1.set(title=f"Internal {kwargs['n_splits']}-Folds CV {kwargs['metric']} - Train")
    ax2.set(title=f"Internal {kwargs['n_splits']}-Folds CV {kwargs['metric']} - Test")
    ax3.set(
        title=f"External {kwargs['n_splits']}-Folds CV {kwargs['eval_metric']} - Train"
    )
    ax4.set(
        title=f"External {kwargs['n_splits']}-Folds CV {kwargs['eval_metric']} - Test"
    )
    plt.show()


def plot_xgb_cv_results(
    cv_results,
    figsize=None,
    linestyle=None,
    train_label=None,
    test_label=None,
    train_color=None,
    train_std_color=None,
    test_color=None,
    test_std_color=None,
):
    """
    Function to plot cv results of XGBoostCVClassifier.
    Parameters
    ----------
    cv_results: Pandas DataFrame()
        Cross-validation results in DataFrame() format
    figsize: tuple, optional, (default=(8, 5))
        Figure size
    linestyle: str, optional, (default="--")
        Style of lines. Complete options are available at
        (https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html)
    train_label: str, optional (default="Train")
        Label in the figure legend for the training line
    test_label: str, optional (default="Test")
        Label in the figure legend for the training line
    train_color: str, optional, (default="navy")
        Color of the training line
    train_std_color: str, optional, (default="#B3C3F3")
        Color of the edge color of the training std bars
    test_color: str, optional, (default="purple")
        Color of the testing line
    test_std_color: str, optional, (default="#D0AAF3")
        Color of the edge color of the testing std bars
    """

    if figsize is None:
        figsize = (8, 5)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    if linestyle is None:
        linestyle = "--"
    elif isinstance(linestyle, str):
        linestyle = linestyle
    else:
        raise TypeError("Only str type is valid for linestyle.")

    if train_label is None:
        train_label = "Train"
    elif isinstance(train_label, str):
        train_label = train_label
    else:
        raise TypeError("Only str type is valid for train_label.")

    if test_label is None:
        test_label = "Test"
    elif isinstance(test_label, str):
        test_label = test_label
    else:
        raise TypeError("Only str type is valid for test_label.")

    if train_color is None:
        train_color = "navy"
    elif isinstance(train_color, str):
        train_color = train_color
    else:
        raise TypeError("Only str type is valid for train_color.")

    if train_std_color is None:
        train_std_color = "#B3C3F3"
    elif isinstance(train_std_color, str):
        train_std_color = train_std_color
    else:
        raise TypeError("Only str type is valid for train_std_color.")

    if test_color is None:
        test_color = "purple"
    elif isinstance(test_color, str):
        test_color = test_color
    else:
        raise TypeError("Only str type is valid for test_color.")

    if test_std_color is None:
        test_std_color = "#D0AAF3"
    elif isinstance(test_std_color, str):
        test_std_color = test_std_color
    else:
        raise TypeError("Only str type is valid for test_std_color.")

    # update metrics capitalizations for title/labels
    metric = cv_results.columns.tolist()[0].split("-")[1]
    metrics = ["AUC", "AUCPR", "Error", "LogLoss"]
    for m in metrics:
        if m.lower() == metric:
            metric = m

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        range(cv_results.shape[0]),
        cv_results.iloc[:, 0],
        yerr=cv_results.iloc[:, 1],
        fmt=linestyle,
        ecolor=train_std_color,
        c=train_color,
        label=train_label,
    )

    ax.errorbar(
        range(cv_results.shape[0]),
        cv_results.iloc[:, 2],
        yerr=cv_results.iloc[:, 3],
        fmt=linestyle,
        ecolor=test_std_color,
        c=test_color,
        label=train_label,
    )

    ax.set_xlabel("# of Boosting Rounds", fontsize=12)
    ax.set_ylabel(f"""{metric}""", fontsize=12)
    ax.set_title(f"""{metric} Evolution vs Boosting Rounds""", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.legend(loc=0, prop={"size": 12}, framealpha=0.0)

    plt.show()


def plot_xgb_feature_importance(
    feature_importance,
    figsize=None,
    color=None,
    marker=None,
    markersize=None,
    markeredgecolor=None,
    markerfacecolor=None,
    markeredgewidth=None,
    fontsize=None,
):
    """Function to plot XGBoost feature importance.
    This function is a helper function based on the feature_importance_
    attribute of the XGBoostCVClassifier class.
    Parameters
    ----------
    feature importance: Pandas DataFrame
        Feature frequency
    figsize: tuple, optional, (default=(8, 5))
        Figure size
    color: str, optional, (default="#87CEEB")
        Color of the vertical lines of lollipops
    marker: str, optional, (default="o")
        Market style of the lollipops. Complete valid
        marker styke can be found at:
        (https://matplotlib.org/2.1.1/api/markers_api.html#module-matplotlib.markers)
    markersize: int or float, optional, (default=10)
        Markersize
    markeredgecolor: str, optional, (default="1F77B4")
        Marker edge color
    markerfacecolor: str, optional, (default="1F77B4")
        Marker face color
    markeredgewidth: int or float, optional, (default=1)
        Marker edge width
    fontsize: int or float, optional, (default=12)
        Fontsize for xlabel and ylabel, and ticks parameters
    """

    # initializing figsize
    if figsize is None:
        figsize = (8, 5)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing color
    if color is None:
        color = "#87CEEB"
    elif isinstance(color, str):
        color = color
    else:
        raise TypeError("Only str type is allowed for color.")

    # initializing marker
    if marker is None:
        marker = "o"
    elif isinstance(marker, str):
        marker = marker
    else:
        raise TypeError("Only str type is allowed for marker.")

    # initializing markersize
    if markersize is None:
        markersize = 10
    elif isinstance(markersize, float) or isinstance(markersize, int):
        markersize = markersize
    else:
        raise TypeError("Only int and float types are allowed for markersize.")

    # initializing markeredgecolor
    if markeredgecolor is None:
        markeredgecolor = "#1F77B4"
    elif isinstance(markeredgecolor, str):
        markeredgecolor = markeredgecolor
    else:
        raise TypeError("Only str type is allowed for markeredgecolor.")

    # initializing markerfacecolor
    if markerfacecolor is None:
        markerfacecolor = "#1F77B4"
    elif isinstance(markerfacecolor, str):
        markerfacecolor = markerfacecolor
    else:
        raise TypeError("Only str type is allowed for markerfacecolor.")

    # initializing markeredgewidth
    if markeredgewidth is None:
        markeredgewidth = 1
    elif isinstance(markeredgewidth, int) or isinstance(markeredgewidth, float):
        markeredgecolor = markeredgecolor
    else:
        raise TypeError("Only int and float types are allowed for markeredgewidth.")

    # initializing fontsize
    if fontsize is None:
        fontsize = 12
    elif isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only int and float types are allowed for fontsize.")

    # define column names
    cols = feature_importance.columns.tolist()
    coly = cols[0]
    colx = cols[1]

    # reindex feature importance
    feature_importance = feature_importance.reindex(
        index=[idx for idx in range(len(feature_importance) - 1, -1, -1)]
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(
        y=feature_importance[coly], xmin=0, xmax=feature_importance[colx], color=color
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

    # put importance values on the plot
    for index, value in enumerate(feature_importance[colx]):
        ax.text(value + 10, index * 1.01, f"{value:.2f}")

    ax.set_xlabel(f"{' '.join(colx.split('_')).title()}", fontsize=fontsize)
    ax.set_ylabel(f"{coly.title()}", fontsize=fontsize)
    ax.set_title("Feature Importance", fontsize=fontsize)
    ax.set(xlim=[None, feature_importance[colx].max() * 1.13])
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.show()
