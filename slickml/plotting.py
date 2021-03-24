import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import shap
import scipy as scp
from sklearn.metrics import accuracy_score


from IPython.core.display import display, HTML
import warnings

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2
warnings.filterwarnings("ignore")
display(HTML("<style>.container { width:95% !important; }</style>"))


def plot_binary_classification_metrics(figsize=None, **kwargs):
    """
    Function to plot binary classification metrics.
    This function is a helper function based on the plotting_dict
    attribute of the BinaryClassificationMetrics class.

    Parameters
    ----------
    figsize: tuple, optional, (default=(12, 12))
        Figure size

    Returns None
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

    # -----------------------------------
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
    ax1.legend(prop={"size": 12}, loc=0, framealpha=0.0)
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

    # -----------------------------------
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

    ax2.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1, framealpha=0.0)
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

    # -----------------------------------
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

    ax3.legend(prop={"size": 12}, loc=0, framealpha=0.0)
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

    # -----------------------------------
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
    ax4.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1, framealpha=0.0)
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
    freq: pandas.DataFrame
        Feature frequency

    figsize: tuple, optional, (default=(8, 4))
        Figure size

    freq_pct: bool, optional, (default=True)
        Flag to show the features frequency in percent

    color: str, optional, (default="#87CEEB")
        Color of the vertical lines of lollipops

    marker: str, optional, (default="o")
        Marker style of the lollipops. Complete valid
        marker style can be found at:
        (https://matplotlib.org/2.1.1/api/markers_api.html#module-matplotlib.markers)

    markersize: int or float, optional, (default=10)
        Markersize

    markeredgecolor: str, optional, (default="#1F77B4")
        Marker edge color

    markerfacecolor: str, optional, (default="#1F77B4")
        Marker face color

    markeredgewidth: int or float, optional, (default=1)
        Marker edge width

    fontsize: int or float, optional, (default=12)
        Fontsize for xlabel and ylabel, and ticks parameters

    Returns None
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
    """Function to plot the cross-validation results ofXGBoostFeatureSelector.
    It visualizes the internal and external performance during the
    selection process. Internal refers to the performance of
    train/test folds during the xgboost.cv() using "metrics" rounds
    to help the best number of boosting round. External refers to
    the performance of xgboost.train() on watchlist using eval_metric.

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

    kwargs: dict()
        Plotting object plotting_cv_

    Returns None
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
    """Function to plot cv results of XGBoostCVClassifier.

    Parameters
    ----------
    cv_results: pandas.DataFrame
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

    Returns None
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
        label=test_label,
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
    feature importance: pandas.DataFrame
        Feature frequency

    figsize: tuple, optional, (default=(8, 5))
        Figure size

    color: str, optional, (default="#87CEEB")
        Color of the horizontal lines of lollipops

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

    Returns None
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


def plot_shap_summary(
    shap_values,
    features,
    plot_type=None,
    figsize=None,
    color=None,
    max_display=None,
    feature_names=None,
    title=None,
    show=True,
    sort=True,
    color_bar=True,
    layered_violin_max_num_bins=None,
    class_names=None,
    class_inds=None,
    color_bar_label=None,
):
    """Function to plot shap summary plot.
    This function is a helper function to plot the shap summary plot
    based on all types of shap explainers including tree, linear, and dnn.

    Parameters
    ----------
    shap_values: numpy.array or pandas.DataFrame
        Calculated SHAP values. For single output explanations like
        binary classificationthis this is a matrix of SHAP values (n_samples, n_features).
        For multi-output explanations this is a list of such matrices of SHAP values

    features: Numpy array or Pandas DataFrame
        The feature matrix that was used to calculate the SHAP values. For the case
        of Numpy array it is recommened to pass the feature_names list as well

    plot_type: str, optional (single-output default="dot", multi-output default="bar")
        The type of summar plot. Options are "bar", "dot", "violin", "layered_violin",
        and "compact_dot" which is recommended for SHAP interactions

    layered_violin_max_num_bins: int, optional, (default=10)
        The number of bins for calculating the violin plots ranges and outliers

    figsize: tuple, optional, (default="auto")
        Figure size

    color: str, optional, (default= "#D0AAF3" for "bar")
        Color of violin and layered violin plots are "RdBl" cmap
        Color of the horizontal lines when plot_type="bar" is "#D0AAF3"

    max_display: int, optional, (default=20)
        Limit to show the number of features in the plot

    feature_names: list[str], optional, (default=None)
        List of feature names to pass. It should follow the order
        of features

    title: str, optional, (default=None)
        Title of the plot

    show: bool, optional, (default=True)
        Flag to show the plot in inteactive environment

    sort: bool, optional, (default=True)
        Flag to plot sorted shap vlues in descending order

    color_bar: bool, optional, (default=True)
        Flag to show color_bar when plot_type is "dot" or "violin"

    class_names: list, optional, (default=None)
        List of class names for multi-output problems

    class_inds: list, optional, (default=True)
        List of class indices for multi-output problems

    color_bar_label: str, optional, (default="Feature Value")
        Label for color bar

    Returns None
    """

    # initializing figsize
    if figsize is None:
        figsize = "auto"
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing color for plot_type="bar"
    if color is None and plot_type == "bar":
        color = "#D0AAF3"
    elif color is None:
        color = None
    else:
        if isinstance(color, str):
            color = color
        else:
            raise TypeError("Only str type is allowed for color.")

    # initializing layered_violin_max_num_bins
    if layered_violin_max_num_bins is None:
        layered_violin_max_num_bins = 10
    elif isinstance(layered_violin_max_num_bins, int):
        layered_violin_max_num_bins = layered_violin_max_num_bins
    else:
        raise TypeError("Only int type is allowed for layered_violin_max_num_bins.")

    # initializing color_bar_label
    if color_bar_label is None:
        color_bar_label = "Feature Value"
    elif isinstance(color_bar_label, int):
        color_bar_label = color_bar_label
    else:
        raise TypeError("Only str type is allowed for color_bar_label.")

    shap.summary_plot(
        shap_values,
        features,
        plot_type=plot_type,
        plot_size=figsize,
        color=color,
        max_display=max_display,
        feature_names=feature_names,
        title=title,
        show=show,
        sort=sort,
        color_bar=color_bar,
        layered_violin_max_num_bins=layered_violin_max_num_bins,
        class_names=class_names,
        class_inds=class_inds,
        color_bar_label=color_bar_label,
    )
    plt.show()


def plot_shap_waterfall(
    shap_values,
    features,
    figsize=None,
    bar_color=None,
    bar_thickness=None,
    line_color=None,
    marker=None,
    markersize=None,
    markeredgecolor=None,
    markerfacecolor=None,
    markeredgewidth=None,
    max_display=None,
    title=None,
    fontsize=None,
):
    """Function to plot shap summary plot.
    This function is a helper function to plot the shap summary plot
    based on all types of shap explainers including tree, linear, and dnn.

    Parameters
    ----------
    shap_values: numpy.array or pandas.DataFrame
        Calculated SHAP values. For single output explanations like
        binary classificationthis this is a matrix of SHAP values (n_samples, n_features).
        For multi-output explanations this is a list of such matrices of SHAP values

    features: numpy.array or pandas.DataFrame
        The feature matrix that was used to calculate the SHAP values. For the case
        of Numpy array it is recommened to pass the feature_names list as well

    figsize: tuple, optional, (default=(8, 5))
        Figure size

    bar_color: str, optional, (default="#B3C3F3")
        Color of the horizontal bar lines

    bar_thickness: float, optional, (default=0.5)
        Thickness (hight) of the horizontal bar lines

    line_color: str, optional, (default="purple")
        Color of the line plot

    marker: str, optional, (default="o")
        Marker style
        marker style can be found at:
        (https://matplotlib.org/2.1.1/api/markers_api.html#module-matplotlib.markers)

    markersize: int or float, optional, (default=7)
        Markersize

    markeredgecolor: str, optional, (default="purple")
        Marker edge color

    markerfacecolor: str, optional, (default="purple")
        Marker face color

    markeredgewidth: int or float, optional, (default=1)
        Marker edge width

    max_display: int, optional, (default=20)
        Limit to show the number of features in the plot

    title: str, optional, (default=None)
        Title of the plot

    fontsize: int or float, optional, (default=12)
        Fontsize for xlabel and ylabel, and ticks parameters

    Returns None
    """

    # initializing figsize
    if figsize is None:
        figsize = (8, 5)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing bar_color
    if bar_color is None:
        bar_color = "#B3C3F3"
    elif isinstance(bar_color, str):
        bar_color = bar_color
    else:
        raise TypeError("Only str type is allowed for bar_color.")

    # initializing bar_thickness
    if bar_thickness is None:
        bar_thickness = 0.5
    elif isinstance(bar_thickness, float):
        bar_thickness = bar_thickness
    else:
        raise TypeError("Only float type is allowed for bar_thickness.")

    # initializing line_color
    if line_color is None:
        line_color = "purple"
    elif isinstance(line_color, str):
        line_color = line_color
    else:
        raise TypeError("Only str type is allowed for line_color.")

    # initializing marker
    if marker is None:
        marker = "o"
    elif isinstance(marker, str):
        marker = marker
    else:
        raise TypeError("Only str type is allowed for marker.")

    # initializing markersize
    if markersize is None:
        markersize = 7
    elif isinstance(markersize, float) or isinstance(markersize, int):
        markersize = markersize
    else:
        raise TypeError("Only int and float types are allowed for markersize.")

    # initializing markeredgecolor
    if markeredgecolor is None:
        markeredgecolor = "purple"
    elif isinstance(markeredgecolor, str):
        markeredgecolor = markeredgecolor
    else:
        raise TypeError("Only str type is allowed for markeredgecolor.")

    # initializing markerfacecolor
    if markerfacecolor is None:
        markerfacecolor = "purple"
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

    # initializing max_display
    if max_display is None:
        max_display = 20
    elif isinstance(max_display, int):
        max_display = max_display
    else:
        raise TypeError("Only int type is allowed for max_display.")

    # initializing fontsize
    if fontsize is None:
        fontsize = 12
    elif isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only int and float types are allowed for fontsize.")

    # main calculation of cum/comp ratios
    feature_names = features.columns
    shap_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    feature_names = feature_names[np.argsort(shap_ratio)[::-1]]
    shap_ratio_order = np.sort(shap_ratio)[::-1]
    cum_sum = np.cumsum(shap_ratio_order)
    feature_names = feature_names[:max_display]
    shap_ratio_order = shap_ratio_order[:max_display]
    cum_sum = cum_sum[:max_display]

    # plotting
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
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1) + 1, 10))
    ax2.set_xticks(np.arange(0, round(shap_ratio_order.max(), -1) + 1, 10))
    ax1.tick_params(axis="both", which="major", labelsize=fontsize)

    ax1.set(
        ylim=[-1, len(feature_names)],
        xlabel="Cumulative Ratio (%)",
        ylabel="Feature",
        title=title,
    )
    ax2.set(xlabel="Composition Ratio (%)")

    plt.show()


def plot_regression_metrics(figsize=None, **kwargs):
    """Function to plot regression metrics.
    This function is a helper function based on the plotting_dict
    attribute of the RegressionMetrics class.

    Parameters
    ----------
    figsize: tuple, optional, (default=(12, 12))
        Figure size

    Returns None
    """

    # initializing figsize
    if figsize is None:
        figsize = (12, 16)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=figsize)

    # subplot 1: actual vs predicted

    sns.regplot(
        kwargs["y_true"],
        kwargs["y_pred"],
        marker="o",
        scatter_kws={"edgecolors": "navy"},
        color="#B3C3F3",
        fit_reg=False,
        ax=ax1,
    )
    ax1.plot(
        [kwargs["y_true"].min(), kwargs["y_true"].max()],
        [kwargs["y_true"].min(), kwargs["y_true"].max()],
        "r--",
        lw=3,
    )

    ax1.set(
        xlabel="Actual Values",
        ylabel="Predicted Values",
        title="Actual-Predicted",
    )
    ax1_ylim = max(max(kwargs["y_pred"]), max(kwargs["y_true"]))
    ax1.tick_params(axis="both", which="major", labelsize=12)

    ax1.text(
        0.05 * min(kwargs["y_true"]),
        0.93 * ax1_ylim,
        f"MAPE = {kwargs['mape']:.3f}",
        fontsize=12,
        transform=ax1.transAxes,
    )

    ax1.text(
        0.05 * min(kwargs["y_true"]),
        0.86 * ax1_ylim,
        f"$R^2$ = {kwargs['r2']:.3f}",
        fontsize=12,
        transform=ax1.transAxes,
    )

    # -----------------------------------
    # subplot 2: Q-Q Normal Plot

    scp.stats.probplot(kwargs["y_residual"], fit=True, dist="norm", plot=ax2)
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

    ax2.tick_params(axis="both", which="major", labelsize=12)

    # -----------------------------------
    # subplot 3: Residuals vs Fitted
    sns.residplot(
        kwargs["y_pred"],
        kwargs["y_true"],
        lowess=True,
        order=1,
        line_kws={"color": "red", "lw": 3, "ls": "--", "alpha": 1},
        scatter_kws={"edgecolors": "navy"},
        color="#B3C3F3",
        robust=True,
        ax=ax3,
    )

    ax3.set(
        xlabel="Predicted Values",
        ylabel="Residuals",
        title="Residuals-Predicted",
    )

    ax3.tick_params(axis="both", which="major", labelsize=12)

    # -----------------------------------
    # subplot 4: Sqrt Standard Residuals vs Fitted

    sns.regplot(
        kwargs["y_pred"],
        kwargs["y_residual_normsq"],
        lowess=True,
        line_kws={"color": "red", "lw": 3, "ls": "--", "alpha": 1},
        scatter_kws={"edgecolors": "navy"},
        color="#B3C3F3",
        ax=ax4,
    )

    ax4.set(
        xlabel="Predicted Values",
        ylabel="Standardized Residuals Norm",
        title="Scale-Location",
    )

    ax4.tick_params(axis="both", which="major", labelsize=12)

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

    ax5.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax5.set_xticklabels(
        ["Less", 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, "More"], rotation=30
    )

    ax5.set(
        ylabel="Frequency",
        title="Prediction Variation",
    )

    ax5.tick_params(axis="both", which="major", labelsize=12)

    ax5_ylim = max(freqs)
    ax5.text(
        0.65,
        ax5_ylim,
        fr"""$\mu$ = {kwargs['mean_y_ratio']:.3f}""",
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

    ax6.tick_params(axis="both", which="major", labelsize=12)
    ax6.legend(prop={"size": 12}, loc=4, framealpha=0.0)

    plt.show()


def plot_glmnet_cv_results(
    figsize=None,
    marker=None,
    markersize=None,
    colors=None,
    linestyle=None,
    fontsize=None,
    grid=True,
    legend=True,
    legendloc=None,
    xlabel=None,
    ylabel=None,
    title=None,
    save_path=None,
    **kwargs,
):
    """Function to plot GLMNetCVClassfier cross-validation results.

    Parameters
    ----------
    figsize: tuple, optional, (default=(8, 5))
        Figure size

    marker: str, optional, (default="o")
        Marker style
        marker style can be found at:
        (https://matplotlib.org/2.1.1/api/markers_api.html#module-matplotlib.markers)

    markersize: int or float, optional, (default=5)
        Markersize

    colors: list or tuple
        Colors of the marker, errorbar line, max_lambda line,
        and best_lambda line, respectively. The default colors
        are ("red", "black", "purple", "navy"). The length of the
        passed tuple/list should be always four.

    linestyle: str, optional (default="--")
        Linestyle of vertical lambda lines

    fontsize: int or float, optional, (default=12)
        Fontsize of the title. The fontsizes of xlabel, ylabel,
        tick_params, and legend are resized with 0.85, 0.85, 0.75,
        and 0.85 fraction of title fontsize, respectively.

    grid : bool, optional (default=True)
        Whether to show (x,y) grid on the plot.

    legend: bool, optional (default=True)
        Whether to show legend on the plot.

    legendloc: int or str, optional (default="best")
        Location of legend

    xlabel : str, optional (default="-Log(Lambda)")
        Xlabel of the plot.

    ylabel : str, optional (default="{n_splits}-Folds CV Mean {metric}"
        Ylabel of the plot.

    title : str, optional (default="Best {lambda_best} with {n} Features"
        Title of the plot.

    save_path: string or None
        The full or relative path to save the image including the image format.
        For example "myplot.png" or "../../myplot.pdf"

    Returns None
    """
    # initializing figsize
    if figsize is None:
        figsize = (8, 5)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing fontsize
    if fontsize is None:
        fontsize = 12
    elif isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only int and float types are allowed for fontsize.")

    # initializing marker
    if marker is None:
        marker = "o"
    elif isinstance(marker, str):
        marker = marker
    else:
        raise TypeError("Only str type is allowed for marker.")

    # initializing markersize
    if markersize is None:
        markersize = 5
    elif isinstance(markersize, float) or isinstance(markersize, int):
        markersize = markersize
    else:
        raise TypeError("Only int and float types are allowed for markersize.")

    # initializing colors
    if colors is None:
        colors = ["red", "black", "purple", "navy"]
    elif (isinstance(colors, tuple) or isinstance(colors, list)) and len(colors) == 4:
        colors = colors
    else:
        raise TypeError("Only tuple or list with length 4 is allowed for colors.")

    # initializing linestyle
    if linestyle is None:
        linestyle = "--"
    elif isinstance(linestyle, str):
        linestyle = linestyle
    else:
        raise TypeError("Only str type is allowed for linestyle.")

    # initializing legendpos
    if legendloc is None:
        legendloc = "best"
    else:
        legendloc = legendloc

    # initializing grid
    if isinstance(grid, bool):
        grid = grid
    else:
        raise TypeError("Only bool type is allowed for grid.")

    # initializing legend
    if isinstance(legend, bool):
        legend = legend
    else:
        raise TypeError("Only bool type is allowed for legend.")

    # initializing xlabel
    if xlabel is None:
        xlabel = r"-$Log(\lambda)$"
    elif isinstance(xlabel, str):
        xlabel = xlabel
    else:
        raise TypeError("Only string type is allowed for xlabel.")

    # initializing ylabel
    if ylabel is None:
        if kwargs["params"]["scoring"] is None:
            if kwargs["module"] == "glmnet.linear":
                ylabel = fr"""{kwargs["params"]["n_splits"]}-Folds CV Mean $R^2$"""
            elif kwargs["module"] == "glmnet.logistic":
                ylabel = f"""{kwargs["params"]["n_splits"]}-Folds CV Mean ACCURACY"""
        else:
            ylabel = f"""{kwargs["params"]["n_splits"]}-Folds CV Mean {' '.join((kwargs["params"]["scoring"]).split("_")).upper()}"""
    elif isinstance(ylabel, str):
        ylabel = ylabel
    else:
        raise TypeError("Only string type is allowed for ylabel.")

    # initializing title
    if title is None:
        title = fr"""Best $\lambda$ = {kwargs["lambda_best"]:.3f} with {len(kwargs["coeff"])} Features"""
    elif isinstance(title, str):
        title = title
    else:
        raise TypeError("Only string type is allowed for title.")

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        -np.log(kwargs["lambda_path"]),
        kwargs["cv_mean_score"],
        yerr=kwargs["cv_standard_error"],
        color=colors[0],
        ecolor=colors[1],
        marker=marker,
        markersize=markersize,
    )

    ax.vlines(
        -np.log(kwargs["lambda_max"]),
        ymin=min(kwargs["cv_mean_score"]) - 0.05,
        ymax=max(kwargs["cv_mean_score"]) + 0.05,
        linestyles=linestyle,
        color=colors[2],
        label=r"max $\lambda$",
    )

    ax.vlines(
        -np.log(kwargs["lambda_best"]),
        ymin=min(kwargs["cv_mean_score"]) - 0.05,
        ymax=max(kwargs["cv_mean_score"]) + 0.05,
        linestyles=linestyle,
        color=colors[3],
        label=r"best $\lambda$",
    )

    ax.set_ylim(
        [min(kwargs["cv_mean_score"]) - 0.05, max(kwargs["cv_mean_score"]) + 0.05]
    )
    ax.set_xlabel(xlabel, fontsize=fontsize * 0.85)
    ax.set_ylabel(
        ylabel,
        fontsize=fontsize * 0.85,
    )
    ax.set_title(
        title,
        fontsize=fontsize,
    )
    ax.tick_params(axis="both", which="major", labelsize=fontsize * 0.75)
    ax.grid(grid)

    if legend:
        ax.legend(loc=legendloc, prop={"size": fontsize * 0.85}, framealpha=0.0)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


def plot_glmnet_coeff_path(
    figsize=None,
    linestyle=None,
    fontsize=None,
    grid=True,
    legend=True,
    legendloc="center",
    xlabel=None,
    ylabel=None,
    title=None,
    bbox_to_anchor=None,
    yscale=None,
    save_path=None,
    **kwargs,
):
    """Function to plot GLMNetCVClassfier coefficients' paths.

    Parameters
    ----------
    figsize: tuple, optional, (default=(8, 5))
        Figure size

    linestyle: str, optional (default="-")
        Linestyle of vertical lambda lines

    fontsize: int or float, optional, (default=12)
        Fontsize of the title. The fontsizes of xlabel, ylabel,
        tick_params, and legend are resized with 0.85, 0.85, 0.75,
        and 0.75 fraction of title fontsize, respectively.

    grid : bool, optional (default=True)
        Whether to show (x,y) grid on the plot.

    legend: bool, optional (default=True)
        Whether to show legend on the plot.

    legendloc: str, optional (default="center")
        Legend location.

    xlabel : str, optional (default="-Log(Lambda")
        Xlabel of the plot.

    ylabel : str, optional (default="Coefficients")
        Ylabel of the plot.

    title : str, optional (default="Best {lambda_best} with {n} Features")
        Title of the plot.

    yscale: str, optional (default="linear")
        Scale for y-axis (coefficients). Valid options are
        "linear", "log", "symlog", "logit". More on:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html

    bbox_to_anchor: tuple, list, optional (default=(1.2, 0.5)
        Relative coordinates for legend location outside of the plot.

    save_path: str, optional (default=None)
        The full or relative path to save the plot including the image format.
        For example "myplot.png" or "../../myplot.pdf"

    Returns None
    """
    # initializing figsize
    if figsize is None:
        figsize = (8, 5)
    elif isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing fontsize
    if fontsize is None:
        fontsize = 12
    elif isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only int and float types are allowed for fontsize.")

    # initializing linestyle
    if linestyle is None:
        linestyle = "-"
    elif isinstance(linestyle, str):
        linestyle = linestyle
    else:
        raise TypeError("Only str type is allowed for linestyle.")

    # initializing grid
    if isinstance(grid, bool):
        grid = grid
    else:
        raise TypeError("Only bool type is allowed for grid.")

    # initializing legendpos
    if legendloc is None:
        legendloc = "center"
    else:
        legendloc = legendloc

    # initializing legend
    if isinstance(legend, bool):
        legend = legend
    else:
        raise TypeError("Only bool type is allowed for legend.")

    # initializing xlabel
    if xlabel is None:
        xlabel = r"-$Log(\lambda)$"
    elif isinstance(xlabel, str):
        xlabel = xlabel
    else:
        raise TypeError("Only string type is allowed for xlabel.")

    # initializing ylabel
    if ylabel is None:
        ylabel = "Coefficients"
    elif isinstance(xlabel, str):
        ylabel = ylabel
    else:
        raise TypeError("Only string type is allowed for ylabel.")

    # initializing title
    if title is None:
        title = fr"""Best $\lambda$ = {kwargs["lambda_best"]:.3f} with {len(kwargs["coeff"])} Features"""
    elif isinstance(title, str):
        title = title
    else:
        raise TypeError("Only string type is allowed for title.")

    # initializing bbox_to_anchor
    if bbox_to_anchor is None:
        bbox_to_anchor = (1.1, 0.5)
    elif isinstance(bbox_to_anchor, tuple) or isinstance(bbox_to_anchor, list):
        bbox_to_anchor = bbox_to_anchor
    else:
        raise TypeError("Only tuple or list type is allowed for bbox_to_anchor.")

    # initializing yscale
    if yscale is None:
        yscale = "linear"
    elif isinstance(yscale, str):
        yscale = yscale
    else:
        raise TypeError("Only string type is allowed for yscale.")

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    for feature, coeff_path in kwargs["coeff_path"].items():
        if feature in kwargs["coeff"]:
            ax.plot(
                -np.log(kwargs["lambda_path"]),
                coeff_path,
                linestyle=linestyle,
                label=feature,
            )

    ax.tick_params(axis="both", which="major", labelsize=fontsize * 0.75)
    ax.set_ylabel(ylabel, fontsize=fontsize * 0.85)
    ax.set_xlabel(xlabel, fontsize=fontsize * 0.85)
    ax.set_title(
        title,
        fontsize=fontsize,
    )
    ax.set_yscale(yscale)
    ax.grid(True)

    if legend:
        ax.legend(
            loc=legendloc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
            prop={"size": fontsize * 0.75},
            framealpha=0.0,
            fancybox=True,
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
