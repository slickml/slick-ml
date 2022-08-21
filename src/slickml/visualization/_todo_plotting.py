import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("ticks")
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["lines.linewidth"] = 2


# TODO(amir): take out the bar chart function into `_base.py` plotting
# and call it for feature frequency and feature importance
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
    save_path=None,
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

    save_path: str, optional (default=None)
        The full or relative path to save the plot including the image format.
        For example "myplot.png" or "../../myplot.pdf"

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
    # TODO(amir): add vline option too ?
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

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


def plot_xfs_cv_results(
    figsize=None,
    int_color=None,
    ext_color=None,
    sharex=False,
    sharey=False,
    save_path=None,
    **kwargs,
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

    save_path: str, optional (default=None)
        The full or relative path to save the plot including the image format.
        For example "myplot.png" or "../../myplot.pdf"

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
        2,
        2,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    sns.distplot(kwargs["int_cv_train"], color=int_color, ax=ax1, axlabel="")
    sns.distplot(kwargs["int_cv_test"], color=int_color, ax=ax2, axlabel="")
    sns.distplot(kwargs["ext_cv_train"], color=ext_color, ax=ax3, axlabel="")
    sns.distplot(kwargs["ext_cv_test"], color=ext_color, ax=ax4, axlabel="")
    ax1.set(title=f"Internal {kwargs['n_splits']}-Folds CV {kwargs['metric']} - Train")
    ax2.set(title=f"Internal {kwargs['n_splits']}-Folds CV {kwargs['metric']} - Test")
    ax3.set(title=f"External {kwargs['n_splits']}-Folds CV {kwargs['eval_metric']} - Train")
    ax4.set(title=f"External {kwargs['n_splits']}-Folds CV {kwargs['eval_metric']} - Test")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

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
    save_path=None,
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

    save_path: str, optional (default=None)
        The full or relative path to save the plot including the image format.
        For example "myplot.png" or "../../myplot.pdf"

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
    metrics = ["AUC", "AUCPR", "Error", "LogLoss", "MAE", "RMSE", "RMSLE"]
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

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

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

    grid: bool, optional (default=True)
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
                ylabel = rf"""{kwargs["params"]["n_splits"]}-Folds CV Mean $R^2$"""
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
        title = rf"""Best $\lambda$ = {kwargs["lambda_best"]:.3f} with {len(kwargs["coeff"])} Features"""
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

    ax.set_ylim([min(kwargs["cv_mean_score"]) - 0.05, max(kwargs["cv_mean_score"]) + 0.05])
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
        title = rf"""Best $\lambda$ = {kwargs["lambda_best"]:.3f} with {len(kwargs["coeff"])} Features"""
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
