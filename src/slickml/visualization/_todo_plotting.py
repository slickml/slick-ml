import matplotlib as mpl
import matplotlib.pyplot as plt
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
