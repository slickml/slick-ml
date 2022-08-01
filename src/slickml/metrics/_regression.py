from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy as scp
import seaborn as sns
from IPython.core.display import display
from matplotlib.figure import Figure
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from slickml.utils import check_var
from slickml.visualization import plot_regression_metrics


# TODO(amir): check the types here again
# TODO(amir): Examples should be added
# TODO(amir): more details should be added in docstring
# TODO(amir): look more into performance of the module; currently I think it takse a while
# my main hunch is the calculation of `REC` takes a while which can be dramatically optimized
# currently, it is implemented via `for loop`; vectorization can be helpul here
@dataclass
class RegressionMetrics:
    """Regression Metrics is a wrapper to calculate all the regression metrics in one place.

    Notes
    -----
    In case of multioutput regression, calculation methods can be chosen among
    ["raw_values", "uniform_average", "variance_weighted"].

    Parameters
    ----------
    y_true: Union[List[float], np.ndarray, pd.Series]
        Ground truth target (response) values

    y_pred: Union[List[float], np.ndarray, pd.Series]
        Predicted target (response) values

    multioutput: str, optional
        Method to calculate the metric for multioutput targets where possible values are
        "raw_values", "uniform_average", and "variance_weighted". "raw_values" returns a full set of
        scores in case of multioutput input. "uniform_average" scores of all outputs are averaged
        with uniform weight. "variance_weighted" scores of all outputs are averaged, weighted by the
        variances of each individual output, by default "uniform_average"

    precision_digits: int, optional
        The number of precision digits to format the scores dataframe, by default 3

    display_df: bool, optional
        Whether to display the formatted scores' dataframe, by default True

    Attributes
    ----------
    y_residual_: np.ndarray
        Residual values (errors) calculated as (y_true - y_pred)

    y_residual_normsq_:  np.ndarray
        Square root of absolute value of y_residual_

    r2_: float
        R^2 score (coefficient of determination) with a possible value between 0.0 and 1.0

    ev_: float
        Explained variance score with a possible value between 0.0 and 1.0

    mae_: float
        Mean absolute error

    mse_: float
        Mean squared error

    msle_: float
        Mean squared log error

    mape_: float
        Mean absolute percentage error

    auc_rec_: float
        Area under REC curve with a possible value between 0.0 and 1.0

    deviation_:  np.ndarray
        List of deviations to plot REC curve.

    accuracy_:  List[float]
        Calculated accuracy at each deviation to plot REC curve.

    y_ratio_:  np.ndarray
        Ratio of y_pred/y_true.

    mean_y_ratio_: float
        Mean value of y_pred/y_true ratio.

    std_y_ratio_: float
        Standard deviation value of y_pred/y_true ratio.

    cv_y_ratio_: float
        Coefficient of variation calculated as std_y_ratio/mean_y_ratio

    metrics_dict_: Dict[str, Union[float, None]]
        Rounded metrics based on the number of precision digits

    metrics_df_: pd.DataFrame
        Pandas DataFrame of all calculated metrics

    plotting_dict_: Dict[str, Any]
        Plotting properties

    Methods
    -------
    plot(figsize=(12, 12), save_path=None, display_plot=False)
        Plots regression metrics

    Examples
    --------
    >>> from slickml.metrics import RegressionMetrics
    >>> m = RegressionMetrics(
    ...     y_true=[3, -0.5, 2, 7],
    ...     y_pred=[2.5, 0.0, 2, 8]
    ... )
    ... m.plot()
    ... m.metrics_df_
    ... m.metrics_dict_
    """

    y_true: Union[List[float], np.ndarray, pd.Series]
    y_pred: Union[List[float], np.ndarray, pd.Series]
    multioutput: Optional[str] = "uniform_average"
    precision_digits: Optional[int] = 3
    display_df: Optional[bool] = True

    def __post_init__(self):
        check_var(
            self.multioutput,
            var_name="multioutput",
            dtypes=str,
            values=(
                "raw_values",
                "variance_weighted",
                "uniform_average",
            ),
        )
        check_var(
            self.precision_digits,
            var_name="precision_digits",
            dtypes=int,
        )
        check_var(
            self.display_df,
            var_name="display_df",
            dtypes=bool,
        )

        # TODO(amir): add `list_to_array()` function into slickml.utils
        # TODO(amir): how numpy works with pd.Series here? kinda fuzzy
        if not isinstance(self.y_true, np.ndarray):
            self.y_true = np.array(self.y_true)
        if not isinstance(self.y_pred, np.ndarray):
            self.y_pred = np.array(self.y_pred)

        # TODO(amir): investigate the option of using @property instead of this for the whole API
        # TODO(amir): maybe adding `fit()` would make more sense ?
        self.y_residual_ = self.y_true - self.y_pred
        self.y_residual_normsq_ = np.sqrt(np.abs(self.y_residual_))
        self.r2_ = self._r2()
        self.ev_ = self._ev()
        self.mae_ = self._mae()
        self.mse_ = self._mse()
        self.msle_ = self._msle()
        self.mape_ = self._mape()
        (
            self.deviation_,
            self.accuracy_,
            self.auc_rec_,
        ) = self._rec_curve()
        (
            self.y_ratio_,
            self.mean_y_ratio_,
            self.std_y_ratio_,
            self.cv_y_ratio_,
        ) = self._ratio_hist()
        self.metrics_dict_ = self._metrics_dict()
        self.metrics_df_ = self._metrics_df()
        self.plotting_dict_ = self._plotting_dict()

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = (12, 16),
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = False,
    ) -> Figure:
        """Plots regression metrics.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size, by default (12, 16)

        save_path : str, optional
            The full or relative path to save the plot including the image format such as
            "myplot.png" or "../../myplot.pdf", by default None

        display_plot : bool, optional
            Whether to show the plot, by default False

        Returns
        -------
        Figure
        """
        return plot_regression_metrics(
            figsize=figsize,
            save_path=save_path,
            display_plot=display_plot,
            **self.plotting_dict_,
        )

    def _r2(self) -> float:
        """Calculates R^2 score.

        Returns
        -------
        float
        """
        return r2_score(
            y_true=self.y_true,
            y_pred=self.y_pred,
            multioutput=self.multioutput,
        )

    def _ev(self) -> float:
        """Calculates explained variance score.

        Returns
        -------
        float
        """
        return explained_variance_score(
            y_true=self.y_true,
            y_pred=self.y_pred,
            multioutput=self.multioutput,
        )

    def _mae(self) -> float:
        """Calculates mean-absolute-error.

        Returns
        -------
        float
        """
        return mean_absolute_error(
            y_true=self.y_true,
            y_pred=self.y_pred,
            multioutput=self.multioutput,
        )

    def _mse(self) -> float:
        """Calculate mean-squared-error.

        Returns
        -------
        float
        """
        return mean_squared_error(
            y_true=self.y_true,
            y_pred=self.y_pred,
            multioutput=self.multioutput,
        )

    # TODO(amir): double check the return type here with mypy
    def _msle(self) -> Union[float, None]:
        """Calculates mean-squared-log-error.

        Returns
        -------
        Union[float, None]
        """
        if min(self.y_true) < 0 or min(self.y_pred) < 0:
            msle = None
        else:
            msle = mean_squared_log_error(
                y_true=self.y_true,
                y_pred=self.y_pred,
                multioutput=self.multioutput,
            )

        return msle

    def _mape(self) -> float:
        """Calculates mean-absolute-percentage-error.

        Returns
        -------
        float
        """
        return mean_absolute_percentage_error(
            y_true=self.y_true,
            y_pred=self.y_pred,
            multioutput=self.multioutput,
        )

    def _rec_curve(self) -> Tuple[np.ndarray, List[float], float]:
        """Calculates the rec curve elements: deviation, accuracy, auc.

        Notes
        -----
        Simpson method is used as the integral method to calculate the area under regression error
        characteristics (REC) and the REC algorithm is implemented based on "Regression error
        characteristic curves" paper [1]_.

        References
        ----------
        .. [1] Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves.
           In Proceedings of the 20th international conference on machine learning
           (ICML-03) (pp. 43-50).
           https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf

        Returns
        -------
        Tuple[np.ndarray, List[float], float]
        """
        begin = 0.0
        end = 1.0
        interval = 0.01
        accuracy = []
        deviation = np.arange(begin, end, interval)

        # TODO(amir): we gotta see if we can simply the for loop here with a better algo
        # main loop to calculate norm and compare with each deviation
        for i in range(len(deviation)):
            count = 0.0
            for j in range(len(self.y_true)):
                calc_norm = np.linalg.norm(self.y_true[j] - self.y_pred[j]) / np.sqrt(
                    np.linalg.norm(self.y_true[j]) ** 2 + np.linalg.norm(self.y_pred[j]) ** 2,
                )
                if calc_norm < deviation[i]:
                    count += 1
            accuracy.append(count / len(self.y_true))

        auc_rec = scp.integrate.simps(accuracy, deviation) / end

        return (deviation, accuracy, auc_rec)

    def _ratio_hist(self) -> Tuple[np.ndarray, float, float, float]:
        """Calculates the histogram elements of y_pred/y_true ratio.

        This would report the coefficient of variation CV as std(ratio)/mean(ratio) based on "A
        Pareto front based evolutionary model for airfoil self-noise prediction" paper _[1].

        References
        ----------
        .. [1] Tahmassebi, A., Gandomi, A. H., & Meyer-Baese, A. (2018, July). A Pareto front based
           evolutionary model for airfoil self-noise prediction. In 2018 IEEE Congress on
           Evolutionary Computation (CEC) (pp. 1-8). IEEE.
           https://www.amirhessam.com/assets/pdf/projects/cec-airfoil2018.pdf

        Returns
        -------
        Tuple[np.ndarray, float, float, float]
        """
        y_ratio = self.y_pred / self.y_true
        mean_y_ratio = np.mean(y_ratio)
        std_y_ratio = np.std(y_ratio)
        cv_y_ratio = std_y_ratio / mean_y_ratio

        return (y_ratio, mean_y_ratio, std_y_ratio, cv_y_ratio)

    # TODO(amir): refactor this into a dataclass with dependency injection
    def _metrics_dict(self) -> Dict[str, Union[float, None]]:
        """Rounded calculated metrics based on the number of precision digits.

        Returns
        -------
        Dict[str, Union[float, None]]
        """
        return {
            "R2 Score": round(
                number=self.r2_,
                ndigits=self.precision_digits,
            ),
            "Explained Variance Score": round(
                number=self.ev_,
                ndigits=self.precision_digits,
            ),
            "Mean Absolute Error": round(
                number=self.mae_,
                ndigits=self.precision_digits,
            ),
            "Mean Squared Error": round(
                number=self.mse_,
                ndigits=self.precision_digits,
            ),
            "Mean Squared Log Error": round(
                number=self.msle_,
                ndigits=self.precision_digits,
            )
            if self.msle_ is not None
            else None,
            "Mean Absolute Percentage Error": round(
                number=self.mape_,
                ndigits=self.precision_digits,
            ),
            "REC AUC": round(
                number=self.auc_rec_,
                ndigits=self.precision_digits,
            ),
            "Coeff. of Variation": round(
                number=self.cv_y_ratio_,
                ndigits=self.precision_digits,
            ),
            "Mean of Variation": round(
                number=self.mean_y_ratio_,
                ndigits=self.precision_digits,
            ),
        }

    def _metrics_df(self) -> pd.DataFrame:
        """Creates a pandas DataFrame of all calculated metrics with custom formatting.

        The resulted dataframe contains all the metrics based on the precision digits and selected
        average method.

        Returns
        -------
        pd.DataFrame
        """
        metrics_df = pd.DataFrame(
            data=self.metrics_dict_,
            index=["Metrics"],
        )
        # TODO(amir): can we do df.reindex() ?
        metrics_df = metrics_df.reindex(
            columns=[
                "R2 Score",
                "Explained Variance Score",
                "Mean Absolute Error",
                "Mean Squared Error",
                "Mean Squared Log Error",
                "Mean Absolute Percentage Error",
                "REC AUC",
                "Coeff. of Variation",
                "Mean of Variation",
            ],
        )

        # TODO(amir): move this to a utility function under utils/format.py since it is repeated
        # that would make it more general and scalable across API
        # Set CSS properties
        th_props = [
            ("font-size", "12px"),
            ("text-align", "left"),
            ("font-weight", "bold"),
        ]

        td_props = [
            ("font-size", "12px"),
            ("text-align", "center"),
        ]

        # Set table styles
        styles = [
            dict(
                selector="th",
                props=th_props,
            ),
            dict(
                selector="td",
                props=td_props,
            ),
        ]
        cm = sns.light_palette(
            "blue",
            as_cmap=True,
        )

        if self.display_df:
            display(
                metrics_df.style.background_gradient(
                    cmap=cm,
                ).set_table_styles(styles),
            )

        return metrics_df

    # TODO(amir): think of a dataclass with the ability of returning all data entries as a dict
    # this can be used along with dependency injection design pattern
    def _plotting_dict(self) -> Dict[str, Any]:
        """Returns the plotting properties.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            "r2": self.r2_,
            "ev": self.ev_,
            "mae": self.mae_,
            "mse": self.mse_,
            "y_pred": self.y_pred,
            "y_true": self.y_true,
            "y_residual": self.y_residual_,
            "y_residual_normsq": self.y_residual_normsq_,
            "auc_rec": self.auc_rec_,
            "y_ratio": self.y_ratio_,
            "cv_y_ratio": self.cv_y_ratio_,
            "std_y_ratio": self.std_y_ratio_,
            "mean_y_ratio": self.mean_y_ratio_,
            "msle": self.msle_,
            "mape": self.mape_,
            "deviation": self.deviation_,
            "accuracy": self.accuracy_,
        }
