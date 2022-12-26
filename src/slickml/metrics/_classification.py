from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from slickml.utils import check_var
from slickml.visualization import plot_binary_classification_metrics


# TODO(amir): update docstrings types in attributes
# TODO(amir): currently `pos_label` in `roc` is defaulted to None
# the None options is ok for [0, 1] and [-1, 1] cases; plan to expose `pos_label`
# with default values to None? what would be the breaking changes ?
@dataclass
class BinaryClassificationMetrics:
    """BinaryClassificationMetrics calculates binary classification metrics in one place.

    Binary metrics are computed based on three methods for calculating the thresholds to binarize
    the prediction probabilities. Threshold computations including:
        1) Youden Index [youden-j-index]_.
        2) Maximizing Precision-Recall.
        3) Maximizing Sensitivity-Specificity.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, pd.Series]
        List of ground truth values such as [0, 1] for binary problems

    y_pred_proba : Union[List[float], np.ndarray, pd.Series]
        List of predicted probabilities for the positive class (class=1) in binary problems
        or ``y_pred_proba[:, 1]`` in scikit-learn API

    threshold : float, optional
        Inclusive threshold value to binarize ``y_pred_prob`` to ``y_pred`` where any value that
        satisfies ``y_pred_prob >= threshold`` will set to ``class=1 (positive class)``. Note that
        for ``">="`` is used instead of ``">"``, by default 0.5

    average_method : str, optional
        Method to calculate the average of any metric. Possible values are ``"micro"``, ``"macro"``,
        ``"weighted"``, ``"binary"``, by default "binary"

    precision_digits : int, optional
        The number of precision digits to format the scores dataframe, by default 3

    display_df : bool, optional
        Whether to display the formatted scores' dataframe, by default True

    Methods
    -------
    plot(figsize=(12, 12), save_path=None, display_plot=False, return_fig=False)
        Plots classification metrics

    get_metrics(dtype="dataframe")
        Returns calculated classification metrics

    Attributes
    ----------
    y_pred_ : np.ndarray
        Predicted class based on the ``threshold``. The threshold value inclusively binarizes
        ``y_pred_prob`` to ``y_pred`` where any value that satisfies ``y_pred_prob >= threshold``
        will set to ``class=1 (positive class)``. Note that for ``">="`` is used instead of ``">"``

    accuracy_ : float
        Accuracy based on the initial ``threshold`` value with a possible value between 0.0 and 1.0

    balanced_accuracy_ : float
        Balanced accuracy based on the initial ``threshold`` value considering the prevalence of the
        classes with a possible value between 0.0 and 1.0

    fpr_list_ : np.ndarray
        List of calculated false-positive-rates based on ``roc_thresholds_``

    tpr_list_ : np.ndarray
        List of calculated true-positive-rates based on ``roc_thresholds_``

    roc_thresholds_ : np.ndarray
        List of thresholds value to calculate ``fpr_list_`` and ``tpr_list_``

    auc_roc_ : float
        Area under ROC curve with a possible value between 0.0 and 1.0

    precision_list_ : np.ndarray
        List of calculated precision based on ``pr_thresholds_``

    recall_list_ : np.ndarray
        List of calculated recall based on ``pr_thresholds_``

    pr_thresholds_ : numpy.ndarray
        List of precision-recall thresholds value to calculate ``precision_list_`` and ``recall_list_``

    auc_pr_ : float
        Area under Precision-Recall curve with a possible value between 0.0 and 1.0

    precision_ : float
        Precision based on the ``threshold`` value with a possible value between 0.0 and 1.0

    recall_ : float
        Recall based on the ``threshold`` value with a possible value between 0.0 and 1.0

    f1_ : float
        F1-score based on the ``threshold`` value (beta=1.0) with a possible value between 0.0 and 1.0

    f2_ : float
        F2-score based on the ``threshold`` value (beta=2.0) with a possible value between 0.0 and 1.0

    f05_ : float
        F(1/2)-score based on the ``threshold`` value (beta=0.5) with a possible value between 0.0 and
        1.0

    average_precision_ : float
        Avearge precision based on the ``threshold`` value and class prevalence with a possible value
        between 0.0 and 1.0

    tn_ : np.int64
        True negative counts based on the ``threshold`` value

    fp_ : np.int64
        False positive counts based on the ``threshold`` valuee

    fn_ : np.int64
        False negative counts based on the ``threshold`` value

    tp_ : np.int64
        True positive counts based on the ``threshold`` value

    threat_score_ : float
        Threat score based on the ``threshold`` value with a possible value between 0.0 and 1.0

    youden_index_ : np.int64
        Index of the calculated Youden index threshold

    youden_threshold_ : float
        Threshold calculated based on Youden Index with a possible value between 0.0 and 1.0

    sens_spec_threshold_ : float
        Threshold calculated based on maximized sensitivity-specificity with a possible value
        between 0.0 and 1.0

    prec_rec_threshold_ : float
        Threshold calculated based on maximized precision-recall with a possible value between 0.0
        and 1.0

    thresholds_dict_ : Dict[str, float]
        Calculated thresholds based on different algorithms including Youden Index
        ``youden_threshold_``, maximizing the area under sensitivity-specificity curve
        ``sens_spec_threshold_``, and maximizing the area under precision-recall curver
        ``prec_rec_threshold_``

    metrics_dict_ : Dict[str, float]
        Rounded metrics based on the number of precision digits

    metrics_df_ : pd.DataFrame
        Pandas DataFrame of all calculated metrics with ``threshold`` set as index

    average_methods_: List[str]
        List of all possible average methods

    plotting_dict_: Dict[str, Any]
        Plotting properties

    References
    ----------
    .. [youden-j-index] https://en.wikipedia.org/wiki/Youden%27s_J_statistic

    Examples
    --------
    >>> from slickml.metrics import BinaryClassificationMetrics
    >>> cm = BinaryClassificationMetrics(
    ...     y_true=[1, 1, 0, 0],
    ...     y_pred_proba=[0.95, 0.3, 0.1, 0.9]
    ... )
    >>> f = cm.plot()
    >>> m = cm.get_metrics()
    """

    y_true: Union[List[int], np.ndarray, pd.Series]
    y_pred_proba: Union[List[float], np.ndarray, pd.Series]
    threshold: Optional[float] = 0.5
    average_method: Optional[str] = "binary"
    precision_digits: Optional[int] = 3
    display_df: Optional[bool] = True

    def __post_init__(self) -> None:
        """Post instantiation validations and assignments."""
        check_var(
            self.y_true,
            var_name="y_true",
            dtypes=(
                np.ndarray,
                pd.Series,
                list,
            ),
        )
        check_var(
            self.y_pred_proba,
            var_name="y_pred_proba",
            dtypes=(
                np.ndarray,
                pd.Series,
                list,
            ),
        )
        check_var(
            self.threshold,
            var_name="threshold",
            dtypes=float,
        )
        check_var(
            self.average_method,
            var_name="average_method",
            dtypes=str,
            values=(
                "micro",
                "macro",
                "weighted",
                "binary",
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
        # TODO(amir): add `values_between` option to `check_var()`

        if self.threshold is not None and (self.threshold < 0.0 or self.threshold > 1.0):
            raise ValueError("The input threshold must have a value between 0.0 and 1.0.")

        # TODO(amir): how we can pull off special cases like this ?
        if self.average_method == "binary" or not self.average_method:
            self.average_method = None

        # TODO(amir): add `list_to_array()` function into slickml.utils
        # TODO(amir): how numpy works with pd.Series here? kinda fuzzy
        if not isinstance(self.y_true, np.ndarray):
            self.y_true = np.array(self.y_true)
        if not isinstance(self.y_pred_proba, np.ndarray):
            self.y_pred_proba = np.array(self.y_pred_proba)

        self.y_pred_ = (self.y_pred_proba >= self.threshold).astype(int)
        self.accuracy_ = self._accuracy()
        self.balanced_accuracy_ = self._balanced_accuracy()
        (
            self.fpr_list_,
            self.tpr_list_,
            self.roc_thresholds_,
        ) = self._roc_curve()
        self.auc_roc_ = self._auc_roc()
        (
            self.precision_list_,
            self.recall_list_,
            self.pr_thresholds_,
        ) = self._precision_recall_curve()
        self.auc_pr_ = self._auc_pr()
        (
            self.precision_,
            self.recall_,
            self.f1_,
        ) = self._precision_recall_f1()
        (
            self.f2_,
            self.f05_,
        ) = self._f2_f50()
        self.average_precision_ = self._average_precision()
        (
            self.tn_,
            self.fp_,
            self.fn_,
            self.tp_,
        ) = self._confusion_matrix()
        self.threat_score_ = self._threat_score()
        self.metrics_dict_ = self._metrics_dict()
        self.metrics_df_ = self._metrics_df()
        (
            self.youden_index_,
            self.youden_threshold_,
        ) = self._threshold_youden()
        (
            self.sens_spec_index_,
            self.sens_spec_threshold_,
        ) = self._threshold_sens_spec()
        (
            self.prec_rec_index_,
            self.prec_rec_threshold_,
        ) = self._threshold_prec_rec()
        self.thresholds_dict_ = self._thresholds_dict()
        self.plotting_dict_ = self._plotting_dict()
        self.average_methods_ = self._average_methods()

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = (12, 12),
        save_path: Optional[str] = None,
        display_plot: Optional[bool] = False,
        return_fig: Optional[bool] = False,
    ) -> Optional[Figure]:
        """Plots classification metrics.

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

        Returns
        -------
        Figure
        """
        return plot_binary_classification_metrics(
            figsize=figsize,
            save_path=save_path,
            display_plot=display_plot,
            return_fig=return_fig,
            **self.plotting_dict_,
        )

    def get_metrics(
        self,
        dtype: Optional[str] = "dataframe",
    ) -> Union[pd.DataFrame, Dict[str, Optional[float]]]:
        """Returns calculated metrics with desired dtypes.

        Currently, available output types are "dataframe" and "dict".

        Parameters
        ----------
        dtype : str, optional
            Results dtype, by default "dataframe"

        Returns
        -------
        Union[pd.DataFrame, Dict[str, Optional[float]]]
        """
        check_var(
            dtype,
            var_name="dtype",
            dtypes=str,
            values=("dataframe", "dict"),
        )

        if dtype == "dataframe":
            return self.metrics_df_
        else:
            return self.metrics_dict_

    def _accuracy(self) -> float:
        """Calculates accuracy score.

        Returns
        -------
        float
        """
        return accuracy_score(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            normalize=True,
        )

    def _balanced_accuracy(self) -> float:
        """Calculates balanced accuracy score.

        Returns
        -------
        float
        """
        return balanced_accuracy_score(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            adjusted=False,
        )

    # TODO(amir): check return types here between ndarray or list
    def _roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates the roc curve elements: fpr, tpr, thresholds.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        fpr_list, tpr_list, roc_thresholds = roc_curve(
            y_true=self.y_true,
            y_score=self.y_pred_proba,
        )

        return (fpr_list, tpr_list, roc_thresholds)

    # TODO(amir): check the API when `average_method="binary"` that does it pass None as the method
    # or keep it as "binary"
    def _auc_roc(self) -> float:
        """Calculates the area under ROC curve (auc_roc).

        Returns
        -------
        float
        """
        return roc_auc_score(
            y_true=self.y_true,
            y_score=self.y_pred_proba,
            average=self.average_method,
        )

    def _precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates precision recall curve elements: precision_list, recall_list, pr_thresholds.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        precision_list, recall_list, pr_thresholds = precision_recall_curve(
            y_true=self.y_true,
            probas_pred=self.y_pred_proba,
        )

        return precision_list, recall_list, pr_thresholds

    def _auc_pr(self) -> float:
        """Calculates the area under Precision-Recal curve (auc_pr).

        Returns
        -------
        float
        """
        return auc(
            self.recall_list_,
            self.precision_list_,
        )

    def _precision_recall_f1(self) -> Tuple[float, float, float]:
        """Calculates precision, recall, and f1-score.

        Returns
        -------
        Tuple[float, float, float]
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            beta=1.0,
            average=self.average_method,
        )

        # updating precision, recall, and f1 for binary average method
        if not self.average_method:
            precision = precision[1]
            recall = recall[1]
            f1 = f1[1]

        return (precision, recall, f1)

    def _f2_f50(self) -> Tuple[float, float]:
        """Calculates f2-score and f0.5-score.

        Returns
        -------
        Tuple[float, float]
        """
        f2 = fbeta_score(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            beta=2.0,
            average=self.average_method,
        )
        f05 = fbeta_score(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            beta=0.5,
            average=self.average_method,
        )

        # updating f2, f0.5 scores for binary average method
        if not self.average_method:
            f2 = f2[1]
            f05 = f05[1]

        return (f2, f05)

    def _average_precision(self) -> float:
        """Calculates average precision.

        Returns
        -------
        float
        """
        return average_precision_score(
            y_true=self.y_true,
            y_score=self.y_pred_proba,
            average=self.average_method,
        )

    def _confusion_matrix(self) -> Tuple[float, float, float, float]:
        """Calculates confusion matrix elements: tn, fp, fn, tp.

        Returns
        -------
        Tuple[float, float, float, float]
        """
        return confusion_matrix(
            y_true=self.y_true,
            y_pred=self.y_pred_,
        ).ravel()

    def _threat_score(self) -> float:
        """Calculates threat score.

        Returns
        -------
        float
        """
        if self.average_method == "weighted":
            w = self.tp_ + self.tn_
            wp = self.tp_ / w
            wn = self.tn_ / w
            threat_score = wp * (self.tp_ / (self.tp_ + self.fp_ + self.fn_)) + wn * (
                self.tn_ / (self.tn_ + self.fn_ + self.fp_)
            )

        elif self.average_method == "macro":
            threat_score = 0.5 * (self.tp_ / (self.tp_ + self.fp_ + self.fn_)) + 0.5 * (
                self.tn_ / (self.tn_ + self.fn_ + self.fp_)
            )

        else:
            threat_score = self.tp_ / (self.tp_ + self.fp_ + self.fn_)

        return threat_score

    def _metrics_dict(self) -> Dict[str, float]:
        """Rounded calculated metrics based on the number of precision digits.

        Returns
        -------
        Dict[str, float]
        """
        return {
            "Accuracy": round(
                number=self.accuracy_,
                ndigits=self.precision_digits,
            ),
            "Balanced Accuracy": round(
                number=self.balanced_accuracy_,
                ndigits=self.precision_digits,
            ),
            "ROC AUC": round(
                number=self.auc_roc_,
                ndigits=self.precision_digits,
            ),
            "PR AUC": round(
                number=self.auc_pr_,
                ndigits=self.precision_digits,
            ),
            "Precision": round(
                number=self.precision_,
                ndigits=self.precision_digits,
            ),
            "Recall": round(
                number=self.recall_,
                ndigits=self.precision_digits,
            ),
            "F-1 Score": round(
                number=self.f1_,
                ndigits=self.precision_digits,
            ),
            "F-2 Score": round(
                number=self.f2_,
                ndigits=self.precision_digits,
            ),
            "F-0.50 Score": round(
                number=self.f05_,
                ndigits=self.precision_digits,
            ),
            "Threat Score": round(
                number=self.threat_score_,
                ndigits=self.precision_digits,
            ),
            "Average Precision": round(
                number=self.average_precision_,
                ndigits=self.precision_digits,
            ),
            "TP": self.tp_,
            "TN": self.tn_,
            "FP": self.fp_,
            "FN": self.fn_,
        }

    def _metrics_df(self) -> pd.DataFrame:
        """Creates a pandas DataFrame of all calculated metrics with custom formatting.

        The resulted dataframe contains all the metrics based on the precision digits and selected
        average method.

        Returns
        -------
        pd.DataFrame
        """
        # update None average_method back to binary for printing
        if not self.average_method:
            self.average_method = "binary"

        metrics_df = pd.DataFrame(
            data=self.metrics_dict_,
            index=[
                f"""Threshold = {self.threshold:.{self.precision_digits}f} | Average =
                {self.average_method.title()}""",
            ],
        )
        # TODO(amir): can we do df.reindex() ?
        metrics_df = metrics_df.reindex(
            columns=[
                "Accuracy",
                "Balanced Accuracy",
                "ROC AUC",
                "PR AUC",
                "Precision",
                "Recall",
                "Average Precision",
                "F-1 Score",
                "F-2 Score",
                "F-0.50 Score",
                "Threat Score",
                "TP",
                "TN",
                "FP",
                "FN",
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
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
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

    def _threshold_youden(self) -> Tuple[int, float]:
        """Calculates the Youden index and Youden threshold.

        Returns
        -------
        Tuple[int, float]
        """
        youden_index = np.argmax(
            np.abs(self.tpr_list_ - self.fpr_list_),
        )
        youden_threshold = self.roc_thresholds_[youden_index]

        return (youden_index, youden_threshold)

    def _threshold_sens_spec(self) -> Tuple[int, float]:
        """Calculates the threshold that maximizes sensitivity-specificity curve.

        Returns
        -------
        Tuple[int, float]
        """
        sens_spec_index = np.argmin(
            abs(self.tpr_list_ + self.fpr_list_ - 1),
        )
        sens_spec_threshold = self.roc_thresholds_[sens_spec_index]

        return (sens_spec_index, sens_spec_threshold)

    def _threshold_prec_rec(self) -> Tuple[int, float]:
        """Calculates the threshold that maximizes precision-recall curve.

        Returns
        -------
        Tuple[int, float]
        """
        prec_rec_index = np.argmin(abs(self.precision_list_ - self.recall_list_))
        prec_rec_threshold = self.pr_thresholds_[prec_rec_index]

        return (prec_rec_index, prec_rec_threshold)

    def _thresholds_dict(self) -> Dict[str, float]:
        """Returns the calculated thresholds as a dictionary.

        Returns
        -------
        Dict[str, float]
        """
        return {
            "Youden": self.youden_threshold_,
            "Sensitivity-Specificity": self.sens_spec_threshold_,
            "Precision-Recall-F1": self.prec_rec_threshold_,
        }

    # TODO(amir): check Any here since it can be Union[np.ndarray, int, float] ?
    def _plotting_dict(self) -> Dict[str, Any]:
        """Returns the plotting properties."""
        return {
            "roc_thresholds": self.roc_thresholds_,
            "pr_thresholds": self.pr_thresholds_,
            "precision_list": self.precision_list_,
            "recall_list": self.recall_list_,
            "y_pred_proba": self.y_pred_proba,
            "y_true": self.y_true,
            "fpr_list": self.fpr_list_,
            "tpr_list": self.tpr_list_,
            "auc_roc": self.auc_roc_,
            "youden_index": self.youden_index_,
            "youden_threshold": self.youden_threshold_,
            "sens_spec_threshold": self.sens_spec_threshold_,
            "prec_rec_threshold": self.prec_rec_threshold_,
            "auc_pr": self.auc_pr_,
            "prec_rec_index": self.prec_rec_index_,
        }

    def _average_methods(self) -> List[str]:
        """Returns the list of average methods.

        Returns
        -------
        List[str]
        """
        return [
            "binary",
            "weighted",
            "macro",
            "micro",
        ]
