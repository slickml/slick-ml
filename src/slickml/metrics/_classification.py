import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display import display
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

from slickml.visualization._todo_plotting import plot_binary_classification_metrics


# TODO(amir): write a protocol for Metric
class BinaryClassificationMetrics:
    """Binary Classification Metrics.
    This is wrapper to calculate all the binary classification
    metrics with both arbitrary and three computed methods for
    calculating the thresholds. Threshold computations including:
    1) Youden Index: (https://en.wikipedia.org/wiki/Youden%27s_J_statistic).
    2) Maximizing Precision-Recall.
    3) Maximizing Sensitivity-Specificity.

    Parameters
    ----------
    y_true: numpy.array[int] or list[int]
        List of ground truth binary values [0, 1]

    y_pred_proba: numpy.array[float] or list[float]
        List of predicted probability for the positive class
        (class=1 or y_pred_proba[:, 1] in scikit-learn)

    threshold: float, optional (default=0.5)
        Threshold value for mapping y_pred_prob to y_pred
        Note that for threshold ">" is used instead of  ">="

    average_method: str, optional (default="binary")
        Method to calculate the average of the metric. Possible values are
        "micro", "macro", "weighted", "binary"

    precision_digits: int, optional (default=3)
        The number of precision digits to format the scores' dataframe

    display_df: boolean, optional (default=True)
        Flag to display the formatted scores' dataframe

    Attributes
    ----------
    y_pred_: numpy.array(int) or list[int]
        Predicted class based on the threshold.
        Positive class for y_pred_proba >= threshold and
        negative for else.

    accuracy_: float value between 0. and 1.
        Classification accuracy based on threshold value

    balanced_accuracy_: float value between 0. and 1.
        Balanced classification accuracy based on threshold value
        considering the prevalence of the classes

    fpr_list_: numpy.array[float] or list[float]
        List of calculated false-positive-rates based on roc_thresholds.
        This can be used for ROC curve plotting

    tpr_list_: numpy.array[float] or list[float]
        List of calculated true-positive-rates based on roc_thresholds
        This can be used for ROC curve plotting

    roc_thresholds_: numpy.array[float] or list[float]
        List of thresholds value to calculate fpr_list_ and tpr_list_

    auc_roc_: float value between 0. and 1.
        Area under ROC curve

    precision_list_: numpy.array[float] or list[float]
        List of calculated precision based on pr_thresholds
        This can be used for ROC curve plotting

    recall_list_: numpy.array[float] or list[float]
        List of calculated recall based on pr_thresholds
        This can be used for ROC curve plotting

    pr_thresholds_: numpy.array[float] or list[float]
        List of thresholds value to calculate precision_list_ and recall_list_

    auc_pr_: float value between 0. and 1.
        Area under Precision-Recall curve

    precision_: float value between 0. and 1.
        Precision based on threshold value

    recall_: float value between 0. and 1.
        Recall based on threshold value

    f1_: float value between 0. and 1.
        F1-score based on threshold value (beta=1.0)

    f2_: float value between 0. and 1.
        F2-score based on threshold value (beta=2.0)

    f05_: float value between 0. and 1.
        F(1/2)-score based on threshold value (beta=0.5)

    average_precision_: float value between 0. and 1.
        Avearge precision based on threshold value and class prevalence

    tn_: integer
        True negative counts based on threshold value

    fp_: integer
        False positive counts based on threshold value

    fn_: integer
        False negative counts based on threshold value

    tp_: integer
        True positive counts based on threshold value

    threat_score_: float value between 0. and 1.
        Threat score based on threshold value

    youden_threshold_: float value between 0. and 1.
        Threshold calculated based on Youden Index

    sens_spec_threshold_: float value between 0. and 1.
        Threshold calculated based on maximized sensitivity-specificity

    prec_rec_threshold_: float value between 0. and 1.
        Threshold calculated based on maximized precision-recall

    thresholds_dict_: dict()
        Dictionary of all calculated thresholds

    metrics_dict_: dict()
        Dictionary of all calculated metrics

    metrics_df_: pandas.DataFrame
        Pandas DataFrame of all calculated metrics with threshold as index

    average_methods_: list[str]
        List of all possible average methods

    plotting_dict_: dict()
        Plotting object as a dictionary consists of all
        calculated metrics which was used to plot the thresholds
    """

    def __init__(
        self,
        y_true,
        y_pred_proba,
        threshold=None,
        average_method=None,
        precision_digits=None,
        display_df=True,
    ):
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        else:
            self.y_true = y_true
        if not isinstance(y_pred_proba, np.ndarray):
            self.y_pred_proba = np.array(y_pred_proba)
        else:
            self.y_pred_proba = y_pred_proba
        if threshold is None:
            self.threshold = 0.5
        else:
            self.threshold = threshold
        if average_method == "binary" or average_method is None:
            self.average_method = None
        else:
            if not isinstance(average_method, str):
                raise TypeError("The input average_method must have str dtype.")
            else:
                if average_method in [
                    "micro",
                    "macro",
                    "weighted",
                ]:
                    self.average_method = average_method
                else:
                    raise ValueError("The input average_method value is not valid.")
        if precision_digits is None:
            self.precision_digits = 3
        else:
            if not isinstance(precision_digits, int):
                raise TypeError("The input precision_digits must have integer dtype.")
            else:
                self.precision_digits = precision_digits
        if not isinstance(display_df, bool):
            raise TypeError("The input display_df must have bool dtype.")
        else:
            self.display_df = display_df
        self.y_pred_ = (self.y_pred_proba >= self.threshold).astype(int)
        self.accuracy_ = self._accuracy()
        self.balanced_accuracy_ = self._balanced_accuracy()
        self.fpr_list_, self.tpr_list_, self.roc_thresholds_ = self._roc_curve()
        self.auc_roc_ = self._auc_roc()
        (
            self.precision_list_,
            self.recall_list_,
            self.pr_thresholds_,
        ) = self._precision_recall_curve()
        self.auc_pr_ = self._auc_pr()
        self.precision_, self.recall_, self.f1_ = self._precision_recall_f1()
        self.f2_, self.f05_ = self._f2_f50()
        self.average_precision_ = self._average_precision()
        self.tn_, self.fp_, self.fn_, self.tp_ = self._confusion_matrix()
        self.threat_score_ = self._threat_score()
        self.metrics_dict_ = self._metrics_dict()
        self.metrics_df_ = self._metrics_df()
        self.youden_index_, self.youden_threshold_ = self._threshold_youden()
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

    def _accuracy(self):
        """
        Function to calculate accuracy score
        """
        accuracy = accuracy_score(y_true=self.y_true, y_pred=self.y_pred_, normalize=True)

        return accuracy

    def _balanced_accuracy(self):
        """
        Function to calculate balanced accuracy score
        """
        balanced_accuracy = balanced_accuracy_score(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            adjusted=False,
        )

        return balanced_accuracy

    def _roc_curve(self):
        """
        Function to calculate the roc curve elements: fpr, tpr, thresholds"""
        fpr_list, tpr_list, roc_thresholds = roc_curve(
            y_true=self.y_true,
            y_score=self.y_pred_proba,
        )

        return fpr_list, tpr_list, roc_thresholds

    def _auc_roc(self):
        """
        Function to calculate the area under ROC curve (auc_roc)
        """
        if self.average_method == "binary":
            self.average_method = None
        auc_roc = roc_auc_score(
            y_true=self.y_true,
            y_score=self.y_pred_proba,
            average=self.average_method,
        )

        return auc_roc

    def _precision_recall_curve(self):
        """
        Function to calculate the precision recall curve elements:
        precision_list, recall_list, pr_thresholds
        """
        precision_list, recall_list, pr_thresholds = precision_recall_curve(
            y_true=self.y_true,
            probas_pred=self.y_pred_proba,
        )

        return precision_list, recall_list, pr_thresholds

    def _auc_pr(self):
        """
        Function to calculate the area under Precision-Recal curve (auc_pr)
        """
        auc_pr = auc(self.recall_list_, self.precision_list_)

        return auc_pr

    def _precision_recall_f1(self):
        """
        Function to calculate precision, recall, and f1-score
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=self.y_true,
            y_pred=self.y_pred_,
            beta=1.0,
            average=self.average_method,
        )

        # updating precision, recall, and f1 for binary average method
        if self.average_method is None:
            precision = precision[1]
            recall = recall[1]
            f1 = f1[1]

        return precision, recall, f1

    def _f2_f50(self):
        """
        Function to calculate f2-score and f0.5-score
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
        if self.average_method is None:
            f2 = f2[1]
            f05 = f05[1]

        return f2, f05

    def _average_precision(self):
        """
        Function to calculate average precision
        """
        average_precision = average_precision_score(
            y_true=self.y_true,
            y_score=self.y_pred_proba,
            average=self.average_method,
        )

        return average_precision

    def _confusion_matrix(self):
        """
        Function to calculate confusion matrix elements: tn, fp, fn, tp
        """
        tn, fp, fn, tp = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred_).ravel()

        return tn, fp, fn, tp

    def _threat_score(self):
        """
        Function to calculate threat score
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

    def _metrics_dict(self):
        """
        Function to create a dictionary of all calculated metrics based on the
        precision digits and average method"""
        metrics_dict = {
            "Accuracy": round(self.accuracy_, self.precision_digits),
            "Balanced Accuracy": round(self.balanced_accuracy_, self.precision_digits),
            "ROC AUC": round(self.auc_roc_, self.precision_digits),
            "PR AUC": round(self.auc_pr_, self.precision_digits),
            "Precision": round(self.precision_, self.precision_digits),
            "Recall": round(self.recall_, self.precision_digits),
            "F-1 Score": round(self.f1_, self.precision_digits),
            "F-2 Score": round(self.f2_, self.precision_digits),
            "F-0.50 Score": round(self.f05_, self.precision_digits),
            "Threat Score": round(self.threat_score_, self.precision_digits),
            "Average Precision": round(self.average_precision_, self.precision_digits),
            "TP": self.tp_,
            "TN": self.tn_,
            "FP": self.fp_,
            "FN": self.fn_,
        }

        return metrics_dict

    def _metrics_df(self):
        """
        Function to create a pandas DataFrame of all calculated metrics based
        on the precision digits and average method"""
        # update None average_method back to binary for printing
        if self.average_method is None:
            self.average_method = "binary"

        metrics_df = pd.DataFrame(
            data=self.metrics_dict_,
            index=[
                f"""Threshold = {self.threshold:.{self.precision_digits}f} | Average =
                {self.average_method.title()}""",
            ],
        )
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

        # Set CSS properties
        th_props = [
            ("font-size", "12px"),
            ("text-align", "left"),
            ("font-weight", "bold"),
        ]

        td_props = [("font-size", "12px"), ("text-align", "center")]

        # Set table styles
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
        ]
        cm = sns.light_palette("blue", as_cmap=True)

        if self.display_df:
            display(metrics_df.style.background_gradient(cmap=cm).set_table_styles(styles))

        return metrics_df

    def _threshold_youden(self):
        """
        Function to calculate youden index as a threshold
        """
        youden_index = np.argmax(np.abs(self.tpr_list_ - self.fpr_list_))
        youden_threshold = self.roc_thresholds_[youden_index]

        return youden_index, youden_threshold

    def _threshold_sens_spec(self):
        """
        Function to calculate the threshold that maximizes
        sensitivity-specificity curve"""
        sens_spec_index = np.argmin(abs(self.tpr_list_ + self.fpr_list_ - 1))
        sens_spec_threshold = self.roc_thresholds_[sens_spec_index]

        return sens_spec_index, sens_spec_threshold

    def _threshold_prec_rec(self):
        """
        Function to calculate the threshold that maximizes precision-recall
        curve"""
        prec_rec_index = np.argmin(abs(self.precision_list_ - self.recall_list_))
        prec_rec_threshold = self.pr_thresholds_[prec_rec_index]

        return prec_rec_index, prec_rec_threshold

    def _thresholds_dict(self):
        """
        Function to return calculated thresholds as a dictionary
        """
        thresholds_dict = {
            "Youden": self.youden_threshold_,
            "Sensitivity-Specificity": self.sens_spec_threshold_,
            "Precision-Recall-F1": self.prec_rec_threshold_,
        }

        return thresholds_dict

    def _plotting_dict(self):
        """
        Function to return the plotting properties as a dictionary
        """
        plotting_dict = {
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

        return plotting_dict

    def _average_methods(self):
        """
        Function to return average methods as a list
        """
        return ["binary", "weighted", "macro", "micro"]

    def plot(self, figsize=None, save_path=None):
        """Function to plot binary classification metrics.
        This function is a helper function based on the plotting_dict
        attribute of the BinaryClassificationMetrics class.

        Parameters
        ----------
        figsize: tuple, optional, (default=(12, 12))
            Figure size

        save_path: str, optional (default=None)
            The full or relative path to save the plot including the image format.
            For example "myplot.png" or "../../myplot.pdf"
        """

        plot_binary_classification_metrics(
            figsize=figsize,
            save_path=save_path,
            **self.plotting_dict_,
        )
