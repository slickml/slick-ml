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


class BinaryClassificationMetrics:
    """Binary Classification Metrics --------------------------
    Parameters:
               - y_true: list[int] List of ground truth binary values [0, 1]
               - y_pred_proba: list[float] List of predicted probability for
               the positive class (class=1 or y_pred_proba[:, 1] in
               scikit-learn)
               - threshold: float, optional (default=0.5) Threshold value for
               mapping y_pred_prob to y_pred (> is used not >=)
               - average_method: string, optional (default="binary") Method to
               calculate the average of the metric. Possible values are
               "micro", "macro", "weighted", "binary"
               - precision_digits: int, optional (default=3) The number of
               precision digits to display scores in final dataframe
               - figsize: tuple, optional (default=(12,12)) Figure size for
               scores vs thresholds plots
               - display_df: boolean, optional (default=True) Flag to display
               metrics DataFrame with CSS formatting

    """

    def __init__(
        self,
        y_true,
        y_pred_proba,
        threshold=0.5,
        average_method="binary",
        precision_digits=3,
        figsize=(12, 12),
        display_df=True,
    ):
        """
        Default constructor
        """
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        else:
            self.y_true = y_true
        if not isinstance(y_pred_proba, np.ndarray):
            self.y_pred_proba = np.array(y_pred_proba)
        else:
            self.y_pred_proba = y_pred_proba
        self.threshold = threshold
        if average_method == "binary":
            self.average_method = None
        else:
            self.average_method = average_method
        self.precision_digits = precision_digits
        self.figsize = figsize
        self.display_df = display_df
        self.y_pred = (self.y_pred_proba > self.threshold).astype(int)
        self.accuracy = self._accuracy()
        self.balanced_accuracy = self._balanced_accuracy()
        self.fpr_list, self.tpr_list, self.roc_thresholds = self._roc_curve()
        self.auc_roc = self._auc_roc()
        (
            self.precision_list,
            self.recall_list,
            self.pr_thresholds,
        ) = self._precision_recall_curve()
        self.auc_pr = self._auc_pr()
        self.precision, self.recall, self.f1 = self._precision_recall_f1()
        self.f2, self.f05 = self._f2_f50()
        self.average_precision = self._average_precision()
        self.tn, self.fp, self.fn, self.tp = self._confusion_matrix()
        self.threat_score = self._threat_score()
        self.metrics_dict = self._metrics_dict()
        self.metrics_df = self._metrics_df()
        self.youden_index, self.youden_threshold = self._threshold_youden()
        (
            self.sens_spec_index,
            self.sens_spec_threshold,
        ) = self._threshold_sens_spec()
        (
            self.prec_rec_index,
            self.prec_rec_threshold,
        ) = self._threshold_prec_rec()
        self.thresholds_dict = self._thresholds_dict()

    def _accuracy(self):
        """
        Function to calculate accuracy score
        """
        accuracy = accuracy_score(
            y_true=self.y_true, y_pred=self.y_pred, normalize=True
        )

        return accuracy

    def _balanced_accuracy(self):
        """
        Function to calculate balanced accuracy score
        """
        balanced_accuracy = balanced_accuracy_score(
            y_true=self.y_true, y_pred=self.y_pred, adjusted=False
        )

        return balanced_accuracy

    def _roc_curve(self):
        """
        Function to calculate the roc curve elements: fpr, tpr, thresholds"""
        fpr_list, tpr_list, roc_thresholds = roc_curve(
            y_true=self.y_true, y_score=self.y_pred_proba
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
            precision_list, recall_list, pr_thresholds"""
        precision_list, recall_list, pr_thresholds = precision_recall_curve(
            y_true=self.y_true, probas_pred=self.y_pred_proba
        )

        return precision_list, recall_list, pr_thresholds

    def _auc_pr(self):
        """
        Function to calculate the area under Precision-Recal curve (auc_pr)
        """
        auc_pr = auc(self.recall_list, self.precision_list)

        return auc_pr

    def _precision_recall_f1(self):
        """
        Function to calculate precision, recall, and f1-score
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=self.y_true,
            y_pred=self.y_pred,
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
            y_pred=self.y_pred,
            beta=2.0,
            average=self.average_method,
        )
        f05 = fbeta_score(
            y_true=self.y_true,
            y_pred=self.y_pred,
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
        tn, fp, fn, tp = confusion_matrix(
            y_true=self.y_true, y_pred=self.y_pred
        ).ravel()

        return tn, fp, fn, tp

    def _threat_score(self):
        """
        Function to calculate threat score
        """
        if self.average_method == "weighted":
            w = self.tp + self.tn
            wp = self.tp / w
            wn = self.tn / w
            threat_score = wp * (
                self.tp / (self.tp + self.fp + self.fn)
            ) + wn * (self.tn / (self.tn + self.fn + self.fp))

        elif self.average_method == "macro":
            threat_score = 0.5 * (
                self.tp / (self.tp + self.fp + self.fn)
            ) + 0.5 * (self.tn / (self.tn + self.fn + self.fp))

        else:
            threat_score = self.tp / (self.tp + self.fp + self.fn)

        return threat_score

    def _metrics_dict(self):
        """
        Function to create a dictionary of all calculated metrics based on the
        precision digits and average method"""
        metrics_dict = {
            "Accuracy": round(self.accuracy, self.precision_digits),
            "Balanced Accuracy": round(
                self.balanced_accuracy, self.precision_digits
            ),
            "ROC AUC": round(self.auc_roc, self.precision_digits),
            "PR AUC": round(self.auc_pr, self.precision_digits),
            "Precision": round(self.precision, self.precision_digits),
            "Recall": round(self.recall, self.precision_digits),
            "F-1 Score": round(self.f1, self.precision_digits),
            "F-2 Score": round(self.f2, self.precision_digits),
            "F-0.50 Score": round(self.f05, self.precision_digits),
            "Threat Score": round(self.threat_score, self.precision_digits),
            "Average Precision": round(
                self.average_precision, self.precision_digits
            ),
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
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
            data=self.metrics_dict,
            index=[
                f"""Threshold = {self.threshold:.3f} | Average =
                {self.average_method.title()}"""
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
            ]
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
            display(
                metrics_df.style.background_gradient(cmap=cm).set_table_styles(
                    styles
                )
            )

        return metrics_df

    def _threshold_youden(self):
        """
        Function to calculate youden index as a threshold
        """
        youden_index = np.argmax(np.abs(self.tpr_list - self.fpr_list))
        youden_threshold = self.roc_thresholds[youden_index]

        return youden_index, youden_threshold

    def _threshold_sens_spec(self):
        """
        Function to calculate the threshold that maximizes
        sensitivity-specificity curve"""
        sens_spec_index = np.argmin(abs(self.tpr_list + self.fpr_list - 1))
        sens_spec_threshold = self.roc_thresholds[sens_spec_index]

        return sens_spec_index, sens_spec_threshold

    def _threshold_prec_rec(self):
        """
        Function to calculate the threshold that maximizes precision-recall
        curve"""
        prec_rec_index = np.argmin(abs(self.precision_list - self.recall_list))
        prec_rec_threshold = self.pr_thresholds[prec_rec_index]

        return prec_rec_index, prec_rec_threshold

    def _thresholds_dict(self):
        """
        Function to return calculated thresholds as a dictionary
        """
        thresholds_dict = {
            "Youden": self.youden_threshold,
            "Sensitivity-Specificity": self.sens_spec_threshold,
            "Precision-Recall-F1": self.prec_rec_threshold,
        }

        return thresholds_dict

    def _plot_thresholds(self):
        """
        Function to plot all the calculated thresholds
        """

        # prepare thresholds for plotting
        thr_set1 = np.arange(
            min(self.roc_thresholds), max(self.roc_thresholds), 0.01
        )
        thr_set2 = np.arange(
            min(self.pr_thresholds), max(self.pr_thresholds), 0.01
        )
        f1_list = [
            2
            * (self.precision_list[i] * self.recall_list[i])
            / (self.precision_list[i] + self.recall_list[i])
            for i in range(len(self.precision_list))
        ]
        queue_rate_list = [
            (self.y_pred_proba >= thr).mean() for thr in thr_set2
        ]
        accuracy_list = [
            accuracy_score(self.y_true, (self.y_pred_proba >= thr).astype(int))
            for thr in thr_set1
        ]

        # subplot 1: roc curve
        plt.figure(figsize=self.figsize)
        plt.subplot(2, 2, 1)
        plt.plot(
            self.fpr_list,
            self.tpr_list,
            color="red",
            label=f"AUC = {self.auc_roc:.3f}",
        )
        plt.plot(
            self.fpr_list[self.youden_index],
            self.tpr_list[self.youden_index],
            marker="o",
            color="navy",
            ms=10,
        )
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("1 - Specificity", fontsize=12)
        plt.ylabel("Sensitivity", fontsize=12)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(prop={"size": 12}, loc=0)
        plt.title("ROC Curve", fontsize=12)
        plt.annotate(
            f"Threshold = {self.youden_threshold:.3f}",
            xy=(
                self.fpr_list[self.youden_index],
                self.tpr_list[self.youden_index],
            ),
            xycoords="data",
            xytext=(
                self.fpr_list[self.youden_index] + 0.4,
                self.tpr_list[self.youden_index] - 0.4,
            ),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        # subplot 2: preferred scores vs thresholds
        plt.subplot(2, 2, 2)
        plt.plot(self.roc_thresholds, 1 - self.fpr_list, label="Specificity")
        plt.plot(self.roc_thresholds, self.tpr_list, label="Sensitivity")
        plt.plot(thr_set1, accuracy_list, label="Accuracy")
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1)
        plt.axvline(self.sens_spec_threshold, color="k", ls="--")
        plt.title(f"Threshold = {self.sens_spec_threshold:.3f}", fontsize=12)
        plt.title("Preferred Scores vs Thresholds", fontsize=12)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        if self.sens_spec_threshold <= 0.5:
            plt.annotate(
                f"Threshold = {self.sens_spec_threshold:.3f}",
                xy=(self.sens_spec_threshold, 0.05),
                xycoords="data",
                xytext=(self.sens_spec_threshold + 0.1, 0.05),
                arrowprops=dict(facecolor="black", shrink=0.05),
                horizontalalignment="left",
                verticalalignment="bottom",
            )
        else:
            plt.annotate(
                f"Threshold = {self.sens_spec_threshold:.3f}",
                xy=(self.sens_spec_threshold, 0.05),
                xycoords="data",
                xytext=(self.sens_spec_threshold - 0.4, 0.05),
                arrowprops=dict(facecolor="black", shrink=0.05),
                horizontalalignment="left",
                verticalalignment="bottom",
            )

        # subplot 3: precision-recall curve
        plt.subplot(2, 2, 3)
        plt.plot(
            self.recall_list,
            self.precision_list,
            color="red",
            label=f"PR AUC ={self.auc_pr:.3f}",
        )
        plt.plot(
            self.recall_list[self.prec_rec_index],
            self.precision_list[self.prec_rec_index],
            marker="o",
            color="navy",
            ms=10,
        )
        plt.axvline(
            x=self.recall_list[self.prec_rec_index],
            ymin=self.recall_list[self.prec_rec_index],
            ymax=self.precision_list[self.prec_rec_index],
            color="navy",
            ls="--",
        )
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.legend(prop={"size": 12}, loc=0)
        plt.title("Precision-Recall Curve", fontsize=12)
        plt.annotate(
            f"Threshold = {self.prec_rec_threshold:.3f}",
            xy=(
                self.recall_list[self.prec_rec_index],
                self.precision_list[self.prec_rec_index],
            ),
            xycoords="data",
            xytext=(
                self.recall_list[self.prec_rec_index] - 0.4,
                self.precision_list[self.prec_rec_index] - 0.4,
            ),
            arrowprops=dict(facecolor="black", shrink=0.05),
            horizontalalignment="left",
            verticalalignment="bottom",
        )

        # subplot 4: preferred Scores vs Thresholds
        plt.subplot(2, 2, 4)
        plt.plot(
            self.pr_thresholds, self.precision_list[1:], label="Precision"
        )
        plt.plot(self.pr_thresholds, self.recall_list[1:], label="Recall")
        plt.plot(self.pr_thresholds, f1_list[1:], label="F1-Score")
        plt.plot(thr_set2, queue_rate_list, label="Queue Rate")
        plt.xlabel("Threshold", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.axvline(self.prec_rec_threshold, color="k", ls="--")
        plt.title("Preferred Scores vs Thresholds", fontsize=12)
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.legend(bbox_to_anchor=(1.2, 0.5), loc="center", ncol=1)
        if self.prec_rec_threshold <= 0.5:
            plt.annotate(
                f"Threshold = {self.prec_rec_threshold:.3f}",
                xy=(self.prec_rec_threshold, 0.03),
                xycoords="data",
                xytext=(self.prec_rec_threshold + 0.1, 0.03),
                arrowprops=dict(facecolor="black", shrink=0.05),
                horizontalalignment="left",
                verticalalignment="bottom",
            )
        else:
            plt.annotate(
                f"Threshold = {self.prec_rec_threshold:.3f}",
                xy=(self.prec_rec_threshold, 0.03),
                xycoords="data",
                xytext=(self.prec_rec_threshold - 0.4, 0.03),
                arrowprops=dict(facecolor="black", shrink=0.05),
                horizontalalignment="left",
                verticalalignment="bottom",
            )

        plt.show()

    @staticmethod
    def _average_methods():
        """
        Function to return average methods as a list
        """
        return ["binary", "weighted", "macro", "micro"]
