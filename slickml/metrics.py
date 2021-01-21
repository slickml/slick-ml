import numpy as np
import scipy as scp
import pandas as pd
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
    r2_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
)

from IPython.core.display import display
from slickml.plotting import plot_binary_classification_metrics, plot_regression_metrics


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
    accuracy: float value between 0. and 1.
        Classification accuracy based on threshold value
    balanced_accuracy: float value between 0. and 1.
        Balanced classification accuracy based on threshold value
        considering the prevalence of the classes
    fpr_list: numpy.array[float] or list[float]
        List of calculated false-positive-rates based on roc_thresholds.
        This can be used for ROC curve plotting
    tpr_list: numpy.array[float] or list[float]
        List of calculated true-positive-rates based on roc_thresholds
        This can be used for ROC curve plotting
    roc_thresholds: numpy.array[float] or list[float]
        List of thresholds value to calculate fpr_list and tpr_list
    auc_roc: float value between 0. and 1.
        Area under ROC curve
    precision_list: numpy.array[float] or list[float]
        List of calculated precision based on pr_thresholds
        This can be used for ROC curve plotting
    recall_list: numpy.array[float] or list[float]
        List of calculated recall based on pr_thresholds
        This can be used for ROC curve plotting
    pr_thresholds: numpy.array[float] or list[float]
        List of thresholds value to calculate precision_list and recall_list
    auc_pr: float value between 0. and 1.
        Area under Precision-Recall curve
    precision: float value between 0. and 1.
        Precision based on threshold value
    recall: float value between 0. and 1.
        Recall based on threshold value
    f1: float value between 0. and 1.
        F1-score based on threshold value (beta=1.0)
    f2: float value between 0. and 1.
        F2-score based on threshold value (beta=2.0)
    f05: float value between 0. and 1.
        F(1/2)-score based on threshold value (beta=0.5)
    average_precision: float value between 0. and 1.
        Avearge precision based on threshold value and class prevalence
    tn: integer
        True negative counts based on threshold value
    fp: integer
        False positive counts based on threshold value
    fn: integer
        False negative counts based on threshold value
    tp: integer
        True positive counts based on threshold value
    threat_score: float value between 0. and 1.
        Threat score based on threshold value
    youden_threshold: float value between 0. and 1.
        Threshold calculated based on Youden Index
    sens_spec_threshold: float value between 0. and 1.
        Threshold calculated based on maximized sensitivity-specificity
    prec_rec_threshold: float value between 0. and 1.
        Threshold calculated based on maximized precision-recall
    thresholds_dict: dict()
        Dictionary of all calculated thresholds
    metrics_dict: dict()
        Dictionary of all calculated metrics
    metrics_df: Pandas DataFrame()
        Pandas DataFrame of all calculated metrics with threshold as index
    average_methods: list[str]
        List of all possible average methods
    plotting_dict: dict()
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
        self.plotting_dict = self._plotting_dict()
        self.average_methods = self._average_methods()

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
            threat_score = wp * (self.tp / (self.tp + self.fp + self.fn)) + wn * (
                self.tn / (self.tn + self.fn + self.fp)
            )

        elif self.average_method == "macro":
            threat_score = 0.5 * (self.tp / (self.tp + self.fp + self.fn)) + 0.5 * (
                self.tn / (self.tn + self.fn + self.fp)
            )

        else:
            threat_score = self.tp / (self.tp + self.fp + self.fn)

        return threat_score

    def _metrics_dict(self):
        """
        Function to create a dictionary of all calculated metrics based on the
        precision digits and average method"""
        metrics_dict = {
            "Accuracy": round(self.accuracy, self.precision_digits),
            "Balanced Accuracy": round(self.balanced_accuracy, self.precision_digits),
            "ROC AUC": round(self.auc_roc, self.precision_digits),
            "PR AUC": round(self.auc_pr, self.precision_digits),
            "Precision": round(self.precision, self.precision_digits),
            "Recall": round(self.recall, self.precision_digits),
            "F-1 Score": round(self.f1, self.precision_digits),
            "F-2 Score": round(self.f2, self.precision_digits),
            "F-0.50 Score": round(self.f05, self.precision_digits),
            "Threat Score": round(self.threat_score, self.precision_digits),
            "Average Precision": round(self.average_precision, self.precision_digits),
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
                metrics_df.style.background_gradient(cmap=cm).set_table_styles(styles)
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

    def _plotting_dict(self):
        """
        Function to return the plotting properties as a dictionary
        """
        plotting_dict = {
            "roc_thresholds": self.roc_thresholds,
            "pr_thresholds": self.pr_thresholds,
            "precision_list": self.precision_list,
            "recall_list": self.recall_list,
            "y_pred_proba": self.y_pred_proba,
            "y_true": self.y_true,
            "fpr_list": self.fpr_list,
            "tpr_list": self.tpr_list,
            "auc_roc": self.auc_roc,
            "youden_index": self.youden_index,
            "youden_threshold": self.youden_threshold,
            "sens_spec_threshold": self.sens_spec_threshold,
            "prec_rec_threshold": self.prec_rec_threshold,
            "auc_pr": self.auc_pr,
            "prec_rec_index": self.prec_rec_index,
        }

        return plotting_dict

    def _average_methods(self):
        """
        Function to return average methods as a list
        """
        return ["binary", "weighted", "macro", "micro"]

    def plot(self, figsize=None):
        """
        Function to plot binary classification metrics.
        This function is a helper function based on the plotting_dict
        attribute of the BinaryClassificationMetrics class.
        Parameters
        ----------
        figsize: tuple, optional, (default=(12, 12))
            Figure size
        """

        plot_binary_classification_metrics(figsize, **self.plotting_dict)


class RegressionMetrics:
    """Regression Metrics.
    This is wrapper to calculate all the regression metrics.
    In case of multioutput regression, calculation methods can be chosen
    among ["raw_values", "uniform_average", "variance_weighted".
    Parameters
    ----------
    y_true: numpy.array[int] or list[float]
        List of ground truth target (response) values
    y_pred: numpy.array[float] or list[float]
        List of predicted target values list[float]
    multioutput: str, optional (default="uniform_average")
        Method to calculate the metric for multioutput targets. Possible values are
        ["raw_values", "uniform_average", "variance_weighted"].
        "raw_values" returns a full set of scores in case of multioutput input.
        "uniform_average" scores of all outputs are averaged with uniform weight.
        "variance_weighted" scores of all outputs are averaged, weighted by the variances
        of each individual output.
    precision_digits: int, optional (default=3)
        The number of precision digits to format the scores' dataframe
    display_df: boolean, optional (default=True)
        Flag to display the formatted scores' dataframe
    Attributes
    ----------
    y_residual: numpy.array[float] or list[float]
        Residual values (errors) calculated as (y_true - y_pred).
    y_residual_normsq:  numpy.array[float] or list[float]
        Square root of absolute value of y_residual.
    r2: float value between 0. and 1
        R2 score (coefficient of determination)
    ev: float value between 0. and 1
        Explained variance score.
    mae: float value between 0. and 1
        Mean absolute error.
    mse: float value between 0. and 1
        Mean squared error.
    msle: float value between 0. and 1
        Mean squared log error.
    mape: float value between 0. and 1
        Mean absolute percentage error.
    auc_rec: float value between 0. and 1
        Area under REC curve.
    deviation:  numpy.array[float] or list[float]
        List of deviations to plot REC curve.
    accuracy:  numpy.array[float] or list[float]
        Calculated accuracy at each deviation to plot REC curve.
    y_ratio:  numpy.array[float] or list[float]
        Ratio of y_pred/y_true.
    mean_y_ratio: float value between 0. and 1
        Mean value of y_pred/y_true ratio.
    std_y_ratio: float
        Standard deviation value of y_pred/y_true ratio.
    cv_y_ratio: float value between 0. and 1
        Coefficient of variation calculated as std_y_ratio/mean_y_ratio
    metrics_dict: dict()
        Dictionary of all calculated metrics
    metrics_df: Pandas DataFrame()
        Pandas DataFrame of all calculated metrics
    plotting_dict: dict()
        Plotting object as a dictionary consists of all
        calculated metrics which was used to plot curves
    """

    def __init__(
        self,
        y_true,
        y_pred,
        multioutput=None,
        precision_digits=None,
        display_df=True,
    ):
        if not isinstance(y_true, np.ndarray):
            self.y_true = np.array(y_true)
        else:
            self.y_true = y_true
        if not isinstance(y_pred, np.ndarray):
            self.y_pred = np.array(y_pred)
        else:
            self.y_pred = y_pred
        if multioutput is None:
            self.multioutput = "uniform_average"
        else:
            if not isinstance(multioutput, str):
                raise TypeError("The input multioutput must have str dtype.")
            else:
                if multioutput in ["raw_values", "variance_weighted"]:
                    self.multioutput = multioutput
                else:
                    raise ValueError("The input multioutput value is not valid.")
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
        self.y_residual = self.y_true - self.y_pred
        self.y_residual_normsq = np.sqrt(np.abs(self.y_residual))
        self.r2 = self._r2()
        self.ev = self._ev()
        self.mae = self._mae()
        self.mse = self._mse()
        self.msle = self._msle()
        self.mape = self._mape()
        self.deviation, self.accuracy, self.auc_rec = self._rec_curve()
        (
            self.y_ratio,
            self.mean_y_ratio,
            self.std_y_ratio,
            self.cv_y_ratio,
        ) = self._ratio_hist()
        self.metrics_dict = self._metrics_dict()
        self.metrics_df = self._metrics_df()
        self.plotting_dict = self._plotting_dict()

    def _r2(self):
        """
        Function to calculate R^2 score
        """
        r2 = r2_score(
            y_true=self.y_true, y_pred=self.y_pred, multioutput=self.multioutput
        )

        return r2

    def _ev(self):
        """
        Function to calculate explained variance score
        """
        ev = explained_variance_score(
            y_true=self.y_true, y_pred=self.y_pred, multioutput=self.multioutput
        )

        return ev

    def _mae(self):
        """
        Function to calculate mean-absolute-error
        """
        mae = mean_absolute_error(
            y_true=self.y_true, y_pred=self.y_pred, multioutput=self.multioutput
        )

        return mae

    def _mse(self):
        """
        Function to calculate mean-squared-error
        """
        mse = mean_squared_error(
            y_true=self.y_true, y_pred=self.y_pred, multioutput=self.multioutput
        )

        return mse

    def _msle(self):
        """
        Function to calculate mean-squared-log-error
        """
        if min(self.y_true) < 0 or min(self.y_pred) < 0:
            msle = None
        else:
            msle = mean_squared_log_error(
                y_true=self.y_true, y_pred=self.y_pred, multioutput=self.multioutput
            )

        return msle

    def _mape(self):
        """
        Function to calculate mean-absolute-percentage-error
        """
        mape = mean_absolute_percentage_error(
            y_true=self.y_true, y_pred=self.y_pred, multioutput=self.multioutput
        )

        return mape

    def _rec_curve(self):
        """
        Function to calculate the rec curve elements: deviation, accuracy, auc.
        Simpson method is used as the integral method to calculate the area under
        regression error characteristics (REC).
        REC is implemented based on the following paper:
        Bi, J., & Bennett, K. P. (2003). Regression error characteristic curves.
        In Proceedings of the 20th international conference on machine learning (ICML-03) (pp. 43-50).
        https://www.aaai.org/Papers/ICML/2003/ICML03-009.pdf
        """
        begin = 0.0
        end = 1.0
        interval = 0.01
        accuracy = []
        deviation = np.arange(begin, end, interval)

        # main loop to calculate norm and compare with each deviation
        for i in range(len(deviation)):
            count = 0.0
            for j in range(len(self.y_true)):
                calc_norm = np.linalg.norm(self.y_true[j] - self.y_pred[j]) / np.sqrt(
                    np.linalg.norm(self.y_true[j]) ** 2
                    + np.linalg.norm(self.y_pred[j]) ** 2
                )
                if calc_norm < deviation[i]:
                    count += 1
            accuracy.append(count / len(self.y_true))

        auc_rec = scp.integrate.simps(accuracy, deviation) / end

        return deviation, accuracy, auc_rec

    def _ratio_hist(self):
        """
        Function to calculate the histogram elements of y_pred/y_true ratio.
        This would report the coefficient of variation CV as std(ratio)/mean(ratio) based on the following paper:
        Tahmassebi, A., Gandomi, A. H., & Meyer-Baese, A. (2018, July). A Pareto front based evolutionary model
        for airfoil self-noise prediction. In 2018 IEEE Congress on Evolutionary Computation (CEC) (pp. 1-8). IEEE.
        https://www.amirhessam.com/assets/pdf/projects/cec-airfoil2018.pdf
        """
        y_ratio = self.y_pred / self.y_true
        mean_y_ratio = np.mean(y_ratio)
        std_y_ratio = np.std(y_ratio)
        cv_y_ratio = std_y_ratio / mean_y_ratio

        return y_ratio, mean_y_ratio, std_y_ratio, cv_y_ratio

    def _metrics_dict(self):
        """
        Function to create a dictionary of all calculated metrics based on the
        precision digits and multioutput method"""
        metrics_dict = {
            "R2 Score": round(self.r2, self.precision_digits),
            "Explained Variance Score": round(self.ev, self.precision_digits),
            "Mean Absolute Error": round(self.mae, self.precision_digits),
            "Mean Squared Error": round(self.mse, self.precision_digits),
            "Mean Squared Log Error": round(self.msle, self.precision_digits)
            if self.msle is not None
            else None,
            "Mean Absolute Percentage Error": round(self.mape, self.precision_digits),
            "REC AUC": round(self.auc_rec, self.precision_digits),
            "Coeff. of Variation": round(self.cv_y_ratio, self.precision_digits),
            "Mean of Variation": round(self.mean_y_ratio, self.precision_digits),
        }

        return metrics_dict

    def _metrics_df(self):
        """
        Function to create a pandas DataFrame of all calculated metrics based
        on the precision digits and average method"""

        metrics_df = pd.DataFrame(
            data=self.metrics_dict,
            index=["Metrics"],
        )
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
                metrics_df.style.background_gradient(cmap=cm).set_table_styles(styles)
            )

        return metrics_df

    def _plotting_dict(self):
        """
        Function to return the plotting properties as a dictionary
        """
        plotting_dict = {
            "r2": self.r2,
            "ev": self.ev,
            "mae": self.mae,
            "mse": self.mse,
            "y_pred": self.y_pred,
            "y_true": self.y_true,
            "y_residual": self.y_residual,
            "y_residual_normsq": self.y_residual_normsq,
            "auc_rec": self.auc_rec,
            "y_ratio": self.y_ratio,
            "cv_y_ratio": self.cv_y_ratio,
            "std_y_ratio": self.std_y_ratio,
            "mean_y_ratio": self.mean_y_ratio,
            "msle": self.msle,
            "mape": self.mape,
            "deviation": self.deviation,
            "accuracy": self.accuracy,
        }

        return plotting_dict

    def plot(self, figsize=None):
        """
        Function to plot binary classification metrics.
        This function is a helper function based on the plotting_dict
        attribute of the BinaryClassificationMetrics class.
        Parameters
        ----------
        figsize: tuple, optional, (default=(12, 12))
            Figure size
        """

        plot_regression_metrics(figsize, **self.plotting_dict)
