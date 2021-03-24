import numpy as np
import pandas as pd
import shap
import glmnet
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from slickml.formatting import Color
from slickml.utilities import df_to_csr
from slickml.plotting import (
    plot_xgb_cv_results,
    plot_xgb_feature_importance,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_glmnet_cv_results,
    plot_glmnet_coeff_path,
)


class XGBoostClassifier:
    """XGBoost Classifier.
    This is wrapper using XGBoost classifier to train a XGBoost
    model with using number of boosting rounds from the inputs. This
    function is pretty useful when feature selection is done and you
    want to train a model on the whole data and test on a separate
    validation set. Main reference is XGBoost Python API:
    (https://xgboost.readthedocs.io/en/latest/python/python_api.html)

    Parameters
    ----------
    num_boost_round: int, optional (default=200)
        Number of boosting round to train the model

    metrics: str or tuple[str], optional (default=("auc"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "auc", "aucpr", "error", "logloss"

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    importance_type: str, optional (default="total_gain")
        Importance type of xgboost.train() with possible values
        "weight", "gain", "total_gain", "cover", "total_cover"

    params: dict, optional
        Set of parameters for evaluation of xboost.train()
        (default={"eval_metric" : "auc",
                  "tree_method": "hist",
                  "objective" : "binary:logistic",
                  "learning_rate" : 0.05,
                  "max_depth": 2,
                  "min_child_weight" : 1,
                  "gamma" : 0.0,
                  "reg_alpha" : 0.0,
                  "reg_lambda" : 1.0,
                  "subsample" : 0.9,
                  "max_delta_step": 1,
                  "verbosity" : 0,
                  "nthread" : 4,
                  "scale_pos_weight" : 1})

    Attributes
    ----------
    feature_importance_: dict()
        Returns a dict() of all feature importance based on
        importance_type at each fold of each iteration during
        selection process

    scaler_: StandardScaler object
        Returns the scaler object if any of scale_mean or scale_std
        was passed True.

    X_train_: pandas.DataFrame
        Returns scaled training data set that passed if if any of
        scale_mean or scale_std was passed as True, else X_train.

    X_test_: pandas.DataFrame
        Returns transformed testing data set using scaler_ object if if any of
        scale_mean or scale_std was passed as True, else X_train.

    d_train_: xgboost.DMatrix object
        Returns the xgboost.DMatrix(X_train_, y_train)

    d_test_: xgboost.DMatrix object
        Returns the xgboost.DMatrix(X_test_, y_test)

    shap_values_train_: numpy.array
        SHAP values from treeExplainer using X_train

    shap_values_test_: numpy.array
        SHAP values from treeExplainer using X_test

    fit(X_train, y_train): instance method
        Returns None and applies the training process using
        the (X_train, y_train) set using xgboost.train()

    predict_proba(X_test, y_test): instance method
        Return the prediction probabilities for positive classes. Please note that
        it only reports the probability of the positive class, while the sklearn
        one returns for both and slicing like pred_proba[:, 1]
        is needed for positive class predictions

    predict(X_test, y_test, threshold=0.5): instance method
        Return the prediction classes based on the passed threshold.
        The default threshold is set at 0.5 while you can find the optimum thresholds
        based on different methods using BinaryClassificationMetrics.

    get_params(): instance method
        Returns params dict

    get_feature_importance(): instance method
        Returns feature importance based on importance_type

    plot_feature_importance(): instance method
        Plots feature importance

    plot_shap_summary(): instance method
        Plot shap values summary
    """

    def __init__(
        self,
        num_boost_round=None,
        metrics=None,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        importance_type=None,
        params=None,
    ):

        if num_boost_round is None:
            self.num_boost_round = 200
        else:
            if not isinstance(num_boost_round, int):
                raise TypeError("The input num_boost_round must have integer dtype.")
            else:
                self.num_boost_round = num_boost_round

        if metrics is None:
            self.metrics = "auc"
        else:
            if not isinstance(metrics, str):
                raise TypeError("The input metrics must be a str dtype.")
            else:
                self.metrics = metrics

        if not isinstance(sparse_matrix, bool):
            raise TypeError("The input sparse_matrix must have bool dtype.")
        else:
            self.sparse_matrix = sparse_matrix

        if not isinstance(scale_mean, bool):
            raise TypeError("The input scale_mean must have bool dtype.")
        else:
            self.scale_mean = scale_mean

        if not isinstance(scale_std, bool):
            raise TypeError("The input scale_std must have bool dtype.")
        else:
            self.scale_std = scale_std

        if importance_type is None:
            self.importance_type = "total_gain"
        else:
            if not isinstance(importance_type, str):
                raise TypeError("The input importance_type must have str dtype.")
            else:
                if importance_type in [
                    "weight",
                    "gain",
                    "total_gain",
                    "cover",
                    "total_cover",
                ]:
                    self.importance_type = importance_type
                else:
                    raise ValueError("The input importance_type value is not valid.")
        params_ = {
            "eval_metric": "auc",
            "tree_method": "hist",
            "objective": "binary:logistic",
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 0.9,
            "max_delta_step": 1,
            "verbosity": 0,
            "nthread": 4,
            "scale_pos_weight": 1,
        }
        if params is None:
            self.params = params_
        else:
            if not isinstance(params, dict):
                raise TypeError("The input params must have dict dtype.")
            else:
                self.params = params_
                for key, val in params.items():
                    self.params[key] = val

    def _dtrain(self, X_train, y_train):
        """
        Function to return dtrain matrix based on
        input parameters including sparse_matrix,
        and scaled using both numpy array and pandas
        DataFrame.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        if isinstance(X_train, np.ndarray):
            self.X_train = pd.DataFrame(
                X_train, columns=[f"F_{i}" for i in range(X_train.shape[1])]
            )
        elif isinstance(X_train, pd.DataFrame):
            self.X_train = X_train
        else:
            raise TypeError(
                "The input X_train must be numpy array or pandas DataFrame."
            )

        if isinstance(y_train, np.ndarray) or isinstance(y_train, list):
            self.y_train = y_train
        else:
            raise TypeError("The input y_train must be numpy array or list.")

        if self.sparse_matrix and self.scale_mean:
            raise ValueError(
                "The scale_mean should be False in conjuction of using sparse_matrix=True."
            )

        if self.scale_mean or self.scale_std:
            self.scaler_ = StandardScaler(
                with_mean=self.scale_mean, with_std=self.scale_std
            )
            self.X_train_ = pd.DataFrame(
                self.scaler_.fit_transform(self.X_train),
                columns=self.X_train.columns.tolist(),
            )
        else:
            self.X_train_ = self.X_train.copy()

        if not self.sparse_matrix:
            dtrain = xgb.DMatrix(data=self.X_train_, label=self.y_train)
        else:
            dtrain = xgb.DMatrix(
                data=df_to_csr(self.X_train_, fillna=0.0, verbose=False),
                label=self.y_train,
                feature_names=self.X_train_.columns.tolist(),
            )

        return dtrain

    def _dtest(self, X_test, y_test=None):
        """
        Function to return dtest matrix based on
        input X_test, y_test including sparse_matrix,
        and scaled using both numpy array and pandas
        DataFrame. It does apply scaler transformation
        in case it was used. Please note that y_test is
        optional since it might not be available while
        validating the model.

        Parameters
        ----------
        X_test: numpy.array or pandas.DataFrame
            Testing/validation features data

        y_test: numpy.array[int] or list[int], optional (default=None)
            List of testing/validation ground truth binary values [0, 1]
        """
        if isinstance(X_test, np.ndarray):
            self.X_test = pd.DataFrame(
                X_test, columns=[f"F_{i}" for i in range(X_test.shape[1])]
            )
        elif isinstance(X_test, pd.DataFrame):
            self.X_test = X_test
        else:
            raise TypeError("The input X_test must be numpy array or pandas DataFrame.")

        if y_test is None:
            self.y_test = None
        elif isinstance(y_test, np.ndarray) or isinstance(y_test, list):
            self.y_test = y_test
        else:
            raise TypeError("The input y_test must be numpy array or list.")

        if self.scale_mean or self.scale_std:
            self.X_test_ = pd.DataFrame(
                self.scaler_.transform(self.X_test),
                columns=self.X_test.columns.tolist(),
            )
        else:
            self.X_test_ = self.X_test.copy()

        if not self.sparse_matrix:
            dtest = xgb.DMatrix(data=self.X_test_, label=self.y_test)
        else:
            dtest = xgb.DMatrix(
                data=df_to_csr(self.X_test_, fillna=0.0, verbose=False),
                label=self.y_test,
                feature_names=self.X_test_.columns.tolist(),
            )

        return dtest

    def _model(self):
        """
        Function to train XGBoost model based on the given number
        of boosting round.
        """
        model = xgb.train(
            params=self.params,
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round - 1,
        )
        return model

    def _imp_to_df(self):
        """
        Function to build convert feature importance to df.
        """

        data = {"feature": [], f"{self.importance_type}": []}
        features_gain = self.model_.get_score(importance_type=self.importance_type)
        for key, val in features_gain.items():
            data["feature"].append(key)
            data[f"{self.importance_type}"].append(val)

        df = (
            pd.DataFrame(data)
            .sort_values(by=f"{self.importance_type}", ascending=False)
            .reset_index(drop=True)
        )

        return df

    def fit(self, X_train, y_train):
        """
        Function to run xgboost.train() method based on the given number of
        boosting round from the inputs using (X_train, y_train) set
        and returns it.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # creating dtrain
        self.dtrain_ = self._dtrain(X_train, y_train)

        # train model
        self.model_ = self._model()

        # feature importance
        self.feature_importance_ = self._imp_to_df()

        return None

    def get_params(self):
        """
        Function to return the train parameters for XGBoost.
        """

        return self.params

    def get_feature_importance(self):
        """
        Function to return the feature importance of the best model
        at each fold of each iteration of feature selection.
        """

        return self.feature_importance_

    def predict_proba(self, X_test, y_test=None):
        """
        Function to return the prediction probabilities for both classes.
        Please note that it only reports the probability of the positive class,
        while the sklearn one returns for both and slicing like pred_proba[:, 1]
        is needed for positive class predictions. Note that y_test is optional while
        it might not be available in validiation.

        Parameters
        ----------
        X_test: numpy.array or pandas.DataFrame
            Validation features data

        y_test: numpy.array[int] or list[int], optional (default=None)
            List of validation ground truth binary values [0, 1]
        """
        self.dtest_ = self._dtest(X_test, y_test)
        self.y_pred_proba_ = self.model_.predict(self.dtest_, output_margin=False)

        return self.y_pred_proba_

    def predict(self, X_test, y_test=None, threshold=0.5):
        """
        Function to return the prediction classes based on the passed threshold.
        The default threshold is set at 0.5 while you can find the optimum thresholds
        based on different methods using BinaryClassificationMetrics.

        Parameters
        ----------
        X_test: numpy.array or pandas.DataFrame
            Validation features data

        y_test: numpy.array[int] or list[int], optional (default=None)
            List of validation ground truth binary values [0, 1]

        threshold: float, optional (default=0.5)
            Threshold to define classes based on probabilities.
            predict_proba >= threshold would be defined as 1 else 0.
        """
        self.dtest_ = self._dtest(X_test, y_test)
        self.y_pred_proba_ = self.model_.predict(self.dtest_, output_margin=False)
        self.y_pred_ = [1 if p >= threshold else 0 for p in self.y_pred_proba_]

        return self.y_pred_

    def plot_feature_importance(
        self,
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

        plot_xgb_feature_importance(
            self.feature_importance_,
            figsize,
            color,
            marker,
            markersize,
            markeredgecolor,
            markerfacecolor,
            markeredgewidth,
            fontsize,
        )

    def plot_shap_summary(
        self,
        validation=True,
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
        Please note that this function should be ran after the predict_proba to
        make sure the X_test is being instansiated.

        Parameters
        ----------
        validation: bool, optional, (default=True)
            Flag to calculate SHAP values of X_test if it is True.
            If validation=False, it calculates the SHAP values of
            X_train and plots the summary plot.

        plot_type: str, optional (single-output default="dot", multi-output default="bar")
            The type of summar plot. Options are "bar", "dot", "violin", "layered_violin",
            and "compact_dot" which is recommended for SHAP interactions

        figsize: tuple, optional, (default="auto")
            Figure size

        color: str, optional, (default= "#D0AAF3" for "bar")
            Color of violin and layered violin plots are "RdBl" cmap
            Color of the horizontal lines when plot_type="bar" is "#D0AAF3"

        max_display: int, optional, (default=20)
            Limit to show the number of features in the plot

        feature_names: str, optional, (default=None)
            List of feature names to pass. It should follow the order
            of fatures

        title: str, optional, (default=None)
            Title of the plot

        show: bool, optional, (default=True)
            Flag to show the plot in inteactive environment

        sort: bool, optional, (default=True)
            Flag to plot sorted shap vlues in descending order

        color_bar: bool, optional, (default=True)
            Flag to show color_bar when plot_type is "dot" or "violin"

        layered_violin_max_num_bins: int, optional, (default=10)
            The number of bins for calculating the violin plots ranges
            and outliers

        class_names: list, optional, (default=None)
            List of class names for multi-output problems

        class_inds: list, optional, (default=True)
            List of class indices for multi-output problems

        color_bar_label: str, optional, (default="Feature Value")
            Label for color bar
        """

        # define tree explainer
        self.explainer_ = shap.TreeExplainer(self.model_)
        self.shap_values_test_ = self.explainer_.shap_values(self.X_test_)
        self.shap_values_train_ = self.explainer_.shap_values(self.X_train_)

        # check the validation flag
        if validation:
            # define shap values for X_test
            shap_values = self.shap_values_test_
            features = self.X_test_
        else:
            # define shap values for X_train
            shap_values = self.shap_values_train_
            features = self.X_train_

        plot_shap_summary(
            shap_values=shap_values,
            features=features,
            plot_type=plot_type,
            figsize=figsize,
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

    def plot_shap_waterfall(
        self,
        validation=True,
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
        """Function to plot shap waterfall plot.
        This function is a helper function to plot the shap waterfall plot
        based on all types of shap explainers including tree, linear, and dnn.
        This would show the cumulitative/composite ratios of shap values per feature.
        Therefore, it can be easily seen with each feature how much explainability we
        can acheieve. Please note that this function should be ran after the predict_proba to
        make sure the X_test is being instansiated.

        Parameters
        ----------
        validation: bool, optional, (default=True)
            Flag to calculate SHAP values of X_test if it is True.
            If validation=False, it calculates the SHAP values of
            X_train and plots the summary plot.

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
        """

        # define tree explainer
        self.explainer_ = shap.TreeExplainer(self.model_)
        self.shap_values_test_ = self.explainer_.shap_values(self.X_test_)
        self.shap_values_train_ = self.explainer_.shap_values(self.X_train_)

        # check the validation flag
        if validation:
            # define shap values for X_test
            shap_values = self.shap_values_test_
            features = self.X_test_
        else:
            # define shap values for X_train
            shap_values = self.shap_values_train_
            features = self.X_train_

        plot_shap_waterfall(
            shap_values,
            features,
            figsize=figsize,
            bar_color=bar_color,
            bar_thickness=bar_thickness,
            line_color=line_color,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            markeredgewidth=markeredgewidth,
            max_display=max_display,
            title=title,
            fontsize=fontsize,
        )


class XGBoostCVClassifier(XGBoostClassifier):
    """XGBoost CV Classifier.
    This is subclass of XGBoostClassifier to run xgboost.cv()
    model with n-folds cross-validation and train model based on
    the best number of boosting round to avoid over-fitting. This
    function is pretty useful when feature selection is done and you
    want to train a model on the whole data and test on a separate
    validation set. In this case, cross-validation part on the train
    set decrease the possibility of over-fitting.
    run xgboost.train(). Main reference is XGBoost Python API:
    (https://xgboost.readthedocs.io/en/latest/python/python_api.html)

    Parameters
    ----------
    num_boost_round: int, optional (default=200)
        Number of boosting round at each fold of xgboost.cv()

    n_splits: int, optional (default=4)
        Number of folds for cross-validation

    metrics: str or tuple[str], optional (default=("auc"))
        Metric used for evaluation at cross-validation
        using xgboost.cv(). Please note that this is different
        than eval_metric that needs to be passed to params dict.
        Possible values are "auc", "aucpr", "error", "logloss"

    early_stopping_rounds: int, optional (default=20)
        The criterion to early abort the xgboost.cv() phase
        if the test metric is not improved

    random_state: int, optional (default=1367)
        Random seed

    stratified: bool, optional (default=True)
        Flag to stratificaiton of the targets to run xgboost.cv() to
        find the best number of boosting round at each fold of
        each iteration

    shuffle: bool, optional (default=True)
        Flag to shuffle data to have the ability of building
        stratified folds in xgboost.cv()

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format.
        This would increase the speed of feature selection for
        relatively large datasets

    scale_mean: bool, optional (default=False)
        Flag to center the data before scaling. This flag should be
        False when using sparse_matrix=True, since it centering the data
        would decrease the sparsity and in practice it does not make any
        sense to use sparse matrix method and it would make it worse.

    scale_std: bool, optional (default=False)
        Flag to scale the data to unit variance
        (or equivalently, unit standard deviation)

    importance_type: str, optional (default="total_gain")
        Importance type of xgboost.train() with possible values
        "weight", "gain", "total_gain", "cover", "total_cover"

    params: dict, optional
        Set of parameters for evaluation of xboost.train()
        (default={"eval_metric" : "auc",
                  "tree_method": "hist",
                  "objective" : "binary:logistic",
                  "learning_rate" : 0.05,
                  "max_depth": 2,
                  "min_child_weight" : 1,
                  "gamma" : 0.0,
                  "reg_alpha" : 0.0,
                  "reg_lambda" : 1.0,
                  "subsample" : 0.9,
                  "max_delta_step": 1,
                  "verbosity" : 0,
                  "nthread" : 4,
                  "scale_pos_weight" : 1})

    callbacks: bool, optional (default=False)
        Flag for printing results during xgboost.cv().
        This would help to track the early stopping criterion

    verbose: bool, optional (default=False)
        Flag to show the final results of xgboost.cv()

    Attributes
    ----------
    feature_importance_: dict()
        Returns a dict() of all feature importance based on
        importance_type at each fold of each iteration during
        selection process

    feature_frequency_: Pandas DataFrame()
        Returns a DataFrame() cosists of total frequency of
        each feature during the selection process

    cv_results_: Pandas DataFrame()
        Return a Pandas DataFrame() of the mean value of the metrics

        n n-folds cross-validation for each boosting round
    scaler_: StandardScaler object
        Returns the scaler object if any of scale_mean or scale_std
        was passed True.

    X_train_: Pandas DataFrame()
        Returns scaled training data set that passed if if any of
        scale_mean or scale_std was passed as True, else X_train.

    X_test_: Pandas DataFrame()
        Returns transformed testing data set using scaler_ object if if any of
        scale_mean or scale_std was passed as True, else X_train.

    d_train_: xgboost.DMatrix object
        Returns the xgboost.DMatrix(X_train_, y_train)

    d_test_: xgboost.DMatrix object
        Returns the xgboost.DMatrix(X_test_, y_test)

    shap_values_train_: Numpy array
        SHAP values from treeExplainer using X_train

    shap_values_test_: Numpy array
        SHAP values from treeExplainer using X_test

    fit(X_train, y_train): instance method
        Returns None and applies the training process using
        the (X_train, y_train) set using xgboost.cv() and xgboost.train()

    predict_proba(X_test, y_test): instance method
        Return the prediction probabilities for positive classes. Please note that
        it only reports the probability of the positive class, while the sklearn
        one returns for both and slicing like pred_proba[:, 1]
        is needed for positive class predictions

    predict(X_test, y_test, threshold=0.5): instance method
        Return the prediction classes based on the passed threshold.
        The default threshold is set at 0.5 while you can find the optimum thresholds
        based on different methods using BinaryClassificationMetrics.

    get_params(): instance method
        Returns params dict

    get_feature_importance(): instance method
        Returns feature importance based on importance_type
        at each fold of each iteration of the selection process

    get_cv_results(): instance method
        Return a Pandas DataFrame() of the mean value of the metrics
        in n-folds cross-validation for each boosting round

    plot_cv_results(): instance method
        Plot cross-validation results

    plot_feature_importance(): instance method
        Plots feature importance

    plot_shap_summary(): instance method
        Plot shap values summary
    """

    def __init__(
        self,
        num_boost_round=None,
        n_splits=None,
        metrics=None,
        early_stopping_rounds=None,
        random_state=None,
        stratified=True,
        shuffle=True,
        sparse_matrix=False,
        scale_mean=False,
        scale_std=False,
        importance_type=None,
        params=None,
        callbacks=False,
        verbose=True,
    ):
        super().__init__(
            num_boost_round,
            metrics,
            sparse_matrix,
            scale_mean,
            scale_std,
            importance_type,
            params,
        )

        if n_splits is None:
            self.n_splits = 4
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must have integer dtype.")
            else:
                self.n_splits = n_splits

        if early_stopping_rounds is None:
            self.early_stopping_rounds = 20
        else:
            if not isinstance(early_stopping_rounds, int):
                raise TypeError(
                    "The input early_stopping_rounds must have integer dtype."
                )
            else:
                self.early_stopping_rounds = early_stopping_rounds

        if random_state is None:
            self.random_state = 1367
        else:
            if not isinstance(random_state, int):
                raise TypeError("The input random_state must have integer dtype.")
            else:
                self.random_state = random_state

        if not isinstance(stratified, bool):
            raise TypeError("The input stratified must have bool dtype.")
        else:
            self.stratified = stratified

        if not isinstance(shuffle, bool):
            raise TypeError("The input shuffle must have bool dtype.")
        else:
            self.shuffle = shuffle

        if not isinstance(callbacks, bool):
            raise TypeError("The input callbacks must have bool dtype.")
        else:
            if callbacks:
                self.callbacks = [
                    xgb.callback.print_evaluation(show_stdv=True),
                    xgb.callback.early_stop(self.early_stopping_rounds),
                ]
            else:
                self.callbacks = None

        if not isinstance(verbose, bool):
            raise TypeError("The input verbose must have bool dtype.")
        else:
            self.verbose = verbose

    def _cv(self):
        """
        Function to return XGBoost cv_results to find the best
        number of boosting rounds.
        """
        cvr = xgb.cv(
            params=self.params,
            dtrain=self.dtrain_,
            num_boost_round=self.num_boost_round,
            nfold=self.n_splits,
            stratified=self.stratified,
            metrics=self.metrics,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.random_state,
            shuffle=self.shuffle,
            callbacks=self.callbacks,
        )

        return cvr

    def _model(self):
        """
        Function to train XGBoost model based on the best number
        of boosting round.
        """
        model = xgb.train(
            params=self.params,
            dtrain=self.dtrain_,
            num_boost_round=len(self.cv_results_) - 1,
        )
        return model

    def fit(self, X_train, y_train):
        """
        Function to run xgboost.cv() method first to find the best number of boosting round
        and train a model based on that on (X_train, y_train) set and returns it.

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # creating dtrain, dtest (dtest here for the sake of plotting)
        self.dtrain_ = self._dtrain(X_train, y_train)

        # run cross-validation
        self.cv_results_ = self._cv()

        if self.verbose:
            print(
                Color.BOLD
                + "*-* "
                + Color.GREEN
                + f"Best Boosting Round = {len(self.cv_results_) - 1}"
                + Color.END
                + Color.BOLD
                + " -*- "
                + Color.F_Red
                + f"{self.n_splits}-Folds CV {self.metrics.upper()}: "
                + Color.END
                + Color.BOLD
                + Color.B_Blue
                + f"Train = {self.cv_results_.iloc[-1][0]:.3f}"
                + " +/- "
                + f"{self.cv_results_.iloc[-1][1]:.3f}"
                + Color.END
                + Color.BOLD
                + " -*- "
                + Color.B_Magenta
                + f"Test = {self.cv_results_.iloc[-1][2]:.3f}"
                + " +/- "
                + f"{self.cv_results_.iloc[-1][3]:.3f}"
                + Color.END
                + Color.BOLD
                + " *-*"
            )

        # train best model
        self.model_ = self._model()

        # feature importance
        self.feature_importance_ = self._imp_to_df()

        return None

    def get_cv_results(self):
        """
        Function to return both internal and external cross-validation
        results as Pandas DataFrame().
        """

        return self.cv_results_

    def plot_cv_results(
        self,
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
        Function to plot the results of xgboost.cv() process and evolution
        of metrics through number of boosting rounds.

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

        plot_xgb_cv_results(
            self.cv_results_,
            figsize,
            linestyle,
            train_label,
            test_label,
            train_color,
            train_std_color,
            test_color,
            test_std_color,
        )


class GLMNetCVClassifier:
    """GLMNet CV Classifier.
    This is wrapper using GLM-Net to train a Regularized Linear Model
    via logitic regression and find the optimal penalty values through
    N-Folds cross validation. This function is pretty useful to train
    a Logit-Net model with the ability of feature reduction. Main
    theoretical reference:
    (https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html)

    Parameters
    ----------
    alpha: float, optional (default=0.5)
        The stability parameter, 0 <= alpha <= 1: 0.0 for Ridge, 1.0 for LASSO

    n_lambda: int, optional (default=100)
        Maximum number of lambda values to compute

    n_splits: int, optional (default=3)
        Number of cross validation folds for computing performance metrics and
        determining lambda_best_ and lambda_max_. If non-zero, must be
        at least 3.

    metric: str or callable, optional (default="roc_auc")
        Metric used for model selection during cross validation.
        Valid options are "accuracy", "roc_auc", "average_precision",
        "precision", "recall". Alternatively, supply a function or callable
        object with the following signature "scorer(estimator, X, y)".
        Note, the metric function affects the selection of "lambda_best_" and
        "lambda_max_", fitting the same data ith different metric methods will result
        in the selection of different models.

    scale : bool, optional (default=True)
        Flag to standardize (mean of 0.0 and std of 1) input features prior to fitting.
        The final coefficients will be on the scale of the original data regardless
        of the value of the scale flag.

    sparse_matrix: bool, optional (default=False)
        Flag to convert data to sparse matrix with csr format. This would increase
        the speed of feature selection for relatively large datasets. Note that when
        "scale=True", you would lose sparsity due to standardization process.

    fit_intercept : bool, optional (default=True)
        Include an intercept term in the model.

    cut_point : float, optional (default=1.0)
        The cut point to use for selecting lambda_best.
        Based on this value, the distance between "lambda_max" and "best_lambda"
        would be cut_point * standard_error(lambda_max)
        arg_max(lambda) for cv_score(lambda) >= cv_score(lambda_max) - cut_point * standard_error(lambda_max)

    min_lambda_ratio: float, optional (default=1e-4)
        In combination with "n_lambda", the ratio of the smallest and largest
        values of lambda computed (min_lambda/max_lambda >= min_lambda_ratio).

    lambda_path: array, optional (default=None)
        In place of supplying "n_lambda", provide an array of specific values
        to compute. The specified values must be in decreasing order. When
        None, the path of lambda values will be determined automatically. A
        maximum of "n_lambda" values will be computed.

    tol: float, optional (default=1e-7)
        Convergence tolerance

    max_iter: int, optional (default=100000)
        Maximum passes over the data

    random_state: int, optional (default=1367)
        Seed for the random number generator. The glmnet solver is not
        deterministic, this seed is used for determining the cv folds.

    max_features: int, optional (default=None)
        Optional maximum number of features with nonzero coefficients after
        regularization. If not set, defaults to X.shape[1] during fit
        Note, this will be ignored if the user specifies lambda_path.

    Attributes
    ----------
    X_train_: pandas.DataFrame()
        Returns training data set.

    X_test_: pandas.DataFrame()
        Returns transformed testing data set.

    y_train_: numpy.array[int] or list[int]
        Returns the list of training ground truth binary values [0, 1]

    y_test_: numpy.array[int] or list[int]
        Returns the list of testing ground truth binary values [0, 1]

    coeff_: pandas.DataFrame
        Return the model's non-zero coefficients

    intercept_: float
        Return the model's intercept

    cv_results_: pandas.DataFrame
        Returns the cross-validation results

    results_: dict()
        Returns the model's total results

    params_: dict()
        Returns model's fitting parameters

    fit(X_train, y_train): instance method
        Returns None and applies the training process using
        the (X_train, y_train) set using glmnet.LogitNet()

    predict_proba(X_test, y_test=None): instance method
        Return the prediction probabilities for positive classes. Please note that
        it only reports the probability of the positive class, while the sklearn
        one returns for both and slicing like pred_proba[:, 1]
        is needed for positive class predictions

    predict(X_test, y_test=None, threshold=0.5): instance method
        Returns the prediction classes based on the passed threshold.
        The default threshold is set at 0.5.

    get_params(): instance method
        Returns params dict

    get_intercept(): instance method
        Returns model's intercept

    get_coeffs(): instance method
        Returns non-zero coefficients DataFrame

    get_cv_results(): instance method
        Returns cross-validation results DataFrame

    get_results(): instance method
        Returns model's total results dict

    plot_cv_results(): instance method
        Returns plot of cross-validation results

    plot_coeff_path(): instance method
        Returns plot of coeff. paths
    """

    def __init__(
        self,
        alpha=None,
        n_lambda=None,
        n_splits=None,
        metric=None,
        scale=True,
        sparse_matrix=False,
        fit_intercept=True,
        cut_point=None,
        min_lambda_ratio=None,
        lambda_path=None,
        tol=None,
        max_iter=None,
        random_state=None,
        max_features=None,
    ):

        if alpha is None:
            self.alpha = 0.5
        else:
            if not isinstance(alpha, float):
                raise TypeError("The input alpha must have float dtype.")
            else:
                self.alpha = alpha

        if n_lambda is None:
            self.n_lambda = 100
        else:
            if not isinstance(n_lambda, int):
                raise TypeError("The input n_lambda must be a int dtype.")
            else:
                self.n_lambda = n_lambda

        if n_splits is None:
            self.n_splits = 3
        else:
            if not isinstance(n_splits, int):
                raise TypeError("The input n_splits must be a int dtype.")
            else:
                self.n_splits = n_splits

        if metric is None:
            self.metric = "roc_auc"
        else:
            if not isinstance(metric, str):
                raise TypeError("The input metric must have str dtype.")
            else:
                if metric in [
                    "accuracy",
                    "roc_auc",
                    "average_precision",
                    "precision",
                    "recall",
                ]:
                    self.metric = metric
                else:
                    raise ValueError("The input metric value is not valid.")

        if not isinstance(scale, bool):
            raise TypeError("The input scale must have bool dtype.")
        else:
            self.scale = scale

        if not isinstance(sparse_matrix, bool):
            raise TypeError("The input sparse_matrix must have bool dtype.")
        else:
            self.sparse_matrix = sparse_matrix

        if self.sparse_matrix and self.scale:
            raise ValueError(
                "The scale should be False in conjuction of using sparse_matrix=True to maintain sparsity."
            )

        if not isinstance(fit_intercept, bool):
            raise TypeError("The input fit_intercept must have bool dtype.")
        else:
            self.fit_intercept = fit_intercept

        if cut_point is None:
            self.cut_point = 1.0
        else:
            if not isinstance(cut_point, float):
                raise TypeError("The input cut_point must have float dtype.")
            else:
                self.cut_point = cut_point

        if min_lambda_ratio is None:
            self.min_lambda_ratio = 1e-4
        else:
            if not isinstance(min_lambda_ratio, float):
                raise TypeError("The input min_lambda_ratio must have float dtype.")
            else:
                self.min_lambda_ratio = min_lambda_ratio

        if lambda_path is None:
            self.lambda_path = None
        else:
            if isinstance(lambda_path, np.ndarray) or isinstance(lambda_path, list):
                self.lambda_path = lambda_path
            else:
                raise TypeError("The input lambda_path must be numpy array or list.")

        if tol is None:
            self.tol = 1e-7
        else:
            if not isinstance(tol, float):
                raise TypeError("The input tol must have float dtype.")
            else:
                self.tol = tol

        if max_iter is None:
            self.max_iter = 100000
        else:
            if not isinstance(max_iter, int):
                raise TypeError("The input max_iter must have int dtype.")
            else:
                self.max_iter = max_iter

        if random_state is None:
            self.random_state = 1367
        else:
            if not isinstance(random_state, int):
                raise TypeError("The input random_state must have int dtype.")
            else:
                self.random_state = random_state

        if max_features is None:
            self.max_features = None
        else:
            if not isinstance(max_features, int):
                raise TypeError("The input max_features must have int dtype.")
            else:
                self.max_features = max_features

    def _dtrain(self, X_train, y_train):
        """
        Function to preprocess X_train, y_train data as
        Pandas DataFrame for the sake of postprocessing.

        Parameters
        ----------
        X_train: numpy.array or Pandas DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        if isinstance(X_train, np.ndarray):
            self.X_train = pd.DataFrame(
                X_train, columns=[f"F_{i}" for i in range(X_train.shape[1])]
            )
        elif isinstance(X_train, pd.DataFrame):
            self.X_train = X_train
        else:
            raise TypeError(
                "The input X_train must be numpy array or pandas DataFrame."
            )

        if isinstance(y_train, np.ndarray) or isinstance(y_train, list):
            self.y_train = y_train
        else:
            raise TypeError("The input y_train must be numpy array or list.")

        return self.X_train, self.y_train

    def _dtest(self, X_test, y_test=None):
        """
        Function to preprocess X_test, y_test data as
        Pandas DataFrame for the sake of postprocessing.
        Note that y_test is optional since it might not
        be available while validating the model.

        Parameters
        ----------
        X_test: numpy.array or Pandas DataFrame
            Testing features data

        y_test: numpy.array[int] or list[int]
            List of testing ground truth binary values [0, 1]
        """
        if isinstance(X_test, np.ndarray):
            self.X_test = pd.DataFrame(
                X_test, columns=[f"F_{i}" for i in range(X_test.shape[1])]
            )
        elif isinstance(X_test, pd.DataFrame):
            self.X_test = X_test
        else:
            raise TypeError("The input X_test must be numpy array or pandas DataFrame.")

        if y_test is None:
            self.y_test = None
        elif isinstance(y_test, np.ndarray) or isinstance(y_test, list):
            self.y_test = y_test
        else:
            raise TypeError("The input y_test must be numpy array or list.")

        return self.X_test, self.y_test

    def _model(self):
        """
        Function to initialize a LogitNet model.
        """

        model = glmnet.LogitNet(
            alpha=self.alpha,
            n_lambda=self.n_lambda,
            min_lambda_ratio=self.min_lambda_ratio,
            lambda_path=self.lambda_path,
            standardize=self.scale,
            fit_intercept=self.fit_intercept,
            cut_point=self.cut_point,
            n_splits=self.n_splits,
            scoring=self.metric,
            n_jobs=-1,
            tol=self.tol,
            max_iter=self.max_iter,
            random_state=self.random_state,
            max_features=self.max_features,
            verbose=False,
        )

        return model

    def _coeff_to_df(self):
        """
        Function to return the non-zero coeff for the best lambda as Pandas DataFrame.
        """
        dct = self._coeff_to_dict()

        return (
            pd.DataFrame(data=dct.items(), columns=["feature", "coeff"])
            .sort_values(by="coeff", ascending=False)
            .reset_index(drop=True)
        )

    def _coeff_to_dict(self):
        """
        Function to return the non-zero coeff for the best lambda as dict.
        """
        idx = list(np.nonzero(np.reshape(self.model_.coef_, (1, -1)))[1])
        dct = dict(
            zip(
                [self.X_train_.columns.tolist()[i] for i in idx],
                [
                    self.model_.coef_.reshape(-1, self.model_.coef_.shape[-1])[0][i]
                    for i in idx
                ],
                #                 [self.model_.coef_[0][i] for i in idx],
            )
        )

        return dct

    def _results(self):
        """
        Function to return model's results as a nested dictionary.
        """
        results = {}
        results["coeff"] = self._coeff_to_dict()
        results["coeff_path"] = dict(
            zip(
                [f"{col}" for col in self.X_train_.columns.tolist()],
                (
                    self.model_.coef_path_.reshape(-1, self.model_.coef_path_.shape[-1])
                ).tolist(),
            )
        )
        results["cv_standard_error"] = self.model_.cv_standard_error_.tolist()
        results["cv_mean_score"] = self.model_.cv_mean_score_.tolist()
        results["lambda_path"] = self.model_.lambda_path_.tolist()
        results["lambda_best"] = self.model_.lambda_best_[0]
        results["lambda_max"] = self.model_.lambda_max_
        results["n_lambda"] = self.model_.n_lambda_
        results["intercept"] = self.model_.intercept_
        results["intercept_path"] = self.model_.intercept_path_.tolist()[0]
        results["params"] = self.model_.get_params()
        results["module"] = self.model_.__module__

        return results

    def _cv_results(self):
        """
        Function to return model's results as a Pandas DataFrame.
        """
        df = pd.DataFrame(
            (self.model_.coef_path_.reshape(-1, self.model_.coef_path_.shape[-1])).T,
            columns=[f"{col}_coeff_path" for col in self.X_train_.columns.tolist()],
        )
        df["intercept_path"] = (
            self.model_.intercept_path_.reshape(
                -1, self.model_.intercept_path_.shape[-1]
            )
        ).T
        df["lambda_path"] = self.model_.lambda_path_
        df["cv_standard_error"] = self.model_.cv_standard_error_
        df["cv_mean_score"] = self.model_.cv_standard_error_

        return df

    def _prep_attributes(self):
        """
        Function to run all the model's attributes while fitting.
        """
        self.coeff_ = self._coeff_to_df()
        self.results_ = self._results()
        self.cv_results_ = self._cv_results()
        self.intercept_ = self.model_.intercept_
        self.params_ = self.model_.get_params()

        return None

    def fit(self, X_train, y_train):
        """
        Function to initialize a LogitNet model using (X, y).

        Parameters
        ----------
        X_train: numpy.array or pandas.DataFrame
            Training features data

        y_train: numpy.array[int] or list[int]
            List of training ground truth binary values [0, 1]
        """
        # preprocessing X, y
        self.X_train_, self.y_train_ = self._dtrain(X_train, y_train)

        # initialize model
        self.model_ = self._model()

        # train model
        if self.sparse_matrix:
            self.model_.fit(
                df_to_csr(self.X_train_, fillna=0.0, verbose=False), self.y_train_
            )
        else:
            self.model_.fit(self.X_train_, self.y_train_)

        # prep attributes
        self._prep_attributes()

        return None

    def predict_proba(self, X_test, y_test=None, lamb=None):
        """
        Function to return the prediction probabilities for positive classes.
        Please note that it only reports the probability of the positive class,
        while the sklearn one returns for both and slicing like pred_proba[:, 1]
        is needed for positive class predictions. Note that y_test is optional while
        it might not be available in validiation.

        Parameters
        ----------
        X_test: numpy.array or pandas.DataFrame
            Validation features data

        y_test: numpy.array[int] or list[int], optional (default=None)
            List of validation ground truth binary values [0, 1]

        lamb: array, optional (default=None)
        Values with shape (n_lambda,) of lambda from lambda_path_
        from which to make predictions. If no values are provided (None),
        the returned predictions will be those corresponding to lambda_best_.
        The values of lamb must also be in the range of lambda_path_,
        values greater than max(lambda_path_) or less than  min(lambda_path_)
        will be clipped.
        """
        self.X_test_, self.y_test_ = self._dtest(X_test, y_test)
        if self.sparse_matrix:
            self.y_pred_proba_ = self.model_.predict_proba(
                df_to_csr(self.X_test_), lamb=lamb
            )[:, 1]
        else:
            self.y_pred_proba_ = self.model_.predict_proba(self.X_test_, lamb=lamb)[
                :, 1
            ]

        return self.y_pred_proba_

    def predict(self, X_test, y_test=None, threshold=0.5, lamb=None):
        """
        Function to return the prediction classes based on the passed threshold.
        The default threshold is set at 0.5 while you can find the optimum thresholds
        based on different methods using BinaryClassificationMetrics.

        Parameters
        ----------
        X_test: numpy.array or pandas.DataFrame
            Validation features data

        y_test: numpy.array[int] or list[int], optional (default=None)
            List of validation ground truth binary values [0, 1]

        threshold: float, optional (default=0.5)
            Threshold to define classes based on probabilities.
            predict_proba >= threshold would be defined as 1 else 0.

        lamb: array, optional (default=None)
        Values with shape (n_lambda,) of lambda from lambda_path_
        from which to make predictions. If no values are provided (None),
        the returned predictions will be those corresponding to lambda_best_.
        The values of lamb must also be in the range of lambda_path_,
        values greater than max(lambda_path_) or less than  min(lambda_path_)
        will be clipped.
        """
        y_pred_proba = self.predict_proba(X_test, y_test=y_test, lamb=lamb)
        self.y_pred_ = [1 if p >= threshold else 0 for p in y_pred_proba]

        return self.y_pred_

    def get_intercept(self):
        """
        Function to return the model's intercept.
        """

        return self.intercept_

    def get_coeffs(self):
        """
        Function to return model's parameters.
        """

        return self._coeff_to_dict()

    def get_params(self):
        """
        Function to return model's parameters.
        """

        return self.params_

    def get_cv_results(self):
        """
        Function to return model's cv results.
        """

        return self.cv_results_

    def get_results(self):
        """
        Function to return model's total results.
        """

        return self.results_

    def plot_cv_results(
        self,
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

        plot_glmnet_cv_results(
            figsize=figsize,
            marker=marker,
            markersize=markersize,
            colors=colors,
            linestyle=linestyle,
            fontsize=fontsize,
            grid=grid,
            legend=legend,
            legendloc=legendloc,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            save_path=save_path,
            **self.results_,
        )

    def plot_coeff_path(
        self,
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

        plot_glmnet_coeff_path(
            figsize=figsize,
            linestyle=linestyle,
            fontsize=fontsize,
            grid=grid,
            legend=legend,
            legendloc=legendloc,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            bbox_to_anchor=bbox_to_anchor,
            yscale=yscale,
            save_path=save_path,
            **self.results_,
        )
