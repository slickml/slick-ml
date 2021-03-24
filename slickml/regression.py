import numpy as np
import pandas as pd
import glmnet

from slickml.utilities import df_to_csr
from slickml.plotting import (
    plot_glmnet_cv_results,
    plot_glmnet_coeff_path,
)


class GLMNetCVRegressor:
    """GLMNet CV Regressor.
    This is wrapper using GLM-Net to train a Regularized Linear Model
    and find the optimal penalty values through N-Folds cross validation.
    This function is pretty useful to train a Elastic-Net model with
    the ability of feature reduction. Main theoretical reference:
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

    metric: str or callable, optional (default="r2")
        Metric used for model selection during cross validation.
        Valid options are "r2", "mean_squared_error", "mean_absolute_error",
        and "median_absolute_error". Alternatively, supply a function or callable
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

    y_train_: numpy.array[float] or list[float]
        Returns the list of training ground truth target values

    y_test_: numpy.array[int] or list[int]
        Returns the list of testing ground truth target values

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
        the (X_train, y_train) set using glmnet.ElasticNet()

    predict(X_test, y_test=None): instance method
        Returns the prediction values

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
            self.metric = "r2"
        else:
            if not isinstance(metric, str):
                raise TypeError("The input metric must have str dtype.")
            else:
                if metric in [
                    "r2",
                    "mean_squared_error",
                    "mean_absolute_error",
                    "median_absolute_error",
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

        y_train: numpy.array[float] or list[float]
            List of training ground truth values
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

        y_test: numpy.array[float] or list[float]
            List of testing ground truth values
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
        Function to initialize a ElasticNet model.
        """

        model = glmnet.ElasticNet(
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
        Function to initialize a ElasticNet model using (X, y).

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

    def predict(self, X_test, lamb=None):
        """
        Function to return the prediction values

        Parameters
        ----------
        X_test: numpy.array or pandas.DataFrame
            Validation features data

        lamb: array, optional (default=None)
        Values with shape (n_lambda,) of lambda from lambda_path_
        from which to make predictions. If no values are provided (None),
        the returned predictions will be those corresponding to lambda_best_.
        The values of lamb must also be in the range of lambda_path_,
        values greater than max(lambda_path_) or less than  min(lambda_path_)
        will be clipped.
        """
        self.y_pred_ = self.model_.predict(X_test, lamb=lamb)

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
        """Function to plot GLMNetCVRegressor cross-validation results.

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
        """Function to plot GLMNetCVRegressor coefficients' paths.

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
