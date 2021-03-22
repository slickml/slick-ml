[![build status](https://travis-ci.com/slickml/slick-ml.svg?branch=master)](https://travis-ci.com/github/slickml/slick-ml)
[![License](https://img.shields.io/github/license/slickml/slick-ml)](https://github.com/slickml/slick-ml/blob/master/LICENSE/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/slickml)](https://pypi.org/project/slickml/)
![PyPI Version](https://img.shields.io/pypi/v/slickml)
[![Issues](https://img.shields.io/github/issues/slickml/slick-ml)](https://github.com/slickml/slick-ml/issues)
[![Forks](https://img.shields.io/github/forks/slickml/slick-ml)](https://github.com/slickml/slick-ml/network/members/)
[![Stars](https://img.shields.io/github/stars/slickml/slick-ml)](https://github.com/slickml/slick-ml/stargazers/)

<p align="center">
<a href="https://www.slickml.com/">
  <img src="https://raw.githubusercontent.com/slickml/slick-ml/master/assets/design/logo.png" width="250"></img></a>
</p>

<h1 align="center">
    SlickML: Slick Machine Learning in Python
</h1>


**SlickML** is an open-source machine learning library written in Python aimed
at accelerating the experimentation time for a ML application. Data Scientist
tasks can often be repetitive such as feature selection, model tuning, or
evaluating metrics for classification and regression problems. SlickML provides
Data Scientist with a toolbox of utility functions to quickly prototype
solutions for a given problem with minimal code.


## Installation

First, install Python 3.6 from https://www.python.org, and then run:

```
pip install slickml
```

Note: in order to avoid any potential conflicts with other Python packages it's
recommended to use a virtual environment, e.g. [Python3
virtualenv](https://docs.python.org/3/library/venv.html) or [Conda
environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for further documentation.


## Quick Start
Here is an exmple using SlickML to quickly run a feature selection pipeline: 
```python
# run feature selection using loaded data
from slickml.feautre_selection import XGBoostFeatureSelector
xfs = XGBoostFeatureSelector()
xfs.fit(X, y)
```
![selection](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/feature_selection.png)

```python
# plot cross-validation results
xfs.plot_cv_results()
```
![xfscv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/xfs_cv_results.png)

```python
# plot feature frequncy after feature selection
xfs.plot_frequency()
```
![frequency](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/feature_frequency.png)

Here is an example using SlickML how to tune hyper-params with Bayesian Optimization:
```python
# apply BayesianOpt to tune parameters of classifier using loaded train/test data
from slickml.optimization import XGBoostClassifierBayesianOpt
xbo = XGBoostClassifierBayesianOpt()
xbo.fit(X_train, y_train)
```
![clfbo](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_hyper_params.png)


```python
# best parameters
best_params = xbo.get_best_params()
best_params

{'colsample_bytree': 0.8213916662259918,
 'gamma': 1.0,
 'learning_rate': 0.23148232373451072,
 'max_depth': 4,
 'min_child_weight': 5.632602921054691,
 'reg_alpha': 1.0,
 'reg_lambda': 0.39468801734425263,
 'subsample': 1.0}
```

Here is an example using SlickML how to train/validate a XGBoostCV classifier:
```python
# train a classifier using loaded train/test data and best params
from slickml.classification import XGBoostCVClassifier
clf = XGBoostCVClassifier(params=best_params)
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)

# plot cross-validation results
clf.plot_cv_results()
```
![clfcv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_cv_results.png)

```python
# plot  features importance
clf.plot_feature_importance()
```
![clfimp](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_feature_importance.png)

```python
# plot SHAP summary violin plot
clf.plot_shap_summary(plot_type="violin")

```
![clfshap](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_shap_summary.png)

```python
# plot SHAP summary layered violin plot
clf.plot_shap_summary(plot_type="layered_violin", layered_violin_max_num_bins=5)

```
![clfshaplv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_shap_summary_lv.png)


```python
# plot SHAP waterfall plot
clf.plot_shap_waterfall()

```
![clfshapwf](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_shap_waterfall.png)

Here is an example using SlickML how to train/validate a GLMNetCV classifier:
```python
# train a classifier using loaded train/test data and your choice of params
from slickml.classification import GLMNetCVClassifier
clf = GLMNetCVClassifier(alpha=0.3, n_splits=4, metric="roc_auc")
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)

# plot cross-validation results
clf.plot_cv_results()
```
![clfglmnetcv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_glmnet_cv_results.png)

```python
# plot coefficients paths
clf.plot_coeff_path()

```
![clfglmnetpath](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_glmnet_paths.png)

Here is an example using SlickML to quickly visualize the binary classification 
metrics based on multiple calculated thresholds:
```python
# plot binary metrics
from slickml.metrics import BinaryClassificationMetrics
clf_metrics = BinaryClassificationMetrics(y_test, y_pred_proba)
clf_metrics.plot()

```
![clfmetrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_metrics.png)

Here is an example using SlickML to quickly visualize the regression metrics:

```python
# plot regression metrics
from slickml.metrics import RegressionMetrics
reg_metrics = RegressionMetrics(y_test, y_pred)
reg_metrics.plot()
```
![regmetrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/reg_metrics.png)

## Contributing

Please read the [Contributing](CONTRIBUTING.md) document to understand the requirements for
submitting pull-requests. Note before starting any major new feature work,
please open an issue describing what you are planning to work on. This will
ensure that interested parties can give valuable feedback on the feature, and
let others know that you are working on it. Whether the contributions consists of adding new features,  optimizing code, or assisting with the documentation, we welcome new contributors of all experience
levels. The SlickML community goals are to be helpful and effective.

## Citing SlickML
If you use SlickML in academic work, please consider citing
https://doi.org/10.1117/12.2304418 .

### Bibtex Entry:
```bib
@inproceedings{tahmassebi2018ideeple,
  title={ideeple: Deep learning in a flash},
  author={Tahmassebi, Amirhessam},
  booktitle={Disruptive Technologies in Information Sciences},
  volume={10652},
  pages={106520S},
  year={2018},
  organization={International Society for Optics and Photonics}
}
```
### APA Entry:

Tahmassebi, A. (2018, May). ideeple: Deep learning in a flash. In Disruptive
Technologies in Information Sciences (Vol. 10652, p. 106520S). International
Society for Optics and Photonics.

