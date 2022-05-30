[![build](https://github.com/slickml/slick-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/slickml/slick-ml/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/slickml/slick-ml)](https://github.com/slickml/slick-ml/blob/master/LICENSE/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/slickml)](https://pypi.org/project/slickml/)
![PyPI Version](https://img.shields.io/pypi/v/slickml)
![Python Version](https://img.shields.io/pypi/pyversions/slickml)
[![Issues](https://img.shields.io/github/issues/slickml/slick-ml)](https://github.com/slickml/slick-ml/issues)
[![Forks](https://img.shields.io/github/forks/slickml/slick-ml)](https://github.com/slickml/slick-ml/network/members/)
[![Stars](https://img.shields.io/github/stars/slickml/slick-ml)](https://github.com/slickml/slick-ml/stargazers/)


<p align="center">
<a href="https://www.slickml.com/">
  <img src="https://raw.githubusercontent.com/slickml/slick-ml/master/assets/designs/logo_clear.png" width="250"></img></a>
</p>

<h1 align="center">
    SlickML🧞: Slick Machine Learning in Python
</h1>


**SlickML** is an open-source machine learning library written in Python aimed
at accelerating the experimentation time for a ML application. Data Scientist
tasks can often be repetitive such as feature selection, model tuning, or
evaluating metrics for classification and regression problems. SlickML provides
Data Scientists with a toolbox to quickly prototype solutions for a given problem with minimal code. 


## 🛠 Installation
To begin with, install [Python version >=3.8,<3.10](https://www.python.org) and simply run 🏃‍♀️ :
```console
pip install slickml
```
📣  Please note that a working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required to build the package. If you do not have `gcc` installed, the following commands depending on your operating system will take care of this requirement.
```console
# Mac Users
brew install gcc

# Linux Users
sudo apt install build-essential       
```

### 🐍 Python Virtual Environments
In order to avoid any potential conflicts with other installed Python packages, it is
recommended to use a virtual environment, e.g. [python poetry](https://python-poetry.org/), [python virtualenv](https://docs.python.org/3/library/venv.html), [pyenv virtualenv](https://github.com/pyenv/pyenv-virtualenv), or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Our recommendation is to use `python-poetry` 🥰 for everything 😁.


## 📌 Quick Start
✅ An example to quickly run a `Feature Selection` pipeline with embedded `Cross-Validation` and `Feature-Importance` visualization: 
```python
from slickml.feautre_selection import XGBoostFeatureSelector
xfs = XGBoostFeatureSelector()
xfs.fit(X, y)
```
![selection](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/feature_selection.png)

```python
xfs.plot_cv_results()
```
![xfscv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/xfs_cv_results.png)

```python
xfs.plot_frequency()
```
![frequency](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/feature_frequency.png)

✅ An example to quickly find the `tuned hyper-parameter` with `Bayesian Optimization`:
```python
from slickml.optimization import XGBoostClassifierBayesianOpt
xbo = XGBoostClassifierBayesianOpt()
xbo.fit(X_train, y_train)
```
![clfbo](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_hyper_params.png)

```python
best_params = xbo.get_best_params()
best_params

{"colsample_bytree": 0.8213916662259918,
 "gamma": 1.0,
 "learning_rate": 0.23148232373451072,
 "max_depth": 4,
 "min_child_weight": 5.632602921054691,
 "reg_alpha": 1.0,
 "reg_lambda": 0.39468801734425263,
 "subsample": 1.0
 }
```

✅ An example to quickly train/validate a `XGBoostCV Classifier` with `Cross-Validation`, `Feature-Importance`, and `Shap` visualizations:
```python
from slickml.classification import XGBoostCVClassifier
clf = XGBoostCVClassifier(params=best_params)
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)

clf.plot_cv_results()
```
![clfcv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_cv_results.png)

```python
clf.plot_feature_importance()
```
![clfimp](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_feature_importance.png)

```python
clf.plot_shap_summary(plot_type="violin")
```
![clfshap](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_shap_summary.png)

```python
clf.plot_shap_summary(plot_type="layered_violin", layered_violin_max_num_bins=5)
```
![clfshaplv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_shap_summary_lv.png)

```python
clf.plot_shap_waterfall()
```
![clfshapwf](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_shap_waterfall.png)


✅ An example to train/validate a `GLMNetCV Classifier` with `Cross-Validation` and `Coefficients` visualizations:
```python
from slickml.classification import GLMNetCVClassifier
clf = GLMNetCVClassifier(alpha=0.3, n_splits=4, metric="roc_auc")
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)

clf.plot_cv_results()
```
![clfglmnetcv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_glmnet_cv_results.png)

```python
clf.plot_coeff_path()
```
![clfglmnetpath](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_glmnet_paths.png)


✅ An example to quickly visualize the `binary classification metrics` based on multiple `thresholds`:
```python
from slickml.metrics import BinaryClassificationMetrics
clf_metrics = BinaryClassificationMetrics(y_test, y_pred_proba)
clf_metrics.plot()
```
![clfmetrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_metrics.png)


✅ An example to quickly visualize some `regression metrics`:
```python
from slickml.metrics import RegressionMetrics
reg_metrics = RegressionMetrics(y_test, y_pred)
reg_metrics.plot()
```
![regmetrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/reg_metrics.png)


## 🧑‍💻🤝 Contributing to SlickML🧞
You can find the details of the development process in our [Contributing](CONTRIBUTING.md) guidelines. We strongly believe that reading and following these guidelines will help us make the contribution process easy and effective for everyone involved 🚀🌙 .


## ❓ 🆘 📲 Need Help?
Please join our [Slack Channel](https://join.slack.com/t/slickml/shared_invite/zt-19taay0zn-V7R4jKNsO3n76HZM5mQfZA) to interact directly with the core team and our small community. This is a good place to discuss your questions and ideas or in general ask for help 👨‍👩‍👧 👫 👨‍👩‍👦 .


## 📚 Citing SlickML🧞
If you use SlickML in an academic work 📃 🧪 🧬 , please consider citing it 🙏 .
### Bibtex Entry:
```bib
@software{slickml2020,
  title={SlickML: Slick Machine Learning in Python},
  author={Tahmassebi, Amirhessam and Smith, Trace},
  url={https://github.com/slickml/slick-ml},
  version={0.2.0},
  year={2021},
}

@article{tahmassebi2021slickml,
  title={Slickml: Slick machine learning in python},
  author={Tahmassebi, Amirhessam and Smith, Trace},
  journal={URL available at: https://github. com/slickml/slick-ml},
  year={2021}
}
```