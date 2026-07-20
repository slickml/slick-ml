<div align="center">

[![build](https://github.com/slickml/slick-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/slickml/slick-ml/actions/workflows/ci.yml)
[![docs](https://github.com/slickml/slick-ml/actions/workflows/cd.yml/badge.svg)](https://github.com/slickml/slick-ml/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/slickml/slick-ml/graph/badge.svg?token=Z7XP51MB4K)](https://codecov.io/gh/slickml/slick-ml)
[![downloads](https://pepy.tech/badge/slickml)](https://pepy.tech/project/slickml)
[![license](https://img.shields.io/github/license/slickml/slick-ml)](https://github.com/slickml/slick-ml/blob/master/LICENSE/)
![pypi_version](https://img.shields.io/pypi/v/slickml)
![python_version](https://img.shields.io/pypi/pyversions/slickml)
[![slack_invite](https://badgen.net/badge/Join/SlickML%20Slack/purple?icon=slack)](https://www.slickml.com/slack-invite)
![twitter_url](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2FSlickML)

</div>

<p align="center">
  <a href="https://www.docs.slickml.com/">
    <img src="https://raw.githubusercontent.com/slickml/slick-ml/master/assets/designs/logo_clear.png" width="250"></img>
  </a>
</p>

<div align="center">
<h1 align="center">SlickML🧞: Slick Machine Learning in Python</h1>
  <p align="center">
    <a href="https://github.com/slickml/slick-ml/releases"> Explore Releases</a>
    🟣 
    <a href="https://github.com/slickml/slick-ml/blob/master/CONTRIBUTING.md"> Become a Contributor</a>
    🟣 
    <a href="https://www.docs.slickml.com/slick-ml/"> Explore API Docs</a>
    🟣 
    <a href="https://www.slickml.com/slack-invite"> Join our Slack</a>
    🟣 
    <a href="https://twitter.com/slickml"> Tweet Us</a>   
  </p>
</div>

## 🧠 SlickML🧞 Philosophy
**SlickML** is an open-source machine learning library written in Python aimed at accelerating the
experimentation time for ML applications with tabular data while maximizing the amount of information
can be inferred. Data Scientists' tasks can often be repetitive such as feature selection, model
tuning, or evaluating metrics for classification and regression problems. We strongly believe that a
good portion of the tasks based on tabular data can be addressed via gradient boosting and generalized
linear models<sup>[1](https://arxiv.org/pdf/2207.08815.pdf)</sup>. SlickML provides Data Scientists
with a toolbox to quickly prototype solutions for a given problem with minimal code while maximizing
the amount of information that can be inferred. Additionally, the prototype solutions can be easily
promoted and served in production with our recommended recipes via various model serving frameworks
including [ZenML](https://github.com/zenml-io/zenml), [BentoML](https://github.com/bentoml/BentoML),
and [Prefect](https://github.com/PrefectHQ/prefect). More details coming soon 🤞 ...


## 📖 Documentation
✨ The API documentation is available at [docs.slickml.com/slick-ml](https://www.docs.slickml.com/slick-ml/).

## 🛠 Installation
To begin with, install [Python version >=3.9,<3.13](https://www.python.org) and to install the library
from [PyPI](https://pypi.org/project/slickml/) simply run 🏃‍♀️ :
```
pip install slickml
```
or if you are a [uv](https://docs.astral.sh/uv/) user, simply run 🏃‍♀️ :
```
uv add slickml
```

📣  Please note that a working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required to build the package. If you do not have `gcc` installed, the following commands depending on your operating system will take care of this requirement.
```
# Mac Users
brew install gcc

# Linux Users
sudo apt install build-essential gfortran
```

The SlickML CLI tool behaves similarly to many other CLIs for basic features. In order to find out
which version of SlickML you are running, simply run 🏃‍♀️ :
```
slickml --version
```

### 🐍 Python Virtual Environments
In order to avoid any potential conflicts with other installed Python packages, it is
recommended to use a virtual environment, e.g. [uv](https://docs.astral.sh/uv/), [python virtualenv](https://docs.python.org/3/library/venv.html), [pyenv virtualenv](https://github.com/pyenv/pyenv-virtualenv), or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Our recommendation is to use `uv` 🥰 for everything 😁.


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
from slickml.optimization import XGBoostBayesianOptimizer
xbo = XGBoostBayesianOptimizer()
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
clf = GLMNetCVClassifier(alpha=0.3, n_splits=4, metric="auc")
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
Special thanks to all of our amazing contributors 👇

<a href="https://github.com/slickml/slick-ml/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=slickml/slick-ml" />
</a>

![Repobeats analytics image](https://repobeats.axiom.co/api/embed/ca865991b0547199fe7a069de7af25645b225e9c.svg "Repobeats analytics image")



## ❓ 🆘 📲 Need Help?
Please join our [Slack Channel](https://www.slickml.com/slack-invite) to interact directly with the core team and our small community. This is a good place to discuss your questions and ideas or in general ask for help 👨‍👩‍👧 👫 👨‍👩‍👦 .


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
