📣 🥁 Changelog & Releases
=========================

- We follow [Semantic Versioning](http://semver.org/) to document any notable changes.
- Please checkout [SlickML Official Releases](https://github.com/slickml/slick-ml/releases) for more details.

---

## 📍 Unreleased Version X.X.X - XXXX-XX-XX

### 🛠 Fixed

### 🔥 Added

---

## 📍 Version 0.2.0-beta - 2022-05-29

### 🛠 Fixed
- [#78](https://github.com/slickml/slick-ml/pull/78) `build` badge using GitHub actions and removed the `travis-ci` badge and dependencies.
- [#77](https://github.com/slickml/slick-ml/pull/77) updated `.flake8`, `.gitingore` entries, `ISSUE_TEMPLATES`, `README.md`, `CONTRIBUTING.md`, `assets/`, `examples/` formats, and `src/` style, `ci.yml` workflow.

### 🔥 Added
- [#77](https://github.com/slickml/slick-ml/pull/77) added poetry essentials and essentials based on [#72](https://github.com/slickml/slick-ml/pull/72) and removed all `setup.py` essentials.
- [#77](https://github.com/slickml/slick-ml/pull/77) added `tox`, `mypy`, `pytest-cov`.
- [#77](https://github.com/slickml/slick-ml/pull/77) added `sphinx-auto-api-doc` based on [#32](https://github.com/slickml/slick-ml/pull/32). 

---

## 📍 Version 0.1.5 - 2021-09-06

### 🛠 Fixed
- [#74](https://github.com/slickml/slick-ml/pull/74) updated `requirements.txt` to the latest versions.
- [#71](https://github.com/slickml/slick-ml/pull/71) updated optimization examples.

### 🔥 Added
- [#71](https://github.com/slickml/slick-ml/pull/71) added `XGBoostRegressorBayesianOpt` and `XGBoostRegressorHyperOpt` classes in optimization. 

---

## 📍 Version 0.1.4 - 2021-05-31

### 🛠 Fixed
- [#70](https://github.com/slickml/slick-ml/pull/70) fixed bugs in `plot_xgb_cv_results`. 
- [#70](https://github.com/slickml/slick-ml/pull/70) fixed bugs in `plot_regression_metrics`. 
- [#70](https://github.com/slickml/slick-ml/pull/70) updated metrics initialization in `XGBoostClassifier` and `XGBoostCVClassifier`.
- [#70](https://github.com/slickml/slick-ml/pull/70) updated notebook examples to go over each class separetely.

### 🔥 Added
- [#70](https://github.com/slickml/slick-ml/pull/70) added `XGBoostRegressor` and `XGBoostCVRegressor` classes.
- [#70](https://github.com/slickml/slick-ml/pull/70) added `NeurIPS 2021` submission pdf.

---

## 📍 Version 0.1.3 - 2021-05-15

### 🛠 Fixed
- [#66](https://github.com/slickml/slick-ml/pull/66) fixed bugs in feature selection algorithm. 
- [#67](https://github.com/slickml/slick-ml/pull/67) fixed bugs in metrics. 
- [#66](https://github.com/slickml/slick-ml/pull/66) updated the order of the functions inside each class.
- [#68](https://github.com/slickml/slick-ml/pull/68) updated `save_path` in plotting functions.
- [#68](https://github.com/slickml/slick-ml/pull/68) updated `bibtex` citations to software.

### 🔥 Added
- [#68](https://github.com/slickml/slick-ml/pull/68) added directories for `JOSS` and `NeurIPS` papers.

---

## 📍 Version 0.1.2 - 2021-04-17

### 🛠 Fixed
- [#63](https://github.com/slickml/slick-ml/pull/63) fixed bugs in RegressionMetrics plotting. Now, the text label positions are dynamic and invariat of the data. Additionally, fixed the bug in coef. shapes in `GLMNet` classes. 
- [#61](https://github.com/slickml/slick-ml/pull/61) updated `metrics.py` attributes API to end with under-score
- [#63](https://github.com/slickml/slick-ml/pull/63) updated all docstrings based on Scikit-Learn API
- [#64](https://github.com/slickml/slick-ml/pull/64) updated `setup.py` with dynamic version and install requirements

### 🔥 Added
- [#60](https://github.com/slickml/slick-ml/pull/60) added `CHANGELOG.md`
- [#63](https://github.com/slickml/slick-ml/pull/63) added `GLMNetCVRegressor` class

---

## 📍 Version 0.1.1 - 2021-03-18

### 🛠 Fixed
- [#54](https://github.com/slickml/slick-ml/pull/54) fixed bug in XGBoostClassifer. dtest has `y_test` as required parameter while it should be optional, since you wont have the `y_true` in production.
- [#56](https://github.com/slickml/slick-ml/pull/56) fixed bugs in plotting
- [#57](https://github.com/slickml/slick-ml/pull/57) updated `requirements.txt`
- [#59](https://github.com/slickml/slick-ml/pull/59) updated docstrings

### 🔥 Added
- [#44](https://github.com/slickml/slick-ml/pull/44) added XGBoostClassifierHyperOpt
- [#57](https://github.com/slickml/slick-ml/pull/57) added GLMNetCVClassifier class, plotting, and examples, `CODE_OF_CONDUCT.md`

---

## 📍 Version 0.0.8 - 2021-02-17

### 🛠 Fixed
- [#47](https://github.com/slickml/slick-ml/pull/47) fixed bugs in HyperOpt `__init__`
- [#52](https://github.com/slickml/slick-ml/pull/52) updated xgboost version to 1.0.0 to remove the conflict with shap version

### 🔥 Added
- [#44](https://github.com/slickml/slick-ml/pull/44) added XGBoostClassifierHyperOpt
- [#49](https://github.com/slickml/slick-ml/pull/49) added Google Colab links to notebooks
- [#51](https://github.com/slickml/slick-ml/pull/51) added regression metrics
- [#52](https://github.com/slickml/slick-ml/pull/52) added SHAP waterfall plot

---

## 📍 Version 0.0.7 - 2020-09-27

### 🛠 Fixed
- [#28](https://github.com/slickml/slick-ml/pull/28) updated feature selection method from run to fit and removed X, y from init and added to fit to be similar to sklearn API.
- [#17](https://github.com/slickml/slick-ml/pull/17) updated plotting to Matplotlib object oriented API
- [#34](https://github.com/slickml/slick-ml/pull/34) fixed formatting and import bugs in source code
- [#38](https://github.com/slickml/slick-ml/pull/38) fixed typos in README and bug in `df_to_csr` function
- [#41](https://github.com/slickml/slick-ml/pull/41) updated requirements for bayesian optimization, design pattern, classification examples

### 🔥 Added
- [#4](https://github.com/slickml/slick-ml/pull/4) added `metrics.py`
- [#6](https://github.com/slickml/slick-ml/pull/6) added logo design
- [#9](https://github.com/slickml/slick-ml/pull/9) added plots for metrics and `utilities.py`
- [#12](https://github.com/slickml/slick-ml/pull/12) added PEP8
- [#15](https://github.com/slickml/slick-ml/pull/15) added `feature_selection.py` and `tests/`
- [#20](https://github.com/slickml/slick-ml/pull/20) added `formatting.py`
- [#23](https://github.com/slickml/slick-ml/pull/23) added examples for feature selection
- [#24](https://github.com/slickml/slick-ml/pull/24) added XGBoostCVClassifier
- [#37](https://github.com/slickml/slick-ml/pull/37) added SHAP summary plots
- [#38](https://github.com/slickml/slick-ml/pull/38) added unit tests for classification
- [#43](https://github.com/slickml/slick-ml/pull/43) added BayesianOpt class

---

## 📍 Version 0.0.1 - 2020-08-31

### 🔥 Added
- [#2](https://github.com/slickml/slick-ml/pull/2) initial ideas
