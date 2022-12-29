📣 🥁 Changelog & Releases
===========================

- We follow [Semantic Versioning](http://semver.org/) to document any notable changes.
- Please checkout [SlickML Official Releases](https://github.com/slickml/slick-ml/releases) for more details.


## 📍 Unreleased Version X.X.X - XXXX-XX-XX
### 🛠 Fixed

### 🔥 Added
---

## 📍 Version 0.2.1 - 2022-12-29
### 🛠 Fixed
- [#174](https://github.com/slickml/slick-ml/pull/174) fixed `badges` in API docs.
### 🔥 Added
- [#177](https://github.com/slickml/slick-ml/pull/177), [#176](https://github.com/slickml/slick-ml/pull/176) added `CLI` basic functionalities for `version` and `help`.
- [#175](https://github.com/slickml/slick-ml/pull/175) added unit-tests to cover `save-path` flag in all visualization modules.
- [#173](https://github.com/slickml/slick-ml/pull/173) added threshold in `.coveragerc` and `codecov.yml` to protect test coverages.

---
## 📍 Version 0.2.0 - 2022-11-27

### 🛠 Fixed
- [#170](https://github.com/slickml/slick-ml/pull/170) enabled more `flake8` plugins and fixed `poe check` command and `mypy` dependencies.
- [#169](https://github.com/slickml/slick-ml/pull/169) refactored `XGBoostHyperOptimizer` class.

### 🔥 Added
- [#171](https://github.com/slickml/slick-ml/pull/171) added `type-stubs` and rolled out type checking with `mypy` across library.

---

## 📍 Version 0.2.0-beta.2 - 2022-11-13

### 🛠 Fixed
- [#167](https://github.com/slickml/slick-ml/pull/167), [#160](https://github.com/slickml/slick-ml/pull/160) fixed `dependencies`, `tox.ini` and `README.md`.
- [#164](https://github.com/slickml/slick-ml/pull/164) refactored `XGBoostBayesianOptimizer` class.
- [#161](https://github.com/slickml/slick-ml/pull/161) fixed `XGBoostFeatureSelector` `callbacks` to work smoothly.
- [#157](https://github.com/slickml/slick-ml/pull/157) fixed `codecov-action` to use `v3`.
- [#156](https://github.com/slickml/slick-ml/pull/156) refactored `XGBoostFeatureSelector` class.
- [#155](https://github.com/slickml/slick-ml/pull/155) fixed default PR reviewers.
### 🔥 Added
- [#162](https://github.com/slickml/slick-ml/pull/162) added `BaseXGBoostEstimator` class.
- [#158](https://github.com/slickml/slick-ml/pull/158) added `conftest.py` for `pytest` unit-tests.
- [#153](https://github.com/slickml/slick-ml/pull/153), [#159](https://github.com/slickml/slick-ml/pull/159) added ascii banner arts to `poe greet` command.

---

## 📍 Version 0.2.0-beta.1 - 2022-10-04

### 🛠 Fixed
- [#143](https://github.com/slickml/slick-ml/pull/143), [#123](https://github.com/slickml/slick-ml/pull/123) fixed `CI/CD` workflows.
- [#141](https://github.com/slickml/slick-ml/pull/141), [#144](https://github.com/slickml/slick-ml/pull/144) refactored `GLMNetCVClassifier`.
- [#137](https://github.com/slickml/slick-ml/pull/137), [#135](https://github.com/slickml/slick-ml/pull/135) refactored `XGBoostCVRegressor`.
- [#133](https://github.com/slickml/slick-ml/pull/133), [#126](https://github.com/slickml/slick-ml/pull/126) refactored `XGBoostCVClassifier`.
- [#147](https://github.com/slickml/slick-ml/pull/147), [#113](https://github.com/slickml/slick-ml/pull/113), [#108](https://github.com/slickml/slick-ml/pull/108), [#109](https://github.com/slickml/slick-ml/pull/109) refactored `Metrics`.
- [#95](https://github.com/slickml/slick-ml/pull/95), [#100](https://github.com/slickml/slick-ml/pull/100), [#100](https://github.com/slickml/slick-ml/pull/110) fixed `Format / Lint`.
- [#96](https://github.com/slickml/slick-ml/pull/96), [#98](https://github.com/slickml/slick-ml/pull/98), [#112](https://github.com/slickml/slick-ml/pull/112) fixed `Utils` functions and transformations.
- [#105](https://github.com/slickml/slick-ml/pull/105), [#150](https://github.com/slickml/slick-ml/pull/150), [#148](https://github.com/slickml/slick-ml/pull/148), [#145](https://github.com/slickml/slick-ml/pull/145), [#114](https://github.com/slickml/slick-ml/pull/114), [#127](https://github.com/slickml/slick-ml/pull/127), [#115](https://github.com/slickml/slick-ml/pull/115),  [#129](https://github.com/slickml/slick-ml/pull/129), [#130](https://github.com/slickml/slick-ml/pull/130), [#117](https://github.com/slickml/slick-ml/pull/117), [#116](https://github.com/slickml/slick-ml/pull/116), [#111](https://github.com/slickml/slick-ml/pull/111), [#124](https://github.com/slickml/slick-ml/pull/124) fixed `Sphinx Auto-Api Docs + README`.

### 🔥 Added
- [#142](https://github.com/slickml/slick-ml/pull/142) added `Poetry v1.2` dependencies.
- [#138](https://github.com/slickml/slick-ml/pull/138) added `codecov.yml`.
- [#131](https://github.com/slickml/slick-ml/pull/131) added `py.typed` to comply with `PEP-561`.
- [#104](https://github.com/slickml/slick-ml/pull/104) added Workflow for API Docs Deploy.
- [#103](https://github.com/slickml/slick-ml/pull/103) added Check-Var Utilities.
- [#99](https://github.com/slickml/slick-ml/pull/99) added PR template.

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
- [#68](https://github.com/slickml/slick-ml/pull/68) updated `save_path` in plotting functions.
- [#68](https://github.com/slickml/slick-ml/pull/68) updated `bibtex` citations to software.
- [#67](https://github.com/slickml/slick-ml/pull/67) fixed bugs in metrics. 
- [#66](https://github.com/slickml/slick-ml/pull/66) fixed bugs in feature selection algorithm. 
- [#66](https://github.com/slickml/slick-ml/pull/66) updated the order of the functions inside each class.

### 🔥 Added
- [#68](https://github.com/slickml/slick-ml/pull/68) added directories for `JOSS` and `NeurIPS` papers.

---

## 📍 Version 0.1.2 - 2021-04-17

### 🛠 Fixed
- [#64](https://github.com/slickml/slick-ml/pull/64) updated `setup.py` with dynamic version and install requirements
- [#63](https://github.com/slickml/slick-ml/pull/63) fixed bugs in RegressionMetrics plotting. Now, the text label positions are dynamic and invariat of the data. Additionally, fixed the bug in coef. shapes in `GLMNet` classes. 
- [#63](https://github.com/slickml/slick-ml/pull/63) updated all docstrings based on Scikit-Learn API
- [#61](https://github.com/slickml/slick-ml/pull/61) updated `metrics.py` attributes API to end with under-score

### 🔥 Added
- [#63](https://github.com/slickml/slick-ml/pull/63) added `GLMNetCVRegressor` class
- [#60](https://github.com/slickml/slick-ml/pull/60) added `CHANGELOG.md`

---

## 📍 Version 0.1.1 - 2021-03-18

### 🛠 Fixed
- [#59](https://github.com/slickml/slick-ml/pull/59) updated docstrings
- [#57](https://github.com/slickml/slick-ml/pull/57) updated `requirements.txt`
- [#56](https://github.com/slickml/slick-ml/pull/56) fixed bugs in plotting
- [#54](https://github.com/slickml/slick-ml/pull/54) fixed bug in XGBoostClassifer. dtest has `y_test` as required parameter while it should be optional, since you wont have the `y_true` in production.

### 🔥 Added
- [#57](https://github.com/slickml/slick-ml/pull/57) added GLMNetCVClassifier class, plotting, and examples, `CODE_OF_CONDUCT.md`
- [#44](https://github.com/slickml/slick-ml/pull/44) added XGBoostClassifierHyperOpt

---

## 📍 Version 0.0.8 - 2021-02-17

### 🛠 Fixed
- [#52](https://github.com/slickml/slick-ml/pull/52) updated xgboost version to 1.0.0 to remove the conflict with shap version
- [#47](https://github.com/slickml/slick-ml/pull/47) fixed bugs in HyperOpt `__init__`

### 🔥 Added
- [#52](https://github.com/slickml/slick-ml/pull/52) added SHAP waterfall plot
- [#51](https://github.com/slickml/slick-ml/pull/51) added regression metrics
- [#49](https://github.com/slickml/slick-ml/pull/49) added Google Colab links to notebooks
- [#44](https://github.com/slickml/slick-ml/pull/44) added XGBoostClassifierHyperOpt

---

## 📍 Version 0.0.7 - 2020-09-27

### 🛠 Fixed
- [#41](https://github.com/slickml/slick-ml/pull/41) updated requirements for bayesian optimization, design pattern, classification examples
- [#38](https://github.com/slickml/slick-ml/pull/38) fixed typos in README and bug in `df_to_csr` function
- [#34](https://github.com/slickml/slick-ml/pull/34) fixed formatting and import bugs in source code
- [#28](https://github.com/slickml/slick-ml/pull/28) updated feature selection method from run to fit and removed X, y from init and added to fit to be similar to sklearn API.
- [#17](https://github.com/slickml/slick-ml/pull/17) updated plotting to Matplotlib object oriented API

### 🔥 Added
- [#43](https://github.com/slickml/slick-ml/pull/43) added BayesianOpt class
- [#38](https://github.com/slickml/slick-ml/pull/38) added unit tests for classification
- [#37](https://github.com/slickml/slick-ml/pull/37) added SHAP summary plots
- [#24](https://github.com/slickml/slick-ml/pull/24) added XGBoostCVClassifier
- [#23](https://github.com/slickml/slick-ml/pull/23) added examples for feature selection
- [#20](https://github.com/slickml/slick-ml/pull/20) added `formatting.py`
- [#15](https://github.com/slickml/slick-ml/pull/15) added `feature_selection.py` and `tests/`
- [#12](https://github.com/slickml/slick-ml/pull/12) added PEP8
- [#9](https://github.com/slickml/slick-ml/pull/9) added plots for metrics and `utilities.py`
- [#6](https://github.com/slickml/slick-ml/pull/6) added logo design
- [#4](https://github.com/slickml/slick-ml/pull/4) added `metrics.py`

---

## 📍 Version 0.0.1 - 2020-08-31

### 🔥 Added
- [#2](https://github.com/slickml/slick-ml/pull/2) initial ideas
