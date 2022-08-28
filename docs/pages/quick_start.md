📌 Quick Start
================

## ✅ Feature Selection
An example to quickly run a Feature Selection pipeline with embedded Cross-Validation and Feature-Importance visualization: 
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

## ✅ Hyper-parameter Tuning
An example to quickly find the tuned hyper-parameter with Bayesian Optimization:
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

## ✅ Classification via `XGBoost`
An example to quickly train/validate a `XGBoostCVClassifier` with Cross-Validation, Feature-Importance, and Shap visualizations:
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


## ✅ Classification via `GLMNet`
An example to train/validate a `GLMNetCVClassifier` with Cross-Validation and Coefficients visualizations:
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

## ✅ Classification Metrics
An example to quickly visualize the binary classification metrics based on multiple thresholds:
```python
from slickml.metrics import BinaryClassificationMetrics
clf_metrics = BinaryClassificationMetrics(y_test, y_pred_proba)
clf_metrics.plot()
```
![clfmetrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_metrics.png)

## ✅ Regression Metrics
An example to quickly visualize some regression metrics:
```python
from slickml.metrics import RegressionMetrics
reg_metrics = RegressionMetrics(y_test, y_pred)
reg_metrics.plot()
```
![regmetrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/reg_metrics.png)
