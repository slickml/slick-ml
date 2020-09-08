# SlickML: Slick Machine Learning in Python

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
xfs = XGBoostFeatureSelector(X, y)
xfs.run()
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

Here is an example using SlickML how to train/validate a XGBoostCV classifier:
```python
# train a classifier using loaded train/test data
from slickml.classification import XGBoostCVClassifier
clf = XGBoostCVClassifier()
clf.fit(X_train, y_train)

# plot cross-validation and feature importance results
clf.plot_cv_results()
clf.plot_feature_importance()
```
![clfcv](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_cv_results.png)
![clfimp](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/clf_feature_importance.png)

Here is an example using SlickML to quickly visualize the binary classification 
metrics based on multiple calculated thresholds:
```python
# plot binary metrics
from slickml.metrics import BinaryClassificationMetrics
y_pred_proba = clf.predict_proba(X_test, y_test)
metrics = BinaryClassificationMetrics(y_test, y_pred_proba)
metrics.plot()

```
![metrics](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/metrics.png)

## Contributing

Please read the Contributing document to understand the requirements for
submitting pull-requests. Note before starting any major new feature work,
please open an issue describing what you are planning to work on. This will
ensure that interested parties can give valuable feedback on the feature, and
let others know that you are working on it. 

Whether the contributions consists of adding new features,  optimizing code, or
assisting with the documentation, we welcome new contributors of all experience
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

