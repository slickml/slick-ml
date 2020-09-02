# SlickML: Slick Machine Learning in Python

**SlickML** is a machine learning library for Python. With SlickML, you can save more time on ML-Automation and ML-Tuning. Machine learing consists of many importants tricks that can only be learned in industry while dealing with production and deployment. Thus, the main philosophy of SlickML is to bring simplicity to produce more effective models with a minimal amount of code.

## Installation

First, install Python 3.6 from https://www.python.org. and then run:

```
pip install slickml
```

Note: in order to avoid any potential conflicts with other Python packages it's recommended to use a virtual environment, e.g. python3 virtualenv (see python3 virtualenv documentation) or conda environments for further documentaiton.


## Quick Start
Here is an example using SlickML to quickly visualize the binary classification metrics based on multiple calculated thresholds:

```python
# train a classifier using loaded train/test data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# import slickml
from slickml.metrics import BinaryClassificationMetrics
example = BinaryClassificationMetrics(y_test, y_pred_proba)
example.plot()

```
![slickml viz1](https://raw.githubusercontent.com/slickml/slick-ml/master/assets/images/metrics2.png)

## Contributing

Please read the Contributing document to understand the requirements for submitting pull-requests. Note before 
starting any major new feature work, please open an issue describing what you are planning to work on. This will 
ensure that interested parties can give valuable feedback on the feature, and let others know that you are working 
on it. 

Whether the contributions consists of adding new features,  optimizing code, or assisting with the documentation, we 
welcome new contributors of all experience levels. The SlickML community goals are to be helpful and effective

## Citing SlickML
If you use SlickML in academic work, please consider citing https://doi.org/10.1117/12.2304418

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

Tahmassebi, A. (2018, May). ideeple: Deep learning in a flash. In Disruptive Technologies in Information Sciences (Vol. 10652, p. 106520S). International Society for Optics and Photonics.

