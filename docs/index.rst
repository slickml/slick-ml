.. SlickML documentation master file, created by sphinx-quickstart on Sat May 28 00:59:59 2022.
   Reference: 
      - https://sphinx-tutorial.readthedocs.io/
      - https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html

.. NOTES: 
   1) You can create a new page under `docs/pages/` in `.md` or `.rst` format
   2) You can add any required sphinx extension to `docs/conf.py`
   3) `docs/pages/releases.md` is a symlink to `CHANGELOG.md` in the root

.. TODOS:
   1) Once we add example notebooks, we can populate them below. I am not still sure on the format
   maybe we can have one examples pages and add the examples there as table with a permalink to
   github project repo. Anything we put in the doc, should be easy to maintain. So, we should
   maximize the use of permalinks and symlinks.

SlickML🧞: Slick Machine Learning in Python
===========================================

|build_status| |docs_status| |codecov| |downloads| |github_stars| |slack_invite| 

**SlickML** is an open-source machine learning library written in Python aimed at accelerating the experimentation time for ML applications
with tabular data while maximizing the amount of information can be inferred. Data Scientists' tasks can often be repetitive such as feature
selection, model tuning, or evaluating metrics for classification and regression problems. We strongly believe that a good portion of the tasks
based on tabular data can be addressed via gradient boosting and generalized linear models<sup>[1](https://arxiv.org/pdf/2207.08815.pdf)</sup>.
SlickML provides Data Scientists with a toolbox to quickly prototype solutions for a given problem with minimal code while maximizing the amound
of information that can be inferred.


.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   Installation <pages/installation>
   Quick Start <pages/quick_start>
   Releases <pages/releases>
   Contributing <pages/contributing>
   Citation <pages/citation>
   License <pages/license>
   Code of Conduct <pages/code_of_conduct>
   Contact Us <pages/contact_us>

.. FUTURE PAGES:
   Examples <pages/examples>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |build_status| image:: https://github.com/slickml/slick-ml/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/slickml/slick-ml/actions/workflows/ci.yml
.. |docs_status| image:: https://github.com/slickml/slick-ml/actions/workflows/cd.yml/badge.svg
   :target: https://github.com/slickml/slick-ml/actions/workflows/cd.yml
.. |codecov| image:: https://codecov.io/gh/slickml/slick-ml/branch/master/graph/badge.svg?token=Z7XP51MB4K
   :target: https://codecov.io/gh/slickml/slick-ml
.. |license| image:: https://img.shields.io/github/license/slickml/slick-ml
   :target: https://github.com/slickml/slick-ml/blob/master/LICENSE
.. |downloads| image:: https://pepy.tech/badge/slickml
   :target: https://pepy.tech/project/slickml
.. |pypi_version| image:: https://img.shields.io/pypi/v/slickml
   :target: https://pypi.org/project/slickml
.. |python_versions| image:: https://img.shields.io/pypi/pyversions/slickml
   :target: https://pypi.org/project/slickml
.. |slack_invite| image:: https://badgen.net/badge/Join/SlickML%20Slack/purple?icon=slack
   :target: https://www.slickml.com/slack-invite
.. |github_stars| image:: https://img.shields.io/github/stars/slickml/slick-ml?color=cyan&label=github&logo=github
   :target: https://github.com/slickml/slick-ml
