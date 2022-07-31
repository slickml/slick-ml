Installation
============

Altair can be installed, along with the example datasets in vega_datasets_, using::

    $ pip install altair vega_datasets

If you are using the conda_ package manager, the equivalent is::

    $ conda install -c conda-forge altair vega_datasets

At this point, you should be able to open `Jupyter Notebook`_ or `JupyterLab`_
and execute any of the code from the :ref:`example-gallery`.
For more information on how to display charts in various notebook environments
and non-notebook IDEs, see :ref:`displaying-charts`.

Dependencies
============

Altair has the following dependencies, all of which are installed automatically
with the above installation commands:

- python 3.6 or newer
- entrypoints_
- jsonschema_
- NumPy_
- Pandas_
- Toolz_

To run Altair's full test suite and build Altair's documentation requires a few
additional dependencies:

- flake8
- pytest
- jinja2
- sphinx
- m2r
- docutils
- vega_datasets_
- ipython

Development Install
===================

The `Altair source repository`_ is available on GitHub. Once you have cloned the
repository and installed all the above dependencies, run the following command
from the root of the repository to install the master version of Altair:

.. code-block:: bash

    $ pip install -e .

To install development dependencies as well, run

.. code-block:: bash

    $ pip install -e .[dev]

If you do not wish to clone the source repository, you can install the
development version directly from GitHub using:

.. code-block:: bash

    $ pip install -e git+https://github.com/altair-viz/altair.git


.. _conda: https://docs.conda.io/
.. _Vega-Lite: http://vega.github.io/vega-lite
.. _JupyterLab: http://jupyterlab.readthedocs.io/
.. _Jupyter Notebook: https://jupyter-notebook.readthedocs.io/
.. _Zeppelin: https://zeppelin.apache.org/
.. _IPython: https://github.com/ipython/ipython

.. _entrypoints: https://github.com/takluyver/entrypoints
.. _jsonschema: https://github.com/Julian/jsonschema
.. _NumPy: http://www.numpy.org/
.. _Pandas: http://pandas.pydata.org
.. _Toolz: https://github.com/pytoolz/toolz
.. _vega_datasets: https://github.com/altair-viz/vega_datasets
.. _Altair source repository: http://github.com/altair-viz/altair
.. _nteract: https://nteract.io
.. _vega: https://pypi.python.org/pypi/vega/

.. To begin with, install [Python version >=3.8,<3.11](https://www.python.org) and simply run 🏃‍♀️ :
.. ```console
.. pip install slickml
.. ```
.. 📣  Please note that a working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required to build the package. If you do not have `gcc` installed, the following commands depending on your operating system will take care of this requirement.
.. ```console
.. # Mac Users
.. brew install gcc

.. # Linux Users
.. sudo apt install build-essential gfortran
.. ```

.. ### 🐍 Python Virtual Environments
.. In order to avoid any potential conflicts with other installed Python packages, it is
.. recommended to use a virtual environment, e.g. [python poetry](https://python-poetry.org/), [python virtualenv](https://docs.python.org/3/library/venv.html), [pyenv virtualenv](https://github.com/pyenv/pyenv-virtualenv), or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Our recommendation is to use `python-poetry` 🥰 for everything 😁.

