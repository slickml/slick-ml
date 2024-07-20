🛠 Installation
=================

- To begin with, install a [Python version >=3.8,<3.12](https://www.python.org).
- A working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required.
If you do not have `gcc` installed, the following commands depending on your operating
system will take care of this requirement. Please note that installing `gcc` sometimes might
take couple minutes ⏳ 🤦‍♂️. You can also check the standalone `gfortran` installers [here](https://github.com/fxcoudert/gfortran-for-macOS/releases).

  ```
  # Mac Users
  brew install gcc

  # Linux Users
  sudo apt install build-essential gfortran
  ```
- Now, to install the library from [PyPI](https://pypi.org/project/slickml/) simply run 🏃‍♀️ :

  ```
  pip install slickml
  ```
  or if you are a [python poetry](https://python-poetry.org/) user, simply run 🏃‍♀️ :
  ```
  poetry add slickml
  ```
- The SlickML does come with CLI tool which behaves similarly to many other CLIs for basic
  features. In order to find out which version of SlickML you are running, simply run 🏃‍♀️ :
  ```
  slickml --version | -v | version
  ```
- If you ever need more information on exactly what a certain command will do, use the ``--help``
or ``-h`` command. For example, to see all available commands, simply run 🏃‍♀️ :
  ```
  slickml --help | -h
  ```
- In order to avoid any potential conflicts with other installed Python packages, it is
recommended to use a virtual environment, e.g. [python poetry](https://python-poetry.org/), [python virtualenv](https://docs.python.org/3/library/venv.html), [pyenv virtualenv](https://github.com/pyenv/pyenv-virtualenv), or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
- We highly recommend to manage your projects using `python-poetry`. All **SlickML** developments are done via [*python-poetry*](https://python-poetry.org/). To begin with, first install `poetry` following the [*installation documentation*](https://python-poetry.org/docs/#installation) depending on your operating system.
- You can also easily [*manage your Python environments*](https://python-poetry.org/docs/managing-environments#managing-environments) and easily switch between environments via `poetry`. To set the `poetry` environment using your preferred `python` version (i.e. `3.9.18`) which is already installed on your system preferably via `pyenv`, simply run 🏃‍♀️ :
  ```
  poetry env use 3.9.18
  ```
