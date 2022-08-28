🛠 Installation
=================

- To begin with, install a [Python version >=3.8,<3.11](https://www.python.org).
- A working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required.
If you do not have `gcc` installed, the following commands depending on your operating
system will take care of this requirement. Please note that installing `gcc` sometimes might
take couple minutes ⏳ 🤦‍♂️.

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
- In order to avoid any potential conflicts with other installed Python packages, it is
recommended to use a virtual environment, e.g. [python poetry](https://python-poetry.org/), [python virtualenv](https://docs.python.org/3/library/venv.html), [pyenv virtualenv](https://github.com/pyenv/pyenv-virtualenv), or [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
- We highly recommend to manage your projects using `python-poetry`. All **SlickML** developments are done via [*python-poetry*](https://python-poetry.org/). To begin with, first install `poetry` following the [*installation documentation*](https://python-poetry.org/docs/#installation) depending on your operating system.
- You can also easily [*manage your Python environments*](https://python-poetry.org/docs/managing-environments#managing-environments) and easily switch between environments via `poetry`. To set the `poetry` environment using your preferred `python` version (i.e. `3.9.9`) which is already installed on your system preferably via `pyenv`, simply run 🏃‍♀️ :
  ```
  poetry env use 3.9.13
  ```
