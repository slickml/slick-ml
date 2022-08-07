🛠 Installation
===============

- To begin with, install a [Python version >=3.8,<3.11](https://www.python.org).
- A working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required.
If you do not have `gcc` installed, the following commands depending on your operating
system will take care of this requirement. Please note that installing `gcc` sometimes might
take couple minutes ⏳ 🤦‍♂️.

  ```console
  # Mac Users
  brew install gcc

  # Linux Users
  sudo apt install build-essential gfortran
  ```
- Now, simply run 🏃‍♀️ :

  ```console
  pip install slickml
  ```
- We highly recommend to manage your projects in your favorite `python environment`. For instance, all **SlickML** developments are done via [*python-poetry*](https://python-poetry.org/). To begin with, first install `poetry` following the [*installation documentation*](https://python-poetry.org/docs/#installation) depending on your operating system.
- You can also easily [*manage your Python environments*](https://python-poetry.org/docs/managing-environments#managing-environments) and easily switch between environments via `poetry`. To set the `poetry` environment using your preferred `python` version (i.e. `3.9.9`), simply run 🏃‍♀️ :
  ```console
  poetry env use 3.9.9
  ```
