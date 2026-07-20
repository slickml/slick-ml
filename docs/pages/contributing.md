🧑‍💻🤝 Contributing to SlickML🧞
================================

## 👋 Introduction
Hello from SlickML🧞 Team 👋 and welcome to our contributing guidelines 🤗 . Here we laid out the details of the development process based on our coding standards, and we hope these guidelines would ease the process for you. Please feel free to apply your revisions if you did not find these guidelines useful.


## 👩‍⚖️  Code of Conduct
We as members, contributors, and leaders pledge to make participation in our community a harassment-free experience for everyone, regardless of age, body size, visible or invisible disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation. We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community. By participating and contributing to this project, you agree to uphold
our [Code of Conduct](code_of_conduct.md) 🙏 .


## 🚀🌙 Getting Started
Please note that before starting any major work, open an issue describing what you are planning to work on. The best way to start is to check the
[*good-first-issue*](https://github.com/slickml/slick-ml/labels/good%20first%20issue) label🏷 on the issue board. In this way, the SlickML team members and other interested parties can give you feedback on the opened *`issue`* 🙋‍♀️ regarding the possible *`idea`* 💡, *`bug`* 🪲, or *`feature`* 🧬. Additionally, it will reduce the chance of duplicated work and it would help us to manage the tasks in a parallel fashion; so your pull request would get merged faster 🏎  🏁 . Whether the contributions consists of adding new features, optimizing the code-base, or assisting with the documentation, we welcome new contributors of all experience levels. The SlickML🧞 community goals are to be helpful and effective 🙌 .


## 📐 Coding Standards
- Long time Pythoneer 🐍 *Tim Peters* succinctly channels the BDFL’s guiding principles for Python’s design into 20 aphorisms, only 19 of which have been written down as [*Zen of Python*](https://peps.python.org/pep-0020/) 🧘‍♀️ .
  1. Beautiful is better than ugly.
  2. Explicit is better than implicit.
  3. Simple is better than complex.
  4. Complex is better than complicated.
  5. Flat is better than nested.
  6. Sparse is better than dense.
  7. Readability counts.
  8. Special cases aren't special enough to break the rules.
  9. Although practicality beats purity.
  10. Errors should never pass silently.
  11. Unless explicitly silenced.
  12. In the face of ambiguity, refuse the temptation to guess.
  13. There should be one-- and preferably only one --obvious way to do it.
  14. Although that way may not be obvious at first unless you're Dutch.
  15. Now is better than never.
  16. Although never is often better than *right* now.
  17. If the implementation is hard to explain, it's a bad idea.
  18. If the implementation is easy to explain, it may be a good idea.
  19. Namespaces are one honking great idea -- let's do more of those!
- We try to follow [*Google Python Style Guide*](https://google.github.io/styleguide/pyguide.html) as much as possible.
- We try to maximize the use of [*Data Classes*](https://peps.python.org/pep-0557/) in our source codes and unit-tests.


## 🐍 🥷 Environment Management

- To begin with, install a [Python version >=3.9,<3.13](https://www.python.org).
- A working [Fortran Compiler](https://gcc.gnu.org/install/) (`gfortran`) is also required. If you do not have `gcc` installed, the following commands depending on your operating system will take care of this requirement. Please note that installing `gcc` sometimes might take couple minutes ⏳ 🤦‍♂️.
  ```
  # Mac Users
  brew install gcc

  # Linux Users
  sudo apt install build-essential gfortran
  ```
- All developments are done via [*uv*](https://docs.astral.sh/uv/). To begin with, first install `uv` following the [*installation documentation*](https://docs.astral.sh/uv/getting-started/installation/) depending on your operating system.
- Once you setup your environment, to install the dependencies (`uv.lock`), simply run 🏃‍♀️ :
  ```
  uv sync
  ```
- We mainly use [*Poe the Poet*](https://poethepoet.natn.io/installation.html), a pythonic task runner that works well with `uv`. Install the CLI once 🏃‍♀️ :
  ```
  uv tool install poethepoet
  ```
- To make sure your environmnet is setup correctly, simply run 🏃‍♀️ :
  ```
  poe greet
  ```
- For more options for task runners, simply run 🏃‍♀️ :
  ```
  poe --help
  ```


### 🛠 Formatting
- To ease the process and reduce headache 💆‍♀️ , we have serialized the required formatting commands to save more time ⏰. To apply all the required `formatting` steps, simply run 🏃‍♀️ :
  ```
  poe format
  ```
- We save a lot of time ⏳ and mental energy 🔋 for more important matters by using [*ruff*](https://docs.astral.sh/ruff/) as our main code formatter (line-length = 100; see [`ruff.toml`](https://github.com/slickml/slick-ml/blob/master/ruff.toml)). To apply formatting, simply run 🏃‍♀️ :
  ```
  poe format
  ```
- To check if the code is formatted correctly (without writing), simply run 🏃‍♀️ :
  ```
  poe format --check
  ```


### 🪓 Linting
- Similar to formatting, to ease the process and reduce headache 💆‍♂️ , we have serialized the required linting commands to save more time ⏰. To apply all the required `linting` steps, simply run 🏃‍♀️ :
  ```
  poe check
  ```
- `poe check` essentially runs `poe format --check`, `ruff check .`, and `poe mypy` behind the scenes in a serial fashion. You can learn more about each step below 👇 .
- To lint our code base we use [*ruff*](https://docs.astral.sh/ruff/). To apply `ruff` to the code base, simply run 🏃‍♀️ :
  ```
  ruff check .
  ```
- We also use [*mypy*](https://github.com/python/mypy) with more specification laid out in [*mypy.ini*](https://github.com/slickml/slick-ml/blob/master/mypy.ini) to check static typing of our code base. To apply `mypy` to the code base, simply run 🏃‍♀️ :
  ```
  poe mypy
  ```
- To run lint + tests + build across supported Python versions, use [*tox*](https://tox.wiki/) ([`tox.ini`](https://github.com/slickml/slick-ml/blob/master/tox.ini)):
  ```
  poe tox
  ```

## 🧪 Testing
- We believe in [Modern Test Driven Development (TDD)](https://testdriven.io/blog/modern-tdd/) and mainly use [*pytest*](https://docs.pytest.org/en/7.1.x/), [*assertpy*](https://github.com/assertpy/assertpy) along with various plugins including [*pytest-cov*](https://github.com/pytest-dev/pytest-cov) with more specification laid out in [*.coveragerc*](https://github.com/slickml/slick-ml/blob/master/.coveragerc) to develop our unit-tests.
- All unit-tests live in `tests/` directory separted from the source code.
- All unit-test files should begin with the word `test` i.e. `test_foo.py`.
- Our naming convention for naming tests is `test_<method_under_test>__<when>__<then>` pattern which would increase the code readbility.
- We use [*pytest-cov*](https://github.com/pytest-dev/pytest-cov) plugin 🔌 helps to populated a coverage report 🗂 for the unit-tests and see the parts of the code that the related unit-tests have not touched 🔎 🕵️‍♀️.
- To run all unit-tests, simply run 🏃‍♀️ :
  ```
  poe test
  ```
- To run a specific test file, simply run 🏃‍♀️ :
  ```
  poe test tests/test_<file_name>.py
  ```


## 📖 Documentation
- We follow [*numpydoc*](https://numpydoc.readthedocs.io/en/latest/format.html) style guidelines for docstrings syntax, and best practices 👌 .
- We use [*Sphinx Auto API*](https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html) 🤖 for generating our API documentation 💪 .
- In order to generate the API documentation 🔖  from source 🌲 , simply run 🏃‍♀️ :
  ```
  poe sphinx
  ```
- The generated API documentation file can be found at `docs/_build/index.html`.
- You can also add a new page in `.rst` or `md` formats under `docs/pages`.
- All `sphinx` configurations that we are using are available in `docs/conf.py`.


## 🔥 Pull Requests
- We currently have `bug-report` and `feature-request` as [*issue-templates*](https://github.com/slickml/slick-ml/issues). As laid out above, please make sure to open-up an issue before start working on a major work and get the core team feedback.
- Try to fix one bug or add one new feature per PR. This would minimize the amount of code changes and it is easier for code-review. Hefty PRs usually do not get merged so fast while it could have been if the work was splitted into multiple PRs clearly laid out in an issue before hand. Therefore, the code reviewer would not be surprised by the work.
- We recommend to follow [*Fork and Pull Request Workflow*](https://github.com/susam/gitpr).
  1. Fork our repository to your own Github account.
  2. Clone the forked repository to your machine.
  3. Create a branch locally; our naming conventions are `bugfix/the-bug-i-fix` and `feature/the-new-feature-i-add` for bug fixes and new features, respectively.
  4. Please use **present** tense verbs for your commit messages i.e. `Fix bug ...`, `Add feature ...`, and avoid using past tense verbs.
  5. Try to `rebase` the commits as much as possible to keep the git history clean.
  6. Follow the `formatting`, `linting`, and `testing` guidelines above.
  7. Finally, to check cross-compatibility of your changes using different operating systems and python versions, simply run 🏃‍♀️ :
     ```
     poe tox
     ```
     🔔 Please note that, we are currently running `tox` against `python versions 3.9, 3.10, and 3.11`. Therefore, you can leverage `pyenv` and install these versions and use `pyenv local` command (i.e. `pyenv local 3.9.X 3.10.Y 3.11.Z`) to activate them before running `poe tox`.
  8. Now, you are ready to push your changes to your forked repository.
  9.  Lastly, open a PR in our repository to the `master` branch and follow the PR template so that we can efficiently review the changes as soon as possible and get your feature/bug-fix merged.
  10. Nicely done! You are all set! You are now officially part of [SlickML contributors](https://github.com/slickml/slick-ml/graphs/contributors).





## ❓ 🆘 📲 Need Help?
Please join our [Slack Channel](https://www.slickml.com/slack-invite) to interact directly with the core team and our small community. This is a good place to discuss your questions and ideas or in general ask for help 👨‍👩‍👧 👫 👨‍👩‍👦 .
