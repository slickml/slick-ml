# Contributing

## Contribute to SlickML

We welcome your pull requests! Any new features need to contain unit tests and must be PEP8 conformant (max-line-length = 100). 
For PEP8 formatting, we  will be using the Python library [black](https://github.com/psf/black) and 
[flake8](https://flake8.pycqa.org/en/latest/) for code linting. If you are unsure, discuss the feature by raising an 
[issue](https://github.com/slickml/slick-ml) before opening a PR.

## Getting started

Best start by reading the [documentation](https://www.freqtrade.io/) to get a feel for what is possible with the bot, or 
head straight to the [Developer-documentation](https://www.freqtrade.io/en/latest/developer/) (WIP) which should help you 
getting started.

## Before sending the PR:

### 1. Run unit tests

All unit tests must pass. Integration tests will be automatically run when the PR is created, and will only be reviewed if
the build successfully  passes. 

#### Test the whole project

```bash
pytest
```

#### Test only one file

```bash
pytest tests/test_<file_name>.py
```

### 2. Test if your code is PEP8 compliant

#### Run Flake8

```bash
flake8 slickml 
```

#### Run Black

```bash
python -m black slickml --line-length=80 
```

### Process: Pull Requests

How to prioritize pull requests, from most to least important:

1. Fixes for broken tests. Broken means broken on any supported platform or Python version.
2. Extra tests to cover corner cases.
3. Minor edits to docs.
4. Bug fixes.
5. Major edits to docs.
6. Features.

Please ensure that each pull request meets all requirements in the Contributing document.

### Responsibilities

- Ensure cross-platform compatibility for every change that's accepted. Windows, Mac & Linux.
- Ensure no malicious code is introduced into the core code.
- Create issues for any major changes and enhancements that you wish to make. 
- Discuss things transparently and get community feedback.
- Keep feature versions as small as possible, preferably one new feature per version.
- Be welcoming to newcomers and encourage diverse new contributors from all backgrounds. See the Python Community Code of Conduct (https://www.python.org/psf/codeofconduct/).
