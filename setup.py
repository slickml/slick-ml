import io
import os
import re
import setuptools


# ==============================================================================
# Utilities
# ==============================================================================
def read(path, encoding="utf-8"):
    """Read the README.md file"""
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


def get_install_requirements(path):
    """Install all the packages in requirements.txt"""
    content = read(path)
    return [req for req in content.split("\n") if req != "" and not req.startswith("#")]


def get_version(path):
    """Obtain the package version from a python file slickml/__init__.py"""
    version_file = read(path)
    version_match = re.search(
        r"""^__version__ = ['"]([^'"]*)['"]""", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# ==============================================================================
# Variables
# ==============================================================================
DESCRIPTION = "SlickML: Slick Machine Learning in Python"
LONG_DESCRIPTION = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
NAME = "slickml"
AUTHOR = "Amirhessam Tahmassebi, Trace Smith"
AUTHOR_EMAIL = "admin@slickml.com"
URL = "http://www.slickml.com"
DOWNLOAD_URL = "https://github.com/slickml/slick-ml/"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.6"
PACKAGES = setuptools.find_packages()
VERSION = get_version("slickml/__init__.py")
INSTALL_REQUIRES = get_install_requirements("requirements.txt"),
# ==============================================================================
# Setup
# ==============================================================================

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
)
