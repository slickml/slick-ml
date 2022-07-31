# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from importlib.metadata import version

# -- Project information -----------------------------------------------------
# TODO(amir): get all the info via importlib.metadata
project = "SlickML"
copyright = "2022, SlickML"
author = "Amirhessam Tahmassebi"
version = version("slickml")
release = version
language = "en"

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "autoapi.extension",
]

# Auto-API directories
autoapi_dirs = [
    "../src/slickml",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".ipynb_checkpoints",
]

# numpydoc configuration
numpydoc_show_class_members = False  # Otherwise Sphinx emits thousands of warnings
numpydoc_class_members_toctree = False

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "../assets/designs/logo_clear.png"
html_favicon = "../assets/designs/logo_clear.png"
html_show_copyright = True
html_show_search_summary = True
html_show_sphinx = True
html_output_encoding = "utf-8"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for Auto-API-Docs -------------------------------------------------
# Reference: https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html
autoapi_type = "python"
autoapi_template_dir = ""
autoapi_file_patterns = ["*.py", "*.pyi"]
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_ignore = ["*migrations*"]
autoapi_add_toctree_entry = True
autoapi_python_class_content = "class"
autoapi_member_order = "alphabetical"
autoapi_python_use_implicit_namespaces = False
autoapi_prepare_jinja_env = None
autoapi_keep_files = False
suppress_warnings = []
