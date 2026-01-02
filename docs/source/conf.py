# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../src/'))


project = 'BioSonic'
copyright = '2025, Lena Gies, Tecumseh Fitch'
author = 'Lena Gies, Tecumseh Fitch, Yannick Jadoul'
release = 'v0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',         # Google/NumPy style docstrings
    'sphinx.ext.viewcode',         # source code
    'sphinx_autodoc_typehints',    # type hints
]

templates_path: list[str] = ['_templates']
exclude_patterns: list[str] = []

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}
extensions.append('sphinx.ext.intersphinx')
autodoc_typehints = 'description'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path: list[str] = ['_static']
