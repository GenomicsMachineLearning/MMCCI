# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../'))

project = 'mmcci'
copyright = '2025, Genomics and Machine Learning Lab'
author = 'Levi Hockey'
release = '1.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    # 'nbsphinx',
    'myst_nb'
]

autodoc_typehints = 'description'
templates_path = ['_templates']
exclude_patterns = [
    '*/_build', 
    'Thumbs.db', 
    '.DS_Store', 
    # '**__init__.py', 
    '**.ipynb_checkpoints',
    '**manuscript_code**'
    ]

nb_execution_mode = "off"
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
# html_show_sphinx = False

autoclass_content = 'both'

def setup(app: Sphinx) -> None:
    app.add_css_file("custom.css")