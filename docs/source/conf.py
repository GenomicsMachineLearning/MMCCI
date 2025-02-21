# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

project = 'MMCCI'
copyright = '2025, Genomics and Machine Learning Lab'
author = 'Levi Hockey'
release = '1.0.1'

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
    '*/build', 
    'Thumbs.db', 
    '.DS_Store', 
    '*/__init__.py', 
    '**.ipynb_checkpoints'
    ]

nb_execution_mode = "off"
autodoc_member_order = "groupwise"
autodoc_typehints = "signature"
autodoc_docstring_signature = True
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")

autoclass_content = 'both'
