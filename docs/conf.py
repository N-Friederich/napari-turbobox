# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'napari-turbobox'
copyright = '2025-2026, Nils Friederich and the napari-TurboBox contributors'
author = 'Nils Friederich'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Render docstring "Attributes:" sections as :ivar: fields rather than
# .. attribute:: directives, so they do not collide with the object
# descriptions autodoc already emits for the same attributes.
napoleon_use_ivar = True

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = []
html_logo = 'logo.svg'
html_favicon = 'logo-small.svg'
