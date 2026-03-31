# Configuration file for the Sphinx documentation builder.

# -- Project information

import os
import sys
# Move up from 'source' to 'docs', then into 'src' where spaAnchor.py lives
sys.path.insert(0, os.path.abspath('../src'))

project = 'spaAnchor'
copyright = '2025, Han Shu et al'
author = 'Han Shu'

release = '1.0'
version = 'latest'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'autoapi.extension',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    # 'numpy': ('http://docs.scipy.org/doc/numpy', None),
    # 'pandas': ('https://pandas.pydata.org/docs', None),
    # 'torch': ('https://pytorch.org/docs/main', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

autoapi_dirs = ['../src']
autodoc_mock_imports = ["torch", "scanpy", "numpy", "pandas"]
autoapi_add_toctree_entry = False
toc_object_entries = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
autodoc_member_order = 'bysource'

html_theme_options = {
    'collapse_navigation': False,  # Important: This keeps the tree expanded
    'navigation_depth': 4,         # Allows for deep nesting
    'titles_only': False           # Shows sub-headers if you want them
}

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

nbsphinx_execute = 'never'
