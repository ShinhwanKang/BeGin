import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'BeGin'
copyright = 'KAIST Data Mining Lab'
author = 'BeGin Team'
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_logo = "logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
}

release = '0.1'
version = '0.1.0'

# -- General configuration

# extensions = [
#     'sphinx.ext.duration',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.doctest',
#     'sphinx.ext.intersphinx',
#     'sphinx.ext.todo',
#     'sphinx.ext.coverage',
#     'sphinx.ext.mathjax',
#     'sphinx.ext.ifconfig',
#     'sphinx.ext.napoleon',
#     "sphinx_rtd_theme",
#     'sphinx.ext.imgmath',
#     'sphinx.ext.ifconfig',
#     'sphinx.ext.viewcode',
#     'sphinx.ext.githubpages',
#     'sphinx.ext.todo',
#     'recommonmark'
# ]

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    "sphinx_rtd_theme",
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'recommonmark'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'torch': ('https://pytorch.org/docs/master', None)
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
exclude_patterns = ['.ipynb_checkpoints/*.rst', '0*0*/.ipynb_checkpoints/*.rst']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'


autodoc_mock_imports = ["torch", "dgl", "numpy", "os", "time", "copy", "itertools", "pickle", "torch_scatter", "sklearn", "ogb", "scipy", "networkx", "tqdm", "qpth", "quadprog", "cvxpy", "rdkit", "dgllife", "pandas"]

# -- Options for EPUB output
epub_show_urls = 'footnote'

add_module_names = False
todo_include_todos = True


# import torch
# from torch import nn
# from torch_scatter import scatter
# import numpy as np
# from sklearn.metrics import roc_auc_score, average_precision_score
# from ogb.linkproppred import Evaluator as EEvaluator
# from ogb.nodeproppred import Evaluator as NEvaluator
