import os
import sys
# sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../../trainer/'))
sys.path.append(os.path.abspath('../../scenarios/'))
# sys.path.insert(0, os.path.abspath('./'))
# sys.path.append(os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'BeGin'
copyright = '2022, Data Mining Lab'
author = 'Jihoon Ko*, Shinhwan Kang*, and Kijung Shin'

release = '0.1'
version = '0.1.0'

# -- General configuration

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
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'recommonmark'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
