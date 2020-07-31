"""Sphinx configuration."""
import datetime

import pkg_resources

# Sphinx configuration below.
project = "amazon-braket-schemas"
version = pkg_resources.require(project)[0].version
release = version
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

source_suffix = ".rst"
master_doc = "index"

autoclass_content = "both"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = "{}doc".format(project)

napoleon_use_rtype = False
