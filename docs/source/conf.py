# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "brieflow"
copyright = "2025, Matteo Di Bernardo, Roshan Kern"
author = "Matteo Di Bernardo, Roshan Kern"
release = "4/8/2025"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# Use Markdown as source format
source_suffix = {
    ".md": "markdown",
}

root_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

# Build docs to test locally
# cd brieflow/
# make -C docs html


