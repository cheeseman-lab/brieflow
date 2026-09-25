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

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 4

templates_path = ["_templates"]
exclude_patterns = []

# Use Markdown as source format
source_suffix = {
    ".md": "markdown",
}

root_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "brieflow"
html_logo = "../../images/brieflow_logo.png"
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500&display=swap",
    "custom.css",
]
pygments_style = "friendly"
pygments_dark_style = "monokai"

_fonts = {
    "font-stack": "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
    "font-stack--monospace": "'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace",
}
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/cheeseman-lab/brieflow/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        **_fonts,
        "color-brand-primary": "#8a4b08",
        "color-brand-content": "#9a5a10",
        "color-background-primary": "#fbfaf7",
        "color-background-secondary": "#f3f0e9",
        "color-background-border": "#e4ded2",
        "color-code-background": "#f4f1ea",
        "color-inline-code-background": "#efe9dd",
        "color-admonition-background": "#f6f3ec",
    },
    "dark_css_variables": {
        **_fonts,
        "color-brand-primary": "#f2b866",
        "color-brand-content": "#f5c67f",
        "color-background-primary": "#17181c",
        "color-background-secondary": "#1f2025",
        "color-background-border": "#2e3037",
        "color-code-background": "#1d1e23",
        "color-inline-code-background": "#26272d",
        "color-admonition-background": "#1f2025",
    },
}

# Build docs to test locally (from the brieflow/ directory):
# pip install -r docs/requirements.txt
# sphinx-build -b html docs/source docs/_build/html
