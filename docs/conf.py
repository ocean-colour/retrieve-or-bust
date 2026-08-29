"""Sphinx configuration for the retrieve-or-bust documentation site.

Published at https://retrieve-or-bust.readthedocs.io/.

Shape follows PAB (flat ``docs/``, ``docs/conf.py``, MyST enabled via
``myst_nb``); the API pattern and the version single-sourcing follow IOPtics.
The theme is neither project's -- see ``html_theme`` below.

Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

# Make the checkout importable for autodoc. conf.py lives at docs/, so the
# repository root is one level up.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "retrieve-or-bust"
author = "J. Xavier Prochaska and collaborators"
copyright = "2026, " + author

# Single-source the version from the package (IOPtics' pattern). The fallback
# keeps conf.py importable in an environment where `robust` is not installed --
# e.g. a linter, or a docs build attempted before `pip install -e .`.
try:
    import robust

    release = robust.__version__
except Exception:
    release = "0.0.dev0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google- and NumPy-style docstrings
    "sphinx.ext.viewcode",  # "[source]" links next to each API entry
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",  # the model pages are equation-heavy
    # myst_nb bundles myst-parser (Markdown) and adds notebook (.ipynb)
    # support; load it INSTEAD of myst_parser -- loading both conflicts
    # (PAB's conf.py carries the same warning, from experience).
    "myst_nb",
    "sphinx_design",  # card grids (front page)
    "sphinx_copybutton",  # copy button on code blocks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"

# -- Source formats ----------------------------------------------------------

# Treat .rst, .md and .ipynb as sources (.md/.ipynb via myst-nb).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# Do NOT execute notebooks at build time -- the quickstart notebook (D2) is
# committed with its outputs, and RTD has neither $OS_COLOR nor the L23 data.
nb_execution_mode = "off"

# Kept short on purpose: every extension is another way for the -W build to
# fail. `linkify` in particular is NOT enabled -- it needs the separate
# linkify-it-py package, and myst-nb raises at build time without it.
myst_enable_extensions = [
    "dollarmath",  # $...$ / $$...$$ -- used throughout the model pages
    "amsmath",  # align/gather environments for the composition law
    "colon_fence",  # ::: directives, easier to read than ``` in prose
    "deflist",  # definition lists for the conventions tables
]
myst_heading_anchors = 3  # let pages link to each other's subsections

# The two reports and the design docs are authored as standalone GitHub
# documents carrying repo-relative links (``design/...``, ``robust/...``).
# Those are not Sphinx cross-reference targets, and CI builds with -W, so a
# missing-xref warning would be a build failure. Suppress just that class.
suppress_warnings = ["myst.xref_missing"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",  # the source order is the pipeline order
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"  # built-in; no sphinx-autodoc-typehints dep

# Deliberately EMPTY (DocQ2): we install the real stack on Read the Docs rather
# than mocking it, so the rendered signatures and the `jaxtyping` shape
# annotations are the real ones. This is cheap because `import robust.rt` needs
# exactly `jax`, `jaxtyping` and `numpy` at module level -- `flax`/`optax`
# (emulator training) and `ocpy` (the L23 loader) are imported inside
# functions, so autodoc never drags in ocpy's geospatial tree.
autodoc_mock_imports = []

# -- Napoleon ----------------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# True: render Attributes sections as :ivar: fields. With False, napoleon
# emits py:attribute directives that collide with the attribute docstrings
# autodoc already picks up -- 52 duplicate-object warnings in BING's build.
# The IOPs/PhaseParams/Geometry/Inelastic pytrees are exactly that shape.
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------

# pydata-sphinx-theme: neither BING/PAB's sphinx_rtd_theme nor IOPtics' furo,
# so the site reads as its own thing (DocQ2). It is also the scientific-Python
# house theme (NumPy, SciPy, pandas, xarray, Matplotlib), and its three-column
# layout keeps a right-hand in-page table of contents on the long, subsection
# -heavy model chapters.
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "retrieve-or-bust"

# Keep these minimal: every option is another chance to fail the -W build.
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ocean-colour/retrieve-or-bust",
            "icon": "fa-brands fa-github",
        },
    ],
    # The five parts of the site as top-navbar tabs rather than one very long
    # sidebar. Entries are documents; their children stay in the left sidebar.
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_prev_next": True,
}
