"""Sphinx configuration for the retrieve-or-bust documentation site.

Published at https://retrieve-or-bust.readthedocs.io/.

Shape follows PAB (flat ``docs/``, ``docs/conf.py``, MyST enabled via
``myst_nb``); the API pattern and the version single-sourcing follow IOPtics.
The theme is neither project's -- see ``html_theme`` below.

Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import re
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

# -- Outbound GitHub links: one constant, one place ---------------------------
#
# Several pages (the development record today; the generated report copies at
# D2 task 6) link out to files that live in the repository but are never
# rendered by the site -- the design docs, the coding plans, the implementation
# records, and the ten milestone notebooks. Those links have to name a git ref.
#
# `main` is the ref the site describes and the ref these URLs will be correct
# against, so it is the default. It is *not* correct today: the docs work lives
# on an unmerged branch, and `main` carries no ``reports/``, ``design/`` or
# ``notebooks/`` directory at all, so every link below 404s until the merge.
# That is a known, accepted state (prompt-doc Q&A Q6: "I will not merge
# cdom-rt into main until we are all done"), not an oversight -- and the reason
# the ref is a single constant rather than fifteen hardcoded strings is so the
# whole site moves in one edit, or one environment variable, when it changes:
#
#     export ROBUST_GITHUB_URL_BASE=\
#         "https://github.com/ocean-colour/retrieve-or-bust/blob/cdom-rt/"
#     python -m sphinx -b html docs docs/_build/html
#
# Hardcoding a branch name here would be worse than the 404: the working
# agreements forbid treating a branch name as fact, and those URLs would rot at
# the merge instead of before it.
github_repo_url = "https://github.com/ocean-colour/retrieve-or-bust"
github_url_base = os.environ.get(
    "ROBUST_GITHUB_URL_BASE", f"{github_repo_url}/blob/main/"
)

# Pages use it through a MyST URL scheme, so a link is written as its
# repo-relative path and nothing else:  [text](gh:notebooks/RT/foo.ipynb).
# Listing the standard schemes alongside is required -- assigning
# `myst_url_schemes` replaces the default set rather than extending it.
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "gh": {
        "url": github_url_base + "{{path}}",
        "title": "{{path}} on GitHub",
        "classes": ["github"],
    },
}

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


# -- Docstring markup repair (temporary; see prompt-doc Q&A Q4) ----------------
#
# Three docstrings in `robust/` carry reStructuredText that docutils cannot
# parse, and autodoc renders docstrings verbatim, so each is a build failure
# under -W. The durable fix is a few characters in the source; D1 task 5 has no
# sanctioned `robust/` edit (and one of the two files is under concurrent
# development by the CDOM effort), so the defects are repaired here, at render
# time, and raised as Q&A Q4 for a source fix at D2 task 5.
#
# These are *general* rules, not text-keyed patches: they keep working if the
# surrounding prose is reworded, and become no-ops once the sources are fixed.
_DOCSTRING_REPAIRS = [
    # `**48**(35)` -- docutils requires whitespace or closing punctuation after
    # a strong end-string; "(" is opening punctuation, so the `**` never
    # closes. Two journal citations in ztt.py are written this way.
    (re.compile(r"(\*\*[^*\s][^*]*\*\*)\("), r"\1 ("),
    # `|delta|` used as mathematical absolute-value bars reads to docutils as a
    # substitution reference. Escape the bars so they render as literal pipes.
    (re.compile(r"(?<![\\|])\|(\u03b4[_A-Za-z]*)\|(?![|])"), r"\\|\1\\|"),
]


def _repair_docstring(app, what, name, obj, options, lines):
    """Apply :data:`_DOCSTRING_REPAIRS` to one autodoc docstring, in place."""
    for i, line in enumerate(lines):
        new = line
        for pattern, replacement in _DOCSTRING_REPAIRS:
            new = pattern.sub(replacement, new)
        if new != line:
            lines[i] = new


def setup(app):
    app.connect("autodoc-process-docstring", _repair_docstring)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


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

# -- Generated figures -------------------------------------------------------
#
# The site's figures are committed once, under ``reports/``. This copies the
# ones the site renders into ``docs/_static/`` (which is gitignored for
# ``fig_*.png``), so there is never a second committed copy to drift.
#
# Called here, at conf.py import time, rather than from a Sphinx event: the
# copies must exist before the *read* phase, because a document that references
# a missing image is a warning and CI builds with -W. Doing it here also means
# Read the Docs and CI produce the copies themselves, from a bare checkout, with
# no extra build step to remember. It is pure pathlib/shutil (no matplotlib) and
# idempotent, so paying for it on every build is free.
#
# Sphinx evaluates conf.py with the working directory set to the config
# directory, which is what makes the relative path below correct -- the same
# assumption the sys.path line at the top of this file already makes.
sys.path.insert(0, os.path.abspath("figures"))

from make_docs_figures import copy_figures  # noqa: E402

copy_figures()

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
