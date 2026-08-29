# Minimal makefile for Sphinx documentation.
#
# Convenience only -- CI and Read the Docs both invoke sphinx directly. The
# strict gate this project actually holds itself to is:
#
#     python -m sphinx -b html -W --keep-going docs docs/_build/html
#
# which is what `make strict` runs (from the repository root: `make -C docs
# strict`).

SPHINXOPTS    ?=
SPHINXBUILD   ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help strict clean Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# The gate: warnings are errors, and --keep-going reports all of them rather
# than stopping at the first.
strict:
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all: route anything else (html, linkcheck, ...) to sphinx's -M builder.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
