"""retrieve-or-bust: our last best effort at IOP retrievals, with AI."""

#: The single source of truth for the package version.
#:
#: ``setup.py`` parses this literal out of this file with a regex (it must not
#: ``import robust`` at build time -- that would drag the JAX stack into
#: packaging), and ``docs/conf.py`` reads it as ``robust.__version__`` for the
#: Sphinx ``release``. Change it here and nowhere else.
__version__ = "0.0.dev0"
