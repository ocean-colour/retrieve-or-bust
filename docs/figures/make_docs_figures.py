"""Put the figures the documentation renders where Sphinx can find them.

**D1 form: copy mode only.** Every figure the site shows already exists as a
committed PNG under ``reports/``, produced by the two report figure scripts
(``reports/make_report_figures.py`` and
``reports/make_inelastic_report_figures.py``) from the validation artefacts in
``design/validation/``. This script copies the named ones into
``docs/_static/`` so the site has a *supply chain* rather than a second
committed copy of the same bytes: the copies are gitignored
(``docs/_static/fig_*.png``), and Read the Docs and CI produce them
themselves because ``docs/conf.py`` calls :func:`copy_figures` at import time.

Two consequences worth stating, because they are the reason for the shape:

* **No matplotlib at module level.** Copying must work in the documentation
  build environment, which installs the Sphinx toolchain plus the real JAX
  stack and nothing else (``docs/requirements.txt``). This module imports
  ``pathlib`` and ``shutil``, and that is the whole of it.
* **Idempotent.** Running it twice is a no-op the second time: a destination
  that already matches the source in size and modification time is left alone
  (``shutil.copy2`` preserves both, so the match is exact). A build that runs
  it on every ``conf.py`` import therefore does not churn ``docs/_static/``.

**D2 extends this file** (D2 task 6): the tuple below grows to all seven
``reports/fig_*.png``, the two reports are copied into ``docs/reports/`` with
their image and repo-relative links rewritten, and an opt-in ``--regenerate``
mode re-runs the two report scripts (which needs ``$OS_COLOR`` and the L23
netCDFs, and must never be reachable from a documentation build).

Run it by hand with::

    python docs/figures/make_docs_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "DOCS_FIGURES",
    "REPORTS_DIR",
    "STATIC_DIR",
    "copy_figures",
]

# Resolved from this file rather than from the working directory: conf.py is
# executed with the cwd set to ``docs/``, a developer runs the script from the
# repository root, and CI does whichever it likes.
_HERE = Path(__file__).resolve().parent
_DOCS_DIR = _HERE.parent
_REPO_ROOT = _DOCS_DIR.parent

#: Where the committed figures live (written by the two report scripts).
REPORTS_DIR = _REPO_ROOT / "reports"

#: Where Sphinx looks for them (``html_static_path = ["_static"]``).
STATIC_DIR = _DOCS_DIR / "_static"

#: The figures the site currently renders, by filename under ``reports/``.
#: One today -- the front-page hero (D1 task 6, DocQ7: reuse the inelastic
#: architecture diagram as-is rather than drawing a new one). D2 task 6 grows
#: this to all seven, when the Reports section renders both reports in full.
DOCS_FIGURES = ("fig_inelastic_architecture.png",)


def _is_up_to_date(src: Path, dst: Path) -> bool:
    """Is ``dst`` already a copy of ``src``?

    Compares size and modification time, which is exact rather than heuristic
    here because :func:`shutil.copy2` copies both across. Content hashing would
    be more thorough and would buy nothing: nothing but this function writes
    these files, and a stale copy with a matching mtime *and* size cannot arise
    from that one writer.

    Parameters
    ----------
    src, dst : pathlib.Path
        Source figure and its destination in ``docs/_static/``.

    Returns
    -------
    bool
        True when the copy can be skipped.
    """
    if not dst.exists():
        return False
    s, d = src.stat(), dst.stat()
    return s.st_size == d.st_size and int(s.st_mtime) == int(d.st_mtime)


def copy_figures(
    names: tuple[str, ...] = DOCS_FIGURES,
    *,
    reports_dir: Path | None = None,
    static_dir: Path | None = None,
) -> list[tuple[str, str]]:
    """Copy the named report figures into ``docs/_static/``.

    Parameters
    ----------
    names : tuple of str, optional
        Filenames under ``reports/``. Defaults to :data:`DOCS_FIGURES`.
    reports_dir : pathlib.Path, optional
        Source directory. Defaults to :data:`REPORTS_DIR`.
    static_dir : pathlib.Path, optional
        Destination directory, created if absent. Defaults to
        :data:`STATIC_DIR`.

    Returns
    -------
    list of (str, str)
        One ``(filename, action)`` pair per figure, where action is
        ``"copied"`` or ``"up-to-date"``.

    Raises
    ------
    FileNotFoundError
        If a named figure is missing from ``reports/``. Deliberately loud: a
        silent skip would surface much later as a broken image in the rendered
        HTML, and the documentation build is run with ``-W``, so failing here
        with the path in the message is the cheaper failure.
    """
    reports_dir = REPORTS_DIR if reports_dir is None else Path(reports_dir)
    static_dir = STATIC_DIR if static_dir is None else Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)

    actions: list[tuple[str, str]] = []
    for name in names:
        src = reports_dir / name
        if not src.is_file():
            raise FileNotFoundError(
                f"make_docs_figures: {src} is missing. The documentation "
                f"renders it, and copy mode only copies -- regenerate it with "
                f"reports/make_inelastic_report_figures.py (needs $OS_COLOR "
                f"and the L23 data), or check out the committed copy."
            )
        dst = static_dir / name
        if _is_up_to_date(src, dst):
            actions.append((name, "up-to-date"))
            continue
        shutil.copy2(src, dst)
        actions.append((name, "copied"))
    return actions


if __name__ == "__main__":
    for _name, _action in copy_figures():
        print(f"{_action:>11}: {STATIC_DIR / _name}")
