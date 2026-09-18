"""Put the figures and the reports the documentation renders where Sphinx can find them.

Two modes, and the difference between them matters:

**Copy mode (the default, and the only mode a documentation build ever runs).**
Pure ``pathlib``/``shutil``/``re``. It does two things:

1. Copies the seven committed ``reports/fig_*.png`` into ``docs/_static/``.
2. Writes ``docs/reports/report_rt_{elastic,inelastic}_model.md`` as *generated
   copies* of the two reports under ``reports/``, with two mechanical rewrites
   and a generated-file banner.

Both outputs are gitignored (``docs/_static/fig_*.png`` and
``docs/reports/report_rt_*.md``), so the repository never carries a second copy
of bytes it already has, and there is nothing to drift. Read the Docs and CI
produce them themselves, because ``docs/conf.py`` calls :func:`copy_assets` at
import time.

**Regenerate mode (``--regenerate``, opt-in, developer machines only).** Re-runs
the two report figure scripts as subprocesses -- ``reports/make_report_figures.py``
and ``reports/make_inelastic_report_figures.py``, reused rather than forked --
and only then copies. It needs matplotlib and the committed validation CSVs, it
fails loudly and early if either is missing, and it is unreachable from a
documentation build: ``conf.py`` calls :func:`copy_assets`, which cannot invoke
it, and the ``--regenerate`` flag exists only on this module's command line.

Two properties the shape depends on:

* **No matplotlib at module level.** Copy mode must work in the documentation
  build environment, which installs the Sphinx toolchain plus the real JAX
  stack and nothing else (``docs/requirements.txt`` -- no matplotlib). The
  imports here are ``argparse``, ``os``, ``re``, ``shutil``, ``subprocess``,
  ``sys``, ``pathlib``; matplotlib is reached only inside the subprocesses
  ``--regenerate`` spawns.
* **Idempotent.** Running copy mode twice is a no-op the second time. A figure
  whose destination matches the source in size and modification time is left
  alone (``shutil.copy2`` preserves both, so the match is exact); a report copy
  whose bytes already equal what would be written is left alone. A build that
  runs it on every ``conf.py`` import therefore does not churn the tree.

Run it by hand with::

    python docs/figures/make_docs_figures.py                # copy mode
    python docs/figures/make_docs_figures.py --regenerate   # redraw, then copy

A note on ``--regenerate``'s prerequisites, because the plan for this script
said it needs ``$OS_COLOR`` and the L23 netCDFs and, checked against the two
scripts, **it does not**. Both draw their numbers from the committed artefacts
in ``design/validation/`` -- that is their own stated rule ("draw from the
committed validation artefacts, never recompute the science") -- so what they
actually need is matplotlib, numpy and three CSVs. The preflight below checks
those, the real ones. What is true of ``$OS_COLOR`` is that regenerating the
*artefacts* (``design/py/run_validation.py``) needs it; redrawing the figures
from them does not.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

__all__ = [
    "DOCS_FIGURES",
    "DOCS_REPORTS",
    "FIGURE_SCRIPTS",
    "GITHUB_REPO_URL",
    "GITHUB_URL_BASE",
    "REGENERATE_INPUTS",
    "REPORTS_DIR",
    "REPORTS_OUT_DIR",
    "STATIC_DIR",
    "copy_assets",
    "copy_figures",
    "copy_reports",
    "regenerate_figures",
    "rewrite_report",
]

# Resolved from this file rather than from the working directory: conf.py is
# executed with the cwd set to ``docs/``, a developer runs the script from the
# repository root, and CI does whichever it likes.
_HERE = Path(__file__).resolve().parent
_DOCS_DIR = _HERE.parent
_REPO_ROOT = _DOCS_DIR.parent

#: Where the committed figures and reports live.
REPORTS_DIR = _REPO_ROOT / "reports"

#: Where Sphinx looks for the figures (``html_static_path = ["_static"]``).
STATIC_DIR = _DOCS_DIR / "_static"

#: Where the generated report pages go (``docs/reports/index.md``'s toctree).
REPORTS_OUT_DIR = _DOCS_DIR / "reports"

#: Every figure the site renders, by filename under ``reports/``. All seven of
#: them: three from the elastic report, four from the inelastic one. The
#: inelastic architecture diagram is also the front page's hero (DocQ7).
DOCS_FIGURES = (
    "fig_architecture.png",
    "fig_inelastic_architecture.png",
    "fig_inelastic_deltas.png",
    "fig_inelastic_rrms_ladder.png",
    "fig_inelastic_unseen_zenith.png",
    "fig_rrms_ladder.png",
    "fig_unseen_zenith.png",
)

#: The two reports rendered in-site, by filename under ``reports/``.
DOCS_REPORTS = (
    "report_rt_elastic_model.md",
    "report_rt_inelastic_model.md",
)

#: The plotting scripts ``--regenerate`` re-runs, in dependency-free order.
FIGURE_SCRIPTS = (
    "make_report_figures.py",
    "make_inelastic_report_figures.py",
)

#: What ``--regenerate`` actually reads, checked before anything is spawned.
#: Repo-relative; see the module docstring on why ``$OS_COLOR`` is not here.
REGENERATE_INPUTS = (
    "design/validation/rrms_per_wavelength.csv",
    "design/validation/rrms_per_wavelength_inelastic.csv",
    "design/validation/metrics_inelastic.csv",
)

# -- Outbound GitHub links: the single source of truth -------------------------
#
# The two reports are standalone GitHub documents: they link to design docs,
# validation artefacts and their own plotting scripts by repo-relative path.
# Rendered in-site those paths mean nothing, so they are rewritten to absolute
# GitHub URLs on this base.
#
# It lives *here* rather than in ``conf.py`` for one reason: ``conf.py`` already
# imports this module (to call :func:`copy_assets` at import time), so a
# constant defined there and read here would be a circular import, while the
# reverse works. ``conf.py`` therefore reads ``github_url_base`` from this
# module -- one definition, two consumers (the ``gh:`` MyST URL scheme the
# hand-written pages use, and the rewriter below). The comment block in
# ``conf.py`` explains the choice of ref; the short version is Q&A Q6: `main` is
# the ref the site describes, `main` does not carry ``reports/``, ``design/`` or
# ``notebooks/`` until the branch merges, and the accepted answer is to link
# `main` anyway rather than bake a branch name into the repository. Override for
# a pre-merge build with::
#
#     export ROBUST_GITHUB_URL_BASE=\
#         "https://github.com/ocean-colour/retrieve-or-bust/blob/cdom-rt/"
GITHUB_REPO_URL = "https://github.com/ocean-colour/retrieve-or-bust"
GITHUB_URL_BASE = os.environ.get(
    "ROBUST_GITHUB_URL_BASE", f"{GITHUB_REPO_URL}/blob/main/"
)

#: Markdown inline links and images: ``[text](target)`` / ``![alt](target)``.
#: Deliberately narrow -- no parentheses inside the target, which is true of
#: every link in both reports (checked) and keeps the pattern readable.
_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")

#: Anything with one of these prefixes is already absolute and left alone.
_ABSOLUTE_PREFIXES = ("http://", "https://", "mailto:", "ftp:", "#", "/")

_BANNER = """\
<!-- GENERATED FILE -- do not edit.
     Written by docs/figures/make_docs_figures.py from {source}.
     Edit that file instead; this copy is gitignored and is rewritten on every
     documentation build. The only differences from the source are this banner,
     image paths pointing into docs/_static/, and repo-relative links rewritten
     to absolute GitHub URLs. -->
"""

_NOTE = """\
:::{{note}}
This page is a verbatim copy of [`{source}`]({url}), the team report as it was
written. Only the image paths and the repo-relative links differ; no wording,
number or section has been changed. It is reproduced here, rather than
summarised, because it is the evidence behind the rest of this site.
:::
"""


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
                f"`python docs/figures/make_docs_figures.py --regenerate`, or "
                f"check out the committed copy."
            )
        dst = static_dir / name
        if _is_up_to_date(src, dst):
            actions.append((name, "up-to-date"))
            continue
        shutil.copy2(src, dst)
        actions.append((name, "copied"))
    return actions


def _github_url(repo_path: str, *, is_dir: bool) -> str:
    """Absolute GitHub URL for a repository-relative path.

    Parameters
    ----------
    repo_path : str
        Path relative to the repository root, POSIX-style, no leading slash.
    is_dir : bool
        True when the target is a directory. GitHub serves directories under
        ``/tree/`` rather than ``/blob/``; it does redirect ``blob`` to ``tree``
        for a directory, but emitting the right one costs one substitution.

    Returns
    -------
    str
        The absolute URL.
    """
    base = GITHUB_URL_BASE
    if is_dir:
        base = base.replace("/blob/", "/tree/", 1)
    return base + repo_path


def rewrite_report(text: str, source_name: str) -> tuple[str, list[tuple[str, str]]]:
    """Rewrite one report's links for in-site rendering.

    Three cases, applied to every Markdown link and image target in ``text``.
    Targets are written relative to ``reports/``, which is where the source
    document lives.

    1. ``fig_*.png`` -- an image beside the report. Becomes
       ``../_static/fig_*.png``, the copy :func:`copy_figures` puts where Sphinx
       can reach it (the generated page sits in ``docs/reports/``).
    2. The other report -- ``report_rt_elastic_model.md`` and its inelastic
       twin. Left exactly as written: both are rendered pages in the same
       directory, so the relative Markdown link already resolves *in-site*,
       which is strictly better for a reader than sending them to GitHub.
    3. Everything else relative -- ``../design/…``, ``../context/…``,
       ``make_report_figures.py`` -- is resolved against ``reports/`` into a
       repository-relative path and turned into an absolute GitHub URL on
       :data:`GITHUB_URL_BASE`.

    Absolute targets (``http://``, ``https://``, ``mailto:``, ``ftp:``, a bare
    ``#anchor``, a leading ``/``) are left alone.

    Parameters
    ----------
    text : str
        The source report, verbatim.
    source_name : str
        Its filename under ``reports/``, used in the banner and the note.

    Returns
    -------
    text : str
        The rewritten document, banner and note prepended.
    targets : list of (str, str)
        One ``(original_target, repo_path)`` pair per link rewritten to GitHub,
        so the caller can check the paths exist in the checkout.
    """
    targets: list[tuple[str, str]] = []

    def _sub(match: re.Match[str]) -> str:
        bang, label, target = match.group(1), match.group(2), match.group(3)
        if target.startswith(_ABSOLUTE_PREFIXES):
            return match.group(0)
        if bang and target.startswith("fig_") and "/" not in target:
            return f"{bang}[{label}](../_static/{target})"
        if target in DOCS_REPORTS:
            return match.group(0)
        is_dir = target.endswith("/")
        repo_path = str(PurePosixPath(os.path.normpath(f"reports/{target}")))
        targets.append((target, repo_path))
        return f"{bang}[{label}]({_github_url(repo_path, is_dir=is_dir)})"

    body = _LINK_RE.sub(_sub, text)

    source_rel = f"reports/{source_name}"
    note = _NOTE.format(source=source_rel, url=_github_url(source_rel, is_dir=False))
    # The note goes *after* the document's H1 so the page keeps its own title;
    # the banner is an HTML comment and renders as nothing.
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            head, tail = lines[: i + 1], lines[i + 1 :]
            body = "\n".join(head + ["", note.rstrip()] + tail)
            break
    else:  # pragma: no cover - both reports open with an H1 (checked)
        body = note + "\n" + body

    return _BANNER.format(source=source_rel) + "\n" + body, targets


def copy_reports(
    names: tuple[str, ...] = DOCS_REPORTS,
    *,
    reports_dir: Path | None = None,
    out_dir: Path | None = None,
    repo_root: Path | None = None,
) -> list[tuple[str, str]]:
    """Write the generated report pages into ``docs/reports/``.

    Parameters
    ----------
    names : tuple of str, optional
        Filenames under ``reports/``. Defaults to :data:`DOCS_REPORTS`.
    reports_dir : pathlib.Path, optional
        Source directory. Defaults to :data:`REPORTS_DIR`.
    out_dir : pathlib.Path, optional
        Destination directory, created if absent. Defaults to
        :data:`REPORTS_OUT_DIR`.
    repo_root : pathlib.Path, optional
        Root the rewritten links are checked against. Defaults to the checkout
        this file lives in.

    Returns
    -------
    list of (str, str)
        One ``(filename, action)`` pair per report, where action is
        ``"written"`` or ``"up-to-date"``. A copy whose bytes already match
        what would be written is not rewritten, which is what makes running
        this on every ``conf.py`` import free.

    Raises
    ------
    FileNotFoundError
        If a source report is missing, or if a link in one points at a
        repository path that does not exist in this checkout. The second is the
        loud version of shipping a 404: these URLs are unverifiable at build
        time (they name a git ref, not this working tree), so the path is the
        only thing that *can* be checked, and it is checked.
    """
    reports_dir = REPORTS_DIR if reports_dir is None else Path(reports_dir)
    out_dir = REPORTS_OUT_DIR if out_dir is None else Path(out_dir)
    repo_root = _REPO_ROOT if repo_root is None else Path(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    actions: list[tuple[str, str]] = []
    for name in names:
        src = reports_dir / name
        if not src.is_file():
            raise FileNotFoundError(
                f"make_docs_figures: {src} is missing. The documentation "
                f"renders it as docs/reports/{name}."
            )
        text, targets = rewrite_report(src.read_text(encoding="utf-8"), name)
        for original, repo_path in targets:
            if not (repo_root / repo_path).exists():
                raise FileNotFoundError(
                    f"make_docs_figures: {src} links to '{original}', which "
                    f"resolves to '{repo_path}' and does not exist in "
                    f"{repo_root}. That link would render as a GitHub 404; fix "
                    f"the link in the report."
                )
        dst = out_dir / name
        if dst.exists() and dst.read_text(encoding="utf-8") == text:
            actions.append((name, "up-to-date"))
            continue
        dst.write_text(text, encoding="utf-8")
        actions.append((name, "written"))
    return actions


def copy_assets() -> list[tuple[str, str]]:
    """Copy mode: the figures and the report pages, in that order.

    This is the whole of what a documentation build runs -- ``docs/conf.py``
    calls it at import time, so Read the Docs and CI produce every generated
    file themselves from a bare checkout. It cannot reach
    :func:`regenerate_figures`.

    Returns
    -------
    list of (str, str)
        The concatenated ``(filename, action)`` pairs from :func:`copy_figures`
        and :func:`copy_reports`.
    """
    return copy_figures() + copy_reports()


def regenerate_figures(
    *,
    repo_root: Path | None = None,
    scripts: tuple[str, ...] = FIGURE_SCRIPTS,
) -> list[str]:
    """Re-run the two report figure scripts, then return what they wrote to.

    The scripts are *reused, not forked*: each is run as
    ``sys.executable <path>``, exactly as its own docstring documents
    (``python reports/make_report_figures.py``), and each writes its PNGs beside
    itself under ``reports/``. Not imported, because both do their work under
    ``if __name__ == "__main__":`` and both mutate ``matplotlib.rcParams`` at
    module scope.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Checkout to run in. Defaults to the one this file lives in.
    scripts : tuple of str, optional
        Filenames under ``reports/``. Defaults to :data:`FIGURE_SCRIPTS`.

    Returns
    -------
    list of str
        The scripts that ran, in order.

    Raises
    ------
    RuntimeError
        If matplotlib is not importable, if a validation CSV is missing, or if
        a script exits non-zero. All three are checked or reported before
        anything else happens, with the reason named -- this mode exists to be
        run deliberately on a developer machine, and a half-finished redraw
        that leaves three of seven figures stale is worse than not starting.
    """
    repo_root = _REPO_ROOT if repo_root is None else Path(repo_root)

    # Preflight, in the order a failure is cheapest to read.
    import importlib.util

    missing_mods = [
        m for m in ("matplotlib", "numpy") if importlib.util.find_spec(m) is None
    ]
    if missing_mods:
        raise RuntimeError(
            f"--regenerate needs {', '.join(missing_mods)}, which "
            f"{sys.executable} cannot import. This mode is for a developer "
            f"machine with the plotting stack (the `ocean14` environment); the "
            f"documentation build environment deliberately has no matplotlib. "
            f"Run without --regenerate to copy the committed figures."
        )
    missing_data = [p for p in REGENERATE_INPUTS if not (repo_root / p).is_file()]
    if missing_data:
        raise RuntimeError(
            f"--regenerate reads the committed validation artefacts and these "
            f"are missing from {repo_root}: {', '.join(missing_data)}. They are "
            f"written by design/py/run_validation.py (which does need $OS_COLOR "
            f"and the L23 netCDFs); the figure scripts only redraw them."
        )
    missing_scripts = [s for s in scripts if not (repo_root / "reports" / s).is_file()]
    if missing_scripts:
        raise RuntimeError(
            f"--regenerate runs {', '.join(missing_scripts)} under "
            f"{repo_root / 'reports'}, and they are not there."
        )

    ran: list[str] = []
    for script in scripts:
        path = repo_root / "reports" / script
        print(f"  running: {path}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"--regenerate: {path} exited {result.returncode}.\n"
                f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            )
        for line in result.stdout.splitlines():
            print(f"    {line}")
        ran.append(script)
    return ran


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Arguments, defaulting to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="make_docs_figures",
        description=(
            "Copy the report figures into docs/_static/ and write the "
            "generated report pages into docs/reports/."
        ),
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "First re-run reports/make_report_figures.py and "
            "reports/make_inelastic_report_figures.py, redrawing the committed "
            "PNGs from design/validation/. Needs matplotlib; never run by a "
            "documentation build."
        ),
    )
    args = parser.parse_args(argv)

    if args.regenerate:
        print("regenerating the report figures (developer mode):")
        regenerate_figures()

    for name, action in copy_figures():
        print(f"{action:>11}: {STATIC_DIR / name}")
    for name, action in copy_reports():
        print(f"{action:>11}: {REPORTS_OUT_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
