#!/usr/bin/env python
"""
M8 task 6 — every elastic model we have, scored on PB24 (the baseline).

This script **measures**; it changes no model code and no model number. It
scores, on the saved splits (``design/ztt_pb24/splits.json``), all 5000 PB24
realisations at every above-water geometry (window and shell) and at the
in-water-only nodes, on all 451 bands:

- standard Gordon (``rrs``);
- ZTT with the TT2017, the L23 and the L23-closure µ∞, each with Sullivan's
  and with PB24's own Fournier–Forand Pbb (all six combinations exist through
  ``rrs_ZTT``'s public ``P_bb=`` input; ``rrs``);
- the L23-trained hybrid with and without its out-of-domain fallback, scored
  on 350–750 nm; 751–800 nm is out of the emulator's domain by construction
  and is reported in its own rows (``rrs``);
- O25 as published (``Rrs``, "calibrated on all 5000"), the O25 form refit on
  each split's training rows (relatively weighted) and, on the realisation
  split, the paper's unweighted objective (``Rrs``);
- the common ``rrs`` table: O25 mapped through the task-4 Fresnel transfer
  (with the downwelling factor, primary, and the θv-only form) beside the
  transfer's own error on PB24's true ``Rrs``.

and runs five diagnostics (µd three ways, Ψ_KLu/µ∞ against KLu/a at nadir,
the eigen µ∞ on every (realisation, λ), the ψ distribution with the
non-physical share, and the files' ``bbw``).

**How the numbers are computed** (also written into ``baseline.md``):
predictions are streamed in chunks of :data:`CHUNK` realisations. For every
value the relative error ``e = pred/true - 1`` is binned into a fixed
histogram of ``log10|e|`` (:data:`HIST_LO`..:data:`HIST_HI`, bin
:data:`HIST_WIDTH` dex; non-finite predictions in the overflow bin), keyed by
split side, angle cut, band and each breakdown. Medians and q90 come from
those histograms (**all values**, quantile resolution ±half a bin); rRMS and
the counts come from exact float64 sums over **all values**, summed per chunk
and then over chunks in chunk order, so a rerun is bit-identical for any
``--workers``. Nothing is thresholded: non-physical predictions (≤ 0 or
non-finite) stay in the median and q90, are counted, and rRMS is reported
with them and, labelled, without.

Every artefact goes to ``design/ztt_pb24/`` with the prefix ``baseline_``;
``baseline.md`` is generated **from the CSVs** after they are written.

    python design/py/run_pb24_baseline.py               # full run (+ eigen if absent)
    python design/py/run_pb24_baseline.py --eigen-only     # build the eigen µ∞ product
    python design/py/run_pb24_baseline.py --skip-eigen     # never build it
    python design/py/run_pb24_baseline.py --check       # rerun, compare byte for byte
    python design/py/run_pb24_baseline.py --quick --out-dir /tmp/x

**Memory, and how to run it on profx** (task 5b, `rt_fit_pb24.md` Q&A/OOM).
Each worker peaks at about :data:`WORKER_PEAK_GB` while scoring a chunk, and
the workers' first loads used to coincide. On 2026-11-14 the default of 12
workers made a ~75 GB burst, systemd-oomd read the resulting memory pressure
as an emergency and killed the desktop session. So now: the default is
:data:`DEFAULT_WORKERS`, more than :data:`MAX_WORKERS` is refused without
``--allow-more-workers``, each worker's start is staggered by
:data:`STAGGER_S` seconds, and the predicted peak is printed at start. Run it
in its own cgroup **from a plain terminal, not from Cursor's**, so that any
kill lands on the job alone and never on the IDE or a Claude session::

    systemd-run --user --scope -p MemoryMax=80G \\
        python design/py/run_pb24_baseline.py --skip-eigen

Do not run it beside another memory-heavy job (a PypeIt DevSuite run).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: Where the committed artefacts land.
OUT_DIR = REPO / "design" / "ztt_pb24"
#: Every artefact's file-name prefix.
PREFIX = "baseline_"

#: Realisations per streamed chunk. Part of the definition of the run (the
#: per-chunk float sums are added in chunk order), so it is a constant. 20 at
#: first; 10 since task 5b. Measured, that halves the per-chunk working set
#: (the start-up allocation burst) but **not** a worker's peak RSS, which is
#: 6.2 GB at either value: ~2 GB of JAX and imports plus ~4 GB of integer
#: histogram accumulators (:func:`new_ints`, every model × pass × group ×
#: :data:`N_BIN` bins) that each worker fills over its whole life. No
#: baseline number had been committed on 20.
CHUNK = 10

#: Worker-pool policy (task 5b/5c). Measured peak RSS of one worker scoring
#: chunks of :data:`CHUNK` realisations, all 451 bands, every model, both
#: passes: 6.2 GB (``/usr/bin/time -v``, ``--workers 0``, 10 and 20
#: realisations alike); the start-up print multiplies it by the worker count.
WORKER_PEAK_GB = 6.5
#: Default and refused-above worker counts. 12 workers caused the 2026-11-14
#: systemd-oomd logout; 4 ran with zero measured memory pressure.
DEFAULT_WORKERS = 6
MAX_WORKERS = 6
#: Seconds between successive workers' starts, so their first chunk loads
#: (the allocation burst) do not coincide. Changes no number.
STAGGER_S = 10.0

#: The --quick sample: 50 realisations drawn with this seed from the 5000,
#: every 10th band (46 bands, 350-800 nm by 10). Named in every artefact.
QUICK_SEED = 6
QUICK_N = 50
QUICK_BAND_STRIDE = 10

#: The relative-error histogram: log10|e| from HIST_LO to HIST_HI in bins of
#: HIST_WIDTH dex, plus an underflow bin (|e| < 10**HIST_LO, including 0) and
#: an overflow bin (|e| >= 10**HIST_HI or a non-finite prediction).
HIST_LO, HIST_HI, HIST_WIDTH = -6.0, 3.0, 0.002
N_FINE = int(round((HIST_HI - HIST_LO) / HIST_WIDTH))
N_BIN = N_FINE + 2

#: Scattering-angle bin edges (degrees). 110.4 is where Ψ_KLu changes sign and
#: 134 where its fitted range ends (record §7.12).
PSI_EDGES = (0.0, 90.0, 100.0, 110.4, 120.0, 134.0, 145.0, 155.0, 165.0, 180.001)
#: Ψ_KLu's fitted range starts here (Q6: a first-class cut).
PSI_FITTED_MIN = 134.0
#: The ZTT sign flip of Ψ_KLu (record §7.12).
PSI_SIGN_FLIP = 110.4

#: Equal-count B_p bins over the 5000 realisations' mean B_p (splits.json).
N_BP_BIN = 5

#: The emulator's ``wave_nm`` feature stops here (Q12).
EMULATOR_WAVE_MAX = 750.0

#: Band groups for the markdown tables (nm, inclusive). The last group is the
#: hybrid's out-of-domain range.
BAND_GROUPS = (
    (350, 400),
    (401, 450),
    (451, 500),
    (501, 550),
    (551, 600),
    (601, 650),
    (651, 700),
    (701, 750),
    (751, 800),
)

#: The eigen µ∞ recipe: FF kernels on a log-spaced B grid (linear in log B),
#: 80 streams, 2° truncation (robust.rt.asymptotic defaults). 97 nodes give a
#: node spacing of 4.5% in B; the interpolation error is measured on
#: EIGEN_CHECK_N realisations (seed EIGEN_CHECK_SEED) against exact kernels.
EIGEN_B_GRID = (0.0008, 0.06, 97)
EIGEN_CHECK_N = 40
EIGEN_CHECK_SEED = 15

#: Within-bin B_p analysis of the eigen µ∞: (log10 bb/a, log10 η_bb) bins of
#: this width; a bin is analysed when it holds at least EIGEN_BIN_MIN points
#: and its B_p spans at least EIGEN_BIN_BP_SPAN (ratio).
EIGEN_BIN_DEX = 0.1
EIGEN_BIN_MIN = 30
EIGEN_BIN_BP_SPAN = 4.0
#: The B_p range Q15's prediction names.
Q15_BP_RANGE = (0.001, 0.036)
#: Eq. (8)'s fit error on L23 (record §7b.7).
L23_EQ8_FIT_ERROR_PCT = 0.21

#: µd diagnostic: bb/a bins (the TT fit range ends at 0.1; L23's maximum 0.59).
MU_D_BBA_EDGES = (0.0, 0.1, 0.5946, np.inf)

#: M5's ZTT non-physical share on PB24's window (record §7.12).
M5_NONPHYSICAL_PCT = 22.0


# =========================================================== labels ==========
@dataclass(frozen=True)
class ModelSpec:
    """One scored model instance. ``label`` is the name used in every artefact.

    Attributes
    ----------
    key : str
        Code id (unique per instance).
    label : str
        The model's name in every CSV, table and figure.
    space : str
        ``"rrs"`` (scored against PB24 ``rrs`` at θ_w) or ``"Rrs"`` (against
        ``Rrs`` at θ_air).
    family : str
        Figure colour family.
    fit : str or None
        The split kind whose training rows the instance was fitted on, or
        ``None`` for a model not fitted on PB24.
    in_water : bool
        Also scored on the in-water-only set (ZTT and Gordon only).
    tables : tuple of str
        ``"native"`` and/or ``"common"`` (the common ``rrs`` table).
    """

    key: str
    label: str
    space: str
    family: str
    fit: str | None = None
    in_water: bool = False
    tables: tuple = ("native", "common")

    @property
    def hybrid(self) -> bool:
        """Whether 751-800 nm is out of this model's domain by construction."""
        return self.family == "hybrid"


GORDON = "standard Gordon"
ZTT_LABELS = {
    ("tt2017", False): "ZTT (TT2017 µ∞), Sullivan Pbb",
    ("tt2017", True): "ZTT (TT2017 µ∞), PB24 FF Pbb",
    ("l23", False): "ZTT (L23 µ∞), Sullivan Pbb",
    ("l23", True): "ZTT (L23 µ∞), PB24 FF Pbb",
    ("l23_closure", False): "ZTT (L23 closure µ∞), Sullivan Pbb",
    ("l23_closure", True): "ZTT (L23 closure µ∞), PB24 FF Pbb",
}
HYBRID = "L23-trained hybrid, no fallback"
HYBRID_FB = "L23-trained hybrid, ZTT fallback out of domain"
O25_PUB = "O25 published (calibrated on all 5000)"
O25_REFIT = "O25 form, refit on PB24 (relatively weighted)"
O25_UNW = "O25 form, refit on PB24 (unweighted, Rrs <= 0.06, paper's objective)"
TRANSFER_DW = "Fresnel + downwelling transfer"
TRANSFER_TV = "Fresnel θv-only transfer"
O25_PUB_RRS = {
    "dw": f"O25 published -> rrs ({TRANSFER_DW})",
    "tv": f"O25 published -> rrs ({TRANSFER_TV})",
}
O25_REFIT_RRS = {
    "dw": f"O25 refit (relatively weighted) -> rrs ({TRANSFER_DW})",
    "tv": f"O25 refit (relatively weighted) -> rrs ({TRANSFER_TV})",
}
TRANSFER_ONLY = {
    "dw": f"transfer alone: PB24 Rrs -> rrs ({TRANSFER_DW})",
    "tv": f"transfer alone: PB24 Rrs -> rrs ({TRANSFER_TV})",
}

#: The split kinds of splits.json, in report order.
SPLIT_KINDS = ("realisation", "bp_band", "geometry")


def build_registry(*, with_published: bool = True) -> tuple[ModelSpec, ...]:
    """Every scored instance, in report order.

    ``with_published=False`` drops the models that need the O25 checkout.
    """
    out = [ModelSpec("gordon", GORDON, "rrs", "gordon", in_water=True)]
    for mu in ("tt2017", "l23", "l23_closure"):
        for ff in (False, True):
            out.append(
                ModelSpec(
                    f"ztt_{mu}_{'ff' if ff else 'sul'}",
                    ZTT_LABELS[(mu, ff)],
                    "rrs",
                    f"ztt_{mu}",
                    in_water=True,
                )
            )
    out.append(ModelSpec("hybrid", HYBRID, "rrs", "hybrid"))
    out.append(ModelSpec("hybrid_fb", HYBRID_FB, "rrs", "hybrid"))
    if with_published:
        out.append(ModelSpec("o25_pub", O25_PUB, "Rrs", "o25_pub", tables=("native",)))
    for kind in SPLIT_KINDS:
        out.append(
            ModelSpec(
                f"o25_refit@{kind}",
                O25_REFIT,
                "Rrs",
                "o25_refit",
                kind,
                tables=("native",),
            )
        )
    out.append(
        ModelSpec(
            "o25_unw@realisation",
            O25_UNW,
            "Rrs",
            "o25_refit",
            "realisation",
            tables=("native",),
        )
    )
    for t in ("dw", "tv"):
        if with_published:
            out.append(
                ModelSpec(
                    f"o25_pub_{t}", O25_PUB_RRS[t], "rrs", "o25_pub", tables=("common",)
                )
            )
        for kind in SPLIT_KINDS:
            out.append(
                ModelSpec(
                    f"o25_refit_{t}@{kind}",
                    O25_REFIT_RRS[t],
                    "rrs",
                    "o25_refit",
                    kind,
                    tables=("common",),
                )
            )
        out.append(
            ModelSpec(
                f"xfer_{t}", TRANSFER_ONLY[t], "rrs", "transfer", tables=("common",)
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class CutSpec:
    """An angle cut: which geometry categories it takes, in which pass."""

    key: str
    label: str
    pass_: str
    gcats: tuple


#: Geometry categories of the air pass: 0 window & ψ ≥ 134, 1 window & ψ <
#: 134, 2 shell. The in-water-only pass has one category.
CUTS = (
    CutSpec("window", "window (θs, θv_air ≤ 70°)", "air", (0, 1)),
    CutSpec("psi134", "window, ψ ≥ 134°", "air", (0,)),
    CutSpec("shell", "shell (θs or θv_air ≥ 80°)", "air", (2,)),
    CutSpec("everything", "everything (window + shell)", "air", (0, 1, 2)),
    CutSpec("in_water", "in-water-only (θ_w 50-87.5°)", "in_water", (0,)),
)
CUT = {c.key: c for c in CUTS}


@dataclass(frozen=True)
class SideSpec:
    """A split side, as the realisation categories (rcat) it takes.

    ``rcat = real_test + 2 * bp_test``. The geometry split's sides are the
    window (train) and the shell (held out) over all realisations, i.e. the
    side ``all`` of those two cuts.
    """

    key: str
    split: str
    side: str
    rcats: tuple


SIDES = (
    SideSpec("all", "none", "all realisations", (0, 1, 2, 3)),
    SideSpec("real_test", "realisation", "held out", (1, 3)),
    SideSpec("real_train", "realisation", "train", (0, 2)),
    SideSpec("bp_test", "bp_band", "held out", (2, 3)),
    SideSpec("bp_train", "bp_band", "train", (0, 1)),
)
SIDE = {s.key: s for s in SIDES}


@dataclass(frozen=True)
class BandRange:
    """A wavelength range a statistic is taken over."""

    key: str
    label: str
    lo: float
    hi: float


BAND_RANGES = (
    BandRange("all", "350-800", 0.0, 1e9),
    BandRange("le750", "350-750", 0.0, EMULATOR_WAVE_MAX),
    BandRange("gt750", "751-800 (hybrid: out of domain by construction)", 750.5, 1e9),
)
BRANGE = {b.key: b for b in BAND_RANGES}


def instance_sides(spec: ModelSpec) -> tuple:
    """The sides an instance is scored on: a refit only on its own split.

    The geometry-split refit is scored on side ``all``, where the window is
    its training side and the shell its held-out side.
    """
    if spec.fit is None:
        return tuple(s.key for s in SIDES)
    return {
        "realisation": ("real_test", "real_train"),
        "bp_band": ("bp_test", "bp_train"),
        "geometry": ("all",),
    }[spec.fit]


def instance_ranges(spec: ModelSpec) -> tuple:
    """Band ranges reported: a hybrid never on the mixed 350-800."""
    return ("le750", "gt750") if spec.hybrid else ("all", "le750", "gt750")


def split_label(spec: ModelSpec, side: SideSpec, cut: CutSpec) -> tuple[str, str]:
    """(split, side) as written in the CSVs."""
    if spec.fit == "geometry":
        return "geometry", ("train" if cut.key in ("window", "psi134") else "held out")
    return side.split, side.side


# ======================================================= stat columns ========
@dataclass(frozen=True)
class Column:
    """A CSV column: its header and how its value is formatted."""

    name: str
    fmt: str = "s"

    def format(self, value) -> str:
        """The cell text."""
        if self.fmt == "s":
            return str(value)
        if self.fmt == "d":
            return str(int(value))
        v = float(value)
        if not np.isfinite(v):
            return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
        return format(v, self.fmt)


STAT_COLUMNS = (
    Column("n_values", "d"),
    Column("n_truth_zero_excluded", "d"),
    Column("median_abs_rel_err_pct", ".4f"),
    Column("q90_abs_rel_err_pct", ".4f"),
    Column("rrms_pct", ".6g"),
    Column("rrms_physical_only_pct", ".6g"),
    Column("nonphysical_pct", ".4f"),
    Column("nonpositive_pct", ".4f"),
    Column("nonfinite_pct", ".4f"),
)


# ===================================================== small utilities =======
def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path: Path, columns, rows) -> None:
    """A CSV whose header *is* ``[c.name for c in columns]`` and whose cells
    are each column's own format of ``row[c.name]`` -- one label object for
    both, so the header cannot drift from the values (M5 task 6)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([c.name for c in columns])
    for row in rows:
        writer.writerow([c.format(row[c.name]) for c in columns])
    write_atomically(path, buffer.getvalue())


def read_csv(path: Path) -> list[dict]:
    """A committed CSV back as dicts of strings (the markdown's only input)."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@contextlib.contextmanager
def float64():
    """JAX in float64 for reference arithmetic, restoring the prior setting."""
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def hist_quantile(h: np.ndarray, q: float) -> float:
    """Quantile of |e| from a :data:`N_BIN` histogram, log-linear in the bin.

    Returns a fraction (not %); ``inf`` when it falls in the overflow bin,
    ``nan`` for an empty histogram.
    """
    total = int(h.sum())
    if total == 0:
        return float("nan")
    target = q * total
    c = np.cumsum(h)
    j = int(np.searchsorted(c, target, side="left"))
    j = min(j, N_BIN - 1)
    if j == N_BIN - 1:
        return float("inf")
    prev = float(c[j - 1]) if j > 0 else 0.0
    frac = (target - prev) / float(h[j]) if h[j] else 0.0
    frac = min(max(frac, 0.0), 1.0)
    if j == 0:
        return frac * 10.0**HIST_LO
    lo = HIST_LO + (j - 1) * HIST_WIDTH
    return float(10.0 ** (lo + frac * HIST_WIDTH))


def histogram_bin(e: np.ndarray) -> np.ndarray:
    """Bin index of each relative error (see :data:`HIST_LO`)."""
    a = np.abs(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log10(a)
        idx = np.floor((lg - HIST_LO) / HIST_WIDTH)
    idx = np.where(np.isfinite(idx), idx, -1.0)
    out = np.clip(idx, 0, N_FINE - 1).astype(np.int64) + 1
    out[a < 10.0**HIST_LO] = 0
    out[~np.isfinite(e) | (a >= 10.0**HIST_HI)] = N_BIN - 1
    return out


# ============================================================ the plan =======
@dataclass
class Plan:
    """What is scored: realisations, bands, splits, labels, and its name.

    Attributes
    ----------
    ids : numpy.ndarray
        Realisation ids, ascending.
    band_idx : numpy.ndarray
        Indices into PB24's 451-band grid.
    wave : numpy.ndarray
        The scored wavelengths (nm).
    cache_dir : str or None
        The hyperspectral cache (``None``: the default).
    sample : str
        The sample's description (the markdown and the eigen provenance).
    tag : str
        Its short name, the ``sample`` cell of every CSV row.
    rcat, bp_bin, water_class : numpy.ndarray
        Per realisation (aligned with ``ids``): split category
        ``real_test + 2 bp_test``, B_p bin, water-class index (0-based).
    bp_edges : numpy.ndarray
        B_p bin edges (from the 5000 realisations' mean B_p).
    class_labels : tuple of str
        Water-class labels (``("n/a",)`` when the sidecar is absent).
    """

    ids: np.ndarray
    band_idx: np.ndarray
    wave: np.ndarray
    cache_dir: str | None
    sample: str
    tag: str
    rcat: np.ndarray
    bp_bin: np.ndarray
    water_class: np.ndarray
    bp_edges: np.ndarray
    class_labels: tuple
    n_real_test: int = 0
    n_bp_test: int = 0

    @property
    def chunks(self) -> list[tuple[int, ...]]:
        """Consecutive runs of :data:`CHUNK` ids (the unit of every sum)."""
        ids = [int(i) for i in self.ids]
        return [tuple(ids[i : i + CHUNK]) for i in range(0, len(ids), CHUNK)]

    def attrs(self, realisation: np.ndarray) -> tuple[np.ndarray, ...]:
        """(rcat, bp_bin, water_class) per sample of a realisation vector."""
        pos = np.searchsorted(self.ids, realisation)
        return self.rcat[pos], self.bp_bin[pos], self.water_class[pos]

    @property
    def le750(self) -> np.ndarray:
        """Band positions inside the emulator's wavelength domain."""
        return np.flatnonzero(self.wave <= EMULATOR_WAVE_MAX)

    @property
    def gt750(self) -> np.ndarray:
        """Band positions beyond it (751-800 nm)."""
        return np.flatnonzero(self.wave > EMULATOR_WAVE_MAX)


def make_plan(*, quick: bool, cache_dir=None, realisations=None) -> Plan:
    """The scored sample, its splits and labels.

    ``realisations`` overrides the sample (the tests pass the fixture's ids).
    """
    from robust.rt.data import pb24 as P
    from robust.rt.data import pb24_cache

    cache = pb24_cache.open_cache(cache_dir)
    available = np.asarray(cache.realisations, dtype=int)
    stored_bands = np.asarray(cache.four_d_bands, dtype=int)
    full_wave = np.asarray(cache.grids["lambda"], dtype=float)
    if realisations is not None:
        ids = np.unique(np.asarray(realisations, dtype=int))
        band_idx = stored_bands
        name = f"custom: {ids.size} realisations {ids.tolist()[:6]}"
    elif quick:
        rng = np.random.default_rng(QUICK_SEED)
        ids = np.sort(
            rng.choice(available, size=min(QUICK_N, available.size), replace=False)
        )
        band_idx = stored_bands[::QUICK_BAND_STRIDE]
        name = (
            f"quick: {ids.size} realisations drawn with seed {QUICK_SEED}, every "
            f"{QUICK_BAND_STRIDE}th band"
        )
    else:
        ids = available
        band_idx = stored_bands
        name = "full: every realisation in the cache"
    missing = sorted(set(ids.tolist()) - set(available.tolist()))
    if missing:
        raise SystemExit(f"realisations {missing[:10]} are not in the cache")
    band_idx = np.intersect1d(band_idx, stored_bands)
    kind = "custom" if realisations is not None else "quick" if quick else "full"
    tag = f"{kind}-{ids.size}x{band_idx.size}" + (
        f"-seed{QUICK_SEED}" if kind == "quick" else ""
    )
    wave = full_wave[band_idx]
    name += (
        f" ({ids.size} realisations, {wave.size} bands {wave[0]:.0f}-{wave[-1]:.0f} nm,"
        f" hyperspectral cache pb24_v1)"
    )

    content = P.read_splits()
    real_test = set(int(i) for i in content["realisation"]["test"])
    bp_test = set(int(i) for i in content["bp_band"]["test"])
    rcat = np.array(
        [int(i in real_test) + 2 * int(i in bp_test) for i in ids.tolist()], dtype=int
    )
    all_ids = np.asarray(content["realisations"], dtype=int)
    bp_mean = np.asarray(content["B_p_mean"], dtype=float)
    edges = np.quantile(bp_mean, np.linspace(0.0, 1.0, N_BP_BIN + 1))
    pos = np.searchsorted(all_ids, ids)
    if not np.array_equal(all_ids[pos], ids):
        raise SystemExit("the splits file does not hold every scored realisation")
    bp_bin = np.clip(
        np.searchsorted(edges[1:-1], bp_mean[pos], side="right"), 0, N_BP_BIN - 1
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classes = np.asarray(P.read_classes(), dtype=int).reshape(-1)
        water_class = classes[ids - 1] - 1
        class_labels = tuple(f"class {k}" for k in range(1, P.N_CLASS + 1))
    except Exception:  # noqa: BLE001 - no sidecar: the breakdown says so
        water_class = np.zeros(ids.size, dtype=int)
        class_labels = ("n/a (class sidecar absent)",)
    return Plan(
        ids=ids,
        band_idx=band_idx,
        wave=wave,
        cache_dir=None if cache_dir is None else str(cache_dir),
        sample=name,
        tag=tag,
        rcat=rcat,
        bp_bin=bp_bin,
        water_class=water_class,
        bp_edges=edges,
        class_labels=class_labels,
        n_real_test=int(sum(i in real_test for i in ids.tolist())),
        n_bp_test=int(sum(i in bp_test for i in ids.tolist())),
    )


@dataclass
class Chunk:
    """One loaded chunk, already on the plan's bands."""

    iops: object
    phase: object
    geom: object
    wave: object
    rrs: np.ndarray
    Rrs: np.ndarray
    realisation: np.ndarray
    n_geom: int
    theta_s: np.ndarray
    theta_v: np.ndarray
    dphi: np.ndarray


def take_bands(iops, phase, geom, sel: np.ndarray):
    """The three containers restricted to band positions ``sel``."""
    import jax.numpy as jnp

    from robust.rt.types import Geometry, IOPs, PhaseParams

    s = jnp.asarray(sel)
    iops2 = IOPs(
        a=iops.a[:, s],
        bb_w=iops.bb_w[:, s],
        bb_p=iops.bb_p[:, s],
        a_ph=None if iops.a_ph is None else iops.a_ph[:, s],
    )
    phase2 = PhaseParams(
        B_p=phase.B_p[:, s],
        ff_B=None if phase.ff_B is None else phase.ff_B[:, s],
        ff_weight=None if phase.ff_weight is None else phase.ff_weight[:, s],
    )
    geom2 = Geometry(
        theta_s=geom.theta_s,
        theta_v=geom.theta_v,
        dphi=geom.dphi,
        diffuse_fraction=(
            None if geom.diffuse_fraction is None else geom.diffuse_fraction[:, s]
        ),
    )
    return iops2, phase2, geom2


def load_chunk(plan: Plan, ids, pass_: str) -> Chunk:
    """Load ``ids`` through the task-3 loader and keep the plan's bands.

    ``angles="all"`` for the air pass, ``"in_water_only"`` for the other;
    ``drop_zero_rrs=False`` so no sample leaves silently (a zero truth is
    counted and excluded per value instead). The Fresnel check and
    ``validate`` were run over the whole release in task 3 and are off here.
    """
    import jax.numpy as jnp

    from robust.rt.data import pb24 as P

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = P.load_batch(
            realisations=list(ids),
            angles="all" if pass_ == "air" else "in_water_only",
            extras=(),
            drop_zero_rrs=False,
            water_classes=None,
            validate=False,
            check_fresnel=False,
            cache_dir=plan.cache_dir,
        )
    wave = np.asarray(b.wave, dtype=float)
    pos = np.searchsorted(wave, plan.wave)
    if not np.array_equal(wave[np.clip(pos, 0, wave.size - 1)], plan.wave):
        raise RuntimeError("the loaded bands do not contain the plan's bands")
    iops, phase, geom = b.iops, b.phase_params, b.geometry
    rrs, Rrs = np.asarray(b.rrs), np.asarray(b.Rrs)
    if pos.size != wave.size:
        iops, phase, geom = take_bands(iops, phase, geom, pos)
        rrs, Rrs = rrs[:, pos], Rrs[:, pos]
    n_geom = b.n_sample // len(ids)
    return Chunk(
        iops=iops,
        phase=phase,
        geom=geom,
        wave=jnp.asarray(plan.wave, dtype=jnp.float32),
        rrs=rrs,
        Rrs=Rrs,
        realisation=np.asarray(b.realisation),
        n_geom=n_geom,
        theta_s=np.asarray(b.theta_s, dtype=float),
        theta_v=np.asarray(b.theta_v, dtype=float),
        dphi=np.asarray(b.dphi, dtype=float),
    )


@dataclass
class GeomTable:
    """The per-geometry labels of one pass (identical for every realisation)."""

    pass_: str
    theta_s: np.ndarray
    theta_v: np.ndarray
    dphi: np.ndarray
    psi: np.ndarray
    gcat: np.ndarray
    ts_idx: np.ndarray
    tv_idx: np.ndarray
    psi_bin: np.ndarray
    ts_nodes: np.ndarray
    tv_nodes: np.ndarray

    @property
    def n_gcat(self) -> int:
        """Geometry categories in this pass."""
        return 3 if self.pass_ == "air" else 1

    @property
    def n(self) -> int:
        """Geometries per realisation."""
        return self.theta_s.size


def geometry_table(chunk: Chunk, pass_: str) -> GeomTable:
    """Labels of the first realisation's geometries; checks the rest repeat."""
    from robust.rt import ztt as Z
    from robust.rt.data import pb24 as P
    from robust.rt.types import Geometry

    g = chunk.n_geom
    ts, tv, ph = chunk.theta_s[:g], chunk.theta_v[:g], chunk.dphi[:g]
    reps = chunk.theta_s.size // g
    for arr, first in ((chunk.theta_s, ts), (chunk.theta_v, tv), (chunk.dphi, ph)):
        if not np.array_equal(arr, np.tile(first, reps)):
            raise RuntimeError("geometries differ between realisations of a chunk")
    in_water = pass_ == "in_water"
    psi = np.asarray(
        Z.geometry_to_paper_angles(
            Geometry(theta_s=ts, theta_v=tv, dphi=ph), theta_v_in_water=in_water
        )[3],
        dtype=float,
    )
    if in_water:
        gcat = np.zeros(g, dtype=int)
    else:
        window = (ts <= P.ANGLE_WINDOW_MAX) & (tv <= P.ANGLE_WINDOW_MAX)
        gcat = np.where(window, np.where(psi >= PSI_FITTED_MIN, 0, 1), 2)
    ts_nodes, tv_nodes = np.unique(ts), np.unique(tv)
    return GeomTable(
        pass_=pass_,
        theta_s=ts,
        theta_v=tv,
        dphi=ph,
        psi=psi,
        gcat=gcat,
        ts_idx=np.searchsorted(ts_nodes, ts),
        tv_idx=np.searchsorted(tv_nodes, tv),
        psi_bin=np.clip(
            np.searchsorted(PSI_EDGES, psi, side="right") - 1, 0, len(PSI_EDGES) - 2
        ),
        ts_nodes=ts_nodes,
        tv_nodes=tv_nodes,
    )


#: Breakdowns other than the band, with where each takes its per-sample index.
BREAKDOWNS = ("theta_s", "theta_v", "psi_bin", "bp_bin", "water_class")


def breakdown_sizes(plan: Plan, gt: GeomTable) -> dict[str, int]:
    """Number of values of each breakdown in a pass."""
    return {
        "theta_s": gt.ts_nodes.size,
        "theta_v": gt.tv_nodes.size,
        "psi_bin": len(PSI_EDGES) - 1,
        "bp_bin": N_BP_BIN,
        "water_class": len(plan.class_labels),
    }


# ============================================================ O25 refit ======
#: The fitted O25 instances: (name, split kind, weighted, rrs_max).
O25_FITS = (
    ("o25_refit@realisation", "realisation", True, None),
    ("o25_refit@bp_band", "bp_band", True, None),
    ("o25_refit@geometry", "geometry", True, None),
    ("o25_unw@realisation", "realisation", False, 0.06),
)
#: fit_o25_table's floor on points per node, applied here too.
O25_MIN_POINTS = 40


def o25_train_mask(plan: Plan, chunk: Chunk, gt: GeomTable, kind: str) -> np.ndarray:
    """Per-sample training mask of a split kind (air pass)."""
    rcat, _, _ = plan.attrs(chunk.realisation)
    if kind == "realisation":
        return (rcat % 2) == 0
    if kind == "bp_band":
        return (rcat // 2) == 0
    window = np.tile(gt.gcat < 2, chunk.realisation.size // gt.n)
    return window


def o25_normal_equations(plan: Plan, chunk: Chunk, gt: GeomTable) -> dict:
    """Per-node float64 normal equations of each O25 fit, for one chunk.

    The relatively weighted fit solves ``min Σ (x·β/Rrs − 1)²`` and the
    unweighted one ``min Σ (x·β − Rrs)²`` with ``x = (w_bw, w_bw², w_bp,
    w_bp²)`` per point (sample × band), exactly :func:`baselines.fit_o25_table`'s
    objectives; the node is the sample's geometry.
    """
    a = np.asarray(chunk.iops.a, dtype=np.float64)
    bbw = np.asarray(chunk.iops.bb_w, dtype=np.float64)
    bbp = np.asarray(chunk.iops.bb_p, dtype=np.float64)
    denom = a + bbw + bbp
    w1, w2 = bbw / denom, bbp / denom
    X = np.stack([w1, w1**2, w2, w2**2], axis=-1)
    y = chunk.Rrs.astype(np.float64)
    nr, g, W = chunk.realisation.size // gt.n, gt.n, y.shape[-1]
    out = {}
    for name, kind, weighted, rrs_max in O25_FITS:
        m = np.broadcast_to(o25_train_mask(plan, chunk, gt, kind)[:, None], y.shape)
        if rrs_max is not None:
            m = m & (y <= rrs_max)
        if weighted:
            Zm, t = X / y[..., None], np.ones_like(y)
        else:
            Zm, t = X, y
        Zw = Zm * m[..., None]
        Zr, Zwr = Zm.reshape(nr, g, W, 4), Zw.reshape(nr, g, W, 4)
        out[name] = (
            np.einsum("rgwi,rgwj->gij", Zwr, Zr),
            np.einsum("rgwi,rgw->gi", Zwr, t.reshape(nr, g, W)),
            m.reshape(nr, g, W).sum(axis=(0, 2)).astype(np.int64),
        )
    return out


def solve_o25(name: str, XtX, Xty, count, gt: GeomTable):
    """An :class:`O25Table` from summed normal equations (trained nodes only).

    Raises if a trained node has fewer than :data:`O25_MIN_POINTS` points or
    the trained nodes do not form a full grid.
    """
    from robust.rt import baselines as B

    trained = count > 0
    thin = trained & (count < O25_MIN_POINTS)
    if thin.any():
        raise ValueError(
            f"{name}: {int(thin.sum())} node(s) under {O25_MIN_POINTS} points"
        )
    ts, tv, ph = gt.theta_s[trained], gt.theta_v[trained], gt.dphi[trained]
    nodes = [np.unique(v) for v in (ts, tv, ph)]
    shape = tuple(n.size for n in nodes)
    if int(trained.sum()) != int(np.prod(shape)):
        raise ValueError(f"{name}: the trained nodes are not a full grid")
    G = np.full((*shape, 4), np.nan)
    for gi in np.flatnonzero(trained):
        idx = tuple(
            int(np.searchsorted(n, v))
            for n, v in zip(
                nodes, (gt.theta_s[gi], gt.theta_v[gi], gt.dphi[gi]), strict=True
            )
        )
        G[idx] = np.linalg.solve(XtX[gi], Xty[gi])
    kind = dict((n, k) for n, k, _, _ in O25_FITS)[name]
    return B.O25Table(
        theta_s=nodes[0],
        theta_v=nodes[1],
        dphi=nodes[2],
        G=G,
        provenance=(
            f"{name}: float64 normal equations on the {kind} split's training rows, "
            f"{int(count.sum())} points, {shape[0]}x{shape[1]}x{shape[2]} nodes"
        ),
    )


def fit_task(task):
    """Worker: the O25 normal equations of one chunk."""
    plan, ids, gt = task
    chunk = load_chunk(plan, ids, "air")
    return o25_normal_equations(plan, chunk, gt)


# ============================================================== models =======
@dataclass
class ModelContext:
    """What the predictions need besides the chunk."""

    registry: tuple
    o25_tables: dict
    published: object = None
    emulator: object = None
    transfers: dict = field(default_factory=dict)


def make_context(registry, o25_tables) -> ModelContext:
    """Load the emulator, the published O25 LUT (if registered) and transfers."""
    from robust.rt import baselines as B
    from robust.rt import conventions as C
    from robust.rt import emulator as E

    published = None
    if any(s.key == "o25_pub" for s in registry):
        published = B.o25_published_table()
    return ModelContext(
        registry=registry,
        o25_tables=o25_tables,
        published=published,
        emulator=E.load_default(),
        transfers={
            "dw": C.default_fresnel_transfer(downwelling=True),
            "tv": C.default_fresnel_transfer(downwelling=False),
        },
    )


def ztt_prediction(chunk: Chunk, mu_model: str, ff: bool, in_water: bool) -> np.ndarray:
    """ZTT with a µ∞ and a Pbb choice. The variant's own combination goes
    through ``rrs_ZTT`` untouched; the crossed ones pass ``P_bb=``."""
    import jax.numpy as jnp

    from robust.rt import ztt as Z

    variant = "tt2017" if mu_model == "tt2017" else "l23"
    native_ff = variant == "l23"
    kwargs = {"variant": variant, "theta_v_in_water": in_water}
    if mu_model == "l23_closure":
        kwargs["mu_inf_model"] = "l23_closure"
    if ff != native_ff:
        psi = Z.geometry_to_paper_angles(chunk.geom, theta_v_in_water=in_water)[3]
        kwargs["P_bb"] = Z.P_bb_from_phase(
            chunk.phase, jnp.asarray(psi)[..., None], use_ff=ff
        )
    return np.asarray(
        Z.rrs_ZTT(chunk.iops, chunk.phase, chunk.geom, chunk.wave, **kwargs)
    )


def hybrid_prediction(chunk: Chunk, plan: Plan, ctx: ModelContext, fallback: bool):
    """The shipped hybrid (``hybrid.rrs_forward``, subsurface ``rrs``; ``forward``
    returns ``Rrs`` through the nadir constants) on 350-750 nm and,
    separately, on 751-800 nm.

    Two calls because the out-of-domain mask is per sample over the bands of
    a call: one call over 350-800 would mark every sample out of domain and
    the fallback would zero everything. Returns the prediction and the
    per-sample out-of-domain mask on 350-750 nm.
    """
    from robust.rt import hybrid as H

    n, W = chunk.rrs.shape
    out = np.empty((n, W), dtype=np.float32)
    ood = np.zeros(n, dtype=bool)
    for sel, inside in ((plan.le750, True), (plan.gt750, False)):
        if sel.size == 0:
            continue
        iops, phase, geom = take_bands(chunk.iops, chunk.phase, chunk.geom, sel)
        wave = chunk.wave[sel]
        out[:, sel] = np.asarray(
            H.rrs_forward(
                iops,
                phase,
                geom,
                wave,
                mode="hybrid",
                emulator=ctx.emulator,
                check_domain=False,
                on_out_of_domain="ztt" if fallback else "warn",
            )
        )
        if inside:
            ood = np.asarray(ctx.emulator.out_of_domain_mask(iops, phase, geom, wave))
    return out, ood


def predictions(chunk: Chunk, plan: Plan, ctx: ModelContext, pass_: str):
    """Yield ``(spec, prediction, truth, extra)`` for every model in a pass.

    ``extra`` is the hybrid's per-sample out-of-domain mask (else ``None``).
    Each model is evaluated once per chunk and dropped after use.
    """
    from robust.rt import baselines as B
    from robust.rt import conventions as C

    in_water = pass_ == "in_water"
    o25_cache: dict = {}

    def o25_Rrs(table_key):
        if table_key not in o25_cache:
            table = ctx.published if table_key == "pub" else ctx.o25_tables[table_key]
            o25_cache[table_key] = B.Rrs_o25(chunk.iops, None, chunk.geom, coeffs=table)
        return o25_cache[table_key]

    for spec in ctx.registry:
        if in_water and not spec.in_water:
            continue
        extra = None
        k = spec.key
        truth = chunk.rrs if spec.space == "rrs" else chunk.Rrs
        if k == "gordon":
            pred = np.asarray(
                B.rrs_gordon(chunk.iops, chunk.phase, chunk.geom, chunk.wave)
            )
        elif k.startswith("ztt_"):
            mu = k[len("ztt_") : k.rindex("_")]
            pred = ztt_prediction(chunk, mu, k.endswith("_ff"), in_water)
        elif k in ("hybrid", "hybrid_fb"):
            pred, extra = hybrid_prediction(chunk, plan, ctx, k == "hybrid_fb")
        elif k == "o25_pub":
            pred = np.asarray(o25_Rrs("pub"))
        elif k.startswith(("o25_refit@", "o25_unw@")):
            pred = np.asarray(o25_Rrs(k))
        elif k.startswith(("o25_pub_", "o25_refit_")):
            t = "dw" if "_dw" in k else "tv"
            table_key = "pub" if k.startswith("o25_pub_") else f"o25_refit@{spec.fit}"
            pred = np.asarray(
                C.Rrs_to_rrs(
                    o25_Rrs(table_key), geometry=chunk.geom, transfer=ctx.transfers[t]
                )
            )
        elif k.startswith("xfer_"):
            pred = np.asarray(
                C.Rrs_to_rrs(
                    chunk.Rrs,
                    geometry=chunk.geom,
                    transfer=ctx.transfers[k[len("xfer_") :]],
                )
            )
        else:  # pragma: no cover - the registry and this switch are one list
            raise KeyError(k)
        yield spec, pred, truth, extra


# ========================================================= accumulators ======
#: Value codes: 0 physical, 1 finite and <= 0, 2 non-finite, 3 truth = 0.
CODE_PHYSICAL, CODE_NONPOS, CODE_NONFINITE, CODE_TRUTH_ZERO = 0, 1, 2, 3


@dataclass
class PassLayout:
    """Group layout of one pass: ``base = rcat * n_gcat + gcat``."""

    pass_: str
    n_base: int
    n_wave: int
    sizes: dict

    def n_groups(self, kind: str) -> int:
        """Groups of a key kind (``band`` or a breakdown)."""
        if kind == "band":
            return self.n_base * self.n_wave
        return self.n_base * 2 * self.sizes[kind]


def layouts(plan: Plan, gts: dict) -> dict:
    """The :class:`PassLayout` of each pass."""
    return {
        p: PassLayout(p, 4 * gt.n_gcat, plan.wave.size, breakdown_sizes(plan, gt))
        for p, gt in gts.items()
    }


def new_ints(lay: PassLayout) -> dict:
    """Integer accumulators of one model in one pass."""
    out = {}
    for kind in ("band", *BREAKDOWNS):
        g = lay.n_groups(kind)
        # int32 suffices per (base, band) group (<= 6.5 M values); the
        # breakdown groups pool bands and can exceed 2**31.
        dtype = np.int32 if kind == "band" else np.int64
        out[f"hist_{kind}"] = np.zeros(g * N_BIN, dtype=dtype)
        out[f"codes_{kind}"] = np.zeros(g * 4, dtype=np.int64)
    out["spectra"] = np.zeros(lay.n_base * 4, dtype=np.int64)
    out["fallback"] = np.zeros(lay.n_base, dtype=np.int64)
    return out


def accumulate(
    ints: dict, lay: PassLayout, plan: Plan, gt: GeomTable, chunk, pred, truth, ood
):
    """Add one model's chunk to its integer accumulators; return its float sums.

    Returns ``{kind: (Σ e² over finite values, Σ e² over physical values)}``.
    """
    n, W = pred.shape
    reps = n // gt.n
    rcat, bp_bin, wclass = plan.attrs(chunk.realisation)
    gcat = np.tile(gt.gcat, reps)
    base = rcat * (lay.n_base // 4) + gcat
    br = (plan.wave > EMULATOR_WAVE_MAX).astype(np.int64)
    per_sample = {
        "theta_s": np.tile(gt.ts_idx, reps),
        "theta_v": np.tile(gt.tv_idx, reps),
        "psi_bin": np.tile(gt.psi_bin, reps),
        "bp_bin": bp_bin,
        "water_class": wclass,
    }

    t64 = truth.astype(np.float64)
    p64 = pred.astype(np.float64)
    tz = t64 == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        e = p64 / np.where(tz, 1.0, t64) - 1.0
    finite = np.isfinite(pred)
    code = np.where(
        finite, np.where(pred > 0, CODE_PHYSICAL, CODE_NONPOS), CODE_NONFINITE
    )
    code = np.where(tz, CODE_TRUTH_ZERO, code).astype(np.int64)
    hb = histogram_bin(e)
    keep = ~tz
    any_tz = bool(tz.any())
    e2_fin = np.where((code == CODE_PHYSICAL) | (code == CODE_NONPOS), e * e, 0.0)
    e2_phy = np.where(code == CODE_PHYSICAL, e * e, 0.0)

    floats = {}
    for kind in ("band", *BREAKDOWNS):
        g = lay.n_groups(kind)
        if kind == "band":
            key = base[:, None] * W + np.arange(W)[None, :]
        else:
            key = (base[:, None] * 2 + br[None, :]) * lay.sizes[kind] + per_sample[
                kind
            ][:, None]
        kf = key.ravel()
        hk = kf * N_BIN + hb.ravel()
        if any_tz:
            hk = hk[keep.ravel()]
        ints[f"hist_{kind}"] += np.bincount(hk, minlength=g * N_BIN)
        ints[f"codes_{kind}"] += np.bincount(kf * 4 + code.ravel(), minlength=g * 4)
        floats[kind] = (
            np.bincount(kf, weights=e2_fin.ravel(), minlength=g),
            np.bincount(kf, weights=e2_phy.ravel(), minlength=g),
        )
    bad = (code == CODE_NONPOS) | (code == CODE_NONFINITE)
    bad_le = bad[:, br == 0].any(axis=1)
    bad_gt = bad[:, br == 1].any(axis=1)
    bad_all = bad.any(axis=1)
    spec = np.stack([np.ones(n), bad_le, bad_gt, bad_all], axis=1)
    # Counts carried as float weights are exact far beyond any count here.
    ints["spectra"] += np.bincount(
        (base[:, None] * 4 + np.arange(4)[None, :]).ravel(),
        weights=spec.ravel().astype(np.float64),
        minlength=lay.n_base * 4,
    ).astype(np.int64)
    if ood is not None:
        ints["fallback"] += np.bincount(
            base, weights=ood.astype(np.float64), minlength=lay.n_base
        ).astype(np.int64)
    return floats


def instance_passes(spec: ModelSpec) -> tuple:
    """Passes a model is scored in."""
    return ("air", "in_water") if spec.in_water else ("air",)


def score_task(task):
    """Worker: score a static list of chunks.

    Integer accumulators (exact, so order-free) are summed here and written to
    ``out_path``; the float sums are returned **per chunk**, so the caller adds
    them in chunk order whatever the number of workers.
    """
    plan, gts, registry, o25_tables, chunk_ids, out_path = task
    ctx = make_context(registry, o25_tables)
    lays = layouts(plan, gts)
    ints = {(p, s.key): new_ints(lays[p]) for s in registry for p in instance_passes(s)}
    chunks = plan.chunks
    per_chunk = {}
    for cid in chunk_ids:
        result = {}
        for p, gt in gts.items():
            chunk = load_chunk(plan, chunks[cid], p)
            for spec, pred, truth, extra in predictions(chunk, plan, ctx, p):
                result[(p, spec.key)] = accumulate(
                    ints[(p, spec.key)], lays[p], plan, gt, chunk, pred, truth, extra
                )
            del chunk
        per_chunk[cid] = result
    if out_path is None:
        return per_chunk, ints
    flat = {
        f"{p}|{k}|{name}": arr for (p, k), d in ints.items() for name, arr in d.items()
    }
    np.savez(out_path, **flat)
    return per_chunk, None


def worker_env() -> dict:
    """Single-threaded BLAS and XLA in each worker: the numbers then do not
    depend on ``--workers`` (and the pool is not oversubscribed)."""
    return {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        ),
    }


def stagger_delay(counter, stagger_s: float) -> float:
    """Return this worker's start delay and advance the shared counter.

    The *n*-th worker to start waits ``n * stagger_s`` seconds (the first
    waits nothing), so the workers' first chunk loads are spread out instead
    of landing in the same second (task 5b). Pure bookkeeping: the delay
    changes no number.
    """
    with counter.get_lock():
        n = int(counter.value)
        counter.value = n + 1
    return n * float(stagger_s)


def _stagger_init(counter, stagger_s: float) -> None:
    """Pool initializer: sleep this worker's :func:`stagger_delay`."""
    delay = stagger_delay(counter, stagger_s)
    if delay > 0:
        time.sleep(delay)


@contextlib.contextmanager
def pool(workers: int, *, stagger_s: float = STAGGER_S):
    """A spawn-context process pool with :func:`worker_env` (or ``None``).

    Each worker's start is staggered by :func:`stagger_delay` so the pool's
    initial allocation burst is spread over ``(workers - 1) * stagger_s``
    seconds.
    """
    if workers <= 0:
        yield None
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    saved = {k: os.environ.get(k) for k in worker_env()}
    os.environ.update(worker_env())
    try:
        with ProcessPoolExecutor(
            workers,
            mp_context=ctx,
            initializer=_stagger_init,
            initargs=(counter, stagger_s),
        ) as ex:
            yield ex
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_fits(plan: Plan, gts: dict, workers: int) -> dict:
    """The O25 refits: normal equations per chunk, summed in chunk order."""
    tasks = [(plan, ids, gts["air"]) for ids in plan.chunks]
    with pool(workers) as ex:
        results = list(ex.map(fit_task, tasks)) if ex else [fit_task(t) for t in tasks]
    tables = {}
    for name, *_ in O25_FITS:
        XtX = sum(r[name][0] for r in results)
        Xty = sum(r[name][1] for r in results)
        cnt = sum(r[name][2] for r in results)
        tables[name] = solve_o25(name, XtX, Xty, cnt, gts["air"])
    return tables


def run_scores(
    plan: Plan, gts: dict, registry, o25_tables, workers: int, scratch: Path
):
    """Score every model; return (integer accumulators, float sums)."""
    n_chunk = len(plan.chunks)
    lays = layouts(plan, gts)
    if workers <= 0:
        per_chunk, ints = score_task(
            (plan, gts, registry, o25_tables, list(range(n_chunk)), None)
        )
        parts = [per_chunk]
    else:
        w = min(workers, n_chunk)
        tasks = [
            (
                plan,
                gts,
                registry,
                o25_tables,
                list(range(i, n_chunk, w)),
                scratch / f"ints_{i}.npz",
            )
            for i in range(w)
        ]
        with pool(w) as ex:
            parts = [r[0] for r in ex.map(score_task, tasks)]
        ints = {
            (p, s.key): new_ints(lays[p]) for s in registry for p in instance_passes(s)
        }
        for t in tasks:
            with np.load(t[-1]) as z:
                for name in z.files:
                    p, k, field_ = name.split("|")
                    ints[(p, k)][field_] += z[name]
            Path(t[-1]).unlink()
    merged = {}
    for part in parts:
        merged.update(part)
    floats = {}
    for cid in range(n_chunk):  # chunk order: the float sums' definition
        for key, kinds in merged[cid].items():
            acc = floats.setdefault(key, {})
            for kind, (s2f, s2p) in kinds.items():
                if kind in acc:
                    acc[kind] = (acc[kind][0] + s2f, acc[kind][1] + s2p)
                else:
                    acc[kind] = (s2f.copy(), s2p.copy())
    return ints, floats


# ============================================================ statistics =====
class Results:
    """Accumulated scores, queried by (model, cut, side, bands, breakdown)."""

    def __init__(self, plan: Plan, gts: dict, registry, ints: dict, floats: dict):
        self.plan, self.gts, self.registry = plan, gts, registry
        self.lays = layouts(plan, gts)
        self.ints, self.floats = ints, floats

    def _bases(self, cut: CutSpec, side: SideSpec) -> np.ndarray:
        ng = self.gts[cut.pass_].n_gcat
        return np.array([r * ng + g for r in side.rcats for g in cut.gcats], dtype=int)

    def _band_positions(self, brange: str, bands=None) -> np.ndarray:
        if bands is not None:
            return np.asarray(bands, dtype=int)
        r = BRANGE[brange]
        return np.flatnonzero((self.plan.wave >= r.lo) & (self.plan.wave <= r.hi))

    def stats(
        self, spec, cut, side, *, brange="all", bands=None, kind="band", value=None
    ):
        """The :data:`STAT_COLUMNS` of one selection, as a dict."""
        lay = self.lays[cut.pass_]
        ints = self.ints[(cut.pass_, spec.key)]
        fl = self.floats[(cut.pass_, spec.key)][kind]
        B = self._bases(cut, side)
        if kind == "band":
            sel = self._band_positions(brange, bands)
            shape = (lay.n_base, lay.n_wave)
            idx = np.ix_(B, sel)
        else:
            brs = {"all": [0, 1], "le750": [0], "gt750": [1]}[brange]
            shape = (lay.n_base, 2, lay.sizes[kind])
            idx = np.ix_(B, brs, [value])
        h = (
            ints[f"hist_{kind}"]
            .reshape(*shape, N_BIN)[idx]
            .reshape(-1, N_BIN)
            .sum(axis=0, dtype=np.int64)
        )
        c = ints[f"codes_{kind}"].reshape(*shape, 4)[idx].reshape(-1, 4).sum(axis=0)
        s2f = float(fl[0].reshape(shape)[idx].sum())
        s2p = float(fl[1].reshape(shape)[idx].sum())
        n = int(c[0] + c[1] + c[2])
        n_fin, n_phy = int(c[0] + c[1]), int(c[0])
        pct = (lambda k: 100.0 * k / n) if n else (lambda k: float("nan"))
        return {
            "n_values": n,
            "n_truth_zero_excluded": int(c[3]),
            "median_abs_rel_err_pct": 100.0 * hist_quantile(h, 0.5),
            "q90_abs_rel_err_pct": 100.0 * hist_quantile(h, 0.9),
            "rrms_pct": 100.0 * np.sqrt(s2f / n_fin) if n_fin else float("nan"),
            "rrms_physical_only_pct": 100.0 * np.sqrt(s2p / n_phy)
            if n_phy
            else float("nan"),
            "nonphysical_pct": pct(int(c[1] + c[2])),
            "nonpositive_pct": pct(int(c[1])),
            "nonfinite_pct": pct(int(c[2])),
        }

    def spectra(self, spec, cut, side, brange) -> tuple[float, float]:
        """(% spectra with a non-physical value in ``brange``, % samples where
        the hybrid's fallback fired)."""
        ints = self.ints[(cut.pass_, spec.key)]
        B = self._bases(cut, side)
        sp = ints["spectra"].reshape(-1, 4)[B].sum(axis=0)
        col = {"le750": 1, "gt750": 2, "all": 3}[brange]
        n = int(sp[0])
        bad = 100.0 * sp[col] / n if n else float("nan")
        fb = (
            100.0 * ints["fallback"][B].sum() / n
            if n and spec.key == "hybrid_fb"
            else float("nan")
        )
        return bad, fb


#: Identity columns shared by the score CSVs.
ID_COLUMNS = (
    Column("sample"),
    Column("model"),
    Column("space"),
    Column("tables"),
    Column("fitted_on"),
    Column("cut"),
    Column("split"),
    Column("side"),
    Column("bands"),
)
SCORE_COLUMNS = (
    *ID_COLUMNS,
    *STAT_COLUMNS,
    Column("spectra_nonphysical_pct", ".4f"),
    Column("fallback_samples_pct", ".4f"),
)
#: The long tables repeat neither ``tables`` nor ``fitted_on`` (both are in
#: the scores CSV, one row per model).
SHORT_ID_COLUMNS = tuple(c for c in ID_COLUMNS if c.name not in ("tables", "fitted_on"))
BAND_COLUMNS = (
    *SHORT_ID_COLUMNS[:-1],
    Column("wavelength_nm", ".0f"),
    Column("domain"),
    *STAT_COLUMNS,
)
BREAKDOWN_COLUMNS = (
    *SHORT_ID_COLUMNS,
    Column("breakdown"),
    Column("value"),
    *STAT_COLUMNS,
)

#: The (cut, side) pairs the per-band CSV carries.
BAND_COMBOS = (
    ("window", "real_test"),
    ("window", "real_train"),
    ("window", "bp_test"),
    ("shell", "all"),
    ("everything", "real_test"),
    ("in_water", "real_test"),
)


def fitted_on(spec: ModelSpec) -> str:
    """The ``fitted_on`` cell."""
    if spec.key in ("o25_pub", "o25_pub_dw", "o25_pub_tv"):
        return "all 5000 (published calibration)"
    if spec.key.startswith("xfer_"):
        return "transfer fitted on the realisation split's training rows (task 4)"
    if spec.fit is None:
        return "not fitted on PB24"
    return {
        "realisation": "realisation split, training rows",
        "bp_band": "bp_band split, training rows",
        "geometry": "geometry split, training rows (window)",
    }[spec.fit]


def id_cells(plan, spec, cut, side, brange=None) -> dict:
    """The identity cells of a row."""
    split, side_label = split_label(spec, side, cut)
    out = {
        "sample": plan.tag,
        "model": spec.label,
        "space": spec.space,
        "tables": "+".join(spec.tables),
        "fitted_on": fitted_on(spec),
        "cut": cut.label,
        "split": split,
        "side": side_label,
    }
    if brange is not None:
        out["bands"] = BRANGE[brange].label
    return out


def cuts_of(spec: ModelSpec) -> tuple:
    """The cuts a model is scored on."""
    return tuple(c for c in CUTS if c.pass_ == "air" or spec.in_water)


def score_rows(res: Results) -> list[dict]:
    """The headline CSV: model × cut × side × band range."""
    rows = []
    for spec in res.registry:
        for cut in cuts_of(spec):
            if spec.fit == "geometry" and cut.key == "everything":
                continue  # mixes its train and held-out sides
            for side_key in instance_sides(spec):
                side = SIDE[side_key]
                for br in instance_ranges(spec):
                    bad, fb = res.spectra(spec, cut, side, br)
                    rows.append(
                        id_cells(res.plan, spec, cut, side, br)
                        | res.stats(spec, cut, side, brange=br)
                        | {"spectra_nonphysical_pct": bad, "fallback_samples_pct": fb}
                    )
    return rows


def band_rows(res: Results) -> list[dict]:
    """The per-band CSV, on :data:`BAND_COMBOS`."""
    rows = []
    for spec in res.registry:
        for cut_key, side_key in BAND_COMBOS:
            cut, side = CUT[cut_key], SIDE[side_key]
            if cut.pass_ == "in_water" and not spec.in_water:
                continue
            if side_key not in instance_sides(spec):
                continue
            if spec.fit == "geometry" and cut_key == "everything":
                continue
            ids = id_cells(res.plan, spec, cut, side)
            for j, w in enumerate(res.plan.wave):
                ood = spec.hybrid and w > EMULATOR_WAVE_MAX
                rows.append(
                    ids
                    | {
                        "wavelength_nm": w,
                        "domain": "out of domain by construction"
                        if ood
                        else "in domain",
                    }
                    | res.stats(spec, cut, side, bands=[j])
                )
    return rows


def breakdown_values(res: Results, pass_: str) -> dict[str, list[str]]:
    """Value labels of every breakdown in a pass, in index order."""
    gt, plan = res.gts[pass_], res.plan
    psi = [
        f"[{lo:g}, {hi:g})"
        for lo, hi in zip(PSI_EDGES[:-1], PSI_EDGES[1:], strict=True)
    ]
    psi[-1] = f"[{PSI_EDGES[-2]:g}, 180]"
    e = plan.bp_edges
    return {
        "theta_s": [f"{v:g}" for v in gt.ts_nodes],
        "theta_v": [f"{v:g}" for v in gt.tv_nodes],
        "psi_bin": psi,
        "bp_bin": [
            f"bin {k + 1}: [{e[k]:.5f}, {e[k + 1]:.5f}]" for k in range(N_BP_BIN)
        ],
        "water_class": list(plan.class_labels),
    }


#: The breakdown names as written in the CSV, per pass.
BREAKDOWN_NAMES = {
    "air": {
        "theta_s": "theta_s (deg)",
        "theta_v": "theta_v_air (deg)",
        "psi_bin": "psi bin (deg)",
        "bp_bin": "B_p bin (realisation mean, equal count over 5000)",
        "water_class": "water class",
    },
    "in_water": {
        "theta_s": "theta_s (deg)",
        "theta_v": "theta_v_water (deg)",
        "psi_bin": "psi bin (deg)",
        "bp_bin": "B_p bin (realisation mean, equal count over 5000)",
        "water_class": "water class",
    },
}
BAND_GROUP_NAME = "band group (nm)"


def breakdown_rows(res: Results) -> list[dict]:
    """The breakdown CSV: model × cut × side × primary band range × value."""
    rows = []
    for spec in res.registry:
        for cut in cuts_of(spec):
            if spec.fit == "geometry" and cut.key == "everything":
                continue
            values = breakdown_values(res, cut.pass_)
            for side_key in instance_sides(spec):
                side = SIDE[side_key]
                for br in instance_ranges(spec)[:1] if not spec.hybrid else ("le750",):
                    ids = id_cells(res.plan, spec, cut, side, br)
                    for kind in BREAKDOWNS:
                        for v, label in enumerate(values[kind]):
                            rows.append(
                                ids
                                | {
                                    "breakdown": BREAKDOWN_NAMES[cut.pass_][kind],
                                    "value": label,
                                }
                                | res.stats(
                                    spec, cut, side, brange=br, kind=kind, value=v
                                )
                            )
                    for lo, hi in BAND_GROUPS:
                        sel = np.flatnonzero(
                            (res.plan.wave >= lo) & (res.plan.wave <= hi)
                        )
                        if sel.size == 0:
                            continue
                        ood = spec.hybrid and lo > EMULATOR_WAVE_MAX
                        rows.append(
                            ids
                            | {"bands": BRANGE["gt750"].label if ood else f"{lo}-{hi}"}
                            | {"breakdown": BAND_GROUP_NAME, "value": f"{lo}-{hi}"}
                            | res.stats(spec, cut, side, bands=sel)
                        )
    return rows


# =========================================================== diagnostics =====
def table_fields(plan: Plan) -> dict:
    """The per-realisation table on the plan's realisations and bands, float64."""
    from robust.rt.data import pb24_cache

    cache = pb24_cache.open_cache(plan.cache_dir)
    names = (
        "aw", "aph", "ag", "aNAP", "bw", "bph", "bNAP", "bbw", "bbph", "bbNAP",
        "mu_d", "KLu", "Esdir_Es_ratio",
    )  # fmt: skip
    out = {
        n: np.asarray(cache.table(n, plan.ids), dtype=np.float64)[..., plan.band_idx]
        for n in names
    }
    out["theta_s"] = np.asarray(cache.grids["theta_s"], dtype=np.float64)
    out["a"] = out["aw"] + out["aph"] + out["ag"] + out["aNAP"]
    out["bbp"] = out["bbph"] + out["bbNAP"]
    out["bb"] = out["bbw"] + out["bbp"]
    out["bp"] = out["bph"] + out["bNAP"]
    out["bb_over_a"] = out["bb"] / out["a"]
    out["eta_bb"] = out["bbw"] / out["bb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["B_p"] = out["bbp"] / out["bp"]
    return out


def q(values, p) -> float:
    """``np.quantile`` that returns nan on an empty array."""
    values = np.asarray(values)
    return float(np.quantile(values, p)) if values.size else float("nan")


MU_D_METHODS = (
    ("tt15", "TT H (V = 15 km), L23 M*_d"),
    ("tt40", "TT H (V = 40 km, PB24's sky), L23 M*_d"),
    ("data", "data H, L23 M*_d"),
    ("data_c", "data H × c(θs′), L23 M*_d [= ZTT (L23 µ∞) on PB24]"),
    ("a2_data", "data H, Table A2 M*_d [= ZTT (TT2017 µ∞) on PB24]"),
)
MU_D = dict(MU_D_METHODS)


def mu_d_models(T: dict) -> dict:
    """Each µd method on (realisation, θs, λ), float64."""
    from robust.rt import ztt as Z

    ts = T["theta_s"][None, :, None]
    X = T["bb_over_a"][:, None, :]
    eta = T["eta_bb"][:, None, :]
    H = 1.0 - T["Esdir_Es_ratio"]
    l23 = {
        "md_star_coeffs": Z.MD_STAR_L23_COEFFS,
        "box": Z.ZTT_L23_BOX,
        "md_star_range": Z.MD_STAR_L23_RANGE,
    }
    with float64():
        return {
            "tt15": np.asarray(Z.mu_d(ts, X, eta, 15.0, **l23)),
            "tt40": np.asarray(Z.mu_d(ts, X, eta, 40.0, **l23)),
            "data": np.asarray(Z.mu_d(ts, X, eta, **l23, H=H)),
            "data_c": np.asarray(Z.mu_d_l23(ts, X, eta, H=H)),
            "a2_data": np.asarray(Z.mu_d(ts, X, eta, H=H)),
        }


MU_D_COLUMNS = (
    Column("sample"),
    Column("method"),
    Column("theta_s_deg"),
    Column("bands"),
    Column("bb_over_a"),
    Column("n", "d"),
    Column("median_signed_dev_pct", ".4f"),
    Column("median_abs_dev_pct", ".4f"),
    Column("q90_abs_dev_pct", ".4f"),
)


def mu_d_rows(plan: Plan, T: dict):
    """µd against PB24's ``mu_d``: per θs × band group, and per θs × bb/a bin."""
    models = mu_d_models(T)
    truth = T["mu_d"]
    wave = plan.wave
    groups = [("350-800", np.ones(wave.size, bool))] + [
        (f"{lo}-{hi}", (wave >= lo) & (wave <= hi)) for lo, hi in BAND_GROUPS
    ]
    bba = T["bb_over_a"]
    bba_bins = [
        (f"[{lo:g}, {hi:g})", (bba >= lo) & (bba < hi))
        for lo, hi in zip(MU_D_BBA_EDGES[:-1], MU_D_BBA_EDGES[1:], strict=True)
    ]
    rows, per_band = [], []
    for key, label in MU_D_METHODS:
        dev = models[key] / truth - 1.0
        for i, ts in enumerate(T["theta_s"]):
            d_ts = dev[:, i, :]
            for g_label, g in groups:
                if not g.any():
                    continue
                v = d_ts[:, g].ravel()
                rows.append(_mu_d_row(plan, label, ts, g_label, "all", v))
            for b_label, b in bba_bins:
                rows.append(_mu_d_row(plan, label, ts, "350-800", b_label, d_ts[b]))
        window_suns = T["theta_s"] <= 70.0
        d_w = dev[:, window_suns, :]
        rows.append(_mu_d_row(plan, label, WINDOW_SUNS, "350-800", "all", d_w.ravel()))
        for b_label, b in bba_bins:
            m = np.broadcast_to(b[:, None, :], d_w.shape)
            rows.append(_mu_d_row(plan, label, WINDOW_SUNS, "350-800", b_label, d_w[m]))
        per_band.append(np.median(dev, axis=0))  # (θs, λ)
    band_cols = (
        Column("sample"),
        Column("theta_s_deg", "g"),
        Column("wavelength_nm", ".0f"),
        *(
            Column(f"median_signed_dev_pct [{label}]", ".4f")
            for _, label in MU_D_METHODS
        ),
    )
    band_out = []
    for i, ts in enumerate(T["theta_s"]):
        for j, w in enumerate(wave):
            row = {"sample": plan.tag, "theta_s_deg": ts, "wavelength_nm": w}
            for (_, label), med in zip(MU_D_METHODS, per_band, strict=True):
                row[f"median_signed_dev_pct [{label}]"] = 100.0 * med[i, j]
            band_out.append(row)
    return rows, band_cols, band_out


#: The pooled-sun label of the µd rows.
WINDOW_SUNS = "0-70 pooled"


def _mu_d_row(plan, label, ts, bands, bba, v) -> dict:
    return {
        "sample": plan.tag,
        "method": label,
        "theta_s_deg": ts if isinstance(ts, str) else f"{ts:g}",
        "bands": bands,
        "bb_over_a": bba,
        "n": v.size,
        "median_signed_dev_pct": 100.0 * q(v, 0.5),
        "median_abs_dev_pct": 100.0 * q(np.abs(v), 0.5),
        "q90_abs_dev_pct": 100.0 * q(np.abs(v), 0.9),
    }


KLU_COLUMNS = (
    Column("sample"),
    Column("mu_inf"),
    Column("theta_s_deg", "g"),
    Column("psi_deg", ".2f"),
    Column("n", "d"),
    Column("n_klu_nonpositive", "d"),
    Column("ratio_q16", ".4f"),
    Column("ratio_median", ".4f"),
    Column("ratio_q84", ".4f"),
)
ORACLE_COLUMNS = (
    Column("sample"),
    Column("model"),
    Column("theta_s_deg"),
    Column("side"),
    Column("n", "d"),
    Column("median_abs_rel_err_pct", ".4f"),
    Column("q90_abs_rel_err_pct", ".4f"),
    Column("rrms_pct", ".6g"),
    Column("nonphysical_pct", ".4f"),
)
ORACLE = "ZTT (L23 µ∞), PB24 FF Pbb + oracle KLu/a (Ψ_KLu/µ∞ := KLu/a)"


def nadir_containers(plan: Plan, T: dict):
    """IOPs/PhaseParams/Geometry of the nadir samples (realisation × θs),
    built as the loader builds them, and PB24's nadir ``rrs``."""
    import jax.numpy as jnp

    from robust.rt.data import pb24 as P
    from robust.rt.data import pb24_cache
    from robust.rt.types import Geometry, IOPs, PhaseParams

    cache = pb24_cache.open_cache(plan.cache_dir)
    rrs = np.asarray(cache.four_d("rrs", plan.ids, theta=0, phi=0, bands=plan.band_idx))
    n, nts, W = rrs.shape
    raw = {
        k: np.asarray(cache.table(k, plan.ids))[..., plan.band_idx]
        for k in ("aw", "aph", "ag", "aNAP", "bbw", "bph", "bNAP", "bbph", "bbNAP")
    }
    ff_B, ff_w = P._ff_mixture(raw)  # the loader's own FF fill (bNAP = 0 case)

    def rep(x):
        x = np.asarray(x, dtype=np.float32)
        return jnp.asarray(np.repeat(x, nts, axis=0))

    bbp = raw["bbph"] + raw["bbNAP"]
    iops = IOPs(
        a=rep(raw["aw"] + raw["aph"] + raw["ag"] + raw["aNAP"]),
        bb_w=rep(raw["bbw"]),
        bb_p=rep(bbp),
        a_ph=rep(raw["aph"]),
    )
    phase = PhaseParams(
        B_p=rep(bbp / (raw["bph"] + raw["bNAP"])), ff_B=rep(ff_B), ff_weight=rep(ff_w)
    )
    df = 1.0 - np.asarray(cache.table("Esdir_Es_ratio", plan.ids))[..., plan.band_idx]
    geom = Geometry(
        theta_s=jnp.asarray(np.tile(T["theta_s"], n).astype(np.float32)),
        theta_v=jnp.zeros(n * nts, dtype=jnp.float32),
        dphi=jnp.zeros(n * nts, dtype=jnp.float32),
        diffuse_fraction=jnp.asarray(df.reshape(n * nts, W).astype(np.float32)),
    )
    return iops, phase, geom, rrs.reshape(n * nts, W)


def klu_rows(plan: Plan, T: dict, eigen):
    """Ψ_KLu/µ∞ against KLu/a at nadir, and the oracle rung in rrs."""
    import jax.numpy as jnp

    from robust.rt import ztt as Z

    ts = T["theta_s"]
    with float64():
        psi = 180.0 - np.asarray(Z.in_water_zenith(ts))
        Psi = np.asarray(Z.psi_KLu(psi))
        X, eta = T["bb_over_a"], T["eta_bb"]
        mus = {
            "TT2017 (Eq. 8 stand-in)": np.asarray(Z.mu_infinity_tt2017(X, eta)),
            "L23 (Eq. 8 refit)": np.asarray(Z.mu_infinity_l23(X, eta)),
            "L23 closure (a ceiling)": np.asarray(Z.mu_infinity_l23_closure(X, eta)),
        }
    if eigen is not None:
        mus["eigen (PB24 IOPs + FF pair)"] = eigen
    klu_a = T["KLu"] / T["a"][:, None, :]
    rows = []
    for label, mu in mus.items():
        ratio = (Psi[None, :, None] / mu[:, None, :]) / klu_a
        for i, t in enumerate(ts):
            r = ratio[:, i, :]
            good = T["KLu"][:, i, :] > 0
            rows.append(
                {
                    "sample": plan.tag,
                    "mu_inf": label,
                    "theta_s_deg": t,
                    "psi_deg": psi[i],
                    "n": int(good.sum()),
                    "n_klu_nonpositive": int((~good).sum()),
                    "ratio_q16": q(r[good], 0.16),
                    "ratio_median": q(r[good], 0.5),
                    "ratio_q84": q(r[good], 0.84),
                }
            )

    iops, phase, geom, truth = nadir_containers(plan, T)
    wave = jnp.asarray(plan.wave, dtype=jnp.float32)
    base = np.asarray(Z.rrs_ZTT(iops, phase, geom, wave))
    psi32 = np.asarray(Z.geometry_to_paper_angles(geom)[3])
    klu = T["KLu"].reshape(-1, plan.wave.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_oracle = np.asarray(Z.psi_KLu(psi32))[:, None] * np.asarray(iops.a) / klu
    oracle = np.asarray(
        Z.rrs_ZTT(
            iops, phase, geom, wave, mu_inf=jnp.asarray(mu_oracle, dtype=jnp.float32)
        )
    )
    nts = ts.size
    ts_s = np.tile(ts, plan.ids.size)
    held = np.repeat(plan.rcat % 2 == 1, nts)
    out = []
    for name, pred in (
        (ZTT_LABELS[("l23", True)] + " (nadir)", base),
        (ORACLE, oracle),
    ):
        e = pred.astype(np.float64) / truth.astype(np.float64) - 1.0
        for side, smask in (
            ("all realisations", np.ones_like(held)),
            ("realisation held out", held),
        ):
            for t_label, tmask in [("all", np.ones_like(held))] + [
                (f"{t:g}", ts_s == t) for t in ts
            ]:
                v = e[smask & tmask]
                fin = np.isfinite(v)
                out.append(
                    {
                        "sample": plan.tag,
                        "model": name,
                        "theta_s_deg": t_label,
                        "side": side,
                        "n": v.size,
                        "median_abs_rel_err_pct": 100.0
                        * q(np.where(fin, np.abs(v), np.inf), 0.5),
                        "q90_abs_rel_err_pct": 100.0
                        * q(np.where(fin, np.abs(v), np.inf), 0.9),
                        "rrms_pct": 100.0 * float(np.sqrt(np.mean(v[fin] ** 2)))
                        if fin.any()
                        else float("nan"),
                        "nonphysical_pct": 100.0
                        * float(
                            np.mean(
                                (pred[smask & tmask] <= 0)
                                | ~np.isfinite(pred[smask & tmask])
                            )
                        )
                        if v.size
                        else float("nan"),
                    }
                )
    return rows, out


BBW_COLUMNS = (
    Column("sample"),
    Column("wavelength_nm", ".0f"),
    Column("bbw_file_median", ".6e"),
    Column("bbw_file_min", ".6e"),
    Column("bbw_file_max", ".6e"),
    Column("bb_w_conventions_clamp", ".6e"),
    Column("bb_w_conventions_extrapolate", ".6e"),
    Column("file_over_conventions_clamp", ".5f"),
    Column("file_over_conventions_extrapolate", ".5f"),
    Column("bbw_over_half_bw_min", ".5f"),
    Column("bbw_over_half_bw_max", ".5f"),
)


def bbw_rows(plan: Plan, T: dict) -> list[dict]:
    """The files' ``bbw`` against ``conventions.bb_w`` per λ."""
    from robust.rt import conventions as C

    with float64():
        clamp = np.asarray(C.bb_w(plan.wave, mode="clamp"))
        extra = np.asarray(C.bb_w(plan.wave, mode="extrapolate"))
    bbw, half = T["bbw"], T["bw"] / 2.0
    rows = []
    for j, w in enumerate(plan.wave):
        med = float(np.median(bbw[:, j]))
        r = bbw[:, j] / half[:, j]
        rows.append(
            {
                "sample": plan.tag,
                "wavelength_nm": w,
                "bbw_file_median": med,
                "bbw_file_min": float(bbw[:, j].min()),
                "bbw_file_max": float(bbw[:, j].max()),
                "bb_w_conventions_clamp": clamp[j],
                "bb_w_conventions_extrapolate": extra[j],
                "file_over_conventions_clamp": med / clamp[j],
                "file_over_conventions_extrapolate": med / extra[j],
                "bbw_over_half_bw_min": float(r.min()),
                "bbw_over_half_bw_max": float(r.max()),
            }
        )
    return rows


PSI_COLUMNS = (
    Column("sample"),
    Column("cut"),
    Column("statistic"),
    Column("value", ".4f"),
)


def psi_rows(plan: Plan, gts: dict) -> list[dict]:
    """ψ in each cut (every realisation shares the geometries, so the share
    of geometries is the share of samples), and a 5° histogram of the window."""
    rows = []
    for cut in CUTS:
        gt = gts[cut.pass_]
        psi = gt.psi[np.isin(gt.gcat, cut.gcats)]
        stats = {
            "n_geometries": psi.size,
            "min_deg": psi.min(),
            "q01_deg": np.quantile(psi, 0.01),
            "median_deg": np.median(psi),
            "max_deg": psi.max(),
            f"share_below_{PSI_SIGN_FLIP:g}_pct": 100.0 * np.mean(psi < PSI_SIGN_FLIP),
            f"share_below_{PSI_FITTED_MIN:g}_pct": 100.0
            * np.mean(psi < PSI_FITTED_MIN),
        }
        for k, v in stats.items():
            rows.append(
                {"sample": plan.tag, "cut": cut.label, "statistic": k, "value": v}
            )
    window = gts["air"].psi[gts["air"].gcat < 2]
    for lo in range(80, 180, 5):
        rows.append(
            {
                "sample": plan.tag,
                "cut": CUT["window"].label,
                "statistic": f"geometries in [{lo}, {lo + 5})",
                "value": float(np.sum((window >= lo) & (window < lo + 5))),
            }
        )
    return rows


# ================================================================ eigen µ∞ ===
def ff_pair(T: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-realisation FF backscatter ratios (B_ph, B_NAP), float64.

    PB24's are constant in λ within a realisation (Pitarch & Brando 2025
    §1.1); the files' ratios vary by float32 rounding only, so the λ-mean is
    taken. A realisation with ``bNAP = 0`` everywhere takes the loader's fill,
    which carries zero weight. Also returns the largest relative λ-variation.
    """
    from robust.rt.data import pb24 as P

    B_ph_l = T["bbph"] / T["bph"]
    with np.errstate(divide="ignore", invalid="ignore"):
        B_nap_l = np.where(T["bNAP"] > 0, T["bbNAP"] / T["bNAP"], np.nan)
    B_ph = B_ph_l.mean(axis=1)
    has = np.isfinite(B_nap_l).any(axis=1)
    B_nap = np.where(
        has, np.nanmean(np.where(has[:, None], B_nap_l, 1.0), axis=1), P.FF_B_NAP_FILL
    )
    var = max(
        float(np.max(np.abs(B_ph_l / B_ph[:, None] - 1.0))),
        float(np.nanmax(np.abs(B_nap_l / B_nap[:, None] - 1.0))) if has.any() else 0.0,
    )
    return B_ph, B_nap, var


def _eigen_task(task):
    """Worker: µ∞ of a block of realisations with interpolated (or exact) kernels."""
    from robust.rt import asymptotic as A

    a, b, B_ph, B_nap, grid, water, mu, w, exact = task
    out = np.empty_like(a)
    for r in range(a.shape[0]):
        if exact:
            k_ph = A.build_kernel(A.ff_phase_numpy(B_ph[r]), mu, w)
            k_nap = A.build_kernel(A.ff_phase_numpy(B_nap[r]), mu, w)
        else:
            k_ph, k_nap = grid.at(B_ph[r]), grid.at(B_nap[r])
        out[r] = A.mu_infinity_eigen(a[r], b[r], mu, w, (water, k_ph, k_nap))
    return out


def build_eigen(plan: Plan, T: dict, path: Path, workers: int) -> dict:
    """Solve the eigen µ∞ on every (realisation, λ) and save the product."""
    from robust.rt import asymptotic as A
    from robust.rt.data import pb24_cache, pb24_eigen

    t0 = time.time()
    lo, hi, n_node = EIGEN_B_GRID
    B_grid = np.geomspace(lo, hi, n_node)
    B_ph, B_nap, lam_var = ff_pair(T)
    if B_ph.min() < lo or max(B_ph.max(), B_nap.max()) > hi or B_nap.min() < lo:
        raise SystemExit("eigen: a realisation's B lies outside the kernel grid")
    mu, w = A.double_gauss()
    water = A.build_kernel(A.water_phase(), mu, w)
    a = T["a"]
    b = np.stack([T["bw"], T["bph"], T["bNAP"]], axis=-1)
    n = plan.ids.size
    rng = np.random.default_rng(EIGEN_CHECK_SEED)
    check = np.sort(rng.choice(n, size=min(EIGEN_CHECK_N, n), replace=False))
    with pool(workers) as ex:
        mapper = ex.map if ex else map
        grid = A.ff_kernel_grid(B_grid, mu, w, mapper=mapper)
        t_kernels = time.time() - t0
        blocks = [np.arange(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]
        tasks = [
            (a[s], b[s], B_ph[s], B_nap[s], grid, water, mu, w, False) for s in blocks
        ]
        mu_inf = np.concatenate(list(mapper(_eigen_task, tasks)))
        t_solve = time.time() - t0 - t_kernels
        cblocks = [check[i : i + 2] for i in range(0, check.size, 2)]
        ctasks = [
            (a[s], b[s], B_ph[s], B_nap[s], None, water, mu, w, True) for s in cblocks
        ]
        exact = np.concatenate(list(mapper(_eigen_task, ctasks)))
    dev = np.abs(mu_inf[check] / exact - 1.0)
    sidecar = pb24_cache.read_sidecar(plan.cache_dir)
    digest = hashlib.sha256(json.dumps(sidecar, sort_keys=True).encode()).hexdigest()
    provenance = {
        "sample": plan.tag,
        "source": "PB24 hyperspectral cache pb24_v1 table: a = aw+aph+ag+aNAP; "
        "b = (bw, bph, bNAP); FF B per realisation = lambda-mean of bbph/bph and "
        "bbNAP/bNAP",
        "cache_sidecar_sha256": digest,
        "n_realisations": int(n),
        "n_bands": int(plan.wave.size),
        "n_solves": int(n * plan.wave.size),
        "kernel_B_grid": {
            "lo": lo,
            "hi": hi,
            "n_nodes": n_node,
            "spacing": "geometric",
            "interpolation": "linear in log B (matrix and truncated fraction)",
        },
        "n_streams": A.N_STREAMS,
        "truncation_deg": A.TRUNCATION_DEG,
        "n_phi": A.N_PHI,
        "water_phase": f"Rayleigh-like, depolarization {A._ztt.WATER_DEPOLARIZATION}",
        "ff_B_lambda_variation_max_rel": lam_var,
        "interpolation_check": {
            "realisations": [int(i) for i in plan.ids[check]],
            "seed": EIGEN_CHECK_SEED,
            "n_solves": int(dev.size),
            "abs_rel_dev_median": float(np.median(dev)),
            "abs_rel_dev_q99": float(np.quantile(dev, 0.99)),
            "abs_rel_dev_max": float(dev.max()),
        },
        "systematic": "the eigen µ∞ read ~1.3% high against L23's asymptotic profiles "
        "(record §7b.5/§7b.7); PB24 has no profiles to check it",
    }
    pb24_eigen.save_mu_inf_eigen(
        path,
        realisations=plan.ids,
        wave=plan.wave,
        mu_inf=mu_inf,
        B_ph=B_ph,
        B_nap=B_nap,
        provenance=provenance,
    )
    print(
        f"eigen: {n * plan.wave.size} solves; kernels {t_kernels:.1f} s, solves "
        f"{t_solve:.1f} s, check {time.time() - t0 - t_kernels - t_solve:.1f} s; "
        f"interp |dev| median {np.median(dev):.2e}, max {dev.max():.2e}; {path} "
        f"({path.stat().st_size:,} B)"
    )
    return provenance


#: The eigen CSV's B_p-span row label.
SPAN_ITEM = (
    f"span of mu_inf over B_p {Q15_BP_RANGE[0]}-{Q15_BP_RANGE[1]} (|slope| x ln ratio)"
)

EIGEN_COLUMNS = (
    Column("sample"),
    Column("section"),
    Column("item"),
    Column("group"),
    Column("n", "d"),
    Column("value_1", ".6g"),
    Column("value_2", ".6g"),
    Column("value_3", ".6g"),
    Column("meaning"),
)


def eq8_design(X: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Eq. (8)'s sixteen terms: X^i L^j, i, j in 0..3 (X = log10 bb/a,
    L = log10 η_bb)."""
    return np.stack([X**i * L**j for i in range(4) for j in range(4)], axis=-1)


def eigen_rows(plan: Plan, T: dict, product) -> list[dict]:
    """Eigen µ∞ diagnostics (rows of :data:`EIGEN_COLUMNS`)."""
    from robust.rt import ztt as Z

    rows = []

    def add(section, item, group, n, v1, v2=float("nan"), v3=float("nan"), meaning=""):
        rows.append(
            {
                "sample": plan.tag,
                "section": section,
                "item": item,
                "group": group,
                "n": n,
                "value_1": v1,
                "value_2": v2,
                "value_3": v3,
                "meaning": meaning,
            }
        )

    pv = product.provenance
    ic = pv["interpolation_check"]
    add(
        "recipe",
        "solves",
        "all",
        pv["n_solves"],
        pv["n_realisations"],
        pv["n_bands"],
        meaning="value_1 realisations, value_2 bands",
    )
    g = pv["kernel_B_grid"]
    add(
        "recipe",
        "kernel B grid",
        g["interpolation"],
        g["n_nodes"],
        g["lo"],
        g["hi"],
        meaning="n nodes; value_1 lo, value_2 hi (geometric)",
    )
    add(
        "recipe",
        "FF B lambda variation",
        "max relative",
        0,
        pv["ff_B_lambda_variation_max_rel"],
    )
    add(
        "interpolation error",
        "|mu_interp/mu_exact - 1|",
        f"{len(ic['realisations'])} realisations, seed {ic['seed']}",
        ic["n_solves"],
        ic["abs_rel_dev_median"],
        ic["abs_rel_dev_q99"],
        ic["abs_rel_dev_max"],
        "value_1 median, value_2 q99, value_3 max",
    )

    rows_idx = product.rows(plan.ids)
    wpos = np.searchsorted(product.wave, plan.wave)
    mu_e = product.mu_inf[np.ix_(rows_idx, wpos)]
    X10, eta = T["bb_over_a"], T["eta_bb"]
    Bp = T["B_p"]
    with float64():
        variants = {
            "TT2017": np.asarray(Z.mu_infinity_tt2017(X10, eta)),
            "L23": np.asarray(Z.mu_infinity_l23(X10, eta)),
            "L23 closure": np.asarray(Z.mu_infinity_l23_closure(X10, eta)),
        }
    add(
        "eigen",
        "mu_inf range",
        "all",
        mu_e.size,
        float(mu_e.min()),
        float(np.median(mu_e)),
        float(mu_e.max()),
        "min, median, max",
    )
    bba_edges = (0.0, 1e-3, 1e-2, 0.1, 0.5946, np.inf)
    bp_edges = (0.0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, np.inf)
    for name, mu_v in variants.items():
        r = mu_v / mu_e - 1.0
        add(
            "variant vs eigen",
            name,
            "all",
            r.size,
            100 * q(r, 0.5),
            100 * q(np.abs(r), 0.5),
            100 * q(np.abs(r), 0.9),
            "% : median signed, median |.|, q90 |.|",
        )
        for lo, hi in zip(bba_edges[:-1], bba_edges[1:], strict=True):
            m = (X10 >= lo) & (X10 < hi)
            add(
                "variant vs eigen",
                name,
                f"bb/a [{lo:g}, {hi:g})",
                int(m.sum()),
                100 * q(r[m], 0.5),
                100 * q(np.abs(r[m]), 0.5),
                100 * q(np.abs(r[m]), 0.9),
                "% : median signed, median |.|, q90 |.|",
            )
        for lo, hi in zip(bp_edges[:-1], bp_edges[1:], strict=True):
            m = (Bp >= lo) & (Bp < hi)
            add(
                "variant vs eigen",
                name,
                f"B_p [{lo:g}, {hi:g})",
                int(m.sum()),
                100 * q(r[m], 0.5),
                100 * q(np.abs(r[m]), 0.5),
                100 * q(np.abs(r[m]), 0.9),
                "% : median signed, median |.|, q90 |.|",
            )

    # (1) Eq. (8)'s two-variable form refit to the eigen µ∞ (diagnostic only):
    # what it cannot absorb, by B_p.
    X, L = np.log10(X10).ravel(), np.log10(eta).ravel()
    y = mu_e.ravel()
    A_ = eq8_design(X, L)
    beta, *_ = np.linalg.lstsq(A_, y, rcond=None)
    resid = A_ @ beta / y - 1.0
    add(
        "Eq. (8) form refit to eigen",
        "residual (fit/eigen - 1)",
        "all",
        y.size,
        100 * q(resid, 0.5),
        100 * q(np.abs(resid), 0.5),
        100 * float(np.sqrt(np.mean(resid**2))),
        "% : median signed, median |.|, rms",
    )
    bpf = Bp.ravel()
    for lo, hi in zip(bp_edges[:-1], bp_edges[1:], strict=True):
        m = (bpf >= lo) & (bpf < hi)
        add(
            "Eq. (8) form refit to eigen",
            "residual by B_p",
            f"B_p [{lo:g}, {hi:g})",
            int(m.sum()),
            100 * q(resid[m], 0.16),
            100 * q(resid[m], 0.5),
            100 * q(resid[m], 0.84),
            "% : q16, median, q84",
        )

    # (2) Within (log bb/a, log η_bb) bins: the slope of ln µ∞ on ln B_p after
    # the bin's own linear trend in X and L, and the span it implies over
    # Q15's B_p range.
    xb = np.floor(X / EIGEN_BIN_DEX).astype(int)
    lb = np.floor(L / EIGEN_BIN_DEX).astype(int)
    key = (xb - xb.min()) * (lb.max() - lb.min() + 1) + (lb - lb.min())
    order = np.argsort(key, kind="stable")
    ks, starts = np.unique(key[order], return_index=True)
    ends = np.append(starts[1:], key.size)
    span_ln = np.log(Q15_BP_RANGE[1] / Q15_BP_RANGE[0])
    lnmu, lnbp = np.log(y), np.log(bpf)
    att, raw_sd, xl_sd, full_sd, wts = [], [], [], [], []
    n_points = 0
    for s0, s1 in zip(starts, ends, strict=True):
        idx = order[s0:s1]
        if idx.size < EIGEN_BIN_MIN or np.exp(np.ptp(lnbp[idx])) < EIGEN_BIN_BP_SPAN:
            continue
        D = np.stack(
            [
                np.ones(idx.size),
                X[idx] - X[idx].mean(),
                L[idx] - L[idx].mean(),
                lnbp[idx] - lnbp[idx].mean(),
            ],
            axis=1,
        )
        c, *_ = np.linalg.lstsq(D, lnmu[idx], rcond=None)
        c2, *_ = np.linalg.lstsq(D[:, :3], lnmu[idx], rcond=None)
        att.append(abs(c[3]) * span_ln * 100.0)
        raw_sd.append(np.std(lnmu[idx]) * 100.0)
        xl_sd.append(np.std(lnmu[idx] - D[:, :3] @ c2) * 100.0)
        full_sd.append(np.std(lnmu[idx] - D @ c) * 100.0)
        wts.append(idx.size)
        n_points += idx.size
    att, xl_sd, full_sd, wts = map(np.asarray, (att, xl_sd, full_sd, wts))

    def wq(v, p):
        if v.size == 0:
            return float("nan")
        o = np.argsort(v)
        c = np.cumsum(wts[o]) / wts.sum()
        return float(v[o][np.searchsorted(c, p)])

    add(
        "B_p at fixed (bb/a, eta_bb)",
        "bins analysed",
        f"{EIGEN_BIN_DEX} dex bins, >= {EIGEN_BIN_MIN} points, B_p span >= "
        f"{EIGEN_BIN_BP_SPAN:g}x",
        n_points,
        len(wts),
        100.0 * n_points / y.size,
        meaning="n points; value_1 bins, value_2 % of all points",
    )
    add(
        "B_p at fixed (bb/a, eta_bb)",
        SPAN_ITEM,
        "point-weighted over bins",
        n_points,
        wq(att, 0.5),
        wq(att, 0.1),
        wq(att, 0.9),
        "% : median, q10, q90",
    )
    add(
        "B_p at fixed (bb/a, eta_bb)",
        "within-bin sd of ln mu_inf after the X, L trend",
        "point-weighted over bins",
        n_points,
        wq(xl_sd, 0.5),
        wq(xl_sd, 0.9),
        float("nan"),
        "% : median, q90",
    )
    add(
        "B_p at fixed (bb/a, eta_bb)",
        "within-bin sd after X, L and ln B_p",
        "point-weighted over bins",
        n_points,
        wq(full_sd, 0.5),
        wq(full_sd, 0.9),
        float("nan"),
        "% : median, q90",
    )
    add(
        "B_p at fixed (bb/a, eta_bb)",
        "share of points where the B_p span exceeds",
        "1% / 2% / 5%",
        n_points,
        100 * float(wts[att > 1].sum() / max(wts.sum(), 1)),
        100 * float(wts[att > 2].sum() / max(wts.sum(), 1)),
        100 * float(wts[att > 5].sum() / max(wts.sum(), 1)),
        "% of analysed points",
    )
    return rows


# ============================================================== markdown =====
def fnum(cell: str, digits: int = 2) -> str:
    """A CSV number for a markdown cell."""
    try:
        v = float(cell)
    except ValueError:
        return cell
    if np.isnan(v):
        return "–"
    if np.isinf(v):
        return "∞"
    if abs(v) >= 1e4:
        return f"{v:.3g}"
    if 0 < abs(v) < 10.0 ** (-digits):
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """A markdown table."""
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def pick(rows, **want) -> list[dict]:
    """Rows whose cells equal ``want``."""
    return [r for r in rows if all(r[k] == v for k, v in want.items())]


def one(rows, **want) -> dict | None:
    """The single row matching ``want`` (None if absent)."""
    hit = pick(rows, **want)
    if len(hit) > 1:
        raise RuntimeError(f"ambiguous selection {want}: {len(hit)} rows")
    return hit[0] if hit else None


#: Predictions, written here before any number of this run existed.
PRED_Q9D = (
    "**Prediction (Q9d, written before task 2 measured it; restated before this "
    "run).** With θv read as above-water and refracted, the minimum ψ in the "
    "window is ≈ 91° (criterion: 89–93°), so Ψ_KLu's sign flip below 110.4° "
    "stays in scope on a small share, and ZTT's non-physical share (values ≤ 0 "
    "or non-finite) in the window falls **far below M5's 22%** (criterion: "
    "below 11%, half of it, for ZTT (TT2017 µ∞), Sullivan Pbb — M5's model)."
)
PRED_Q10 = (
    "**Prediction (Q10).** Feeding µd PB24's own diffuse fraction removes most "
    "of the θs trend of µd's error (criterion: the span over θs = 0–70° of the "
    "median signed deviation, all bands, shrinks to less than half of TT's H at "
    "V = 15 km and at V = 40 km); what remains is M*_d's bb/a extrapolation "
    "(criterion: with the data H, the median |deviation| at bb/a ≥ 0.59, beyond "
    "anything L23 held, is at least twice that at bb/a < 0.1)."
)
PRED_KLU = (
    "**Prediction (from M7 on L23, written before this run).** At nadir ZTT's "
    "Ψ_KLu/µ∞(L23) sits ~10% above PB24's KLu/a at small θs, as on L23 (11–13%; "
    "criterion: median ratio 1.05–1.20 at θs ≤ 30°), and the oracle rung "
    "(Ψ_KLu/µ∞ := KLu/a) **worsens** rrs, as it did on L23 (2.90% → 9.54%; "
    "criterion: the oracle's median error at nadir, realisation held out, "
    "exceeds the L23 variant's)."
)
PRED_Q15 = (
    "**Prediction (Q15).** At fixed (bb/a, η_bb) the eigen µ∞ moves by several "
    f"per cent across B_p {Q15_BP_RANGE[0]}–{Q15_BP_RANGE[1]}, well above "
    f"the {L23_EQ8_FIT_ERROR_PCT}% fit error of Eq. (8) on L23 (criterion: the "
    "point-weighted median of the within-bin B_p span ≥ 2%, i.e. ≈ 10× the "
    "L23 fit error)."
)


def verdict(ok: bool | None) -> str:
    """The outcome word."""
    return (
        "**not testable on this sample**"
        if ok is None
        else ("**confirmed**" if ok else "**not confirmed**")
    )


def primary_bands(row: dict) -> bool:
    """A row's band range is its model's primary one: 350-800, or for the
    hybrid 350-750 plus its separate out-of-domain 751-800 row."""
    if row["model"] in (HYBRID, HYBRID_FB):
        return row["bands"] in (BRANGE["le750"].label, BRANGE["gt750"].label)
    return row["bands"] == BRANGE["all"].label


def headline_section(scores, plan) -> list[str]:
    """The headline tables."""
    lines = []
    stat_heads = [
        "median %",
        "q90 %",
        "rRMS %",
        "rRMS, physical only %",
        "non-physical %",
        "spectra with a non-physical value %",
        "n values",
    ]
    stat_keys = [
        "median_abs_rel_err_pct",
        "q90_abs_rel_err_pct",
        "rrms_pct",
        "rrms_physical_only_pct",
        "nonphysical_pct",
        "spectra_nonphysical_pct",
        "n_values",
    ]
    for table_name, title in (
        ("native", "native space"),
        ("common", "the common `rrs` table"),
    ):
        for cut_key in ("window", "everything"):
            rows = []
            for r in scores:
                if table_name not in r["tables"].split("+"):
                    continue
                if (
                    r["cut"] != CUT[cut_key].label
                    or r["split"] != "realisation"
                    or r["side"] != "held out"
                ):
                    continue
                if not primary_bands(r):
                    continue
                rows.append(
                    [
                        r["model"],
                        r["space"],
                        r["bands"],
                        *(fnum(r[k], 0 if k == "n_values" else 2) for k in stat_keys),
                    ]
                )
            lines += [
                f"### {CUT[cut_key].label}, realisation split held out — {title}",
                "",
                md_table(["model", "space", "bands", *stat_heads], rows),
                "",
            ]
    return lines


def wide_table(rows_csv, models, column_key, column_values, stat, **want) -> str:
    """Rows = models, columns = values of ``column_key``; headers are the
    values themselves (the same strings the CSV carries). A value no model
    has any data for (e.g. θs = 80° inside the window) is left out."""
    column_values = [
        v
        for v in column_values
        if any(
            r[column_key] == v and r["n_values"] != "0"
            for r in rows_csv
            if all(r[k] == w for k, w in want.items())
        )
    ]
    body = []
    for m in models:
        cells = []
        for v in column_values:
            hit = one(rows_csv, model=m, **{column_key: v}, **want)
            cells.append(fnum(hit[stat]) if hit else "")
        if any(cells):
            body.append([m, *cells])
    return md_table(["model", *column_values], body)


def write_markdown(out: Path, plan: Plan, registry, info: dict) -> None:
    """``baseline.md``, generated from the committed CSVs only."""
    scores = read_csv(out / f"{PREFIX}scores.csv")
    brk = read_csv(out / f"{PREFIX}breakdowns.csv")
    mud = read_csv(out / f"{PREFIX}mu_d.csv")
    klu = read_csv(out / f"{PREFIX}klu.csv")
    ora = read_csv(out / f"{PREFIX}klu_oracle.csv")
    psi = read_csv(out / f"{PREFIX}psi.csv")
    bbw = read_csv(out / f"{PREFIX}bbw.csv")
    eig_path = out / f"{PREFIX}eigen.csv"
    eig = read_csv(eig_path) if eig_path.is_file() else []
    labels = list(dict.fromkeys(s.label for s in registry))
    native = list(dict.fromkeys(s.label for s in registry if "native" in s.tables))
    common = list(dict.fromkeys(s.label for s in registry if "common" in s.tables))

    L = [
        "# M8 task 6 — the baseline: every elastic model we have, scored on PB24",
        "",
        "*Generated by `design/py/run_pb24_baseline.py` from the `baseline_*.csv` "
        "files "
        "beside it (every number below is read back from a CSV). This task measures; "
        "no model code or number changed.*",
        "",
        f"**Sample:** {plan.sample}; the `sample` cell of every CSV row is "
        f"`{plan.tag}`. Realisation split: {plan.n_real_test} held out; "
        f"B_p-band split: {plan.n_bp_test} held out (`design/ztt_pb24/splits.json`).",
        "",
        "## How the numbers are made",
        "",
        "- **Error:** `e = pred/true − 1` per value (sample × band). Median and q90 of "
        f"`|e|` come from a fixed histogram of log10|e| ({HIST_LO:g} to {HIST_HI:g} in "
        f"bins of {HIST_WIDTH} dex, log-linear inside a bin: "
        f"±{100 * (10 ** (HIST_WIDTH / 2) - 1):.2f}% "
        "of the quoted value at worst) over **all values**; rRMS and every count are "
        "exact float64 sums over **all values**. Streamed in chunks of "
        f"{CHUNK} realisations; the float sums are added in chunk order, so a rerun is "
        "bit-identical for any `--workers`.",
        "- **Nothing thresholded.** A non-physical prediction (≤ 0 or non-finite) "
        "stays "
        "in the median and q90 (non-finite as |e| = ∞) and is counted (`non-physical "
        "%`). "
        "rRMS is over the finite values; `rRMS, physical only` drops the non-physical "
        "ones "
        "and is labelled so. The only exclusion is a zero **truth** (`rrs` = 0 at the "
        "in-water 87.5° node), counted in `n_truth_zero_excluded`; the air sets have "
        "none.",
        "- **Spaces (Q12).** Gordon, ZTT and the hybrid in `rrs` at θ_w (PB24's `rrs` "
        "interpolated to the refracted angle, task 3); O25 in `Rrs` at θ_air. The "
        "common "
        "table puts O25 in `rrs` through the task-4 Fresnel transfer, with the "
        "downwelling "
        "factor (primary) and θv-only, beside the transfer's own error on PB24's true "
        "`Rrs`.",
        "- **µd.** Every ZTT and hybrid evaluation uses PB24's own diffuse fraction "
        "(`Geometry.diffuse_fraction`, filled by the loader), as the package does "
        "since task 4.",
        "- **Cuts.** window: θs and θv_air ≤ 70°; ψ ≥ 134°: inside the window; shell: "
        "θs or θv_air ≥ 80°; everything: window + shell; in-water-only: the in-water "
        "nodes "
        "50–87.5° (ZTT and Gordon only, `theta_v_in_water=True`). The geometry split's "
        "held-out side *is* the shell and its training side the window, so for every "
        "model "
        "not fitted on PB24 it is the `all realisations` row of those two cuts.",
        "- **Hybrid.** Scored on 350–750 nm. Its `wave_nm` feature stops at 750 nm, so "
        "751–800 nm is out of domain by construction and is reported in separate rows "
        f"(`bands` = '{BRANGE['gt750'].label}'), never inside a 350–800 number. Its "
        "out-of-domain mask is per sample over a call's bands, so the two ranges are "
        "separate calls.",
        "- **O25 refits** are float64 normal equations of `fit_o25_table`'s "
        "objectives, "
        "per (θs, θv, Δφ) node, on each split's training rows: the realisation and B_p "
        "splits' training realisations at all 1300 geometries; the geometry split's "
        "window (its shell nodes are clamped by the table, as `O25Table` does).",
        "",
        "## Models",
        "",
        md_table(
            ["model", "space", "tables", "fitted on"],
            [
                [
                    lab,
                    s.space,
                    "+".join(s.tables),
                    "; ".join(
                        dict.fromkeys(fitted_on(t) for t in registry if t.label == lab)
                    ),
                ]
                for lab in labels
                for s in [next(t for t in registry if t.label == lab)]
            ],
        ),
        "",
        "ZTT: all six (µ∞, Pbb) combinations exist through `rrs_ZTT`'s public inputs: "
        '`variant="tt2017"` is TT2017 µ∞ + Table A2 µd + Sullivan, `variant="l23"` '
        '(the package default) L23 µ∞ + L23 µd + FF, `mu_inf_model="l23_closure"` the '
        "closure µ∞ on the L23 variant; the crossed Pbb is passed as `P_bb=`. "
        "The hybrid is the shipped L23-trained emulator on its own TT2017 backbone.",
        "",
        "## Headline",
        "",
        *headline_section(scores, plan),
    ]

    # Every split, window and shell: median.
    def split_cell(model, split, side, cut_key):
        hits = [
            r
            for r in scores
            if r["model"] == model
            and r["split"] == split
            and r["side"] == side
            and r["cut"] == CUT[cut_key].label
            and r["bands"] in (BRANGE["all"].label, BRANGE["le750"].label)
        ]
        hits = [
            h
            for h in hits
            if not (
                h["bands"] == BRANGE["le750"].label
                and not model.startswith("L23-trained hybrid")
            )
        ]
        return fnum(hits[0]["median_abs_rel_err_pct"]) if hits else ""

    cols = [
        ("realisation", "held out", "window"),
        ("realisation", "train", "window"),
        ("bp_band", "held out", "window"),
        ("bp_band", "train", "window"),
        ("geometry", "held out", "shell"),
        ("geometry", "train", "window"),
    ]
    heads = [f"{sp} {sd} ({CUT[c].label.split(' (')[0]})" for sp, sd, c in cols]
    body = []
    for m in labels:
        cells = []
        for sp, sd, c in cols:
            if sp == "geometry":
                cell = split_cell(m, "geometry", sd, c) or split_cell(
                    m, "none", "all realisations", c
                )
            else:
                cell = split_cell(m, sp, sd, c)
            cells.append(cell)
        body.append([m, *cells])
    L += [
        "## Every split (median |e| %, primary bands)",
        "",
        "Geometry split for models not fitted on PB24: the all-realisation window "
        "(train) and shell (held out).",
        "",
        md_table(["model", *heads], body),
        "",
        "## Every cut (realisation split held out; median |e| % and non-physical %)",
        "",
    ]
    cut_heads = [c.label for c in CUTS]
    for stat, title in (
        ("median_abs_rel_err_pct", "median |e| %"),
        ("nonphysical_pct", "non-physical %"),
    ):
        body = []
        for m in labels:
            cells = []
            for c in CUTS:
                hit = [
                    r
                    for r in scores
                    if r["model"] == m
                    and r["cut"] == c.label
                    and r["split"] == "realisation"
                    and r["side"] == "held out"
                    and (
                        r["bands"] == BRANGE["all"].label
                        or (
                            m.startswith("L23-trained hybrid")
                            and r["bands"] == BRANGE["le750"].label
                        )
                    )
                ]
                cells.append(fnum(hit[0][stat]) if hit else "")
            body.append([m, *cells])
        L += [f"**{title}**", "", md_table(["model", *cut_heads], body), ""]

    L += ["## Breakdowns (window, realisation split held out, median |e| %)", ""]
    for name in (BAND_GROUP_NAME, *BREAKDOWN_NAMES["air"].values()):
        sub = [
            r
            for r in brk
            if r["breakdown"] == name
            and r["cut"] == CUT["window"].label
            and r["split"] == "realisation"
            and r["side"] == "held out"
        ]
        values = list(dict.fromkeys(r["value"] for r in sub))
        L += [
            f"### {name}",
            "",
            wide_table(
                sub,
                native + [m for m in common if m not in native],
                "value",
                values,
                "median_abs_rel_err_pct",
            ),
            "",
        ]
    sub = [
        r
        for r in brk
        if r["breakdown"] == BREAKDOWN_NAMES["in_water"]["theta_v"]
        and r["cut"] == CUT["in_water"].label
        and r["split"] == "realisation"
        and r["side"] == "held out"
    ]
    values = list(dict.fromkeys(r["value"] for r in sub))
    L += [
        "### in-water-only set, per in-water view zenith",
        "",
        wide_table(sub, labels, "value", values, "median_abs_rel_err_pct"),
        "",
    ]

    L += diagnostics_markdown(plan, scores, mud, klu, ora, psi, bbw, eig, info)
    L += [
        "## Files",
        "",
        "- `baseline_scores.csv`: model × cut × split side × band range (the "
        "headline).",
        "- `baseline_per_band.csv`: per band, on "
        + ", ".join(
            f"{CUT[c].label} / {SIDE[s].split} {SIDE[s].side}" for c, s in BAND_COMBOS
        )
        + ".",
        "- `baseline_breakdowns.csv`: per θs, θv, ψ bin, B_p bin, water class and "
        "band group.",
        "- `baseline_mu_d.csv`, `baseline_mu_d_per_band.csv`, `baseline_klu.csv`, "
        "`baseline_klu_oracle.csv`, `baseline_eigen.csv`, `baseline_psi.csv`, "
        "`baseline_bbw.csv`, "
        "`baseline_o25_refit.csv`: the diagnostics and the refit tables.",
        "- `baseline_fig_*.png`: figures drawn from the CSVs.",
        "",
    ]
    write_atomically(out / f"{PREFIX}report.md", "\n".join(L) + "\n")


def diagnostics_markdown(plan, scores, mud, klu, ora, psi, bbw, eig, info) -> list[str]:
    """The diagnostics, each prediction written before its numbers."""
    L = ["## Diagnostics", ""]

    # --- ψ and the non-physical share (Q9d) ---------------------------------
    L += [
        "### ψ in the window and the share of non-physical `rrs` (Q9d)",
        "",
        PRED_Q9D,
        "",
    ]
    stats = [r for r in psi if not r["statistic"].startswith("geometries in")]
    cuts = list(dict.fromkeys(r["cut"] for r in stats))
    names = list(dict.fromkeys(r["statistic"] for r in stats))
    L += [
        md_table(
            ["statistic", *cuts],
            [
                [n, *(fnum(one(stats, cut=c, statistic=n)["value"]) for c in cuts)]
                for n in names
            ],
        ),
        "",
    ]
    rows = []
    for m in dict.fromkeys(r["model"] for r in scores if r["space"] == "rrs"):
        cells = []
        for c in ("window", "psi134", "shell", "in_water"):
            hit = [
                r
                for r in scores
                if r["model"] == m
                and r["cut"] == CUT[c].label
                and r["split"] == "none"
                and (
                    r["bands"] == BRANGE["all"].label
                    or (
                        m.startswith("L23-trained hybrid")
                        and r["bands"] == BRANGE["le750"].label
                    )
                )
            ]
            cells += [
                fnum(hit[0]["nonphysical_pct"]) if hit else "",
                fnum(hit[0]["spectra_nonphysical_pct"]) if hit else "",
            ]
        rows.append([m, *cells])
    heads = [
        f"{CUT[c].label}: {k}"
        for c in ("window", "psi134", "shell", "in_water")
        for k in ("values %", "spectra %")
    ]
    L += [
        "Non-physical share, all realisations (values ≤ 0 or non-finite; spectra with "
        "at least one):",
        "",
        md_table(["model", *heads], rows),
        "",
    ]
    psi_min = float(one(stats, cut=CUT["window"].label, statistic="min_deg")["value"])
    m5 = one(
        scores,
        model=ZTT_LABELS[("tt2017", False)],
        cut=CUT["window"].label,
        split="none",
        bands=BRANGE["all"].label,
    )
    share = float(m5["nonphysical_pct"])
    fb = [
        r
        for r in scores
        if r["model"] == HYBRID_FB
        and r["cut"] == CUT["window"].label
        and r["split"] == "none"
        and r["bands"] == BRANGE["le750"].label
    ]
    L += [
        f"**Outcome.** Window ψ_min = {psi_min:.1f}°: "
        f"{verdict(89.0 <= psi_min <= 93.0)}. "
        f"ZTT (TT2017 µ∞), Sullivan Pbb is non-physical on {share:.2f}% of window "
        "values "
        f"against M5's {M5_NONPHYSICAL_PCT:g}%: "
        f"{verdict(share < M5_NONPHYSICAL_PCT / 2)}.",
        "",
    ]
    if fb:
        L += [
            "The hybrid's out-of-domain fallback fired on "
            f"{fnum(fb[0]['fallback_samples_pct'])}% of window samples "
            "(350–750 nm): the shipped emulator was trained at nadir, Δφ = 0 and B_p "
            "0.010–0.018, so "
            "almost every PB24 sample is outside its domain and the fallback is the "
            "TT2017 backbone.",
            "",
        ]

    # --- µd (Q10) -------------------------------------------------------------
    L += ["### µd three ways against PB24's `mu_d` (Q10)", "", PRED_Q10, ""]
    methods = list(dict.fromkeys(r["method"] for r in mud))
    suns = list(dict.fromkeys(r["theta_s_deg"] for r in mud))
    body = [
        [
            m,
            *(
                fnum(
                    one(mud, method=m, theta_s_deg=t, bands="350-800", bb_over_a="all")[
                        "median_signed_dev_pct"
                    ]
                )
                for t in suns
            ),
        ]
        for m in methods
    ]
    L += [
        "Median signed deviation µd_model/µd_PB24 − 1 (%), all bands, per θs (deg):",
        "",
        md_table(["method", *suns], body),
        "",
    ]
    bbas = list(dict.fromkeys(r["bb_over_a"] for r in mud if r["bb_over_a"] != "all"))
    body = [
        [
            m,
            *(
                fnum(
                    one(
                        mud,
                        method=m,
                        theta_s_deg=WINDOW_SUNS,
                        bands="350-800",
                        bb_over_a=b,
                    )["median_abs_dev_pct"]
                )
                for b in bbas
            ),
        ]
        for m in methods
    ]
    L += [
        "Median |deviation| (%), window suns pooled, per bb/a:",
        "",
        md_table(["method", *bbas], body),
        "",
    ]
    win = [t for t in suns if t != WINDOW_SUNS and float(t) <= 70.0]

    def span(method):
        v = [
            float(
                one(
                    mud, method=method, theta_s_deg=t, bands="350-800", bb_over_a="all"
                )["median_signed_dev_pct"]
            )
            for t in win
        ]
        return max(v) - min(v)

    s15, s40, sd, sdc = (
        span(MU_D["tt15"]),
        span(MU_D["tt40"]),
        span(MU_D["data"]),
        span(MU_D["data_c"]),
    )
    lo_b, hi_b = bbas[0], bbas[-1]
    d_lo = float(
        one(
            mud,
            method=MU_D["data"],
            theta_s_deg=WINDOW_SUNS,
            bands="350-800",
            bb_over_a=lo_b,
        )["median_abs_dev_pct"]
    )
    d_hi_row = one(
        mud,
        method=MU_D["data"],
        theta_s_deg=WINDOW_SUNS,
        bands="350-800",
        bb_over_a=hi_b,
    )
    d_hi = float(d_hi_row["median_abs_dev_pct"]) if int(d_hi_row["n"]) else float("nan")
    L += [
        "**Outcome.** Span over θs 0–70° of the median signed deviation: TT H V = 15 "
        "km "
        f"{s15:.2f} pp, V = 40 km {s40:.2f} pp, data H {sd:.2f} pp, data H × c(θs′) "
        f"{sdc:.2f} pp. "
        "The data H removes most of the θs trend: "
        f"{verdict(sd < 0.5 * s15 and sd < 0.5 * s40)}. "
        f"With the data H the median |deviation| is {d_lo:.2f}% at bb/a {lo_b} and "
        f"{fnum(str(d_hi))}% "
        f"at bb/a {hi_b}: the remainder is M*_d's bb/a extrapolation "
        f"{verdict(None if np.isnan(d_hi) else d_hi >= 2 * d_lo)}.",
        "",
    ]

    # --- Ψ_KLu / µ∞ against KLu/a (nadir) ------------------------------------
    L += ["### Ψ_KLu/µ∞ against KLu/a at nadir (ψ = 180° − θs_w)", "", PRED_KLU, ""]
    variants = list(dict.fromkeys(r["mu_inf"] for r in klu))
    suns = list(dict.fromkeys(r["theta_s_deg"] for r in klu))
    psis = [
        fnum(one(klu, mu_inf=variants[0], theta_s_deg=t)["psi_deg"], 1) for t in suns
    ]
    body = [
        [
            v,
            *(
                f"{fnum(one(klu, mu_inf=v, theta_s_deg=t)['ratio_median'], 3)}"
                for t in suns
            ),
        ]
        for v in variants
    ]
    L += [
        "Median of (Ψ_KLu/µ∞)/(KLu/a), per θs (ψ):",
        "",
        md_table(
            ["µ∞", *(f"{t}° ({p}°)" for t, p in zip(suns, psis, strict=True))], body
        ),
        "",
    ]
    nk = sum(int(r["n_klu_nonpositive"]) for r in klu if r["mu_inf"] == variants[0])
    L += [f"Non-positive KLu values excluded from the ratio (counted): {nk}.", ""]
    models = list(dict.fromkeys(r["model"] for r in ora))
    tss = list(dict.fromkeys(r["theta_s_deg"] for r in ora))
    body = [
        [
            m,
            *(
                fnum(
                    one(ora, model=m, theta_s_deg=t, side="realisation held out")[
                        "median_abs_rel_err_pct"
                    ]
                )
                for t in tss
            ),
        ]
        for m in models
    ]
    L += [
        "Oracle rung in `rrs` at nadir, realisation held out, median |e| % per θs:",
        "",
        md_table(["model", *tss], body),
        "",
    ]
    small = [
        float(one(klu, mu_inf="L23 (Eq. 8 refit)", theta_s_deg=t)["ratio_median"])
        for t in suns
        if float(t) <= 30.0
    ]
    base = float(
        one(ora, model=models[0], theta_s_deg="all", side="realisation held out")[
            "median_abs_rel_err_pct"
        ]
    )
    orc = float(
        one(ora, model=ORACLE, theta_s_deg="all", side="realisation held out")[
            "median_abs_rel_err_pct"
        ]
    )
    L += [
        "**Outcome.** Median ratio for the L23 µ∞ at θs ≤ 30°: "
        f"{', '.join(f'{v:.3f}' for v in small)} — "
        f"{verdict(all(1.05 <= v <= 1.20 for v in small))}. Oracle rung: median |e| "
        f"{orc:.2f}% against "
        f"{base:.2f}% for the L23 variant — it worsens rrs: {verdict(orc > base)}.",
        "",
    ]

    # --- eigen µ∞ (Q15) ---------------------------------------------------------
    L += ["### The eigen µ∞ on every (realisation, λ) (Q15)", "", PRED_Q15, ""]
    if not eig:
        L += [
            "**Not run on this sample** (`--skip-eigen` and no product): no outcome.",
            "",
        ]
    else:
        body = [
            [
                r["section"],
                r["item"],
                r["group"],
                r["n"],
                fnum(r["value_1"], 4),
                fnum(r["value_2"], 4),
                fnum(r["value_3"], 4),
                r["meaning"],
            ]
            for r in eig
            if r["section"]
            in ("recipe", "interpolation error", "eigen", "B_p at fixed (bb/a, eta_bb)")
        ]
        L += [
            md_table(["section", "item", "group", "n", "1", "2", "3", "columns"], body),
            "",
        ]
        vs = [r for r in eig if r["section"] == "variant vs eigen"]
        groups = list(dict.fromkeys(r["group"] for r in vs))
        names = list(dict.fromkeys(r["item"] for r in vs))
        body = [
            [
                g,
                *(
                    fnum(one(vs, item=n, group=g)["value_1"])
                    + " / "
                    + fnum(one(vs, item=n, group=g)["value_2"])
                    for n in names
                ),
            ]
            for g in groups
        ]
        L += [
            "Each Eq. (8) variant against the eigen µ∞, median signed / median |.| "
            "deviation (%):",
            "",
            md_table(["group", *names], body),
            "",
        ]
        rf = [r for r in eig if r["section"] == "Eq. (8) form refit to eigen"]
        body = [
            [
                r["item"],
                r["group"],
                r["n"],
                fnum(r["value_1"], 3),
                fnum(r["value_2"], 3),
                fnum(r["value_3"], 3),
                r["meaning"],
            ]
            for r in rf
        ]
        L += [
            "Eq. (8)'s sixteen-term form refit to the eigen µ∞ by linear least "
            "squares (diagnostic only, not shipped), and what it leaves by B_p:",
            "",
            md_table(["item", "group", "n", "1", "2", "3", "columns"], body),
            "",
        ]
        spn = one(
            eig,
            section="B_p at fixed (bb/a, eta_bb)",
            group="point-weighted over bins",
            item=SPAN_ITEM,
        )
        med = float(spn["value_1"])
        L += [
            "**Outcome.** The point-weighted median span of µ∞ across B_p "
            f"{Q15_BP_RANGE[0]}–{Q15_BP_RANGE[1]} "
            f"at fixed (bb/a, η_bb) is {med:.2f}% (q10 {float(spn['value_2']):.2f}%, "
            f"q90 {float(spn['value_3']):.2f}%), "
            f"against Eq. (8)'s {L23_EQ8_FIT_ERROR_PCT}% fit error on L23: "
            f"{verdict(None if np.isnan(med) else med >= 2.0)}.",
            "",
        ]

    # --- bb_w ---------------------------------------------------------------------
    show = [r for r in bbw if float(r["wavelength_nm"]) % 50 == 0]
    L += [
        "### The files' `bbw` against `conventions.bb_w`",
        "",
        md_table(
            [
                "λ (nm)",
                "file bbw (median)",
                "bb_w clamp",
                "file/clamp",
                "file/extrapolate",
                "bbw/(bw/2) min",
                "max",
            ],
            [
                [
                    r["wavelength_nm"],
                    r["bbw_file_median"],
                    r["bb_w_conventions_clamp"],
                    fnum(r["file_over_conventions_clamp"], 4),
                    fnum(r["file_over_conventions_extrapolate"], 4),
                    fnum(r["bbw_over_half_bw_min"], 4),
                    fnum(r["bbw_over_half_bw_max"], 4),
                ]
                for r in show
            ],
        ),
        "",
        "Every model here is fed the files' own `bbw` (task 3); `conventions.bb_w` is "
        "L23's table "
        "and is shown for the record (the task-5 note: O25's published calibration "
        "used bw/2).",
        "",
    ]
    return L


# =============================================================== figures =====
#: Categorical palette (the dataviz reference instance, light mode), assigned
#: to model families in this fixed order; line style is the second encoding.
FAMILY_COLOURS = {
    "ztt_l23": "#2a78d6",
    "o25_pub": "#eb6834",
    "o25_refit": "#1baf7a",
    "ztt_tt2017": "#eda100",
    "ztt_l23_closure": "#e87ba4",
    "hybrid": "#008300",
    "gordon": "#6250d6",
    "transfer": "#e34948",
}
INK, INK_MUTED = "#0b0b0b", "#52514e"


def style_of(spec: ModelSpec) -> dict:
    """Colour by family; dashed = Sullivan Pbb / the fallback / θv-only."""
    dashed = (
        spec.key.endswith(("_sul", "_tv"))
        or spec.key.startswith("o25_refit_tv")
        or spec.key == "hybrid_fb"
        or spec.key.startswith("o25_unw")
    )
    return {
        "color": FAMILY_COLOURS[spec.family],
        "linestyle": "--" if dashed else "-",
        "linewidth": 1.6,
    }


def save_png(fig, path: Path) -> None:
    """Atomic, deterministic PNG (no software/date metadata)."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".png")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fig.savefig(tmp, dpi=130, metadata={"Software": None})
        plt.close(fig)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def make_figures(out: Path, registry, plan: Plan) -> None:
    """Five figures, each drawn from the committed CSVs."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_validation import apply_house_style

    plt = apply_house_style()
    by_label = {}
    for s in registry:
        by_label.setdefault(s.label, s)
    band = read_csv(out / f"{PREFIX}per_band.csv")
    brk = read_csv(out / f"{PREFIX}breakdowns.csv")
    scores = read_csv(out / f"{PREFIX}scores.csv")
    mud = read_csv(out / f"{PREFIX}mu_d.csv")
    eig_path = out / f"{PREFIX}eigen.csv"
    native = [lab for lab, s in by_label.items() if "native" in s.tables]

    def legend(ax):
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False
        )

    # 1. per band, window, realisation held out.
    fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
    for lab in native:
        sub = pick(
            band,
            model=lab,
            cut=CUT["window"].label,
            split="realisation",
            side="held out",
        )
        if not sub:
            continue
        w = np.array([float(r["wavelength_nm"]) for r in sub])
        v = np.array([float(r["median_abs_rel_err_pct"]) for r in sub])
        ax.plot(w, v, label=lab, **style_of(by_label[lab]))
    ax.axvspan(EMULATOR_WAVE_MAX, plan.wave.max(), color="#e1e0d9", alpha=0.6, lw=0)
    ax.text(
        EMULATOR_WAVE_MAX + 2,
        0.98,
        "hybrid out of\ndomain",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("median |pred/true − 1| (%)")
    ax.set_title(
        "PB24 window, realisation split held out: error per band (native spaces)",
        loc="left",
    )
    legend(ax)
    save_png(fig, out / f"{PREFIX}fig_per_band.png")

    # 2. per θv_air and per ψ bin.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, kind, xlabel in (
        (axes[0], BREAKDOWN_NAMES["air"]["theta_v"], "θv_air (deg)"),
        (axes[1], BREAKDOWN_NAMES["air"]["psi_bin"], "ψ bin (deg)"),
    ):
        for lab in native:
            sub = pick(
                brk,
                model=lab,
                cut=CUT["window"].label,
                split="realisation",
                side="held out",
                breakdown=kind,
            )
            if not sub:
                continue
            x = np.arange(len(sub))
            v = np.array([float(r["median_abs_rel_err_pct"]) for r in sub])
            ax.plot(
                x, v, marker="o", markersize=4, label=lab, **style_of(by_label[lab])
            )
            ax.set_xticks(
                x,
                [r["value"] for r in sub],
                rotation=45 if "psi" in kind else 0,
                fontsize=8,
            )
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("median |pred/true − 1| (%)")
    axes[0].set_title("per view zenith (window, realisation held out)", loc="left")
    axes[1].set_title("per scattering-angle bin", loc="left")
    legend(axes[1])
    save_png(fig, out / f"{PREFIX}fig_per_view_and_psi.png")

    # 3. µd three ways.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for (_, label), c in zip(MU_D_METHODS, colours, strict=True):
        sub = [
            r
            for r in pick(mud, method=label, bands="350-800", bb_over_a="all")
            if r["theta_s_deg"] != WINDOW_SUNS
        ]
        x = np.array([float(r["theta_s_deg"]) for r in sub])
        axes[0].plot(
            x,
            [float(r["median_signed_dev_pct"]) for r in sub],
            marker="o",
            markersize=4,
            color=c,
            lw=1.6,
            label=label,
        )
        sub = [
            r
            for r in pick(mud, method=label, theta_s_deg=WINDOW_SUNS, bands="350-800")
            if r["bb_over_a"] != "all"
        ]
        axes[1].plot(
            range(len(sub)),
            [float(r["median_abs_dev_pct"]) for r in sub],
            marker="o",
            markersize=4,
            color=c,
            lw=1.6,
            label=label,
        )
        axes[1].set_xticks(range(len(sub)), [r["bb_over_a"] for r in sub])
    axes[0].axhline(0.0, color=INK_MUTED, lw=0.8)
    axes[0].set_xlabel("θs (deg)")
    axes[0].set_ylabel("median µd_model/µd_PB24 − 1 (%)")
    axes[0].set_title("µd against PB24's mu_d, all bands", loc="left")
    axes[1].set_xlabel("bb/a")
    axes[1].set_ylabel("median |deviation| (%), θs 0–70 pooled")
    axes[1].set_title("what is left, by bb/a", loc="left")
    legend(axes[1])
    save_png(fig, out / f"{PREFIX}fig_mu_d.png")

    # 4. eigen µ∞: what Eq. (8)'s form leaves by B_p, and the variants.
    if eig_path.is_file():
        eig = read_csv(eig_path)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
        rf = pick(eig, section="Eq. (8) form refit to eigen", item="residual by B_p")
        rf = [r for r in rf if int(r["n"]) > 0]
        x = np.arange(len(rf))
        med = np.array([float(r["value_2"]) for r in rf])
        lo = np.array([float(r["value_1"]) for r in rf])
        hi = np.array([float(r["value_3"]) for r in rf])
        axes[0].fill_between(
            x, lo, hi, color="#86b6ef", alpha=0.5, lw=0, label="16–84%"
        )
        axes[0].plot(
            x, med, color="#2a78d6", marker="o", markersize=4, lw=1.6, label="median"
        )
        axes[0].axhline(0.0, color=INK_MUTED, lw=0.8)
        axes[0].set_xticks(
            x, [r["group"].replace("B_p ", "") for r in rf], rotation=30, fontsize=8
        )
        axes[0].set_xlabel("B_p")
        axes[0].set_ylabel("Eq. (8) form fit / eigen µ∞ − 1 (%)")
        axes[0].set_title("what a µ∞(bb/a, η_bb) cannot absorb", loc="left")
        axes[0].legend(frameon=False, fontsize=8)
        vs = [
            r
            for r in pick(eig, section="variant vs eigen")
            if r["group"].startswith("bb/a") and int(r["n"]) > 0
        ]
        for name, c in zip(
            ("TT2017", "L23", "L23 closure"),
            ("#eda100", "#2a78d6", "#e87ba4"),
            strict=True,
        ):
            sub = [r for r in vs if r["item"] == name]
            axes[1].plot(
                range(len(sub)),
                [float(r["value_1"]) for r in sub],
                color=c,
                marker="o",
                markersize=4,
                lw=1.6,
                label=name,
            )
            axes[1].set_xticks(
                range(len(sub)),
                [r["group"].replace("bb/a ", "") for r in sub],
                fontsize=8,
            )
        axes[1].axhline(0.0, color=INK_MUTED, lw=0.8)
        axes[1].set_xlabel("bb/a")
        axes[1].set_ylabel("median µ∞_variant / µ∞_eigen − 1 (%)")
        axes[1].set_title("Eq. (8) variants against the eigen µ∞", loc="left")
        axes[1].legend(frameon=False, fontsize=8)
        save_png(fig, out / f"{PREFIX}fig_eigen_mu_inf.png")

    # 5. non-physical share per model and cut (all realisations).
    rrs_models = [lab for lab, s in by_label.items() if s.space == "rrs"]
    cuts = [c for c in CUTS if c.key != "everything"]
    fig, axes = plt.subplots(
        1, len(cuts), figsize=(13, 5.2), layout="constrained", sharey=True
    )
    for ax, cut in zip(axes, cuts, strict=True):
        vals = []
        for lab in rrs_models:
            br = BRANGE["le750"].label if by_label[lab].hybrid else BRANGE["all"].label
            hit = pick(scores, model=lab, cut=cut.label, split="none", bands=br)
            vals.append(float(hit[0]["nonphysical_pct"]) if hit else np.nan)
        y = np.arange(len(rrs_models))
        ax.barh(
            y,
            vals,
            color=[FAMILY_COLOURS[by_label[m].family] for m in rrs_models],
            height=0.7,
        )
        for yi, v in zip(y, vals, strict=True):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:.1f}", va="center", fontsize=7, color=INK)
        ax.set_title(cut.label, loc="left", fontsize=9)
        ax.set_xlabel("non-physical values (%)")
        ax.invert_yaxis()
    axes[0].set_yticks(np.arange(len(rrs_models)), rrs_models, fontsize=7)
    save_png(fig, out / f"{PREFIX}fig_nonphysical.png")


# ============================================================ O25 tables =====
O25_COLUMNS = (
    Column("sample"),
    Column("instance"),
    Column("theta_s_deg", "g"),
    Column("theta_v_air_deg", "g"),
    Column("dphi_deg", "g"),
    Column("Gw0", ".8e"),
    Column("Gw1", ".8e"),
    Column("Gp0", ".8e"),
    Column("Gp1", ".8e"),
)


def o25_rows(plan, tables: dict) -> list[dict]:
    """Every refit LUT node, so the refits are on the record."""
    rows = []
    for name, t in tables.items():
        for i, ts in enumerate(t.theta_s):
            for j, tv in enumerate(t.theta_v):
                for k, ph in enumerate(t.dphi):
                    G = t.G[i, j, k]
                    rows.append(
                        {
                            "sample": plan.tag,
                            "instance": name,
                            "theta_s_deg": ts,
                            "theta_v_air_deg": tv,
                            "dphi_deg": ph,
                            "Gw0": G[0],
                            "Gw1": G[1],
                            "Gp0": G[2],
                            "Gp1": G[3],
                        }
                    )
    return rows


# ================================================================== run ======
def published_available() -> bool:
    """Whether the O25 checkout with the published LUTs is on disk."""
    from robust.rt import baselines as B

    try:
        data = B.o25_checkout_dir() / "Data"
    except Exception:  # noqa: BLE001 - absent checkout: the model is dropped
        return False
    return all((data / f).is_file() for f in B.O25_PUBLISHED_FILES)


def eigen_path_for(plan: Plan, out: Path, quick: bool) -> Path:
    """The full product beside the cache; anything smaller next to the outputs
    (never committed: the name has no ``baseline_`` prefix)."""
    from robust.rt.data import pb24_eigen

    if not quick and plan.cache_dir is None:
        return pb24_eigen.default_eigen_path()
    return out / "eigen_mu_inf_sample.npz"


def load_eigen(path: Path, plan: Plan):
    """The product if it exists and covers the plan, else ``None``."""
    from robust.rt.data import pb24_eigen

    if not path.is_file():
        return None
    product = pb24_eigen.load_mu_inf_eigen(path)
    if not (
        set(plan.ids.tolist()) <= set(product.realisations.tolist())
        and set(plan.wave.tolist()) <= set(product.wave.tolist())
    ):
        raise SystemExit(
            f"{path} does not cover this sample; rebuild it (--eigen-only)"
        )
    return product


def run(
    out: Path,
    plan: Plan,
    *,
    workers: int,
    eigen: str,
    eigen_path: Path,
    registry=None,
    figures: bool = True,
) -> dict:
    """Everything: (eigen) → fits → scores → CSVs → diagnostics → markdown → figures.

    ``eigen``: ``"auto"`` builds the product if absent, ``"skip"`` never
    builds it, ``"only"``/``"rebuild"`` build it (``"only"`` then stops).
    Returns wall times.
    """
    times = {}
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    T = table_fields(plan)
    if eigen in ("only", "rebuild") or (eigen == "auto" and not eigen_path.is_file()):
        build_eigen(plan, T, eigen_path, workers)
        times["eigen"] = time.time() - t0
    if eigen == "only":
        return times
    if registry is None:
        registry = build_registry(with_published=published_available())

    t = time.time()
    gts = {
        p: geometry_table(load_chunk(plan, plan.chunks[0], p), p)
        for p in ("air", "in_water")
    }
    tables = run_fits(plan, gts, workers)
    times["fits"] = time.time() - t
    t = time.time()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as scratch:
        ints, floats = run_scores(plan, gts, registry, tables, workers, Path(scratch))
    times["scores"] = time.time() - t
    t = time.time()
    res = Results(plan, gts, registry, ints, floats)
    write_csv(out / f"{PREFIX}scores.csv", SCORE_COLUMNS, score_rows(res))
    write_csv(out / f"{PREFIX}per_band.csv", BAND_COLUMNS, band_rows(res))
    write_csv(out / f"{PREFIX}breakdowns.csv", BREAKDOWN_COLUMNS, breakdown_rows(res))
    write_csv(out / f"{PREFIX}o25_refit.csv", O25_COLUMNS, o25_rows(plan, tables))
    del res, ints, floats
    times["tables"] = time.time() - t

    t = time.time()
    product = load_eigen(eigen_path, plan)
    mu_rows, band_cols, band_out = mu_d_rows(plan, T)
    write_csv(out / f"{PREFIX}mu_d.csv", MU_D_COLUMNS, mu_rows)
    write_csv(out / f"{PREFIX}mu_d_per_band.csv", band_cols, band_out)
    eig_arr = None
    if product is not None:
        eig_arr = product.mu_inf[
            np.ix_(product.rows(plan.ids), np.searchsorted(product.wave, plan.wave))
        ]
    k_rows, o_rows = klu_rows(plan, T, eig_arr)
    write_csv(out / f"{PREFIX}klu.csv", KLU_COLUMNS, k_rows)
    write_csv(out / f"{PREFIX}klu_oracle.csv", ORACLE_COLUMNS, o_rows)
    write_csv(out / f"{PREFIX}bbw.csv", BBW_COLUMNS, bbw_rows(plan, T))
    write_csv(out / f"{PREFIX}psi.csv", PSI_COLUMNS, psi_rows(plan, gts))
    eig_csv = out / f"{PREFIX}eigen.csv"
    if product is not None:
        write_csv(eig_csv, EIGEN_COLUMNS, eigen_rows(plan, T, product))
    else:
        eig_csv.unlink(missing_ok=True)
    times["diagnostics"] = time.time() - t

    write_markdown(out, plan, registry, times)
    if figures:
        t = time.time()
        make_figures(out, registry, plan)
        times["figures"] = time.time() - t
    times["total"] = time.time() - t0
    return times


def artefacts(directory: Path) -> list[str]:
    """The run's artefact names in a directory."""
    return sorted(
        p.name
        for p in directory.glob(f"{PREFIX}*")
        if p.suffix in (".csv", ".md", ".png")
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"{QUICK_N} realisations (seed {QUICK_SEED}), "
        f"every {QUICK_BAND_STRIDE}th band",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"processes (0: in-process); more than {MAX_WORKERS} is refused "
        "without --allow-more-workers (task 5b)",
    )
    parser.add_argument(
        "--allow-more-workers",
        action="store_true",
        help=f"permit --workers above {MAX_WORKERS}; each worker peaks at "
        f"~{WORKER_PEAK_GB} GB and the box has been logged out by oomd at 12",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--realisations", type=int, nargs="*", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--skip-eigen", action="store_true", help="never build the eigen product"
    )
    group.add_argument(
        "--eigen-only", action="store_true", help="build the eigen product and stop"
    )
    group.add_argument(
        "--rebuild-eigen",
        action="store_true",
        help="rebuild the eigen product, then run",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rerun into a temporary directory and compare byte for byte",
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.workers > MAX_WORKERS and not args.allow_more_workers:
        raise SystemExit(
            f"refusing --workers {args.workers}: more than {MAX_WORKERS} workers "
            f"(~{args.workers * WORKER_PEAK_GB:.0f} GB peak) caused the 2026-11-14 "
            "systemd-oomd logout; pass --allow-more-workers to override"
        )
    peak = max(args.workers, 1) * WORKER_PEAK_GB
    print(
        f"workers: {args.workers} (predicted peak ~{peak:.0f} GB "
        f"at ~{WORKER_PEAK_GB} GB each, starts staggered by {STAGGER_S:.0f} s); "
        "run inside `systemd-run --user --scope -p MemoryMax=80G` from a plain "
        "terminal (see the docstring)"
    )

    out = args.out_dir.resolve()
    if args.quick and out == OUT_DIR.resolve():
        raise SystemExit(
            "refusing --quick with the default --out-dir: it would overwrite "
            f"{OUT_DIR} with small-sample numbers"
        )
    if args.cache_dir is None and not os.environ.get("OS_COLOR"):
        raise SystemExit("$OS_COLOR is not set, so the PB24 cache cannot be found")
    t0 = time.time()
    plan = make_plan(
        quick=args.quick, cache_dir=args.cache_dir, realisations=args.realisations
    )
    eigen_path = eigen_path_for(plan, out, args.quick or args.realisations is not None)
    print(
        f"sample: {plan.sample}; {len(plan.chunks)} chunks; eigen product {eigen_path}"
    )
    eigen = (
        "skip"
        if args.skip_eigen
        else "only"
        if args.eigen_only
        else "rebuild"
        if args.rebuild_eigen
        else "auto"
    )

    if not args.check:
        times = run(
            out,
            plan,
            workers=args.workers,
            eigen=eigen,
            eigen_path=eigen_path,
            figures=not args.no_figures,
        )
        print("wall times (s): " + ", ".join(f"{k} {v:.1f}" for k, v in times.items()))
        print(f"wrote {len(artefacts(out))} artefacts to {out}")
        return 0

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        tmp = Path(tmp)
        bad = []
        if not args.skip_eigen and eigen_path.is_file():
            t = time.time()
            rebuilt = tmp / "eigen_check.npz"
            build_eigen(plan, table_fields(plan), rebuilt, args.workers)
            same = rebuilt.read_bytes() == eigen_path.read_bytes()
            print(
                f"{eigen_path.name}: {'byte-identical' if same else 'DIFFERS'} "
                f"({time.time() - t:.1f} s)"
            )
            if not same:
                bad.append(eigen_path.name)
        times = run(
            tmp,
            plan,
            workers=args.workers,
            eigen="skip",
            eigen_path=eigen_path,
            figures=not args.no_figures,
        )
        names = sorted(set(artefacts(out)) | set(artefacts(tmp)))
        png_diff = []
        for name in names:
            a, b = out / name, tmp / name
            same = a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()
            if name.endswith(".png"):
                if not same:
                    png_diff.append(name)
                print(
                    f"{name}: {'byte-identical' if same else 'differs'} (figure, "
                    "reported)"
                )
                continue
            print(f"{name}: {'byte-identical' if same else 'DIFFERS'}")
            if not same:
                bad.append(name)
        print(
            "wall times (s): "
            + ", ".join(f"{k} {v:.1f}" for k, v in times.items())
            + f"; check total {time.time() - t0:.1f}"
        )
        if bad:
            print(f"CHECK FAILED: {bad}")
            return 1
        print(
            "CHECK OK: every CSV and markdown byte-identical "
            f"({len(names) - len(png_diff)} files; PNGs differing: {len(png_diff)})"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
