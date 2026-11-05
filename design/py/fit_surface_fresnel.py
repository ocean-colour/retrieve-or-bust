#!/usr/bin/env python
"""
Fit the Fresnel surface transfer on PB24 and score it held out (M8 task 4).

``conventions.FresnelTransfer`` is the Lee form with a Fresnel-scaled
numerator, ``Rrs = A0 T(θv)/T(0) · rrs / (1 − B rrs)``, ``T`` the flat-surface
unpolarised transmittance at the above-water view zenith and ``rrs`` taken at
the refracted angle (``rt_fit_pb24.md`` Q9). Two numbers replace M5's
2600-number table, which was fitted across mismatched directions (Q9c).

**The fit.** ``Rrs = A0 r rrs + B (Rrs rrs)`` with ``r = T(θv)/T(0)`` is linear
in ``(A0, B)``; the shipped fit divides each row by ``Rrs`` (relative
weighting, ``1 = A0 r rrs/Rrs + B rrs``), so the objective is the relative
error it is scored by. The normal equations are accumulated in float64 over
fixed chunks of realisations in a fixed order and solved once: no optimiser,
no seed, so a rerun is bit-identical (``--check``).

**The rows.** PB24's hyperspectral cache (all 451 bands), the **training side
of the realisation split** in ``design/ztt_pb24/splits.json`` (all 4000
training realisations), the **window** geometries (θs ≤ 70°, θv_air ≤ 70°;
832 per realisation). ``rrs`` is interpolated linearly in θ to
``asin(sin θv_air / 1.34)`` exactly as ``robust.rt.data.pb24.load_batch``
does (the script asserts agreement with the loader on three realisations).

**The score.** Held out on the 1000 test realisations, every one of the 1300
air geometries, all bands: ``|pred/true − 1|`` of ``Rrs`` predicted from the
true ``rrs``, in the window and the shell, per θv_air, per θs and per band
group, beside three comparators:

(a) Lee's nadir constants ``A = 0.52, B = 1.7``;
(b) M5's shipped table (``conventions.default_transfer()``, fitted on the
    mismatched pairing), evaluated on correctly paired data;
(c) a per-cell table refitted on the same training rows (the
    ``pb24.fit_transfer`` form: one unweighted lstsq per geometry cell,
    2600 numbers, all 1300 cells -- so in the shell it is fitted, not
    extrapolated);

and two variants of the two-number form, reported and not shipped: the
unweighted fit, and the ``Q`` form ``Rrs = A0 r rrs / (1 − c Q rrs)``.

**The downwelling form** (``FresnelTransfer(downwelling=True)``), fitted on
the same rows and also shipped: ``A = A0 r · t_d/T(0)`` with
``t_d = (1 − H) T(θs) + H T̄``, ``H = 1 − Esdir_Es_ratio`` at the row's θs and
λ. ``rrs`` is normalised by ``Ed(0−)``, ``Rrs`` by ``Ed(0+)``, so the
surface's downwelling transmittance belongs in the transfer; a scratch probe
on the ``--quick`` sample predicted and confirmed that the θv-only form's
large-θs failure is exactly this term.

Quantiles come from streaming histograms of ``log10 |pred/true − 1|``
(10,000 bins over [1e-8, 1e2], i.e. 0.23% relative resolution, quantile =
bin centre); the maxima are exact.

    python design/py/fit_surface_fresnel.py              # fit, score, write md+csv
    python design/py/fit_surface_fresnel.py --quick --dry-run   # development
    python design/py/fit_surface_fresnel.py --check      # refit == shipped, bit for bit

Outputs: ``design/ztt_pb24/surface_fresnel.md`` and ``surface_fresnel.csv``.
The fitted constants are pasted into ``robust/rt/conventions.py``
(``FRESNEL_A0``, ``FRESNEL_B``, ``FRESNEL_N_TRAINING``,
``FRESNEL_TRAINING_DIGEST``) by hand; ``--check`` keeps them honest.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from robust.rt import conventions as C  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402
from robust.rt.data import pb24_cache as PC  # noqa: E402
from robust.rt.types import Geometry  # noqa: E402

#: Realisations per cache read. Part of the fit's definition: the float64 sums
#: are accumulated chunk by chunk, so changing it can change the last bit.
CHUNK = 50

#: ``--quick``: training and test realisations (the first n of each side).
QUICK = (200, 50)

#: Band groups (nm, inclusive) for the per-λ cut.
BAND_GROUPS = (
    ("350-399", 350.0, 399.0),
    ("400-499", 400.0, 499.0),
    ("500-599", 500.0, 599.0),
    ("600-699", 600.0, 699.0),
    ("700-800", 700.0, 800.0),
)

#: Histogram of log10 |pred/true - 1|.
LOG_LO, LOG_HI, N_BINS = -8.0, 2.0, 10_000

#: Quantiles reported.
QUANTILES = (0.5, 0.9, 0.99)

#: The models scored, in column order.
MODELS = (
    "fresnel",
    "fresnel_td",
    "nadir",
    "m5_table",
    "cell_refit",
    "fresnel_unweighted",
    "fresnel_Q",
)
LABELS = {
    "fresnel": "Fresnel θv-only (shipped)",
    "fresnel_td": "Fresnel + downwelling (shipped)",
    "nadir": "nadir constants (0.52, 1.7)",
    "m5_table": "M5 table (shipped, mismatched)",
    "cell_refit": "per-cell refit (2600 numbers)",
    "fresnel_unweighted": "Fresnel, unweighted fit",
    "fresnel_Q": "Fresnel, Q form",
}

OUT_DIR = ROOT / "design" / "ztt_pb24"


# ----------------------------------------------------------------- reading ---
def read_H(cache, ids):
    """``H = 1 − Esdir_Es_ratio``, ``(n, θs, λ)`` on the 4-D fields' bands."""
    ratio = np.asarray(cache.table("Esdir_Es_ratio", ids), dtype=float)
    return 1.0 - ratio[..., cache.four_d_bands]


def read_pairs(cache, ids, theta_nodes, theta_w_air, *, with_Q=True):
    """``rrs`` (refracted, interpolated), ``Rrs`` and ``Q`` for ``ids``, float64.

    Shapes ``(n, θs, θv_air, φ, λ)``: exactly the loader's arithmetic
    (``np.asarray(..., float)`` then :func:`pb24.interp_in_water`).
    """
    rrs = P.interp_in_water(
        np.asarray(cache.four_d("rrs", ids), dtype=float), theta_nodes, theta_w_air
    )
    Rrs = np.asarray(cache.four_d("Rrs", ids), dtype=float)
    Q = None
    if with_Q:
        Q = P.interp_in_water(
            np.asarray(cache.four_d("Q", ids), dtype=float), theta_nodes, theta_w_air
        )
    return rrs, Rrs, Q


def cell_normal_equations(rrs, Rrs):
    """Per-cell sums for the unweighted M5 form ``Rrs = A rrs + B (Rrs rrs)``.

    Returns five ``(θs, θv, φ)`` arrays: Σx0², Σx0x1, Σx1², Σx0y, Σx1y.
    """
    x0, x1, y = rrs, Rrs * rrs, Rrs
    axes = (0, 4)
    return np.stack(
        [
            np.sum(x0 * x0, axis=axes),
            np.sum(x0 * x1, axis=axes),
            np.sum(x1 * x1, axis=axes),
            np.sum(x0 * y, axis=axes),
            np.sum(x1 * y, axis=axes),
        ]
    )


def solve_cells(sums):
    """``(A, B)`` per cell from :func:`cell_normal_equations` sums."""
    s00, s01, s11, s0y, s1y = sums
    det = s00 * s11 - s01 * s01
    A = (s11 * s0y - s01 * s1y) / det
    B = (s00 * s1y - s01 * s0y) / det
    return A, B


# ------------------------------------------------------------------ the fit ---
def fit(cache, train_ids, grids, *, progress=True):
    """Accumulate every fit's normal equations over the training realisations.

    Returns a dict with the Fresnel fits (relative, unweighted, Q form), the
    per-cell table, and row counts.
    """
    theta_nodes = grids["theta"]
    theta_w_air = P.theta_air_to_water(theta_nodes)
    ts, tv = grids["theta_s"], grids["theta"]
    window_cell = (ts[:, None] <= P.ANGLE_WINDOW_MAX) & (
        tv[None, :] <= P.ANGLE_WINDOW_MAX
    )
    theta_v_b = tv[None, None, :, None, None]

    acc = {
        key: [np.zeros((2, 2)), np.zeros(2), 0, 0]
        for key in ("relative", "td", "unweighted", "Q")
    }
    cells = np.zeros((5, ts.size, tv.size, grids["phi"].size))
    t0 = time.time()
    for start in range(0, len(train_ids), CHUNK):
        ids = train_ids[start : start + CHUNK]
        rrs, Rrs, Q = read_pairs(cache, ids, theta_nodes, theta_w_air)
        cells += cell_normal_equations(rrs, Rrs)

        # Window rows only, for the two-number forms.
        sel = np.broadcast_to(window_cell[None, :, :, None, None], rrs.shape)
        tvw = np.broadcast_to(theta_v_b, rrs.shape)[sel]
        r_w, R_w, Q_w = rrs[sel], Rrs[sel], Q[sel]
        td = C._downwelling_ratio_np(
            ts[None, :, None, None, None], read_H(cache, ids)[:, :, None, None, :]
        )
        td_w = np.broadcast_to(td, rrs.shape)[sel]
        for key, kwargs in (
            ("relative", {"relative": True}),
            ("td", {"relative": True, "factor": td_w}),
            ("unweighted", {"relative": False}),
            ("Q", {"relative": True, "Q": Q_w}),
        ):
            XtX, Xty, n_rows, n_drop = C.fresnel_normal_equations(
                r_w, R_w, tvw, **kwargs
            )
            acc[key][0] = acc[key][0] + XtX
            acc[key][1] = acc[key][1] + Xty
            acc[key][2] += n_rows
            acc[key][3] += n_drop
        if progress:
            done = min(start + CHUNK, len(train_ids))
            print(
                f"  fit: {done}/{len(train_ids)} realisations "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )

    out = {}
    for key, (XtX, Xty, n_rows, n_drop) in acc.items():
        out[key] = (*C.solve_fresnel(XtX, Xty), n_rows, n_drop)
    A, B = solve_cells(cells)
    out["cells"] = C.SurfaceTransfer(
        theta_s=ts.astype(float),
        theta_v=tv.astype(float),
        dphi=grids["phi"].astype(float),
        A=A,
        B=B,
        provenance=(
            f"PB24 hyp, {len(train_ids)} realisation-split training realisations, "
            "1300 cells x 451 bands; unweighted normal equations per cell "
            "(fit_surface_fresnel.py)"
        ),
    )
    return out


def self_check(cache, train_ids, grids, cache_dir):
    """The script's pairing and per-cell fit against the package's, on 3 ids.

    (i) ``rrs``/``Rrs`` equal ``pb24.load_batch``'s bit for bit (float64 here);
    (ii) the per-cell normal-equation solution equals ``pb24.fit_transfer``'s
    lstsq on the same rows.
    """
    ids = tuple(int(i) for i in train_ids[:3])
    batch = P.load_batch(
        realisations=ids,
        angles="all",
        extras=(),
        validate=False,
        check_fresnel=False,
        cache_dir=cache_dir,
    )
    theta_nodes = grids["theta"]
    rrs, Rrs, _ = read_pairs(
        cache, ids, theta_nodes, P.theta_air_to_water(theta_nodes), with_Q=False
    )
    n_wave = rrs.shape[-1]
    mine_rrs = rrs.reshape(-1, n_wave)
    mine_Rrs = Rrs.reshape(-1, n_wave)
    # x64 is on in this script, so the batch holds the loader's float64 values.
    assert np.array_equal(mine_rrs, np.asarray(batch.rrs, dtype=float)), (
        "pairing differs from load_batch"
    )
    assert np.array_equal(mine_Rrs, np.asarray(batch.Rrs, dtype=float))

    ref = P.fit_transfer(batch, np.ones(batch.n_sample, dtype=bool))
    A, B = solve_cells(cell_normal_equations(rrs, Rrs))
    dA = float(np.max(np.abs(A / ref.A - 1.0)))
    dB = float(np.max(np.abs(B / ref.B - 1.0)))
    assert dA < 1e-4 and dB < 1e-4, (dA, dB)
    return dA, dB


# ------------------------------------------------------------------ scoring ---
class Hist:
    """Histograms of log10 error per (model, θs, θv_air, band group)."""

    def __init__(self, n_ts, n_tv, n_group):
        self.shape = (len(MODELS), n_ts, n_tv, n_group)
        self.counts = np.zeros(self.shape + (N_BINS,), dtype=np.int64)
        self.maxima = np.zeros(self.shape)
        self.n_nonfinite = np.zeros(len(MODELS), dtype=np.int64)

    def add(self, m, err, group_of_band):
        """``err`` is ``(n, θs, θv, φ, λ)``; ``group_of_band`` maps λ -> group."""
        n_ts, n_tv, n_g = self.shape[1:]
        finite = np.isfinite(err)
        self.n_nonfinite[m] += int((~finite).sum())
        err = np.where(finite, err, 10.0**LOG_HI)
        logs = np.log10(np.maximum(err, 10.0**LOG_LO))
        b = np.clip(
            ((logs - LOG_LO) / (LOG_HI - LOG_LO) * N_BINS).astype(np.int64),
            0,
            N_BINS - 1,
        )
        i_ts = np.arange(n_ts)[None, :, None, None, None]
        i_tv = np.arange(n_tv)[None, None, :, None, None]
        g = group_of_band[None, None, None, None, :]
        key = ((i_ts * n_tv + i_tv) * n_g + g) * N_BINS + b
        self.counts[m] += np.bincount(
            key.ravel(), minlength=n_ts * n_tv * n_g * N_BINS
        ).reshape(n_ts, n_tv, n_g, N_BINS)
        for gi in range(n_g):
            sub = err[..., group_of_band == gi].max(axis=(0, 3, 4))
            self.maxima[m, :, :, gi] = np.maximum(self.maxima[m, :, :, gi], sub)

    def summary(self, m, cell_mask, groups=None):
        """``(n, q50, q90, q99, max)`` over the cells in ``cell_mask`` (θs, θv)."""
        groups = slice(None) if groups is None else groups
        counts = self.counts[m][cell_mask][:, groups].reshape(-1, N_BINS).sum(axis=0)
        mx = self.maxima[m][cell_mask][:, groups].max()
        n = int(counts.sum())
        cdf = np.cumsum(counts) / n
        centres = 10.0 ** (
            LOG_LO + (np.arange(N_BINS) + 0.5) * (LOG_HI - LOG_LO) / N_BINS
        )
        qs = [float(centres[np.searchsorted(cdf, q)]) for q in QUANTILES]
        return (n, *qs, float(mx))


def score(cache, test_ids, grids, transfers, *, progress=True):
    """Stream the held-out errors of every model into :class:`Hist`."""
    theta_nodes = grids["theta"]
    theta_w_air = P.theta_air_to_water(theta_nodes)
    ts, tv, ph = grids["theta_s"], grids["theta"], grids["phi"]
    wave = grids["lambda"][cache.four_d_bands]
    group_of_band = np.full(wave.size, -1)
    for gi, (_, lo, hi) in enumerate(BAND_GROUPS):
        group_of_band[(wave >= lo) & (wave <= hi)] = gi
    assert (group_of_band >= 0).all()

    # Coefficients at every node, (θs, θv, φ), float64 under x64.
    mesh = np.meshgrid(ts, tv, ph, indexing="ij")
    geometry = Geometry(
        theta_s=jnp.asarray(mesh[0], dtype=float),
        theta_v=jnp.asarray(mesh[1], dtype=float),
        dphi=jnp.asarray(mesh[2], dtype=float),
    )
    coef = {}
    for name, transfer in transfers.items():
        if name.startswith("_"):
            continue
        A, B = transfer.coefficients(geometry)
        coef[name] = (
            np.asarray(A, dtype=float)[None, ..., None],
            np.asarray(B, dtype=float)[None, ..., None],
        )
    ratio = C._fresnel_ratio_np(tv)[None, None, :, None, None]
    A0_q, c_q = transfers["_Q"]
    td_transfer = transfers["_td"]
    mesh_b = [jnp.asarray(m[None], dtype=float) for m in mesh]

    hist = Hist(ts.size, tv.size, len(BAND_GROUPS))
    t0 = time.time()
    for start in range(0, len(test_ids), CHUNK):
        ids = test_ids[start : start + CHUNK]
        rrs, Rrs, Q = read_pairs(cache, ids, theta_nodes, theta_w_air)
        H = read_H(cache, ids)[:, :, None, None, :]
        # The package's own spectral path, as a caller with a PB24 batch uses it.
        A_td, B_td = td_transfer.spectral_coefficients(
            Geometry(
                theta_s=mesh_b[0],
                theta_v=mesh_b[1],
                dphi=mesh_b[2],
                diffuse_fraction=jnp.asarray(H),
            )
        )
        A_td, B_td = np.asarray(A_td), np.asarray(B_td)
        if start == 0:
            A_np = (
                td_transfer.A0
                * ratio
                * C._downwelling_ratio_np(ts[None, :, None, None, None], H)
            )
            assert np.allclose(A_td, A_np, rtol=1e-12, atol=0), "class != numpy"
        for m, name in enumerate(MODELS):
            if name == "fresnel_Q":
                pred = A0_q * ratio * rrs / (1.0 - c_q * Q * rrs)
            elif name == "fresnel_td":
                pred = A_td * rrs / (1.0 - B_td * rrs)
            else:
                A, B = coef[name]
                pred = A * rrs / (1.0 - B * rrs)
            hist.add(m, np.abs(pred / Rrs - 1.0), group_of_band)
        if progress:
            done = min(start + CHUNK, len(test_ids))
            print(
                f"  score: {done}/{len(test_ids)} realisations "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
    return hist


# ------------------------------------------------------------------ writing ---
def pct(x):
    """A fraction as a percentage string."""
    return f"{100 * x:.2f}"


def tables(hist, grids):
    """CSV rows and markdown for every cut."""
    ts, tv = grids["theta_s"], grids["theta"]
    window = (ts[:, None] <= P.ANGLE_WINDOW_MAX) & (tv[None, :] <= P.ANGLE_WINDOW_MAX)
    sets = {"window": window, "shell": ~window}

    rows = []
    md = []
    for set_name, mask in sets.items():
        cuts = [("all", "", mask, None)]
        for j, v in enumerate(tv):
            cell = mask & (np.arange(tv.size)[None, :] == j)
            if cell.any():
                cuts.append(("theta_v_air", f"{v:g}", cell, None))
        for i, s in enumerate(ts):
            cell = mask & (np.arange(ts.size)[:, None] == i)
            if cell.any():
                cuts.append(("theta_s", f"{s:g}", cell, None))
        for gi, (label, _, _) in enumerate(BAND_GROUPS):
            cuts.append(("band_nm", label, mask, [gi]))

        summaries = {}
        for cut, value, cell, groups in cuts:
            for m, name in enumerate(MODELS):
                n, q50, q90, q99, mx = hist.summary(m, cell, groups)
                summaries[(cut, value, name)] = (q50, q90, q99, mx)
                rows.append(
                    {
                        "set": set_name,
                        "cut": cut,
                        "value": value,
                        "model": name,
                        "label": LABELS[name],
                        "n": n,
                        "median_pct": pct(q50),
                        "q90_pct": pct(q90),
                        "q99_pct": pct(q99),
                        "max_pct": pct(mx),
                    }
                )

        md.append(f"### {set_name.capitalize()}\n")
        md.append(
            "Median |pred/true − 1| (%), q90 and q99 in brackets for the overall row.\n"
        )
        header = "| cut | " + " | ".join(LABELS[m] for m in MODELS) + " |"
        md.append(header)
        md.append("|---" * (len(MODELS) + 1) + "|")
        q = [summaries[("all", "", name)] for name in MODELS]
        md.append(
            "| **all** | "
            + " | ".join(f"**{pct(a)}** ({pct(b)}, {pct(c)})" for a, b, c, _ in q)
            + " |"
        )
        for cut, label in (
            ("theta_v_air", "θv_air"),
            ("theta_s", "θs"),
            ("band_nm", "λ"),
        ):
            for c_, value, _, _ in cuts:
                if c_ != cut:
                    continue
                cells = [summaries[(cut, value, name)][0] for name in MODELS]
                md.append(
                    f"| {label} = {value} | " + " | ".join(pct(x) for x in cells) + " |"
                )
        md.append("")
    return rows, "\n".join(md)


def write_csv(path, rows):
    """Write ``rows`` (dicts) as CSV."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buf.getvalue())


# --------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", default=None, help="hyperspectral cache")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK} realisations")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument(
        "--check",
        action="store_true",
        help="refit and compare with the shipped constants bit for bit; no scoring",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.quick and args.out_dir == OUT_DIR and not args.dry_run:
        parser.error("--quick fits on a subsample; pass --out-dir or --dry-run")
    if args.quick and args.check:
        parser.error("--check refits the shipped constants; it cannot be --quick")

    content = P.read_splits()
    every = np.asarray(content["realisations"], dtype=int)
    test_ids = np.asarray(sorted(content["realisation"]["test"]), dtype=int)
    train_ids = np.setdiff1d(every, test_ids)
    assert np.intersect1d(train_ids, test_ids).size == 0
    if args.quick:
        train_ids, test_ids = train_ids[: QUICK[0]], test_ids[: QUICK[1]]
    digest = C.realisation_digest(train_ids)
    print(
        f"realisation split: fitting on {train_ids.size} training realisations "
        f"(digest {digest[:16]}), scoring on {test_ids.size} held-out"
    )

    cache = PC.open_cache(args.cache_dir)
    g = cache.grids
    grids = {
        "theta_s": g["theta_s"].astype(float),
        "theta": g["theta"].astype(float),
        "phi": g["phi"].astype(float),
        "lambda": g["lambda"].astype(float),
    }
    assert cache.four_d_bands.size == 451 or args.cache_dir, "expected all 451 bands"

    dA, dB = self_check(cache, train_ids, grids, args.cache_dir)
    print(
        f"self-check: pairing == load_batch (bit for bit); per-cell "
        f"normal equations vs pb24.fit_transfer: max rel diff A {dA:.1e}, B {dB:.1e}"
    )

    t0 = time.time()
    fitted = fit(cache, train_ids, grids)
    A0, B, n_rows, n_drop = fitted["relative"]
    print(
        f"fit ({time.time() - t0:.0f} s): A0 = {A0!r}, B = {B!r}; "
        f"{n_rows} rows, {n_drop} dropped"
    )
    A0_td, B_td, _, _ = fitted["td"]
    print(f"  downwelling: A0 = {A0_td!r}, B = {B_td!r}")
    for key in ("unweighted", "Q"):
        a, b, n_r, n_d = fitted[key]
        print(f"  {key}: A0 = {a!r}, second = {b!r} ({n_r} rows, {n_d} dropped)")

    if args.check:
        shipped = (
            C.FRESNEL_A0,
            C.FRESNEL_B,
            C.FRESNEL_TD_A0,
            C.FRESNEL_TD_B,
            C.FRESNEL_N_TRAINING,
            C.FRESNEL_TRAINING_DIGEST,
        )
        got = (A0, B, A0_td, B_td, int(train_ids.size), digest)
        ok = shipped == got
        verdict = "OK, bit-identical" if ok else "MISMATCH"
        print(f"check: shipped {shipped}\n       refit   {got}\n       {verdict}")
        return 0 if ok else 1

    transfers = {
        "fresnel": C.FresnelTransfer(A0=A0, B=B),
        "_td": C.FresnelTransfer(A0=A0_td, B=B_td, downwelling=True),
        "nadir": C.FresnelTransfer(A0=C.A_RRS, B=C.B_RRS, n=1.0),
        "m5_table": C.default_transfer(),
        "cell_refit": fitted["cells"],
        "fresnel_unweighted": C.FresnelTransfer(*fitted["unweighted"][:2]),
        "_Q": fitted["Q"][:2],
    }
    # n = 1.0 makes T(θ)/T(0) = 1 identically: the nadir constants at every angle.
    m5_overlap = int(np.intersect1d(test_ids, np.arange(1, 401)).size)

    t0 = time.time()
    hist = score(cache, test_ids, grids, transfers)
    print(f"scored ({time.time() - t0:.0f} s)")
    rows, md_tables = tables(hist, grids)

    shipped_note = (
        "equal to the shipped constants"
        if (A0, B, A0_td, B_td)
        == (C.FRESNEL_A0, C.FRESNEL_B, C.FRESNEL_TD_A0, C.FRESNEL_TD_B)
        else "**not yet** the shipped constants (paste them into conventions.py)"
    )
    a_u, b_u, _, _ = fitted["unweighted"]
    a_q, c_q, _, _ = fitted["Q"]
    quick = " --quick" if args.quick else ""
    md = f"""# The Fresnel surface transfer on PB24 (M8 task 4)

*Generated by `design/py/fit_surface_fresnel.py`{quick}. Do not edit by hand.*

`Rrs = A0 · T(θv)/T(0) · rrs / (1 − B rrs)`, `T` the unpolarised flat-surface
Fresnel transmittance at the above-water view zenith (n = 1.34), `rrs` at the
refracted angle (`rt_fit_pb24.md` Q9).

## Fit

- **Rows:** PB24 hyperspectral cache, all {grids["lambda"].size} bands; the
  realisation split's **training** side of `design/ztt_pb24/splits.json`,
  **{train_ids.size} realisations** (digest `{digest}`); window geometries
  (θs ≤ 70°, θv_air ≤ 70°, 832 per realisation); **{n_rows:,} rows**
  ({n_drop} dropped as non-positive or non-finite).
- **Shipped, θv-only** (`FRESNEL_A0`, `FRESNEL_B`; relative weighting,
  `1 = A0 r rrs/Rrs + B rrs`): **A0 = {A0:.6f}, B = {B:.6f}**.
- **Shipped, with the downwelling factor** (`FRESNEL_TD_A0`, `FRESNEL_TD_B`;
  `r → r · t_d/T(0)`, `t_d = (1 − H) T(θs) + H T̄`, T̄ = {C.diffuse_transmittance():.5f},
  H from `Esdir_Es_ratio`): **A0 = {A0_td:.6f}, B = {B_td:.6f}**.
- Both {shipped_note}.
- Unweighted (`Rrs = A0 r rrs + B Rrs rrs`): A0 = {a_u:.6f}, B = {b_u:.6f}.
- `Q` form (`Rrs = A0 r rrs / (1 − c Q rrs)`, relative): A0 = {a_q:.6f},
  c = {c_q:.6f} (Lee's B = 1.7 ≈ r̄·Q with Q ≈ 3.5 would give c ≈ 0.49).
- Deterministic: float64 normal equations over chunks of {CHUNK} realisations
  in id order; `--check` refits and compares bit for bit.
- Self-check: the script's pairing equals `pb24.load_batch` bit for bit (float64)
  on three realisations, and its per-cell normal-equation fit equals
  `pb24.fit_transfer`'s lstsq to {max(dA, dB):.1e} relative.

## Held-out error

{test_ids.size} realisation-split **test** realisations, every air geometry
(window: 832 per realisation; shell: 468), all bands. Error is
`|Rrs_pred / Rrs_true − 1|` with `Rrs_pred` from the true `rrs`. Quantiles from
histograms of log10 error (0.23% relative resolution); maxima exact in the CSV.

Comparators: **nadir constants** `A = 0.52, B = 1.7` at every angle; **M5
table** = `conventions.default_transfer()` (`surface_pb24.npz`; OLCI,
mismatched pairing, Q9c), evaluated here on correctly paired data
({m5_overlap} of the {test_ids.size} test realisations are in M5's 1–400 fitting
pool, which can only flatter it); **per-cell refit** = one unweighted lstsq
per (θs, θv_air, Δφ) cell on the same {train_ids.size} training realisations,
all 1300 cells (2600 numbers; in the shell it is *fitted*, while the
two-number forms extrapolate from the window).

{md_tables}
"""
    print(md_tables)
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "surface_fresnel.md").write_text(md)
    write_csv(args.out_dir / "surface_fresnel.csv", rows)
    print(f"wrote {args.out_dir / 'surface_fresnel.md'} and surface_fresnel.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
