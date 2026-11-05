#!/usr/bin/env python
"""
Reproduce the published O25 LUTs from PB24, and score the published model (M8 task 5).

O25 (Pitarch et al. 2025, *Remote Sens. Environ.* 329, 114920, §2.2) models
``Rrs = (Gw0 + Gw1 w_bw) w_bw + (Gp0 + Gp1 w_bp) w_bp`` with four coefficient
LUTs over ``(theta_s, theta_v, dphi)``, fitted per geometry node by
**unweighted** least squares on the PB24 **OLCI** set (12 bands x 5000
realisations = 60,000 points per node), keeping points with
``Rrs <= 0.06 sr^-1``. This script refits that recipe through
``robust.rt.baselines.fit_o25_table(train=all, weighted=False, rrs_max=0.06)``
and compares it with the published files read by
``baselines.o25_published_table()`` from the O25 checkout (``$O25_DIR``,
default ``../O25``; GPL-3.0, so nothing of it is stored in this repository
except the comparison below).

**The data.** ``robust.rt.data.pb24.load_batch(resolution="olci",
angles="all")``: all 5000 realisations, all 1300 above-water geometries, the 12
OLCI bands 400-753 nm, ``Rrs`` at its node, the files' own water IOPs
(``a = aw + aph + ag + aNAP``, ``bb_w = bbw``, ``bb_p = bbph + bbNAP``). JAX
runs in float64 here, so every number is NumPy/JAX float64 arithmetic on the
files' float32 values. The OLCI files are separate HydroLight runs from the
hyperspectral ones (task 1 log), so the hyperspectral set cut to OLCI bands is
*not* the calibration set and is not used.

**The prediction** (``rt_fit_pb24.md`` task 5, written before any fit): the
four G's agree with the published files to better than 1% at every in-window
node (theta_s <= 70, theta_v <= 70), the residual traceable to float32 storage
and the Rrs cut; a larger disagreement means a zenith or azimuth convention
differs, diagnosed before anything is scored.

**The diagnosis**, each hypothesis re-measured here on all 5000:
(a) azimuth mirror, (b) zenith axis swap, (c) block order, (d) the cut per
point vs per spectrum, (e) weighted vs unweighted, (f) the 87.5/87.75 label,
and (g) the water backscattering the calibration used (``bw/2`` against the
files' ``bbw``).

**The score** (Q11 ii): the published O25 with the true IOPs in, in ``Rrs``,
``|pred/true - 1|`` over every (sample, band) point, in the window, the shell
and everything, per theta_v and per theta_s, beside our paper-recipe refit.
Both are **calibrated on all 5000** realisations: they have seen every PB24
test set. The split-wise refit and the cross-model tables are task 6's.

    python design/py/reproduce_o25.py            # all 5000; writes the artefacts
    python design/py/reproduce_o25.py --quick    # 100 realisations, writes nothing
    python design/py/reproduce_o25.py --check    # recompute; compare bytes

Outputs in ``design/ztt_pb24/``: ``o25_refit_lut.npz`` (our refit, our work on
open data), ``o25_reproduction.csv`` / ``.md`` and ``o25_published_score.csv``
/ ``.md``. ``--check`` compares the two CSVs and the LUT byte for byte. Nothing
from the GPL-3.0 checkout is written: the CSV holds our refit only, and the
comparison with the published tables is reported as quantiles in the markdown
(Q17a).
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from robust.rt import baselines as B  # noqa: E402
from robust.rt.data import pb24  # noqa: E402
from robust.rt.types import Geometry, IOPs  # noqa: E402

OUT_DIR = ROOT / "design" / "ztt_pb24"
LUT_FILE = "o25_refit_lut.npz"
REPRO_CSV = "o25_reproduction.csv"
REPRO_MD = "o25_reproduction.md"
SCORE_CSV = "o25_published_score.csv"
SCORE_MD = "o25_published_score.md"
CHECKED = (LUT_FILE, REPRO_CSV, SCORE_CSV)

#: --quick: every 50th realisation (100 of 5000).
QUICK_STEP = 50
COEFFS = ("Gw0", "Gw1", "Gp0", "Gp1")
WINDOW = pb24.ANGLE_WINDOW_MAX
#: Half a unit in the last place of the published text files (6 decimals).
FILE_HALF_ULP = 5e-7
SCORE_QUANTILES = (0.5, 0.9)
#: The prediction's threshold, relative.
PREDICTED_MAX = 0.01


# ----------------------------------------------------------------- data ------


def load(realisations):
    """The OLCI batch, every air geometry, float64; plus the files' ``bw``."""
    batch = pb24.load_batch(
        realisations=realisations,
        angles="all",
        resolution="olci",
        extras=(),
        check_fresnel=False,
        validate=False,
    )
    report = batch.report
    n_real = report.n_realisation
    if report.n_geometry != 1300 or batch.n_sample != n_real * 1300:
        raise RuntimeError(f"expected 1300 geometries per realisation: {report}")
    if report.n_dropped_zero_rrs:
        raise RuntimeError("load_batch dropped samples; the recipe keeps every one")
    if batch.Rrs.dtype != np.float64:
        raise RuntimeError("JAX is not in float64")
    Rrs = np.asarray(batch.Rrs)
    if not np.all(np.isfinite(Rrs)):
        raise RuntimeError("non-finite Rrs in the OLCI set")
    # Water is the same in every realisation; bw is not on the batch.
    from ocpy.hydrolight import pb24 as ocpy_pb24

    ds = ocpy_pb24.load_ds(int(batch.realisation[0]), "olci")
    bw = np.asarray(ds["bw"].values, dtype=float)
    bbw_file = np.asarray(ds["bbw"].values, dtype=float)
    if not np.array_equal(np.asarray(batch.iops.bb_w)[0], bbw_file):
        raise RuntimeError("bb_w on the batch is not the file's bbw")
    if not np.all(np.asarray(batch.iops.bb_w) == bbw_file):
        raise RuntimeError("bbw differs between realisations")
    return batch, bw


def geometry_index(batch):
    """Per-sample node indices ``(i_ts, i_tv, i_ph)`` on the published nodes."""
    return tuple(
        np.searchsorted(np.asarray(nodes), np.asarray(values))
        for nodes, values in (
            (B.O25_PUBLISHED_THETA_S, batch.geometry.theta_s),
            (B.O25_PUBLISHED_THETA_V, batch.geometry.theta_v),
            (B.O25_PUBLISHED_DPHI, batch.geometry.dphi),
        )
    )


def window_mask_nodes():
    """``(10, 10, 13)`` True where theta_s <= 70 and theta_v <= 70."""
    ts = np.asarray(B.O25_PUBLISHED_THETA_S)[:, None, None]
    tv = np.asarray(B.O25_PUBLISHED_THETA_V)[None, :, None]
    return np.broadcast_to((ts <= WINDOW) & (tv <= WINDOW), (10, 10, 13))


# ------------------------------------------------------------------ fits -----


def refit(batch, *, train=None, weighted=False, rrs_max=B.O25_RRS_CEILING, iops=None):
    """One ``fit_o25_table`` call; ``train`` defaults to every sample."""
    if train is None:
        train = np.ones(batch.n_sample, dtype=bool)
    return B.fit_o25_table(
        batch.iops if iops is None else iops,
        batch.Rrs,
        batch.geometry,
        train=train,
        weighted=weighted,
        rrs_max=rrs_max,
    )


def points_per_node(batch, idx, rrs_max=B.O25_RRS_CEILING):
    """``(10, 10, 13)`` count of points with ``Rrs <= rrs_max`` per node."""
    kept = (np.asarray(batch.Rrs) <= rrs_max).sum(axis=-1)
    flat = (idx[0] * 10 + idx[1]) * 13 + idx[2]
    return (
        np.bincount(flat, weights=kept, minlength=1300)
        .astype(np.int64)
        .reshape(10, 10, 13)
    )


def relative(G, P):
    """``G / P - 1``."""
    return G / P - 1.0


def summary(rel, mask):
    """median / q90 / max of |rel| and signed median, per coefficient, in mask."""
    out = {}
    for c, name in enumerate(COEFFS):
        values = rel[..., c][mask]
        absval = np.abs(values)
        out[name] = (
            float(np.median(absval)),
            float(np.quantile(absval, 0.9)),
            float(absval.max()),
            float(np.median(values)),
        )
    return out


def raw_blocks(table_o25):
    """The files' ``(130, 10)`` arrays rebuilt from an ``azimuth="o25"`` table."""
    G = np.asarray(table_o25.G)  # (ts, tv, block, 4)
    return [G[..., c].transpose(2, 0, 1).reshape(130, 10) for c in range(4)]


# ------------------------------------------------------------- scoring -------


def score_rows(label, pred, true, groups):
    """Rows ``(model, cut, n_points, median, q90)`` of ``|pred/true - 1|``."""
    err = np.abs(pred / true - 1.0)
    rows = []
    for cut, mask in groups:
        values = err[mask].reshape(-1)
        q = np.quantile(values, SCORE_QUANTILES)
        rows.append((label, cut, int(values.size), float(q[0]), float(q[1])))
    return rows


def score_groups(batch):
    """Sample-level masks for every cut; the last group is band-level."""
    ts = np.asarray(batch.geometry.theta_s)
    tv = np.asarray(batch.geometry.theta_v)
    inside = (ts <= WINDOW) & (tv <= WINDOW)
    groups = [("window", inside), ("shell", ~inside), ("all", np.ones_like(inside))]
    groups += [(f"theta_v={v:g}", tv == v) for v in B.O25_PUBLISHED_THETA_V]
    groups += [(f"theta_s={v:g}", ts == v) for v in B.O25_PUBLISHED_THETA_S]
    groups += [
        (f"window, theta_v={v:g}", inside & (tv == v))
        for v in B.O25_PUBLISHED_THETA_V
        if v <= WINDOW
    ]
    return groups


def predict(batch, table):
    """``Rrs_o25`` with a table, float64, as NumPy."""
    return np.asarray(B.Rrs_o25(batch.iops, None, batch.geometry, coeffs=table))


# ------------------------------------------------------------- writers -------


def npz_bytes(arrays: dict) -> bytes:
    """A deterministic ``.npz``: sorted members, fixed timestamps, deflate."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(
                member, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def repro_csv_bytes(G, n_points) -> bytes:
    """One row per node: our refit per coefficient, and the point count.

    Only our own numbers are written. The published values and the per-node
    relative differences are deliberately **not** in this file (Q17a: nothing
    from the GPL-3.0 O25 repository enters this BSD-3 tree); the markdown
    reports the relative differences as quantiles, and anyone with the
    checkout can recompute the per-node differences by reading it next to
    this file.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    header = ["theta_s", "theta_v", "dphi_pb24", "dphi_o25", "in_window", "n_points"]
    header += [f"{name}_refit" for name in COEFFS]
    writer.writerow(header)
    inside = window_mask_nodes()
    for i, ts in enumerate(B.O25_PUBLISHED_THETA_S):
        for j, tv in enumerate(B.O25_PUBLISHED_THETA_V):
            for k, ph in enumerate(B.O25_PUBLISHED_DPHI):
                row = [
                    f"{ts:g}",
                    f"{tv:g}",
                    f"{ph:g}",
                    f"{180.0 - ph:g}",
                    int(inside[i, j, k]),
                    int(n_points[i, j, k]),
                ]
                row += [f"{G[i, j, k, c]:.9e}" for c in range(4)]
                writer.writerow(row)
    return buf.getvalue().encode()


def score_csv_bytes(rows) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["model", "cut", "n_points", "median_abs_rel_pct", "q90_abs_rel_pct"]
    )
    for model, cut, n, med, q90 in rows:
        writer.writerow([model, cut, n, f"{med * 100:.4f}", f"{q90 * 100:.4f}"])
    return buf.getvalue().encode()


def pct(x, digits=3):
    return f"{x * 100:.{digits}f}%"


def summary_table(stats_in, stats_out) -> str:
    lines = [
        "| coefficient | window: median | q90 | max | signed median "
        "| outside: median | q90 | max | signed median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in COEFFS:
        a, b = stats_in[name], stats_out[name]
        lines.append(
            f"| {name} | {pct(a[0])} | {pct(a[1])} | {pct(a[2])} | {pct(a[3])} "
            f"| {pct(b[0])} | {pct(b[1])} | {pct(b[2])} | {pct(b[3])} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- main -------


def run(realisations, *, quick: bool):
    t0 = time.perf_counter()
    batch, bw = load(realisations)
    t_load = time.perf_counter() - t0
    n_real = batch.report.n_realisation
    print(
        f"loaded {n_real} realisations x 1300 geometries x {batch.n_wave} bands "
        f"= {batch.n_sample * batch.n_wave:,} points ({t_load:.1f} s)",
        flush=True,
    )
    idx = geometry_index(batch)
    inside = window_mask_nodes()

    published = B.o25_published_table()
    as_coded = B.o25_published_table(azimuth="o25")
    P = np.asarray(published.G)

    t1 = time.perf_counter()
    table = refit(batch)
    t_fit = time.perf_counter() - t1
    print(f"paper-recipe refit: {t_fit:.1f} s", flush=True)
    for ours, theirs in zip(
        (table.theta_s, table.theta_v, table.dphi),
        (published.theta_s, published.theta_v, published.dphi),
        strict=True,
    ):
        if not np.array_equal(ours, theirs):
            raise RuntimeError("refit nodes differ from the published nodes")
    G = np.asarray(table.G)
    n_points = points_per_node(batch, idx)
    rel = relative(G, P)
    main_in, main_out = summary(rel, inside), summary(rel, ~inside)

    # ------------------------------------------------ diagnosis --------------
    diag = {}
    diag["as coded by O25 (block b at dphi_pb24 = 15 b)"] = relative(G, as_coded.G)
    diag["(b) zenith swap (rows <-> columns), PB24 azimuth"] = relative(
        G, P.transpose(1, 0, 2, 3)
    )
    diag["(b) zenith swap and O25's azimuth labelling"] = relative(
        G, np.asarray(as_coded.G).transpose(1, 0, 2, 3)
    )
    raw = raw_blocks(as_coded)
    ts_major = np.stack(
        [r.reshape(10, 13, 10).transpose(0, 2, 1) for r in raw], axis=-1
    )  # rows theta_s-major: (ts, block, tv) -> (ts, tv, block)
    diag["(c) block order: theta_s-major rows, mirrored"] = relative(
        G, ts_major[:, :, ::-1]
    )
    diag["(c) block order: theta_s-major rows, as labelled"] = relative(G, ts_major)

    t2 = time.perf_counter()
    Rrs = np.asarray(batch.Rrs)
    spectrum_ok = np.all(Rrs <= B.O25_RRS_CEILING, axis=-1)
    variants = {
        "(d) cut per spectrum (any band > 0.06 drops the spectrum)": dict(
            train=spectrum_ok, rrs_max=None
        ),
        "(d') no cut": dict(rrs_max=None),
        "(e) relatively weighted": dict(weighted=True),
    }
    refits = {}
    for name, kwargs in variants.items():
        refits[name] = refit(batch, **kwargs)
        diag[name] = relative(np.asarray(refits[name].G), P)
    half_bw = IOPs(
        a=batch.iops.a,
        bb_w=jax.numpy.broadcast_to(jax.numpy.asarray(bw / 2.0), batch.iops.a.shape),
        bb_p=batch.iops.bb_p,
    )
    G_half = np.asarray(refit(batch, iops=half_bw).G)
    diag["(g) bb_w = bw/2 instead of the files' bbw"] = relative(G_half, P)
    t_diag = time.perf_counter() - t2
    print(f"diagnosis refits: {t_diag:.1f} s", flush=True)

    # (f) the 87.5 / 87.75 label: score the published table both ways at PB24's
    # theta_s = 87.75 samples. Evaluating at a node clamps either way.
    relabelled = B.O25Table(
        theta_s=np.r_[np.asarray(B.O25_PUBLISHED_THETA_S)[:-1], 87.5],
        theta_v=published.theta_v,
        dphi=published.dphi,
        G=published.G,
        provenance="published, last theta_s node labelled 87.5 as O25's code",
    )
    last = np.asarray(batch.geometry.theta_s) == B.O25_PUBLISHED_THETA_S[-1]
    sub = IOPs(
        a=batch.iops.a[last], bb_w=batch.iops.bb_w[last], bb_p=batch.iops.bb_p[last]
    )
    sub_geo = Geometry(
        theta_s=batch.geometry.theta_s[last],
        theta_v=batch.geometry.theta_v[last],
        dphi=batch.geometry.dphi[last],
    )
    f_a = np.asarray(B.Rrs_o25(sub, None, sub_geo, coeffs=published))
    f_b = np.asarray(B.Rrs_o25(sub, None, sub_geo, coeffs=relabelled))
    f_max = float(np.max(np.abs(f_a - f_b)))

    # predicted Gw0 shift under (g): Gw0 scales as 1 / (bw/2 / bbw)
    g_ratio = (bw / 2.0) / np.asarray(batch.iops.bb_w)[0]
    g_pred = float(np.mean(g_ratio) - 1.0)

    # ------------------------------------------------ score -----------------
    t3 = time.perf_counter()
    groups = score_groups(batch)
    pred_pub = predict(batch, published)
    pred_fit = predict(batch, table)
    pred_coded = predict(batch, as_coded)
    rows = score_rows("O25 published (calibrated on all 5000)", pred_pub, Rrs, groups)
    rows += score_rows(
        "O25 paper-recipe refit (calibrated on all 5000)", pred_fit, Rrs, groups
    )
    rows += score_rows(
        "O25 published, azimuth as O25's code labels it (wrong for PB24)",
        pred_coded,
        Rrs,
        groups[:3],
    )
    low = Rrs <= B.O25_RRS_CEILING
    for label, pred in (
        ("O25 published (calibrated on all 5000)", pred_pub),
        ("O25 paper-recipe refit (calibrated on all 5000)", pred_fit),
    ):
        err = np.abs(pred / Rrs - 1.0)[low]
        q = np.quantile(err, SCORE_QUANTILES)
        rows.append(
            (label, "all, Rrs <= 0.06", int(err.size), float(q[0]), float(q[1]))
        )
    t_score = time.perf_counter() - t3
    pub_vs_fit = float(np.max(np.abs(pred_fit / pred_pub - 1.0)))
    share_high = float(1.0 - low.mean())
    print(f"scoring: {t_score:.1f} s", flush=True)

    lut = {
        "theta_s": table.theta_s,
        "theta_v": table.theta_v,
        "dphi": table.dphi,
        "G": G,
        "coefficients": np.array(COEFFS),
        "n_points": n_points,
        "provenance": np.array(
            f"O25 form refit on PB24 (OLCI set, {n_real} realisations, all 1300 "
            f"geometries, 12 bands) by design/py/reproduce_o25.py: the paper's "
            f"recipe, {table.provenance}; dphi in PB24's convention "
            f"(radiance propagation azimuth, 0 towards the sun); calibrated on all "
            f"{n_real} realisations"
        ),
    }
    return dict(
        n_real=n_real,
        n_points_total=int(batch.n_sample * batch.n_wave),
        lut=lut,
        lut_bytes=npz_bytes(lut),
        repro_csv=repro_csv_bytes(G, n_points),
        score_csv=score_csv_bytes(rows),
        rows=rows,
        G=G,
        P=P,
        G_half=G_half,
        n_points=n_points,
        main=(main_in, main_out),
        diag={k: (summary(v, inside), summary(v, ~inside)) for k, v in diag.items()},
        f_max=f_max,
        g_pred=g_pred,
        pub_vs_fit=pub_vs_fit,
        share_high=share_high,
        published=published,
        times=dict(load=t_load, fit=t_fit, diag=t_diag, score=t_score),
        quick=quick,
    )


def diag_line(name, stats, predicted):
    w_in, w_out = stats
    med = max(w_in[c][0] for c in COEFFS)
    mx = max(w_in[c][2] for c in COEFFS)
    mx_out = max(w_out[c][2] for c in COEFFS)
    return f"| {name} | {predicted} | {pct(med, 3)} | {pct(mx, 3)} | {pct(mx_out, 2)} |"


def repro_markdown(res, wall) -> str:
    main_in, main_out = res["main"]
    max_in = max(main_in[c][2] for c in COEFFS)
    holds = max_in < PREDICTED_MAX
    coded_in = res["diag"]["as coded by O25 (block b at dphi_pb24 = 15 b)"][0]
    coded_med = max(coded_in[c][0] for c in COEFFS)
    coded_max = max(coded_in[c][2] for c in COEFFS)
    half = relative(res["G_half"], res["P"])
    abs_half = np.abs(res["G_half"] - res["P"])
    within_ulp = float(np.mean(abs_half <= 2 * FILE_HALF_ULP))
    half_in, half_out = res["diag"]["(g) bb_w = bw/2 instead of the files' bbw"]
    n_pts = res["n_points"]
    sample = (
        f"all {res['n_real']} realisations"
        if not res["quick"]
        else f"--quick: {res['n_real']} realisations (every {QUICK_STEP}th)"
    )
    predictions = {
        "as coded by O25 (block b at dphi_pb24 = 15 b)": (
            "if PB24's phi and O25's dphi agree: < 1% everywhere"
        ),
        "(b) zenith swap (rows <-> columns), PB24 azimuth": (
            "agrees only where theta_s = theta_v; >> 1% elsewhere"
        ),
        "(b) zenith swap and O25's azimuth labelling": ">> 1%",
        "(c) block order: theta_s-major rows, mirrored": "scrambled: >> 1%",
        "(c) block order: theta_s-major rows, as labelled": "scrambled: >> 1%",
        "(d) cut per spectrum (any band > 0.06 drops the spectrum)": (
            "if it were theirs: < 1%; else O(1%) shifts"
        ),
        "(d') no cut": "if the cut were absent: < 1%; else several %",
        "(e) relatively weighted": "if theirs: < 1%; else tens of % (L23: 4x rRMS)",
        "(g) bb_w = bw/2 instead of the files' bbw": (
            f"if theirs: Gw0 offset closes (predicted shift "
            f"{pct(res['g_pred'], 3)}), residual at the files' 6-decimal floor"
        ),
    }
    diag_rows = [
        "| hypothesis (published file read as ...) | predicts | window: worst "
        "coefficient median | window max | outside max |",
        "|---|---|---|---|---|",
        f"| **adopted: (a) azimuth mirrored, dphi_pb24 = 180 - dphi_o25** | "
        f"< 1% in the window (task prediction) | "
        f"{pct(max(main_in[c][0] for c in COEFFS))} | {pct(max_in)} | "
        f"{pct(max(main_out[c][2] for c in COEFFS), 2)} |",
    ]
    diag_rows += [
        diag_line(name, stats, predictions[name]) for name, stats in res["diag"].items()
    ]
    diag_rows.append(
        f"| (f) last theta_s node labelled 87.5 (O25's code) vs 87.75 (PB24) | "
        f"0 at every PB24 sample (nodes clamp) | max abs Rrs difference at "
        f"theta_s = 87.75: {res['f_max']:.1e} | - | - |"
    )
    t = res["times"]
    return f"""# O25 reproduced from PB24 (M8 task 5)

Generated by `design/py/reproduce_o25.py`; {sample}, all 1300 geometries, the
12 OLCI bands ({res["n_points_total"]:,} points). The published LUT is read
from the O25 checkout ({res["published"].provenance}).

**Recipe** (Pitarch et al. 2025 §2.2): `fit_o25_table(train=all True,
weighted=False, rrs_max=0.06)`: one unweighted float64 lstsq per node, points
(sample x band) with Rrs > 0.06 sr^-1 dropped **per point**, the files' own
water IOPs (`bb_w = bbw`). Points per node after the cut: min {n_pts.min():,},
median {int(np.median(n_pts)):,}, max {n_pts.max():,} (of
{res["n_real"] * 12:,}).

## Outcome of the prediction

The prediction (written before any fit) was: the four G's agree with the
published files to better than 1% at every in-window node, the residual
traceable to float32 storage and the Rrs cut.

- **Magnitude: {"confirmed" if holds else "NOT confirmed"}** once the azimuth
  axis is read in O25's convention. With dphi_pb24 = 180 - dphi_o25 the largest
  in-window relative difference over all four coefficients is **{pct(max_in)}**
  ({"<" if holds else ">="} 1%).
- **But not on the first reading.** Read as O25's own code labels the blocks
  against PB24's phi (block b at phi = 15 b), the agreement is a worst-coefficient
  median {pct(coded_med, 1)} and a maximum {pct(coded_max, 1)}: a convention
  difference, as the prediction said a large disagreement would be. PB24's phi is
  the azimuth of radiance *propagation*, zero towards the sun (the sensor looks
  away from the sun); O25's dphi = 0 is "looking at the sun" (its Fig. 8). Both
  papers are self-consistent; the mirror is the translation. The reader
  (`baselines.o25_published_table`) stores the table in PB24's convention.
- **Mechanism: not as predicted.** The residual is not float32 storage or the
  cut. It is a uniform **{pct(main_in["Gw0"][3])}** offset in Gw0 (signed
  median), which (g) traces to the water backscattering: the calibration used
  bb_w = bw/2, while the files carry bbw = (0.4992-0.4999) bw. With bw/2 the
  largest relative difference anywhere (all 1300 nodes) is
  {pct(max(max(half_in[c][2], half_out[c][2]) for c in COEFFS), 4)}, and
  {within_ulp:.1%} of the 5200 values are within two half-units of the files'
  sixth decimal ({2 * FILE_HALF_ULP:.0e}); the median |relative difference| is
  {pct(float(np.median(np.abs(half))), 4)}. That is the text files' rounding
  floor. The committed refit keeps the files' bbw, as the task specified.

## Relative difference refit / published - 1 (adopted reading)

Window: theta_s <= 70 and theta_v <= 70 (8 x 8 x 13 = 832 nodes); outside: the
other 468. Quantiles of |relative difference| over nodes.

{summary_table(main_in, main_out)}

With bb_w = bw/2 (hypothesis g):

{summary_table(half_in, half_out)}

## Diagnosis

Each alternative reading of the files, or each alternative recipe, against our
refit (the adopted reading's refit for the readings; the published table,
adopted reading, for the recipe variants). The hypotheses were first run in a
scratch probe while investigating, so the "predicts" column states what each
hypothesis implies if it were the cause, not a blind forecast; the numbers are
re-measured here.

{chr(10).join(diag_rows)}

(a) and (b) are also the positive controls: un-mirroring the adopted table
or swapping its zenith axes makes the agreement far worse, so the test can
tell the readings apart.

## Wall time

Load {t["load"]:.0f} s, paper-recipe refit {t["fit"]:.0f} s, diagnosis refits
{t["diag"]:.0f} s, scoring {t["score"]:.0f} s; total {wall:.0f} s.

Per-node values: `o25_reproduction.csv` (our refit per coefficient, the point
count, dphi in both conventions). The published values and the per-node
relative differences are not written (Q17a: nothing from the GPL-3.0 checkout
enters the tree); the quantiles above are the comparison. Refit LUT:
`o25_refit_lut.npz`. `--check` recomputes and compares both, and
`o25_published_score.csv`, byte for byte.
"""


def score_markdown(res) -> str:
    rows = res["rows"]
    by = {(m, c): (n, med, q) for m, c, n, med, q in rows}
    pub = "O25 published (calibrated on all 5000)"
    fit = "O25 paper-recipe refit (calibrated on all 5000)"
    coded = "O25 published, azimuth as O25's code labels it (wrong for PB24)"

    def line(cut):
        n, a, b = by[(pub, cut)]
        _, c, d = by[(fit, cut)]
        cells = " | ".join(f"{v * 100:.3f}" for v in (a, b, c, d))
        return f"| {cut} | {n:,} | {cells} |"

    head = (
        "| cut | points | published: median % | q90 % | refit: median % | q90 % |\n"
        "|---|---|---|---|---|---|"
    )
    cuts = [c for m, c, *_ in rows if m == pub]
    main = [c for c in cuts if c in ("window", "shell", "all", "all, Rrs <= 0.06")]
    per_v = [c for c in cuts if c.startswith("theta_v=")]
    per_vw = [c for c in cuts if c.startswith("window, theta_v=")]
    per_s = [c for c in cuts if c.startswith("theta_s=")]
    coded_lines = "\n".join(
        f"| {cut} | {by[(coded, cut)][1] * 100:.3f} | {by[(coded, cut)][2] * 100:.3f} |"
        for cut in ("window", "shell", "all")
    )
    sample = (
        f"all {res['n_real']} realisations"
        if not res["quick"]
        else f"--quick: {res['n_real']} realisations"
    )
    return f"""# O25 published, scored on PB24 (M8 task 5)

Generated by `design/py/reproduce_o25.py`. **Both models were calibrated on all
5000 PB24 realisations** (the published LUT by Pitarch et al. 2025, the refit by
the same recipe here), so these are in-sample numbers on every PB24 split; the
split-wise refit is task 6's.

Sample: {sample}, PB24 OLCI set (12 bands, 400-753 nm), all 1300 above-water
geometries; true IOPs in (`a`, `bb_w = bbw`, `bb_p` from the files), `Rrs`
out, `|Rrs_pred/Rrs - 1|` over every (sample, band) point, in `Rrs` (O25's
native space; no surface transfer). Window: theta_s <= 70 and theta_v_air <= 70.
Published table: `baselines.o25_published_table()` (azimuth in PB24's
convention). {pct(res["share_high"], 2)} of the points have Rrs > 0.06 sr^-1,
outside O25's calibration range; they are scored anyway ("all"), and once
without them.

The refit and the published table predict the same Rrs to within
{pct(res["pub_vs_fit"], 3)} at every point, which is the labelling made
visible: our refit *is* the published model to that precision.

{head}
{chr(10).join(line(c) for c in main)}

Per theta_v_air (every theta_s):

{head}
{chr(10).join(line(c) for c in per_v)}

Per theta_v_air inside the window (theta_s <= 70):

{head}
{chr(10).join(line(c) for c in per_vw)}

Per theta_s (every theta_v_air):

{head}
{chr(10).join(line(c) for c in per_s)}

The cost of the azimuth convention: the published table indexed as O25's code
labels the blocks, against PB24's dphi:

| cut | median % | q90 % |
|---|---|---|
{coded_lines}

Values: `o25_published_score.csv`.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"every {QUICK_STEP}th realisation; prints, writes nothing",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recompute on all 5000; compare the committed CSVs and LUT byte for byte",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.quick and args.check:
        parser.error("--check recomputes the committed artefacts; not with --quick")

    start = time.perf_counter()
    realisations = (
        tuple(range(1, pb24.N_REALISATION + 1, QUICK_STEP)) if args.quick else None
    )
    res = run(realisations, quick=args.quick)
    wall = time.perf_counter() - start
    repro_md = repro_markdown(res, wall)
    score_md = score_markdown(res)
    outputs = {
        LUT_FILE: res["lut_bytes"],
        REPRO_CSV: res["repro_csv"],
        SCORE_CSV: res["score_csv"],
        REPRO_MD: repro_md.encode(),
        SCORE_MD: score_md.encode(),
    }

    if args.check:
        bad = []
        for name in CHECKED:
            path = args.out_dir / name
            same = path.is_file() and path.read_bytes() == outputs[name]
            print(f"check: {name}: {'byte-identical' if same else 'DIFFERS'}")
            if not same:
                bad.append(name)
        print(f"wall time {wall:.1f} s")
        return 1 if bad else 0

    if args.quick:
        print(repro_md)
        print(score_md)
        print(f"wall time {wall:.1f} s (--quick: nothing written)")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        (args.out_dir / name).write_bytes(payload)
        print(f"wrote {args.out_dir / name} ({len(payload):,} bytes)")
    print(f"wall time {wall:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
