#!/usr/bin/env python
"""
Fit ZTT's PB24 terms -- M8 task 8a, ``rt_fit_pb24.md`` (Q21, Q22; "ZTT (PB24)").

Three stages, all deterministic (streamed Householder QR in float64 over
chunks of realisations in id order; no optimiser), fitted on the realisation
split's **training** realisations only and scored on its held-out ones:

(h) **The diffuse-fraction table** ``H(θs', λ) = 1 − Esdir_Es_ratio`` (Q21a).
    An above-surface quantity, identical in every file (asserted bit for bit
    over the training realisations, and reported over all 5,000). Written as
    ``robust/rt/files/diffuse_fraction_pb24.npz`` (fixed zip timestamps, no
    build time, so a rebuild is byte-identical) and served by
    :func:`robust.rt.ztt.diffuse_fraction_pb24`, bilinear in (cos θs', λ).
    The interpolation is tested by leaving the 30° and 60° nodes out.
(d) **µd** (Q21b): Equation (17)'s eight ``M*_d`` coefficients fitted *jointly*
    with the atmospheric correction c(θs') = 1 + c1 u + c2 u², u = 1 − cos θs',
    on TT's ``M+_d`` evaluated **with the table H**, against PB24's own
    ``mu_d`` on the window suns (θs ≤ 70°), all bands. M7's exact
    linearisation: with t = µd / M+_d(θ, H) and g_θ = 1/c(θ), the rows
    ``M*_d(x)/t − g_θ = 0`` (θ > 0) and ``M*_d(x)/t = 1`` (θ = 0) are linear in
    (m*, g_10 … g_70). With eight suns the seven free nodes over-determine the
    quadratic, so its fit to the nodes is reported as a test of the form.
    The (X, L) = (log10 bb/a, log10 η_bb) box is PB24's training range
    (bb/a to ~20, η_bb to ~3e-6); outputs are bounded as M7's.
(i) **µ∞** (Q22a): Equation (8)'s sixteen coefficients refit to the task-6 eigen
    product (``pb24_mu_inf_eigen_v1.npz``, PB24's IOPs and FF pair) on every
    training (realisation, λ), relative residuals, same box, **no B_p term**
    (JXP, Q22): the B_p trend the form cannot absorb is reported per bin.

Held-out errors are evaluated **through the functions the package ships**
(``ztt.mu_infinity``, ``ztt.mu_d`` with ``box`` / ``md_plus_corr`` / ``H``) and
cross-checked against this script's own design matrices (the M7 gate).

Writes ``design/ztt_pb24/fit_*``: ``fit_coefficients.json`` (the source of the
``*_PB24_*`` constants in ``ztt.py``), ``fit_metrics.{json,md}``,
``fit_probes.json`` (points the tests re-evaluate through ``ztt``, no data
needed) and two figures. Reads only the per-realisation table of the cache
(no 4-D arrays): about a minute, ~2 GB.

    python design/py/fit_ztt_pb24.py                   # fit, write
    python design/py/fit_ztt_pb24.py --dry-run         # fit, report, write nothing
    python design/py/fit_ztt_pb24.py --check           # refit, compare byte for byte
    python design/py/fit_ztt_pb24.py --quick --out-dir /tmp/x   # 200/50 realisations
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # fits and metrics in float64

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robust.rt import conventions as C  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402
from robust.rt.data import pb24_cache as PC  # noqa: E402
from robust.rt.data import pb24_eigen  # noqa: E402

OUT = ROOT / "design" / "ztt_pb24"
PREFIX = "fit_"
H_FILE = ROOT / "robust" / "rt" / Z.H_PB24_FILE

#: Realisations per QR update. Part of the fit's definition: the streamed
#: factorisation is exact in exact arithmetic for any chunking, but the last
#: bits depend on it, so a rerun must use the same value.
CHUNK = 250

#: ``--quick``: training and test realisations (the first n of each side).
QUICK = (200, 50)

#: The suns the µd fit uses: the window (θs ≤ 70°, Q21). The two shell suns are
#: scored as extrapolation.
SUN_MAX_DEG = P.ANGLE_WINDOW_MAX

#: bb/a bins for the held-out tables: task 6's edges (0.1, 0.5946 = L23's
#: maximum) plus the PB24 extension beyond them.
BBA_EDGES = (0.0, 1e-3, 1e-2, 0.1, 0.5946, 2.0, np.inf)
BP_EDGES = (0.0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, np.inf)

#: Probe points committed for the tests (seeded).
N_PROBES = 12
PROBE_SEED = P.SPLIT_SEED

_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


# -------------------------------------------------------- streamed LSQ -----
class StreamingLSQ:
    """Linear least squares by a streamed Householder QR.

    Rows arrive in chunks; the ``(n_col + 1) × (n_col + 1)`` triangular factor
    ``R`` of the augmented matrix ``[D | b]`` is updated by factorising
    ``[R; D_chunk | b_chunk]``. The solution, the residual norm and the
    condition number of ``D`` come from ``R`` alone, so nothing of size
    ``n_rows`` is ever held, and the conditioning is that of ``D`` itself (not
    squared, as the normal equations would).
    """

    def __init__(self, n_col: int):
        self.n_col = int(n_col)
        self.R = None
        self.n_rows = 0

    def add(self, D: np.ndarray, b: np.ndarray) -> None:
        """Append rows ``D (n, n_col)`` with targets ``b (n,)``."""
        D = np.asarray(D, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if D.ndim != 2 or D.shape[1] != self.n_col or b.shape != (D.shape[0],):
            raise ValueError(f"StreamingLSQ: bad shapes {D.shape}, {b.shape}")
        M = np.concatenate([D, b[:, None]], axis=1)
        if self.R is not None:
            M = np.concatenate([self.R, M], axis=0)
        self.R = np.linalg.qr(M, mode="r")
        self.n_rows += D.shape[0]

    def solve(self) -> tuple[np.ndarray, float, float]:
        """``(coefficients, residual 2-norm, cond(D))``."""
        if self.R is None:
            raise ValueError("StreamingLSQ: no rows")
        R, qtb = self.R[:-1, :-1], self.R[:-1, -1]
        coeffs = np.linalg.solve(R, qtb)
        return coeffs, float(abs(self.R[-1, -1])), float(np.linalg.cond(R))


# ---------------------------------------------------- design matrices -----
def mu_inf_design(X, Lg):
    """Equation (8)'s 16 columns in ``ztt.mu_infinity``'s coefficient order:
    m[4j + i] multiplies L^(3−i) X^(3−j)."""
    return np.stack(
        [Lg ** (3 - i) * X ** (3 - j) for j in range(4) for i in range(4)], axis=-1
    )


def md_star_design(X, Lg):
    """Equation (17)'s 8 columns in ``ztt.Md_star``'s order."""
    one = np.ones_like(X)
    return np.stack([Lg * X**3, X**3, Lg * X**2, X**2, Lg * X, X, Lg, one], axis=-1)


def mu_d_rows(X, Lg, theta, t, nodes):
    """Rows of the joint µd fit: ``(design (n, 8 + n_free), rhs (n,))``.

    ``nodes`` are the sun zeniths used (the first must be 0°); the free
    unknowns are g_θ = 1/c(θ) for the non-zero nodes, in ``nodes`` order.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes[0] != 0.0:
        raise ValueError("mu_d_rows: the first node must be 0 deg (c(0) = 1)")
    D = md_star_design(X, Lg) / t[:, None]
    G = np.stack([-(theta == z).astype(float) for z in nodes[1:]], axis=-1)
    return np.concatenate([D, G], axis=1), (theta == 0.0).astype(float)


def corr_from_nodes(nodes, c_nodes):
    """``(c1, c2)`` of c(θ) = 1 + c1 u + c2 u² fitted to the free nodes by LSQ
    (over-determined with more than two nodes), and the quadratic's value at
    each node."""
    u = 1.0 - np.cos(np.deg2rad(np.asarray(nodes, dtype=float)))
    A = np.stack([u, u**2], axis=-1)
    sol, *_ = np.linalg.lstsq(A, np.asarray(c_nodes) - 1.0, rcond=None)
    return (float(sol[0]), float(sol[1])), 1.0 + A @ sol


def fit_mu_d(X, Lg, theta, t, nodes):
    """In-memory joint fit (tests): ``(m8, (c1, c2), {node: c_node})``."""
    lsq = StreamingLSQ(8 + len(nodes) - 1)
    D, rhs = mu_d_rows(X, Lg, theta, t, nodes)
    lsq.add(D, rhs)
    sol, _, _ = lsq.solve()
    c_nodes = 1.0 / sol[8:]
    corr, _ = corr_from_nodes(nodes[1:], c_nodes)
    return sol[:8], corr, dict(zip(nodes[1:], c_nodes, strict=True))


def fit_mu_inf(X, Lg, target):
    """In-memory relative-residual fit (tests): ``(m16, cond)``."""
    lsq = StreamingLSQ(16)
    lsq.add(mu_inf_design(X, Lg) / target[:, None], np.ones_like(target))
    m16, _, cond = lsq.solve()
    return m16, cond


# ------------------------------------------------------------- stats -----
def output_range(values, *, cosine=False):
    """M7's smooth output bound: the training values' range padded by five
    ``OUT_CLAMP_WIDTH``; an average cosine is capped at 1 − ``OUT_CLAMP_WIDTH``."""
    pad = 5 * Z.OUT_CLAMP_WIDTH
    lo = float(np.min(values)) - pad
    hi = float(np.max(values)) + pad
    if cosine:
        hi = min(hi, 1.0 - Z.OUT_CLAMP_WIDTH)
    return [lo, hi]


def pct(values, qs=(1, 16, 50, 84, 99)):
    return [float(v) for v in np.nanpercentile(np.ravel(values), qs)]


def describe(rel):
    """Relative errors: median |.|, mean |.|, q90 |.|, signed percentiles."""
    rel = np.ravel(rel)
    return {
        "n": int(rel.size),
        "mape": float(np.nanmedian(np.abs(rel))),
        "mean_abs": float(np.nanmean(np.abs(rel))),
        "q90_abs": float(np.nanpercentile(np.abs(rel), 90)),
        "signed_pct_1_16_50_84_99": pct(rel),
    }


def binned(rel, key, edges, labels=None):
    """``describe`` per bin of ``key``."""
    out = {}
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (key >= lo) & (key < hi)
        label = labels[k] if labels else f"[{lo:g}, {hi:g})"
        out[label] = describe(rel[m]) if m.any() else {"n": 0}
    return out


# ------------------------------------------------------------ inputs -----
def split_ids(quick: bool):
    """``(train_ids, test_ids)`` of the realisation split, ascending."""
    content = P.read_splits()
    every = np.asarray(content["realisations"], dtype=int)
    test = np.asarray(sorted(content["realisation"]["test"]), dtype=int)
    train = np.setdiff1d(every, test)
    assert np.intersect1d(train, test).size == 0 and test.size > 0
    if quick:
        train, test = train[: QUICK[0]], test[: QUICK[1]]
    return train, test


def read_table(cache, ids, names):
    """Table fields for ``ids``, float64."""
    return {n: np.asarray(cache.table(n, ids), dtype=np.float64) for n in names}


IOP_NAMES = ("aw", "aph", "ag", "aNAP", "bph", "bNAP", "bbw", "bbph", "bbNAP")


def iop_logs(tab):
    """``(X, L, B_p)`` per (realisation, λ) from the table's components."""
    a = tab["aw"] + tab["aph"] + tab["ag"] + tab["aNAP"]
    bbp = tab["bbph"] + tab["bbNAP"]
    bb = tab["bbw"] + bbp
    with np.errstate(divide="ignore", invalid="ignore"):
        B_p = bbp / (tab["bph"] + tab["bNAP"])
    return np.log10(bb / a), np.log10(tab["bbw"] / bb), B_p


# ----------------------------------------------------------- stage (h) -----
def h_table(cache, train_ids, all_ids):
    """The H table from the training realisations, and the independence check."""
    E = np.asarray(cache.table("Esdir_Es_ratio", train_ids))  # float32 (n, θs, λ)
    diff_train = float(np.max(np.abs(E - E[:1])))
    if diff_train != 0.0:
        raise RuntimeError(
            f"Esdir_Es_ratio differs between training realisations (max {diff_train})"
        )
    E_all = np.asarray(cache.table("Esdir_Es_ratio", all_ids))
    diff_all = float(np.max(np.abs(E_all - E[:1])))
    H = 1.0 - E[0].astype(np.float64)  # exact: float32 values promoted
    theta_s = np.asarray(cache.grids["theta_s"], dtype=np.float64)
    wave = np.asarray(cache.grids["lambda"], dtype=np.float64)
    return (
        theta_s,
        wave,
        H,
        {"train_max_abs_diff": diff_train, "all_max_abs_diff": diff_all},
    )


def h_file_bytes(theta_s, wave, H, provenance: dict) -> bytes:
    """The ``.npz`` bytes, deterministic (fixed member timestamps)."""
    arrays = {
        "theta_s": np.asarray(theta_s, dtype=np.float64),
        "wave": np.asarray(wave, dtype=np.float64),
        "H": np.asarray(H, dtype=np.float64),
        "provenance": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.save(member, arr, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def h_interpolation_check(theta_s, wave, H):
    """Leave each interior node out and interpolate it (linear in sec θs', the
    package's variable) from its neighbours; the node spacing is then twice the
    real one, so this bounds the shipped interpolation from above."""
    sec = 1.0 / np.cos(np.deg2rad(theta_s))
    out = {}
    for k in range(1, theta_s.size - 1):
        t = (sec[k] - sec[k - 1]) / (sec[k + 1] - sec[k - 1])
        H_int = (1 - t) * H[k - 1] + t * H[k + 1]
        rel = H_int / H[k] - 1
        plus_true = np.asarray(Z.Md_plus(theta_s[k], H=H[k]))
        plus_int = np.asarray(Z.Md_plus(theta_s[k], H=H_int))
        prel = plus_int / plus_true - 1
        out[f"{theta_s[k]:g}"] = {
            "H_rel_err_median": float(np.median(np.abs(rel))),
            "H_rel_err_max": float(np.max(np.abs(rel))),
            "Md_plus_rel_err_median": float(np.median(np.abs(prel))),
            "Md_plus_rel_err_max": float(np.max(np.abs(prel))),
        }
    return out


# ----------------------------------------------------------- stage (d) -----
def fit_mu_d_streamed(cache, train_ids, theta_s, H, *, progress=True):
    """Stream the joint µd rows over the training realisations (window suns)."""
    nodes = theta_s[theta_s <= SUN_MAX_DEG]
    k_win = np.where(theta_s <= SUN_MAX_DEG)[0]
    plus = np.asarray(Z.Md_plus(nodes[:, None], H=H[k_win]))  # (n_sun, λ)
    lsq = StreamingLSQ(8 + nodes.size - 1)
    box = np.array([np.inf, -np.inf, np.inf, -np.inf])
    md_vals = []
    t0 = time.time()
    for start in range(0, train_ids.size, CHUNK):
        ids = train_ids[start : start + CHUNK]
        tab = read_table(cache, ids, IOP_NAMES + ("mu_d",))
        X, Lg, _ = iop_logs(tab)
        box = [
            min(box[0], X.min()),
            max(box[1], X.max()),
            min(box[2], Lg.min()),
            max(box[3], Lg.max()),
        ]
        mud = tab["mu_d"][:, k_win, :]  # (n, n_sun, λ)
        t = mud / plus[None]
        shape = mud.shape
        X3 = np.broadcast_to(X[:, None, :], shape)
        L3 = np.broadcast_to(Lg[:, None, :], shape)
        th3 = np.broadcast_to(nodes[None, :, None], shape)
        D, rhs = mu_d_rows(X3.ravel(), L3.ravel(), th3.ravel(), t.ravel(), nodes)
        lsq.add(D, rhs)
        md_vals.append(md_star_design(X.ravel(), Lg.ravel()))
        if progress:
            print(
                f"  mu_d: {min(start + CHUNK, train_ids.size)}/{train_ids.size} "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
    sol, resid, cond = lsq.solve()
    md8 = sol[:8]
    c_nodes = 1.0 / sol[8:]
    corr, c_quad = corr_from_nodes(nodes[1:], c_nodes)
    bare = np.concatenate(md_vals) @ md8
    return {
        "md8": md8,
        "corr": corr,
        "nodes": nodes,
        "c_nodes": c_nodes,
        "c_quad_at_nodes": c_quad,
        "box": [float(v) for v in box],
        "md_star_range": output_range(bare),
        "n_rows": lsq.n_rows,
        "resid_norm": resid,
        "cond": cond,
    }


# ----------------------------------------------------------- stage (i) -----
def fit_mu_inf_streamed(cache, train_ids, product, box, *, progress=True):
    """Stream Equation (8)'s relative-residual rows over the training rows."""
    lsq = StreamingLSQ(16)
    wpos = np.searchsorted(product.wave, np.asarray(cache.grids["lambda"], float))
    vals = []
    t0 = time.time()
    for start in range(0, train_ids.size, CHUNK):
        ids = train_ids[start : start + CHUNK]
        X, Lg, _ = iop_logs(read_table(cache, ids, IOP_NAMES))
        mu_e = product.mu_inf[np.ix_(product.rows(ids), wpos)]
        D = mu_inf_design(X.ravel(), Lg.ravel())
        lsq.add(D / mu_e.ravel()[:, None], np.ones(mu_e.size))
        vals.append(D)
        if progress:
            print(
                f"  mu_inf: {min(start + CHUNK, train_ids.size)}/{train_ids.size} "
                f"({time.time() - t0:.0f} s)",
                flush=True,
            )
    m16, resid, cond = lsq.solve()
    bare = np.concatenate(vals) @ m16
    return {
        "m16": m16,
        "mu_inf_range": output_range(bare, cosine=True),
        "n_rows": lsq.n_rows,
        "resid_norm": resid,
        "cond": cond,
    }


# --------------------------------------------------------- evaluation -----
def evaluate(cache, test_ids, product, theta_s, wave, H, d, i):
    """Held-out metrics through the shipped evaluators; the M7 agreement gate."""
    tab = read_table(cache, test_ids, IOP_NAMES + ("mu_d",))
    X, Lg, B_p = iop_logs(tab)
    X10, eta = 10.0**X, 10.0**Lg
    box = d["box"]
    md8, corr = tuple(float(v) for v in d["md8"]), d["corr"]
    m16 = tuple(float(v) for v in i["m16"])

    # -- µd, every sun, through ztt.mu_d with the table H --------------------
    ts = theta_s[None, :, None]
    Xb, eb = X10[:, None, :], eta[:, None, :]
    Hb = H[None]
    pb24 = {
        "md_star_coeffs": md8,
        "md_plus_corr": corr,
        "box": box,
        "md_star_range": d["md_star_range"],
    }
    mud_fit = np.asarray(Z.mu_d(ts, Xb, eb, H=Hb, **pb24))
    mud_l23 = np.asarray(Z.mu_d_l23(ts, Xb, eb, H=Hb))
    mud_a2 = np.asarray(Z.mu_d(ts, Xb, eb, H=Hb))
    truth = tab["mu_d"]
    win = theta_s <= SUN_MAX_DEG
    bba3 = np.broadcast_to(X10[:, None, :], truth.shape)
    mu_d_metrics = {}
    for name, model in (
        ("PB24_fit", mud_fit),
        ("L23_fit_data_H", mud_l23),
        ("TT_A2_data_H", mud_a2),
    ):
        rel = model / truth - 1
        mu_d_metrics[name] = {
            "per_theta_s": {
                f"{z:g}": describe(rel[:, k, :]) for k, z in enumerate(theta_s)
            },
            "window_pooled": describe(rel[:, win, :]),
            "window_by_bb_over_a": binned(rel[:, win, :], bba3[:, win, :], BBA_EDGES),
        }
    # The quadratic against the free nodes (form test), and c at the nodes.
    c_form = {
        f"{z:g}": {"c_node": float(cn), "c_quadratic": float(cq)}
        for z, cn, cq in zip(
            d["nodes"][1:], d["c_nodes"], d["c_quad_at_nodes"], strict=True
        )
    }
    # Agreement: the evaluator reproduces the script's design where unguarded.
    inside = (
        (X > box[0] + 10 * Z.CLAMP_WIDTH)
        & (X < box[1] - 10 * Z.CLAMP_WIDTH)
        & (Lg > box[2] + 10 * Z.CLAMP_WIDTH)
        & (Lg < box[3] - 10 * Z.CLAMP_WIDTH)
    )
    bare_md = md_star_design(X, Lg) @ np.asarray(d["md8"])
    out_w = 10 * Z.OUT_CLAMP_WIDTH
    md_un = (
        inside
        & (bare_md > d["md_star_range"][0] + out_w)
        & (bare_md < d["md_star_range"][1] - out_w)
    )
    k_sel = np.where(md_un)
    plus = np.asarray(Z.Md_plus(theta_s[:, None], H=H))  # (θs, λ)
    corr_f = np.asarray(Z.md_plus_correction(theta_s, corr))  # (θs,)
    own = bare_md[k_sel][None, :] * plus[:, k_sel[1]] * corr_f[:, None]  # (θs, m)
    via = mud_fit[k_sel[0], :, k_sel[1]]  # (m, θs)
    agree_mud = float(np.max(np.abs(via.T / own - 1))) if k_sel[0].size else 0.0

    # -- µ∞, through ztt.mu_infinity ----------------------------------------
    wpos = np.searchsorted(product.wave, wave)
    mu_e = product.mu_inf[np.ix_(product.rows(test_ids), wpos)]
    mu_fit = np.asarray(
        Z.mu_infinity(X10, eta, m16, box=box, out_range=i["mu_inf_range"])
    )
    mu_l23 = np.asarray(Z.mu_infinity_l23(X10, eta))
    mu_tt = np.asarray(Z.mu_infinity_tt2017(X10, eta))
    mu_inf_metrics = {}
    for name, model in (("PB24_fit", mu_fit), ("L23_fit", mu_l23), ("TT2017", mu_tt)):
        rel = model / mu_e - 1
        mu_inf_metrics[name] = {
            "all": describe(rel),
            "by_bb_over_a": binned(rel, X10, BBA_EDGES),
            "by_B_p": binned(rel, B_p, BP_EDGES),
        }
    bare_mu = mu_inf_design(X, Lg) @ np.asarray(i["m16"])
    mu_un = (
        inside
        & (bare_mu > i["mu_inf_range"][0] + out_w)
        & (bare_mu < i["mu_inf_range"][1] - out_w)
    )
    agree_mu = (
        float(np.max(np.abs(mu_fit[mu_un] / bare_mu[mu_un] - 1)))
        if mu_un.any()
        else 0.0
    )
    guard = {
        "fraction_heldout_changed_gt_1e-4": float(
            np.mean(np.abs(mu_fit / bare_mu - 1) > 1e-4)
        ),
        "heldout_mape_bare": float(np.median(np.abs(bare_mu / mu_e - 1))),
        "heldout_mape_guarded": float(np.median(np.abs(mu_fit / mu_e - 1))),
    }
    outside = float(
        np.mean((X < box[0]) | (X > box[1]) | (Lg < box[2]) | (Lg > box[3]))
    )
    metrics = {
        "mu_d_heldout": mu_d_metrics,
        "c_form_at_nodes": c_form,
        "mu_inf_heldout_vs_eigen": mu_inf_metrics,
        "mu_inf_guard_effect": guard,
        "gate_evaluator_agreement": {"mu_inf": agree_mu, "mu_d": agree_mud},
        "fraction_heldout_outside_box": outside,
    }
    arrays = {
        "X": X, "Lg": Lg, "B_p": B_p, "theta_s": theta_s,
        "mud_rel": {
            "PB24_fit": mud_fit / truth - 1,
            "L23_fit_data_H": mud_l23 / truth - 1,
            "TT_A2_data_H": mud_a2 / truth - 1,
        },
        "mu_rel": {
            "PB24_fit": mu_fit / mu_e - 1,
            "L23_fit": mu_l23 / mu_e - 1,
            "TT2017": mu_tt / mu_e - 1,
        },
    }  # fmt: skip
    return metrics, arrays


def probes(cache, all_ids, product, theta_s, wave, H, d, i):
    """Seeded probe points, evaluated through the shipped evaluators with the
    fitted numbers; the tests re-evaluate them through the ``*_pb24`` functions."""
    rng = np.random.default_rng(PROBE_SEED)
    n = min(N_PROBES, all_ids.size)
    ids = rng.choice(all_ids, n, replace=False)
    ks = rng.integers(0, theta_s.size, n)
    ws = rng.integers(0, wave.size, n)
    tab = read_table(cache, np.sort(ids), IOP_NAMES)
    X, Lg, _ = iop_logs(tab)
    pos = {int(r): p for p, r in enumerate(np.sort(ids))}
    wpos = np.searchsorted(product.wave, wave)
    out = []
    for r, k, w in zip(ids, ks, ws, strict=True):
        p = pos[int(r)]
        x10, eta = float(10.0 ** X[p, w]), float(10.0 ** Lg[p, w])
        th = float(theta_s[k])
        Hv = float(H[k, w])
        mu = Z.mu_infinity(
            x10, eta, tuple(i["m16"]), box=d["box"], out_range=i["mu_inf_range"]
        )
        mud = Z.mu_d(
            th, x10, eta,
            md_star_coeffs=tuple(d["md8"]), md_plus_corr=d["corr"], box=d["box"],
            md_star_range=d["md_star_range"], H=Hv,
        )  # fmt: skip
        out.append(
            {
                "realisation": int(r),
                "theta_s": th,
                "wave_nm": float(wave[w]),
                "bb_over_a": x10,
                "eta_bb": eta,
                "H": Hv,
                "mu_inf_eigen": float(product.mu_inf[product.rows(int(r))[0], wpos[w]]),
                "mu_inf_pb24": float(mu),
                "mu_d_pb24": float(mud),
            }
        )
    return out


# ------------------------------------------------------------ outputs -----
def fmt(d):
    """One table cell: median |err| %, signed 16/50/84 %, q90 |err| %."""
    if d.get("n", 0) == 0:
        return "– | – | –"
    s = d["signed_pct_1_16_50_84_99"]
    return (
        f"{100 * d['mape']:.2f} | {100 * s[1]:+.2f} / {100 * s[2]:+.2f} / "
        f"{100 * s[3]:+.2f} | {100 * d['q90_abs']:.2f}"
    )


def report(coefficients, metrics, *, quick: bool) -> str:
    c, m = coefficients, metrics
    f = c["fit"]
    L = [
        "# ZTT fitted to PB24 — H, µd, µ∞ (M8 task 8a)",
        "",
        f"*Generated by `design/py/fit_ztt_pb24.py`{' --quick' if quick else ''}. "
        "Do not edit by hand.*",
        "",
        f"Fitted on the realisation split's **{f['n_train_realisations']} training "
        f"realisations** (digest `{f['train_digest']}`), scored on its "
        f"{f['n_test_realisations']} held-out ones; all 451 bands. Errors are "
        "relative, in %: median |err|, signed 16 / 50 / 84 %, and q90 |err|.",
        "",
        "## (h) The diffuse-fraction table",
        "",
        f"- `Esdir_Es_ratio` identical across the training realisations: max |Δ| = "
        f"{f['H_train_max_abs_diff']:g}; across all {f['n_all_realisations']}: "
        f"{f['H_all_max_abs_diff']:g}.",
        f"- Shipped as `{Z.H_PB24_FILE}` ({f['H_file_bytes']} bytes), "
        f"{f['H_shape'][0]} suns × {f['H_shape'][1]} bands; served bilinearly in "
        "(sec θs′, λ) by `ztt.diffuse_fraction_pb24`.",
        "- Leave-one-node-out interpolation, linear in sec θs′ (prediction, "
        "written for cos θs′ and the 30°/60° nodes: within 1% of H, < 0.3% in "
        "µd; the shipped spacing is half this):",
        "",
        "| node | H rel. err. median | max | M⁺_d rel. err. median | max |",
        "|---|---|---|---|---|",
    ]
    for node, v in m["H_interpolation_check"].items():
        L.append(
            f"| {node}° | {100 * v['H_rel_err_median']:.3f}% "
            f"| {100 * v['H_rel_err_max']:.3f}% "
            f"| {100 * v['Md_plus_rel_err_median']:.3f}% "
            f"| {100 * v['Md_plus_rel_err_max']:.3f}% |"
        )
    L += [
        "",
        "## (d) µd — M*_d refit jointly with c(θs′), table H in M⁺_d",
        "",
        f"- Rows: {f['mu_d_n_rows']:,} (window suns θs ≤ {SUN_MAX_DEG:g}°); design "
        f"condition {f['mu_d_design_condition']:.3g}.",
        f"- c(θs′) = 1 + {c['md_plus_corr_c1_c2'][0]:.5f} u "
        f"+ {c['md_plus_corr_c1_c2'][1]:.5f} u², "
        "u = 1 − cos θs′, fitted to the seven free nodes (form test):",
        "",
        "| θs′ | c at node (free) | quadratic | ratio − 1 |",
        "|---|---|---|---|",
    ]
    for node, v in m["c_form_at_nodes"].items():
        L.append(
            f"| {node}° | {v['c_node']:.5f} | {v['c_quadratic']:.5f} | "
            f"{100 * (v['c_quadratic'] / v['c_node'] - 1):+.3f}% |"
        )
    L += [
        "",
        "Held out, vs PB24's `mu_d`, per sun (80° and 87.75° are extrapolation):",
        "",
        "| θs′ | **PB24 fit** | L23 fit, data H × c_L23 | Table A2, data H |",
        "|---|---|---|---|",
    ]
    md = m["mu_d_heldout"]
    for z in md["PB24_fit"]["per_theta_s"]:
        L.append(
            f"| {z}° | {fmt(md['PB24_fit']['per_theta_s'][z])} | "
            f"{fmt(md['L23_fit_data_H']['per_theta_s'][z])} | "
            f"{fmt(md['TT_A2_data_H']['per_theta_s'][z])} |"
        )
    L += [
        "",
        "Window suns pooled, by bb/a (prediction: < 0.5% below 0.5946, < 3% above):",
        "",
        "| bb/a | **PB24 fit** | L23 fit, data H × c_L23 | Table A2, data H |",
        "|---|---|---|---|",
    ]
    for b in md["PB24_fit"]["window_by_bb_over_a"]:
        L.append(
            f"| {b} | {fmt(md['PB24_fit']['window_by_bb_over_a'][b])} | "
            f"{fmt(md['L23_fit_data_H']['window_by_bb_over_a'][b])} | "
            f"{fmt(md['TT_A2_data_H']['window_by_bb_over_a'][b])} |"
        )
    L += [
        "",
        "## (i) µ∞ — Equation (8) refit to the eigen µ∞ (no B_p term, Q22)",
        "",
        f"- Rows: {f['mu_inf_n_rows']:,}; design condition "
        f"{f['mu_inf_design_condition']:.3g}; clamp box (X, L) = "
        f"{[round(v, 4) for v in c['clamp_box_X_lo_hi_L_lo_hi']]}; held-out rows "
        f"outside it: {100 * m['fraction_heldout_outside_box']:.2f}%.",
        f"- Output guard: {[round(v, 4) for v in c['mu_inf_out_range']]}; held-out µ∞ "
        "changed by > 1e-4: "
        f"{100 * m['mu_inf_guard_effect']['fraction_heldout_changed_gt_1e-4']:.2f}%.",
        "",
        "Held out, vs the eigen µ∞ (prediction: ≈ 0.9% median, the B_p trend "
        "remaining):",
        "",
        "| group | **PB24 fit** | L23 fit | TT2017 |",
        "|---|---|---|---|",
    ]
    mi = m["mu_inf_heldout_vs_eigen"]
    L.append(
        f"| all | {fmt(mi['PB24_fit']['all'])} | {fmt(mi['L23_fit']['all'])} | "
        f"{fmt(mi['TT2017']['all'])} |"
    )
    for key in ("by_bb_over_a", "by_B_p"):
        for b in mi["PB24_fit"][key]:
            label = ("bb/a " if key == "by_bb_over_a" else "B_p ") + b
            L.append(
                f"| {label} | {fmt(mi['PB24_fit'][key][b])} "
                f"| {fmt(mi['L23_fit'][key][b])} "
                f"| {fmt(mi['TT2017'][key][b])} |"
            )
    g = m["gate_evaluator_agreement"]
    L += [
        "",
        "## Gate",
        "",
        f"- Shipped evaluators vs the script's design matrices where no guard acts: "
        f"max rel. diff µ∞ {g['mu_inf']:.1e}, µd {g['mu_d']:.1e}.",
        "- `--check` refits and compares every file byte for byte.",
        "",
    ]
    return "\n".join(L)


def outputs(coefficients, metrics, probes, *, quick: bool) -> dict[str, str]:
    dump = {"indent": 2, "sort_keys": True}
    return {
        f"{PREFIX}coefficients.json": json.dumps(coefficients, **dump) + "\n",
        f"{PREFIX}metrics.json": json.dumps(metrics, **dump) + "\n",
        f"{PREFIX}probes.json": json.dumps(probes, **dump) + "\n",
        f"{PREFIX}metrics.md": report(coefficients, metrics, quick=quick),
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
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def figures(arrays, metrics, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta_s = arrays["theta_s"]
    md = metrics["mu_d_heldout"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in (
        ("PB24_fit", "PB24 fit (table H × c_PB24 × M*_d refit)"),
        ("L23_fit_data_H", "L23 fit, data H × c_L23"),
        ("TT_A2_data_H", "Table A2, data H"),
    ):
        med = [
            100 * md[name]["per_theta_s"][f"{z:g}"]["signed_pct_1_16_50_84_99"][2]
            for z in theta_s
        ]
        axes[0].plot(theta_s, med, "o-", label=label)
        bins = md[name]["window_by_bb_over_a"]
        axes[1].plot(
            range(len(bins)),
            [100 * v.get("mape", np.nan) for v in bins.values()],
            "o-",
            label=label,
        )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axvline(SUN_MAX_DEG + 5, color="grey", ls=":")
    axes[0].set_xlabel("θs′ (deg)")
    axes[0].set_ylabel("median µd_model/µd_PB24 − 1 (%)")
    axes[0].set_title(
        "µd held out, per sun (right of the dotted line: extrapolation)",
        loc="left",
        fontsize=9,
    )
    axes[1].set_xticks(range(len(bins)))
    axes[1].set_xticklabels(list(bins), rotation=30, fontsize=8)
    axes[1].set_ylabel("median |deviation| (%), window suns")
    axes[1].set_title("what is left, by bb/a", loc="left", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_png(fig, out / f"{PREFIX}fig_mu_d.png")

    mi = metrics["mu_inf_heldout_vs_eigen"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("PB24_fit", "L23_fit", "TT2017"):
        for ax, key in zip(axes, ("by_bb_over_a", "by_B_p"), strict=True):
            bins = mi[name][key]
            ax.plot(
                range(len(bins)),
                [
                    100 * v["signed_pct_1_16_50_84_99"][2] if v.get("n") else np.nan
                    for v in bins.values()
                ],
                "o-",
                label=name,
            )
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(list(bins), rotation=30, fontsize=8)
    axes[0].set_ylabel("median µ∞_fit / µ∞_eigen − 1 (%)")
    axes[0].set_title("by bb/a (held out)", loc="left", fontsize=9)
    axes[1].set_title("by B_p: the trend Eq. (8) cannot absorb", loc="left", fontsize=9)
    for ax in axes:
        ax.axhline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_png(fig, out / f"{PREFIX}fig_mu_inf.png")


# --------------------------------------------------------------- main -----
def run(*, quick: bool, cache_dir=None, eigen_path=None, progress=True):
    train_ids, test_ids = split_ids(quick)
    cache = PC.open_cache(cache_dir)
    all_ids = cache.realisations
    if not np.isin(np.concatenate([train_ids, test_ids]), all_ids).all():
        # A cache miniature (the test fixture): the split restricted to it.
        train_ids = np.intersect1d(train_ids, all_ids)
        test_ids = np.intersect1d(test_ids, all_ids)
        print(f"cache holds {all_ids.size} realisations; split restricted to them")
    digest = C.realisation_digest(train_ids)
    product = pb24_eigen.load_mu_inf_eigen(eigen_path)
    print(
        f"realisation split: fitting on {train_ids.size} training realisations "
        f"(digest {digest[:16]}), scoring on {test_ids.size} held-out",
        flush=True,
    )

    theta_s, wave, H, h_check = h_table(cache, train_ids, all_ids)
    h_prov = {
        "about": "PB24 diffuse fraction H(theta_s', lambda) = 1 - Esdir_Es_ratio "
        "(Pitarch & Brando 2025; RADTRAN-X, V = 40 km, US Navy maritime aerosol, "
        "354.9 DU). Identical in every file; taken from the realisation split's "
        "training realisations. M8 task 8a, design/py/fit_ztt_pb24.py.",
        "n_training_realisations": int(train_ids.size),
        "training_digest": digest,
        "theta_s_deg": [float(v) for v in theta_s],
        "wave_nm_range": [float(wave[0]), float(wave[-1])],
        "n_wave": int(wave.size),
    }
    h_bytes = h_file_bytes(theta_s, wave, H, h_prov)
    h_interp = h_interpolation_check(theta_s, wave, H)

    d = fit_mu_d_streamed(cache, train_ids, theta_s, H, progress=progress)
    i = fit_mu_inf_streamed(cache, train_ids, product, d["box"], progress=progress)
    metrics, arrays = evaluate(cache, test_ids, product, theta_s, wave, H, d, i)
    metrics["H_interpolation_check"] = h_interp
    pr = probes(cache, all_ids, product, theta_s, wave, H, d, i)

    coefficients = {
        "about": "ZTT fitted to PB24 (M8 task 8a, rt_fit_pb24.md Q21-Q22). Report as "
        "'ZTT (PB24)'. mu_d: Eq. (17) M*_d jointly with c(theta_s') on TT's M+_d "
        "with PB24's own diffuse fraction, against PB24 mu_d, window suns. mu_inf: "
        "Eq. (8) to the eigen mu_inf on PB24 IOPs and FF pair, no B_p term.",
        "mu_inf_coeffs_m1_m16": [float(v) for v in i["m16"]],
        "md_star_coeffs_m1_m8": [float(v) for v in d["md8"]],
        "md_plus_corr_c1_c2": list(d["corr"]),
        "clamp_box_X_lo_hi_L_lo_hi": d["box"],
        "mu_inf_out_range": i["mu_inf_range"],
        "md_star_out_range": d["md_star_range"],
        "fit": {
            "split": (
                f"realisations, SPLIT_SEED={P.SPLIT_SEED}, "
                f"test_fraction={P.TEST_FRACTION}"
            ),
            "quick": quick,
            "n_train_realisations": int(train_ids.size),
            "n_test_realisations": int(test_ids.size),
            "n_all_realisations": int(all_ids.size),
            "train_digest": digest,
            "chunk": CHUNK,
            "sun_max_deg": float(SUN_MAX_DEG),
            "mu_d_nodes_deg": [float(v) for v in d["nodes"]],
            "c_at_nodes": {
                f"{z:g}": float(v)
                for z, v in zip(d["nodes"][1:], d["c_nodes"], strict=True)
            },
            "mu_d_n_rows": int(d["n_rows"]),
            "mu_d_design_condition": d["cond"],
            "mu_d_residual_norm": d["resid_norm"],
            "mu_inf_n_rows": int(i["n_rows"]),
            "mu_inf_design_condition": i["cond"],
            "mu_inf_residual_norm": i["resid_norm"],
            "eigen_product": Path(
                eigen_path or pb24_eigen.default_eigen_path()
            ).name,
            "eigen_version": int(product.provenance["version"]),
            "H_train_max_abs_diff": h_check["train_max_abs_diff"],
            "H_all_max_abs_diff": h_check["all_max_abs_diff"],
            "H_shape": list(H.shape),
            "H_file_bytes": len(h_bytes),
            "H_file": Z.H_PB24_FILE,
        },
    }  # fmt: skip
    return coefficients, metrics, pr, arrays, h_bytes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", default=None, help="hyperspectral cache")
    parser.add_argument("--eigen", default=None, help="eigen µ∞ product (.npz)")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK} realisations")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument(
        "--check", action="store_true", help="refit and compare byte for byte"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--h-file", type=Path, default=H_FILE, help="where the H table is written"
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.quick and args.out_dir == OUT and not args.dry_run:
        parser.error("--quick fits on a subsample; pass --out-dir or --dry-run")
    if args.quick and args.h_file == H_FILE and not args.dry_run:
        parser.error("--quick must not overwrite the shipped H table; pass --h-file")
    if args.quick and args.check:
        parser.error("--check compares the shipped files; it cannot be --quick")

    t0 = time.time()
    coefficients, metrics, pr, arrays, h_bytes = run(
        quick=args.quick,
        cache_dir=args.cache_dir,
        eigen_path=args.eigen,
        progress=not args.quiet,
    )
    files = outputs(coefficients, metrics, pr, quick=args.quick)
    print(files[f"{PREFIX}metrics.md"])
    print(f"fit done in {time.time() - t0:.0f} s")

    if args.check:
        stale = [
            n for n, text in files.items() if (args.out_dir / n).read_text() != text
        ]
        if args.h_file.read_bytes() != h_bytes:
            stale.append(args.h_file.name)
        if not args.no_figures:
            with tempfile.TemporaryDirectory() as tmp:
                figures(arrays, metrics, Path(tmp))
                for name in (f"{PREFIX}fig_mu_d.png", f"{PREFIX}fig_mu_inf.png"):
                    if (Path(tmp) / name).read_bytes() != (
                        args.out_dir / name
                    ).read_bytes():
                        stale.append(name)
        print("check:", "identical" if not stale else f"DIFFERS: {stale}")
        return 1 if stale else 0
    if args.dry_run:
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out_dir / name).write_text(text)
    args.h_file.parent.mkdir(parents=True, exist_ok=True)
    args.h_file.write_bytes(h_bytes)
    if not args.no_figures:
        figures(arrays, metrics, args.out_dir)
    print(f"wrote {sorted(files)} to {args.out_dir}, H table to {args.h_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
