#!/usr/bin/env python
"""
Fit and score the nadir-only ZTT on PB24 -- M8 task 9, ``rt_fit_pb24.md``
(Q34–Q45; "ZTT (PB24_nadir)").

**Since task 9c (Q45)** the variant ``"pb24_nadir"`` is task 8a's H, µd and µ∞
with **TT's own Ψ_KLu and f_L**, floored: it has nothing fitted here, and this
script is where it is scored (the ladder, the gate, its bb/a domain, ``Rrs``,
L23). The fit described below -- Ψ_KLu from PB24's measured K_Lu, f_L by
closure, the variant as first built in task 9b -- failed its gate and is kept
as a record (``angular_model="pb24_klu"``); its constants are still written
by, and checked against, this script.

At nadir (θv = 0) Equation (12) is

    rrs = (1/µd) [β(ψ)/bb] / D,
    D   = (a/bb)(1 + Ψ_KLu(ψ)/µ∞) + f_L(ψ, λ)(1 − 1/b̃b) + 1/b̃b,

with ψ = 180° − θs_w, and each term has its own PB24 observable, so the fit is
**sequential and linear** (Q37 (a)); nothing alternates but f_L's bilinear form:

(1) **µd, µ∞ and H** are task 8a's (:func:`robust.rt.ztt.mu_d_pb24`,
    :func:`robust.rt.ztt.mu_infinity_pb24`, the PB24 diffuse fraction): not
    refit here.
(2) **Ψ_KLu(ψ) from PB24's measured nadir ``KLu``** (Q35 (b), Q36 (a)).
    ``t = (KLu/a) · µ∞`` per (realisation, sun, band) is the measured
    K_Lu/K∞; each window sun gets one node value, an average of ``t`` over the
    training rows; ``Ψ_KLu = 1 + F(ψ)``, ``F`` a polynomial in
    ``u = 1 + cos ψ``, is fitted to the eight nodes (relative, weighted by row
    count). IOP-independent, as TT wrote it. The order is the smallest whose
    worst node misfit is within :data:`NODE_TOL`.

    ``t`` is *not* IOP-independent (its 16–84% range at one sun is a factor
    of 1.5), so the average matters. Two are fitted (:data:`WEIGHTINGS`):
    ``"geometric"``, the plain geometric mean of ``t`` (every row equal), and
    ``"rrs"``, the least-squares mean with each row weighted by the share of
    Equation (12)'s denominator its K_Lu term carries,
    ``c = (a/bb)/(µ∞ D*)`` -- the same measured ``t``, averaged for the rows
    where K_Lu decides ``rrs``. The geometric mean ships
    (:data:`SHIPPED_WEIGHTING`, the design of Q37 (a)); the other is a
    diagnostic, added after the first look at the numbers to test whether
    the plain average was the reason rung (b) is poor (it is not).
(3) **f_L(ψ, λ) = FL_AVE(λ) · (s0 + s1 sin ψ)** -- TT's Equation (31) form --
    by closure on nadir ``rrs`` with Ψ_KLu fixed: ``D* = [β/bb]/(µd · rrs)``,
    ``D* − (a/bb)(1 + Ψ/µ∞) − 1/b̃b = (1 − 1/b̃b) · f_L``, rows weighted
    ``1/D*``; ``FL_AVE`` per band (Q25), a fixed number of alternating passes.
(4) **The floor on D** of task 8b (:func:`robust.rt.ztt.denominator_floor_pb24`).

Fitted on the realisation split's training realisations at the **window suns**
(θs ≤ 70°), all 451 bands; scored on its held-out realisations. Reported
beside it: the nadir ladder (each term added alone and removed alone), the D2
ceiling (measured KLu/a), an IOP-dependent Ψ_KLu and a direct fit of KLu/a
(diagnostics, not shipped: Q35 (c), Q36 (b)), the B_p-band split (refit on its
own training side, Q38), ``Rrs`` through the Fresnel transfer (Q39), the 80°
and 87.75° suns, and L23 (Q42, Q44: never fitted, only scored).

Writes ``design/ztt_pb24/nadir_*``: ``nadir_coefficients.json`` (the source of
``PSI_KLU_PB24_NADIR_COEFFS``, ``FL_AVE_PB24_NADIR``, ``FL_S_PB24_NADIR``),
``nadir_metrics.{json,md}``, ``nadir_per_band.csv``, ``nadir_probes.json``
and three figures. Everything is in memory (14.4 M training rows); about two
minutes, one process.

    python design/py/fit_ztt_pb24_nadir.py
    python design/py/fit_ztt_pb24_nadir.py --check      # byte for byte
    python design/py/fit_ztt_pb24_nadir.py --quick --out-dir /tmp/x
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # fits and metrics in float64

import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402
from robust.rt.data import pb24_cache as PC  # noqa: E402
from robust.rt.types import Geometry, IOPs, PhaseParams  # noqa: E402

OUT = ROOT / "design" / "ztt_pb24"
PREFIX = "nadir_"

#: ``--quick``: training and test realisations (the first n of each side).
QUICK = (200, 50)

#: The window suns (Q34): θs ≤ 70° is fitted and gated; 80° and 87.75° are
#: reported as extrapolation.
ANGLE_MAX_DEG = P.ANGLE_WINDOW_MAX

#: Terms of F(ψ) tried, and the rule written before the numbers: the smallest
#: count whose worst relative misfit to the eight node values is ≤ NODE_TOL.
F_TERMS_TRIED = (2, 3, 4, 5, 6)
NODE_TOL = 0.01

#: Node averages of the measured K_Lu/K∞ (see the module docstring).
WEIGHTINGS = ("geometric", "rrs")
SHIPPED_WEIGHTING = "geometric"

#: Alternating passes of the f_L fit: a fixed count, convergence reported.
N_ROUNDS = 12
#: FL_AVE(λ) is shipped rounded to this many decimals.
FL_DECIMALS = 6
#: Rounds of the direct KLu/a diagnostic (bilinear N(ψ) · Q(X, L)).
N_ROUNDS_DIRECT = 8

D_FLOOR = Z.D_FLOOR_PB24
FOOTPRINT_TOL = 1e-4

#: Breakdown bins.
BBA_EDGES = (0.0, 1e-3, 1e-2, 0.1, 0.5946, 2.0, np.inf)
ETA_EDGES = (0.0, 0.01, 0.1, 0.5, 0.9, np.inf)
BP_EDGES = (0.0, 0.003, 0.005, 0.01, 0.02, 0.03, np.inf)

#: The acceptance gate, written before the fit (``rt_fit_pb24.md`` Q42).
GATE_MEDIAN = 0.04
GATE_NONPHYSICAL = 0.01
GATE_L23_REFERENCE = 2.90
GATE_L23_TOLERANCE = 0.5

#: Predictions written before the fit (Q35, Q36), restated in the report.
PREDICTIONS = (
    "rung (a) (8a's terms, TT's Ψ_KLu and f_L): 4–5% median, non-physical > 1%",
    "rung (b) (Ψ_KLu from KLu, f_L by closure): 3.5–4.5% median; the clear-water "
    "bins (bb/a < 0.01) remain worst",
    "an IOP-dependent Ψ_KLu (rung (c), diagnostic): 3.0–3.5%, within 0.5 pp of the "
    "D2 ceiling; KLu/a itself to ≤ 5% median |dev|",
    "factorised Ψ_KLu/µ∞ against a direct fit of KLu/a: under 0.2 pp apart in rrs",
)

N_PROBES = 12
PROBE_SEED = P.SPLIT_SEED

IOP_NAMES = ("aw", "aph", "ag", "aNAP", "bph", "bNAP", "bbw", "bbph", "bbNAP")

NAME_C = "Ψ_KLu = 1 + F(ψ)·(1, X, L) — rung (c), not shipped"
NAME_DIRECT = "direct KLu/a = N(θs)·Q(X, L), no µ∞ — Q36 (b), not shipped"
#: The variant, as redefined by Q45 (M8 task 9c): 8a's µd, µ∞ and H with TT's
#: own Ψ_KLu and f_L, floored. Specs are (µd, µ∞, Ψ_KLu, f_L, floor); "nadir"
#: names the measured-K_Lu pair fitted here (``angular_model="pb24_klu"``).
FINAL = "+ floor = ZTT (PB24_nadir)"
RUNG_A = "+ µ∞ (PB24) = rung (a)"
RUNG_B = "rung (b): + Ψ_KLu from KLu, + f_L by closure (no floor)"
KLU = "measured-K_Lu fit, floored (9b's variant; a record)"
BASE = "ZTT (L23 µ∞), PB24 FF Pbb"
GORDON = "standard Gordon"
SPEC_BASE = ("l23", "l23", "tt", "tt", False)
SPEC_RUNG_A = ("pb24", "pb24", "tt", "tt", False)
SPEC_FINAL = ("pb24", "pb24", "tt", "tt", True)
SPEC_KLU = ("pb24", "pb24", "nadir", "nadir", True)

#: The ladder to the variant.
LADDER = (
    (BASE, SPEC_BASE),
    ("+ µd (PB24)", ("pb24", "l23", "tt", "tt", False)),
    (RUNG_A, SPEC_RUNG_A),
    (FINAL, SPEC_FINAL),
)
#: The measured-K_Lu fit of task 9b (Q35 (b)), continued from rung (a): scored
#: beside the ladder as the record of why the variant keeps TT's pair.
RECORD = (
    (
        "rung (a) + Ψ_KLu from KLu (TT's f_L, no floor)",
        ("pb24", "pb24", "nadir", "tt", False),
    ),
    (RUNG_B, ("pb24", "pb24", "nadir", "nadir", False)),
    (KLU, SPEC_KLU),
)
#: Each term alone: (added to BASE, changed alone in FINAL). For the three
#: terms the variant has, "changed" is removed; for the two it does not have,
#: it is the measured-K_Lu term swapped in.
ALONE = {
    "µd": (("pb24", "l23", "tt", "tt", False), ("l23", "pb24", "tt", "tt", True)),
    "µ∞": (("l23", "pb24", "tt", "tt", False), ("pb24", "l23", "tt", "tt", True)),
    "floor": (("l23", "l23", "tt", "tt", True), SPEC_RUNG_A),
    "Ψ_KLu from KLu (swapped in)": (
        ("l23", "l23", "nadir", "tt", False),
        ("pb24", "pb24", "nadir", "tt", True),
    ),
    "f_L by closure (swapped in)": (
        ("l23", "l23", "tt", "nadir", False),
        ("pb24", "pb24", "tt", "nadir", True),
    ),
}


# ----------------------------------------------------------------- bases -----
def u_of(psi_deg) -> np.ndarray:
    """``u = 1 + cos ψ`` (0 at ψ = 180°)."""
    return 1.0 + np.cos(np.deg2rad(np.asarray(psi_deg, dtype=np.float64)))


U_MAX = float(u_of(Z.PSI_KLU_PB24_NADIR_PSI_MIN))


def legendre_basis(t, n: int) -> np.ndarray:
    """``P_0 … P_{n−1}(t)`` by the three-term recurrence, stacked last."""
    t = np.asarray(t, dtype=np.float64)
    cols = [np.ones_like(t)]
    if n > 1:
        cols.append(t.copy())
    for k in range(1, n - 1):
        cols.append(((2 * k + 1) * t * cols[k] - k * cols[k - 1]) / (k + 1))
    return np.stack(cols[:n], axis=-1)


def psi_basis(psi_deg, n: int) -> np.ndarray:
    """Legendre basis in ``τ = 2u/U_MAX − 1`` (τ ∈ [−1, 1] over the nadir ψ)."""
    return legendre_basis(2.0 * u_of(psi_deg) / U_MAX - 1.0, n)


def legendre_to_u_poly(p) -> tuple[float, ...]:
    """Legendre coefficients in ``τ = 2u/U_MAX − 1`` → monomials in ``u``,
    highest order first (``jnp.polyval``'s convention)."""
    from numpy.polynomial import Legendre, Polynomial

    in_t = Legendre(np.asarray(p, dtype=np.float64)).convert(kind=Polynomial)
    in_u = in_t(Polynomial([-1.0, 2.0 / U_MAX]))
    return tuple(float(v) for v in in_u.coef[::-1])


# -------------------------------------------------------- model pieces -----
def pieces(iops, phase, geometry, wave, H) -> dict:
    """Equation (12)'s fixed pieces for any batch, through the shipped
    functions, as float64 NumPy ``(n, W)`` (ψ and cos θv ``(n,)``).

    ``H`` is the diffuse fraction ``(n, W)`` or ``(n, 1)`` used by both µd
    variants. The same function serves PB24's nadir rows, L23 and the probes.
    """
    import jax.numpy as jnp

    _, theta_s_air, theta_v_paper, psi = Z.geometry_to_paper_angles(geometry)
    psi_b = jnp.asarray(psi)[:, None]
    ts_b = jnp.asarray(theta_s_air)[:, None]
    wave = jnp.asarray(np.asarray(wave, dtype=np.float64))
    bb = iops.bb
    bb_over_a, eta_bb = bb / iops.a, iops.bb_w / bb
    num = Z.backward_phase_over_bb(
        iops, Z.P_bb_from_phase(phase, psi_b, use_ff=True), psi_b
    )
    Hj = jnp.asarray(H)
    f64 = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    shape = np.broadcast_shapes(f64(iops.a).shape, f64(num).shape)
    full = lambda x: np.broadcast_to(f64(x), shape)  # noqa: E731
    return {
        "psi": f64(psi),
        "cos_tv": np.abs(f64(jnp.cos(jnp.deg2rad(theta_v_paper)))),
        "num": full(num),
        "a_over_bb": full(iops.a / bb),
        "bb_over_a": full(bb_over_a),
        "eta_bb": full(eta_bb),
        "inv_bt": full(1.0 / Z.bb_tilde(iops, phase.B_p)),
        "mu_d_pb24": full(Z.mu_d_pb24(ts_b, bb_over_a, eta_bb, H=Hj)),
        "mu_d_l23": full(Z.mu_d_l23(ts_b, bb_over_a, eta_bb, H=Hj)),
        "mu_inf_pb24": full(Z.mu_infinity_pb24(bb_over_a, eta_bb)),
        "mu_inf_l23": full(Z.mu_infinity_l23(bb_over_a, eta_bb)),
        "wave": f64(wave),
    }


def psi_term(kind, psi, model) -> np.ndarray:
    """Ψ_KLu at ``psi (...)`` for ``kind`` in {"tt", "nadir"}."""
    import jax.numpy as jnp

    if kind == "tt":
        return np.asarray(Z.psi_KLu(jnp.asarray(psi)), dtype=np.float64)
    return np.asarray(
        Z.psi_KLu_pb24_nadir(jnp.asarray(psi), model["coeffs"]), dtype=np.float64
    )


def fl_term(kind, psi, wave, model) -> np.ndarray:
    """f_L ``(..., W)`` at ``psi (...)`` for ``kind`` in {"tt", "nadir"}."""
    import jax.numpy as jnp

    psi_b = jnp.asarray(psi)[..., None]
    wave = jnp.asarray(wave)
    if kind == "tt":
        return np.asarray(Z.f_L(psi_b, wave), dtype=np.float64)
    return np.asarray(
        Z.f_L_pb24_nadir(
            psi_b,
            wave,
            fl_ave=jnp.asarray(model["FL"]),
            fl_wave=jnp.asarray(model["fl_wave"]),
            s=model["s"],
        ),
        dtype=np.float64,
    )


def variant_rrs(pc: dict, model, spec=SPEC_FINAL):
    """``rrs (n, W)`` of a (µd, µ∞, Ψ_KLu, f_L, floor) combination from
    :func:`pieces`; the default is "ZTT (PB24_nadir)". Returns
    ``(rrs, D_unfloored)``."""
    mud, muinf, psi_kind, fl_kind, floor = spec
    Psi = psi_term(psi_kind, pc["psi"], model)[:, None]
    fl = fl_term(fl_kind, pc["psi"], pc["wave"], model)
    K = pc["cos_tv"][:, None] * Psi / pc[f"mu_inf_{muinf}"]
    D = pc["a_over_bb"] * (1.0 + K) + fl * (1.0 - pc["inv_bt"]) + pc["inv_bt"]
    Df = np.asarray(Z.denominator_floor_pb24(D, *D_FLOOR)) if floor else D
    with np.errstate(divide="ignore", invalid="ignore"):
        return pc["num"] / (pc[f"mu_d_{mud}"] * Df), D


# ------------------------------------------------------------- the data -----
@dataclass
class Nadir:
    """PB24 at nadir for a set of realisations: ``(n, S, W)`` arrays over the
    ten suns (``S``) and the bands, per-(realisation, band) ``(n, W)`` arrays
    broadcast on demand, and the fixed pieces of Equation (12)."""

    ids: np.ndarray
    theta_s: np.ndarray  # (S,)
    psi: np.ndarray  # (S,)
    wave: np.ndarray  # (W,)
    rrs: np.ndarray
    Rrs: np.ndarray
    H: np.ndarray
    klu_over_a: np.ndarray
    mu_d_data: np.ndarray
    gordon: np.ndarray
    B_p: np.ndarray  # (n, 1, W)
    pc: dict  # pieces, each (n, S, W) (psi, cos_tv: (n·S,))

    @property
    def window(self) -> np.ndarray:
        return self.theta_s <= ANGLE_MAX_DEG

    def take(self, ids) -> Nadir:
        """The subset of realisations ``ids`` (in the order given)."""
        rows = np.searchsorted(self.ids, np.asarray(ids, dtype=int))
        if not np.array_equal(self.ids[rows], ids):
            raise KeyError("realisation not loaded")
        sub = lambda x: x[rows]  # noqa: E731
        pc = {
            k: (v if k in ("wave", "psi", "cos_tv") else sub(v))
            for k, v in self.pc.items()
        }
        return Nadir(
            ids=self.ids[rows],
            theta_s=self.theta_s,
            psi=self.psi,
            wave=self.wave,
            rrs=sub(self.rrs),
            Rrs=sub(self.Rrs),
            H=sub(self.H),
            klu_over_a=sub(self.klu_over_a),
            mu_d_data=sub(self.mu_d_data),
            gordon=sub(self.gordon),
            B_p=sub(self.B_p),
            pc=pc,
        )


def load_nadir(cache, ids) -> Nadir:
    """Every nadir quantity of ``ids`` (ascending) from the cache."""
    import jax.numpy as jnp

    ids = np.asarray(ids, dtype=int)
    bands = np.asarray(cache.four_d_bands, dtype=int)
    wave = np.asarray(cache.grids["lambda"], dtype=np.float64)[bands]
    theta_s = np.asarray(cache.grids["theta_s"], dtype=np.float64)
    if cache.grids["theta"][0] != 0.0 or cache.grids["phi"][0] != 0.0:
        raise RuntimeError("the first view node is not nadir")
    tab = {
        k: np.asarray(cache.table(k, ids), dtype=np.float64)
        for k in (*IOP_NAMES, "KLu", "mu_d", "Esdir_Es_ratio")
    }
    raw = {k: tab[k][:, bands] for k in IOP_NAMES}
    a = raw["aw"] + raw["aph"] + raw["ag"] + raw["aNAP"]
    bbp, bbw = raw["bbph"] + raw["bbNAP"], raw["bbw"]
    B_p = bbp / (raw["bph"] + raw["bNAP"])
    ff_B, ff_w = P._ff_mixture(raw)
    n, S, W = ids.size, theta_s.size, wave.size
    H = (1.0 - tab["Esdir_Es_ratio"])[..., bands]  # (n, S, W)

    def rep(x):
        return jnp.asarray(np.repeat(np.asarray(x, dtype=np.float64), S, axis=0))

    iops = IOPs(a=rep(a), bb_w=rep(bbw), bb_p=rep(bbp))
    phase = PhaseParams(B_p=rep(B_p), ff_B=rep(ff_B), ff_weight=rep(ff_w))
    geom = Geometry(
        theta_s=jnp.asarray(np.tile(theta_s, n)),
        theta_v=jnp.zeros(n * S),
        dphi=jnp.zeros(n * S),
    )
    pc = pieces(iops, phase, geom, wave, H.reshape(n * S, W))
    pc = {
        k: (v if k in ("wave", "psi", "cos_tv") else v.reshape(n, S, W).copy())
        for k, v in pc.items()
    }
    gordon = np.asarray(
        B.rrs_gordon(
            IOPs(a=jnp.asarray(a), bb_w=jnp.asarray(bbw), bb_p=jnp.asarray(bbp))
        ),
        dtype=np.float64,
    )

    def four(name):
        x = np.asarray(cache.four_d(name, ids, theta=0, phi=0), dtype=np.float64)
        return x.reshape(n, S, W)

    return Nadir(
        ids=ids,
        theta_s=theta_s,
        psi=pc["psi"][:S].copy(),
        wave=wave,
        rrs=four("rrs"),
        Rrs=four("Rrs"),
        H=H,
        klu_over_a=tab["KLu"][..., bands] / a[:, None, :],
        mu_d_data=tab["mu_d"][..., bands],
        gordon=np.broadcast_to(gordon[:, None, :], (n, S, W)),
        B_p=B_p[:, None, :],
        pc=pc,
    )


def azimuth_spread(cache, ids) -> float:
    """Largest relative spread of nadir ``rrs`` across the 13 azimuth nodes
    (they are one direction; reported, to justify taking φ = 0)."""
    x = np.asarray(cache.four_d("rrs", ids, theta=0), dtype=np.float64)
    ref = x[:, :, :1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmax(np.abs(x / ref - 1.0)))


def split_ids(kind: str, quick: bool):
    """``(train_ids, test_ids)`` of a saved split, ascending."""
    content = P.read_splits()
    every = np.asarray(content["realisations"], dtype=int)
    test = np.asarray(sorted(content[kind]["test"]), dtype=int)
    train = np.setdiff1d(every, test)
    assert np.intersect1d(train, test).size == 0 and test.size > 0
    if quick:
        train, test = train[: QUICK[0]], test[: QUICK[1]]
    return train, test


# ------------------------------------------------------------- the fits -----
def klu_target(T: Nadir) -> tuple[np.ndarray, np.ndarray]:
    """``t = (KLu/a) · µ∞(PB24)`` and its validity mask, ``(n, S, W)``."""
    t = T.klu_over_a * T.pc["mu_inf_pb24"]
    return t, np.isfinite(t) & (t > 0)


def fit_psi(T: Nadir, weighting: str, n_terms=None) -> dict:
    """Ψ_KLu(ψ) from the measured nadir KLu (step 2 of the module docstring).

    Node values are per-sun averages of ``t`` (``weighting`` in
    :data:`WEIGHTINGS`); the polynomial is fitted to the window nodes for
    every count in :data:`F_TERMS_TRIED`, and the shipped count follows the
    :data:`NODE_TOL` rule unless ``n_terms`` is given (the B_p-band refit
    reuses the realisation split's choice).
    """
    t, good = klu_target(T)
    win = T.window
    if weighting == "geometric":
        lt = np.where(good, np.log(np.where(good, t, 1.0)), 0.0)
        nodes_all = np.exp(lt.sum(axis=(0, 2)) / good.sum(axis=(0, 2)))
    elif weighting == "rrs":
        pc = T.pc
        with np.errstate(divide="ignore", invalid="ignore"):
            Ds = pc["num"] / (pc["mu_d_pb24"] * T.rrs)
            c = pc["a_over_bb"] / (pc["mu_inf_pb24"] * Ds)
        good = good & np.isfinite(c) & (Ds > 0)
        c2 = np.where(good, c * c, 0.0)
        nodes_all = np.sum(c2 * np.where(good, t, 0.0), axis=(0, 2)) / c2.sum(
            axis=(0, 2)
        )
    else:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    n_k = good.sum(axis=(0, 2)).astype(np.float64)
    nodes, psi_w, w = nodes_all[win], T.psi[win], np.sqrt(n_k[win])
    tried = {}
    for n in F_TERMS_TRIED:
        A = psi_basis(psi_w, n) / nodes[:, None] * w[:, None]
        y = (1.0 - 1.0 / nodes) * w
        p, *_ = np.linalg.lstsq(A, y, rcond=None)
        fitted = 1.0 + psi_basis(psi_w, n) @ p
        tried[n] = {
            "p_legendre": p,
            "max_node_misfit": float(np.max(np.abs(fitted / nodes - 1.0))),
            "cond": float(np.linalg.cond(A)),
        }
    if n_terms is None:
        ok = [n for n in F_TERMS_TRIED if tried[n]["max_node_misfit"] <= NODE_TOL]
        n_terms = ok[0] if ok else F_TERMS_TRIED[-1]
    return {
        "weighting": weighting,
        "n_terms": int(n_terms),
        "coeffs": legendre_to_u_poly(tried[n_terms]["p_legendre"]),
        "nodes_all": nodes_all,
        "n_rows": int(good[:, win].sum()),
        "n_bad": int((~good[:, win]).sum()),
        "tried": tried,
    }


def closure_stats(T: Nadir, K: np.ndarray, mud: str = "pb24") -> dict:
    """Sufficient statistics of the f_L closure fit on the window suns.

    ``K (n, S, W)`` is the model's KLu/a (Ψ_KLu/µ∞ at nadir). With
    ``y = D* − (a/bb)(1 + K) − 1/b̃b`` and ``gL = 1 − 1/b̃b``, the model is
    ``y = gL · h(sun, band)`` and rows are weighted ``1/D*``, so the sums
    ``Σ (w gL)²`` and ``Σ w² gL y`` per (sun, band) are everything any
    ``h`` needs.
    """
    win, pc = T.window, T.pc
    mu_d = T.mu_d_data if mud == "data" else pc[f"mu_d_{mud}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        Ds = (pc["num"] / (mu_d * T.rrs))[:, win]
    known = (pc["a_over_bb"] * (1.0 + K) + pc["inv_bt"])[:, win]
    gL = (1.0 - pc["inv_bt"])[:, win]
    good = np.isfinite(Ds) & (Ds > 0) & (T.rrs[:, win] > 0) & np.isfinite(known)
    w = np.where(good, 1.0 / np.where(good, Ds, 1.0), 0.0)
    y = np.where(good, Ds - known, 0.0)
    return {
        "G2": np.sum((w * gL) ** 2, axis=0),
        "Gy": np.sum(w * w * gL * y, axis=0),
        "yy": float(np.sum((w * y) ** 2)),
        "n": int(good.sum()),
        "n_bad": int((~good).sum()),
        "D_star_min": float(np.min(Ds[good])),
        "D_star_median": float(np.median(Ds[good])),
    }


def closure_rrms(st: dict, h: np.ndarray) -> float:
    """Weighted D-space rRMS of ``h (S_window, W)`` from the statistics."""
    ss = st["yy"] - 2.0 * np.sum(h * st["Gy"]) + np.sum(h * h * st["G2"])
    return float(np.sqrt(max(ss, 0.0) / st["n"]))


def s_basis(psi_deg, n_s: int) -> np.ndarray:
    """``(1, sin ψ[, cos ψ])`` stacked last."""
    psi = np.deg2rad(np.asarray(psi_deg, dtype=np.float64))
    return np.stack([np.ones_like(psi), np.sin(psi), np.cos(psi)][:n_s], axis=-1)


def solve_fl(st: dict, psi_w, n_s: int = 2, n_rounds: int = N_ROUNDS) -> dict:
    """``f_L = FL(λ) · (b(ψ) · c)`` by alternating linear passes with a fixed
    count: ``c`` given ``FL`` (an ``n_s`` system), ``FL`` given ``c`` (one
    division per band), ``FL`` renormalised to mean 1."""
    b = s_basis(psi_w, n_s)  # (S, n_s)
    FL = np.ones(st["G2"].shape[1])
    history, c = [], None
    for _ in range(n_rounds):
        A = np.einsum("ki,kj,k->ij", b, b, st["G2"] @ FL**2)
        c = np.linalg.solve(A, b.T @ (st["Gy"] @ FL))
        sk = b @ c
        FL_new = (sk @ st["Gy"]) / (sk**2 @ st["G2"])
        scale = float(FL_new.mean())
        FL_new, c = FL_new / scale, c * scale
        history.append(
            {
                "max_rel_change": float(np.max(np.abs(FL_new / FL - 1.0))),
                "cond": float(np.linalg.cond(A)),
            }
        )
        FL = FL_new
    return {"FL": FL, "c": c, "history": history, "h": np.outer(b @ c, FL)}


def fit_fl(T: Nadir, K: np.ndarray, mud: str = "pb24") -> dict:
    """The shipped f_L form fitted by closure against ``K``; ``FL`` rounded."""
    st = closure_stats(T, K, mud)
    sol = solve_fl(st, T.psi[T.window], 2)
    FL = np.round(sol["FL"], FL_DECIMALS)
    s = (float(sol["c"][0]), float(sol["c"][1]))
    return {"FL": FL, "s": s, "stats": st, "solution": sol}


def K_of(T: Nadir, psi_kind: str, model, muinf: str = "pb24") -> np.ndarray:
    """The model's KLu/a ``(n, S, W)``: Ψ_KLu(ψ)/µ∞."""
    return psi_term(psi_kind, T.psi, model)[None, :, None] / T.pc[f"mu_inf_{muinf}"]


def fit_model(T: Nadir, weighting: str, n_terms=None) -> tuple[dict, dict]:
    """The whole nadir fit on ``T`` for one node weighting: ``(model, details)``.
    ``details["rrms_train"]`` is the D-space closure residual the choice
    between weightings is made on."""
    psi = fit_psi(T, weighting, n_terms)
    model = {"coeffs": psi["coeffs"], "fl_wave": T.wave}
    fl = fit_fl(T, K_of(T, "nadir", model))
    model.update(FL=fl["FL"], s=fl["s"])
    rrms = closure_rrms(fl["stats"], fl["solution"]["h"])
    return model, {"psi": psi, "fl": fl, "rrms_train": rrms}


# --------------------------------------------------- diagnostics (fits) -----
def box_inputs(T):
    """``X, L`` clamped to the PB24 box (as 8a's fits), for a :class:`Nadir`
    (``(n, S, W)``) or a :func:`pieces` dict (``(n, W)``)."""
    import jax.numpy as jnp

    pc = T.pc if isinstance(T, Nadir) else T
    X, L = Z.log_inputs(
        jnp.asarray(pc["bb_over_a"]), jnp.asarray(pc["eta_bb"]), Z.ZTT_PB24_BOX
    )
    return np.asarray(X, dtype=np.float64), np.asarray(L, dtype=np.float64)


def K_iop_flat(pc: dict, coef: np.ndarray) -> np.ndarray:
    """:func:`K_iop` on a :func:`pieces` dict, ``(n, W)`` (the L23 score)."""
    X, L = box_inputs(pc)
    Fk = psi_basis(pc["psi"], coef.shape[0]) @ coef  # (n, 3)
    Psi = 1.0 + Fk[:, 0, None] + Fk[:, 1, None] * X + Fk[:, 2, None] * L
    return Psi / pc["mu_inf_pb24"]


def K_direct_flat(pc: dict, fit: dict, theta_s, node_theta_s) -> np.ndarray:
    """:func:`K_direct` on a :func:`pieces` dict, ``(n, W)``; the per-sun
    factor is interpolated linearly between the window nodes."""
    X, L = box_inputs(pc)
    N = np.interp(np.asarray(theta_s, dtype=np.float64), node_theta_s, fit["N"])
    return N[:, None] * (_bicubic(X, L) @ fit["q"])


def fit_psi_iop(T: Nadir, n_terms: int) -> np.ndarray:
    """Diagnostic, rung (c): ``Ψ = 1 + F(ψ) · (1, X, L)`` by relative least
    squares on ``Ψ/t − 1`` (window suns). Returns ``(n_terms, 3)`` Legendre
    coefficients."""
    t, good = klu_target(T)
    X, L = box_inputs(T)
    win = np.flatnonzero(T.window)
    Pk = psi_basis(T.psi[win], n_terms)  # (Sw, n)
    AtA = np.zeros((n_terms * 3, n_terms * 3))
    Aty = np.zeros(n_terms * 3)
    for j, k in enumerate(win):
        g = good[:, k]
        it = np.where(g, 1.0 / np.where(g, t[:, k], 1.0), 0.0)
        m = np.stack([np.ones_like(it), X[:, k], L[:, k]], axis=-1) * it[..., None]
        M = np.einsum("nwi,nwj->ij", m, m)
        v = np.einsum("nwi,nw->i", m, np.where(g, 1.0 - it, 0.0))
        AtA += np.kron(np.outer(Pk[j], Pk[j]), M)
        Aty += np.kron(Pk[j], v)
    return np.linalg.lstsq(AtA, Aty, rcond=None)[0].reshape(n_terms, 3)


def K_iop(T: Nadir, coef: np.ndarray) -> np.ndarray:
    """KLu/a of the rung-(c) diagnostic, ``(n, S, W)``."""
    X, L = box_inputs(T)
    Fk = psi_basis(T.psi, coef.shape[0]) @ coef  # (S, 3)
    Psi = (
        1.0 + Fk[None, :, 0, None] + Fk[None, :, 1, None] * X + Fk[None, :, 2, None] * L
    )
    return Psi / T.pc["mu_inf_pb24"]


def _bicubic(X, L) -> np.ndarray:
    """Sixteen products ``x^i l^j`` (i, j ≤ 3) of the box-scaled inputs."""
    lo_x, hi_x, lo_l, hi_l = Z.ZTT_PB24_BOX
    x = (2.0 * X - (lo_x + hi_x)) / (hi_x - lo_x)
    ell = (2.0 * L - (lo_l + hi_l)) / (hi_l - lo_l)
    return np.stack([x**i * ell**j for i in range(4) for j in range(4)], axis=-1)


def fit_direct(T: Nadir) -> dict:
    """Diagnostic, Q36 (b): ``KLu/a = N(sun) · Q(X, L)`` with ``Q`` Equation
    (8)'s sixteen-term form fitted to ``KLu/a`` itself (no eigen µ∞) and a
    free factor per window sun; relative least squares, alternating."""
    win = np.flatnonzero(T.window)
    g = T.klu_over_a
    good = np.isfinite(g) & (g > 0)
    X, L = box_inputs(T)
    basis = _bicubic(X[:, 0], L[:, 0])  # (n, W, 16); X, L do not depend on the sun
    N = np.ones(win.size)
    q = None
    for _ in range(N_ROUNDS_DIRECT):
        AtA, Aty = np.zeros((16, 16)), np.zeros(16)
        for j, k in enumerate(win):
            ig = np.where(good[:, k], 1.0 / np.where(good[:, k], g[:, k], 1.0), 0.0)
            A = basis * (N[j] * ig)[..., None]
            AtA += np.einsum("nwi,nwj->ij", A, A)
            Aty += np.einsum("nwi,nw->i", A, good[:, k].astype(np.float64))
        q = np.linalg.lstsq(AtA, Aty, rcond=None)[0]  # rank-deficient on a miniature
        Q = basis @ q
        for j, k in enumerate(win):
            r = np.where(good[:, k], Q / np.where(good[:, k], g[:, k], 1.0), 0.0)
            N[j] = np.sum(r) / np.sum(r * r)
        q, N = q * N[0], N / N[0]
    return {"q": q, "N": N}


def fit_form_ceiling(T: Nadir) -> dict:
    """Diagnostic: the least-squares optimum of any IOP-independent Ψ_KLu(ψ)
    through ``rrs``. Per window sun, one free ``Ψ`` and a free ``f_L`` per
    band, fitted jointly by weighted least squares in D-space (the closure fit
    of task 8b without its parametric forms). It bounds TT's form at nadir in
    the fit's own objective (relative error over *all* rows, turbid ones
    included) -- not in the median, which a Ψ tuned to clear water can beat
    while failing elsewhere. Not a physical K_Lu (Q37 (b) was declined; this
    is reported, never shipped)."""
    win = np.flatnonzero(T.window)
    pc = T.pc
    Psi = np.full(T.theta_s.size, np.nan)
    fl = np.full((T.theta_s.size, T.wave.size), np.nan)
    for k in win:
        with np.errstate(divide="ignore", invalid="ignore"):
            Ds = pc["num"][:, k] / (pc["mu_d_pb24"][:, k] * T.rrs[:, k])
        good = np.isfinite(Ds) & (Ds > 0) & (T.rrs[:, k] > 0)
        w2 = np.where(good, 1.0 / np.where(good, Ds, 1.0) ** 2, 0.0)
        gF = pc["a_over_bb"][:, k] / pc["mu_inf_pb24"][:, k]
        gL = 1.0 - pc["inv_bt"][:, k]
        y = np.where(good, Ds - pc["a_over_bb"][:, k] - pc["inv_bt"][:, k], 0.0)
        a = np.sum(w2 * gF * gF)
        b = np.sum(w2 * gF * gL, axis=0)
        d = np.sum(w2 * gL * gL, axis=0)
        r0 = np.sum(w2 * gF * y)
        r = np.sum(w2 * gL * y, axis=0)
        Psi[k] = (r0 - np.sum(b * r / d)) / (a - np.sum(b * b / d))
        fl[k] = (r - b * Psi[k]) / d
    return {"Psi": Psi, "fl": fl}


def K_direct(T: Nadir, fit: dict) -> np.ndarray:
    """KLu/a of the direct diagnostic on the window suns (NaN elsewhere)."""
    X, L = box_inputs(T)
    Q = _bicubic(X[:, 0], L[:, 0]) @ fit["q"]
    N = np.full(T.theta_s.size, np.nan)
    N[T.window] = fit["N"]
    return N[None, :, None] * Q[:, None, :]


# -------------------------------------------------------------- scoring -----
def rrs_from(T: Nadir, K, fl, *, floor: bool, mud: str = "pb24"):
    """``rrs (n, S, W)`` from a KLu/a model ``K`` and ``fl`` (broadcastable)."""
    pc = T.pc
    D = pc["a_over_bb"] * (1.0 + K) + fl * (1.0 - pc["inv_bt"]) + pc["inv_bt"]
    Df = np.asarray(Z.denominator_floor_pb24(D, *D_FLOOR)) if floor else D
    mu_d = T.mu_d_data if mud == "data" else pc[f"mu_d_{mud}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return pc["num"] / (mu_d * Df), D


def spec_rrs(T: Nadir, spec, model):
    """``rrs (n, S, W)`` of a ladder spec."""
    mud, muinf, psi_kind, fl_kind, floor = spec
    fl = fl_term(fl_kind, T.psi, T.wave, model)[None]
    return rrs_from(T, K_of(T, psi_kind, model, muinf), fl, floor=floor, mud=mud)[0]


def stat(pred, truth, mask=None) -> dict:
    """Median and q90 of ``|pred/truth − 1|`` (non-finite counted as ∞, so
    non-physical values stay in the median), rRMS over the finite errors, and
    the non-physical share (``pred ≤ 0`` or non-finite)."""
    if mask is not None:
        m = np.broadcast_to(mask, pred.shape)
        pred, truth = pred[m], truth[m]
    if pred.size == 0:
        return {"n": 0, "median_abs": None, "q90_abs": None}
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(pred / truth - 1.0)
    fin = np.isfinite(e)
    rrms = float(np.sqrt(np.mean(e[fin] ** 2))) if fin.any() else None
    e = np.sort(np.where(fin, e, np.inf), axis=None)
    return {
        "n": int(e.size),
        "median_abs": float(e[(e.size - 1) // 2]),
        "q90_abs": float(e[int(0.9 * (e.size - 1))]),
        "rrms_finite": rrms,
        "nonphysical_fraction": float(np.mean((pred <= 0) | ~np.isfinite(pred))),
    }


def edges_names(prefix, edges):
    return [
        f"{prefix}[{a:g}, {b:g})" for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def summary(pred, truth, T: Nadir, classes=None) -> dict:
    """A model's scores on ``T``: the window, each sun, and the breakdowns
    (window suns only) by bb/a, η_bb, B_p and water class."""
    win = T.window[None, :, None]
    out = {"window": stat(pred, truth, win), "per_theta_s": {}, "groups": {}}
    inside = T.pc["bb_over_a"] <= Z.NADIR_BB_OVER_A_MAX
    out["in_domain"] = stat(pred, truth, win & inside)
    out["in_domain"]["share_of_window"] = float(np.mean(inside[:, T.window]))
    out["per_theta_s_in_domain"] = {}
    for k, ts in enumerate(T.theta_s):
        out["per_theta_s"][f"{ts:g}"] = stat(pred[:, k], truth[:, k])
        out["per_theta_s_in_domain"][f"{ts:g}"] = stat(
            pred[:, k], truth[:, k], inside[:, k]
        )
    for prefix, values, edges in (
        ("bb/a", T.pc["bb_over_a"], BBA_EDGES),
        ("eta_bb", T.pc["eta_bb"], ETA_EDGES),
        ("B_p", np.broadcast_to(T.B_p, pred.shape), BP_EDGES),
    ):
        idx = np.clip(
            np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2
        )
        for i, name in enumerate(edges_names(prefix, edges)):
            out["groups"][name] = stat(pred, truth, win & (idx == i))
    if classes is not None:
        for c in range(1, P.N_CLASS + 1):
            m = (classes == c)[:, None, None] & win
            out["groups"][f"class {c}"] = stat(pred, truth, m)
    return out


def klu_summary(K, T: Nadir) -> dict:
    """A KLu/a model against PB24's, held out: ratio statistics pooled over
    the window, per sun and per B_p bin."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = K / T.klu_over_a
    r = np.where(np.isfinite(r) & (T.klu_over_a > 0), r, np.nan)

    def st(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return {
            "median_ratio": float(np.median(v)),
            "median_abs_dev": float(np.median(np.abs(v - 1.0))),
            "q16_q84": [float(np.quantile(v, 0.16)), float(np.quantile(v, 0.84))],
        }

    win = T.window
    out = {"window": st(r[:, win]), "per_theta_s": {}, "per_B_p": {}}
    for k, ts in enumerate(T.theta_s):
        out["per_theta_s"][f"{ts:g}"] = st(r[:, k])
    Bp = np.broadcast_to(T.B_p, r.shape)[:, win]
    idx = np.clip(np.searchsorted(BP_EDGES, Bp, side="right") - 1, 0, len(BP_EDGES) - 2)
    for i, name in enumerate(edges_names("B_p", BP_EDGES)):
        out["per_B_p"][name] = st(r[:, win][idx == i])
    return out


# ------------------------------------------------------ Rrs (Q39) ----------
def transfer_fits(T: Nadir) -> dict:
    """The two Fresnel-transfer forms refit on nadir training rows (window
    suns): θv-only and with the downwelling factor."""
    win = T.window
    rrs, Rrs = T.rrs[:, win], T.Rrs[:, win]
    td = C._downwelling_ratio_np(T.theta_s[win][None, :, None], T.H[:, win])
    out = {}
    for name, factor in (("theta_v only", None), ("downwelling", td)):
        XtX, Xty, n, bad = C.fresnel_normal_equations(rrs, Rrs, 0.0, factor=factor)
        A0, Bc = C.solve_fresnel(XtX, Xty)
        out[name] = {"A0": A0, "B": Bc, "n_rows": n, "n_dropped": bad}
    return out


def apply_transfer(T: Nadir, rrs, A0, Bc, downwelling: bool):
    """``Rrs = A rrs / (1 − B rrs)`` at nadir, ``(n, S, W)``."""
    A = A0
    if downwelling:
        A = A0 * C._downwelling_ratio_np(T.theta_s[None, :, None], T.H)
    return A * rrs / (1.0 - Bc * rrs)


def transfers(nadir_fits: dict) -> dict:
    """Name -> ``(A0, B, downwelling)``: the shipped pair and the nadir refits."""
    return {
        "shipped, θv only (FRESNEL_A0/B)": (C.FRESNEL_A0, C.FRESNEL_B, False),
        "shipped, downwelling (FRESNEL_TD_A0/B)": (
            C.FRESNEL_TD_A0,
            C.FRESNEL_TD_B,
            True,
        ),
        "nadir refit, θv only": (
            nadir_fits["theta_v only"]["A0"],
            nadir_fits["theta_v only"]["B"],
            False,
        ),
        "nadir refit, downwelling": (
            nadir_fits["downwelling"]["A0"],
            nadir_fits["downwelling"]["B"],
            True,
        ),
    }


#: The transfer used for the variant's ``Rrs`` score: the shipped downwelling
#: form (task 4's better transfer), decided before the numbers; the nadir
#: refits are reported beside it.
RRS_TRANSFER = "shipped, downwelling (FRESNEL_TD_A0/B)"


# ------------------------------------------------------------- L23 ---------
def l23_check(model, extras=None) -> dict | None:
    """L23 held-out rRMS (%) of the nadir variant, per zenith -- the
    cross-dataset clause of the gate (Q42, Q44). ``None`` without the data.

    ``extras`` maps a diagnostic's name to ``(K_fn(pieces, theta_s), f_L
    model)``; each is scored with TT's H, floored, beside the variant."""
    import jax.numpy as jnp

    from robust.rt import validation as V
    from robust.rt.data import l23 as L

    try:
        batch = L.load_batch()
    except Exception:  # noqa: BLE001  - the L23 files are absent
        return None
    splits = L.make_splits(batch)
    truth = np.asarray(C.Rrs_to_rrs(batch.Rrs), dtype=np.float64)
    args = (batch.iops, batch.phase_params, batch.geometry, batch.wave)
    wave = np.asarray(batch.wave, dtype=np.float64)
    ts = np.asarray(batch.geometry.theta_s, dtype=np.float64)
    skies = {
        "TT's H (V = 15 km) -- the gate": np.asarray(
            Z.diffuse_fraction(jnp.asarray(ts))
        )[:, None],
        "PB24's H table (the variant's default call)": np.asarray(
            Z.diffuse_fraction_pb24(jnp.asarray(ts), jnp.asarray(wave))
        ),
    }
    preds = {
        "ZTT (L23 µ∞) -- the M7 default": np.asarray(Z.rrs_ZTT(*args, variant="l23"))
    }
    for sky, H in skies.items():
        pc = pieces(*args, H)
        preds[f"ZTT (PB24_nadir), {sky}"] = variant_rrs(pc, model)[0]
        preds[f"measured-K_Lu fit (9b), {sky}"] = variant_rrs(pc, model, SPEC_KLU)[0]
    pc = pieces(*args, skies["TT's H (V = 15 km) -- the gate"])
    for name, (K_fn, fl_model) in (extras or {}).items():
        K = K_fn(pc, ts)
        fl = fl_term("nadir", pc["psi"], pc["wave"], fl_model)
        D = pc["a_over_bb"] * (1.0 + K) + fl * (1.0 - pc["inv_bt"]) + pc["inv_bt"]
        Df = np.asarray(Z.denominator_floor_pb24(D, *D_FLOOR))
        preds[f"{name}, TT's H"] = pc["num"] / (pc["mu_d_pb24"] * Df)
    held = np.asarray(splits.scene_test)
    zen = np.asarray(batch.zenith)
    out = {}
    for name, pred in preds.items():
        row = {"held-out": float(V.rrms(truth[held], pred[held]))}
        for z in L.ZENITHS:
            m = held & (zen == z)
            row[f"{z}"] = float(V.rrms(truth[m], pred[m]))
        row["nonphysical_fraction"] = float(np.mean(~(pred[held] > 0)))
        out[name] = row
    return out


# --------------------------------------------------------------- probes -----
def probes(T: Nadir, raw_iops: dict, model) -> list[dict]:
    """Seeded (realisation, sun, band) points with every input and the
    variant's pieces, for the tests to re-evaluate without the data."""
    rng = np.random.default_rng(PROBE_SEED)
    n = N_PROBES
    rows = rng.integers(0, T.ids.size, n)
    suns = rng.integers(0, T.theta_s.size, n)
    bands = rng.integers(0, T.wave.size, n)
    Psi = psi_term("tt", T.psi, model)
    fl = fl_term("tt", T.psi, T.wave, model)
    rrs, D = rrs_from(T, K_of(T, "tt", model), fl[None], floor=True)
    Df = np.asarray(Z.denominator_floor_pb24(D, *D_FLOOR))
    Psi_k = psi_term("nadir", T.psi, model)
    fl_k = fl_term("nadir", T.psi, T.wave, model)
    rrs_k = rrs_from(T, K_of(T, "nadir", model), fl_k[None], floor=True)[0]
    out = []
    for r, k, w in zip(rows, suns, bands, strict=True):
        r, k, w = int(r), int(k), int(w)
        out.append(
            {
                "realisation": int(T.ids[r]),
                "theta_s": float(T.theta_s[k]),
                "wave_nm": float(T.wave[w]),
                "a": float(raw_iops["a"][r, w]),
                "bb_w": float(raw_iops["bb_w"][r, w]),
                "bb_p": float(raw_iops["bb_p"][r, w]),
                "B_p": float(T.B_p[r, 0, w]),
                "ff_B": [float(v) for v in raw_iops["ff_B"][r, w]],
                "ff_weight": [float(v) for v in raw_iops["ff_weight"][r, w]],
                "H": float(T.H[r, k, w]),
                "psi": float(T.psi[k]),
                "psi_KLu": float(Psi[k]),
                "f_L": float(fl[k, w]),
                "D_no_floor": float(D[r, k, w]),
                "D_floored": float(Df[r, k, w]),
                "rrs_variant": float(rrs[r, k, w]),
                "bb_over_a": float(T.pc["bb_over_a"][r, k, w]),
                "psi_KLu_klu_fit": float(Psi_k[k]),
                "f_L_klu_fit": float(fl_k[k, w]),
                "rrs_klu_fit": float(rrs_k[r, k, w]),
                "rrs_PB24": float(T.rrs[r, k, w]),
            }
        )
    return out


def raw_iops_of(cache, ids) -> dict:
    """``a, bb_w, bb_p, ff_B, ff_weight`` per (realisation, band) for probes."""
    bands = np.asarray(cache.four_d_bands, dtype=int)
    raw = {
        k: np.asarray(cache.table(k, ids), dtype=np.float64)[:, bands]
        for k in IOP_NAMES
    }
    ff_B, ff_w = P._ff_mixture(raw)
    return {
        "a": raw["aw"] + raw["aph"] + raw["ag"] + raw["aNAP"],
        "bb_w": raw["bbw"],
        "bb_p": raw["bbph"] + raw["bbNAP"],
        "ff_B": ff_B,
        "ff_weight": ff_w,
    }


# ------------------------------------------------------------------ run -----
def run(*, quick: bool, cache_dir=None, progress=True, with_l23=True):
    t0 = time.time()

    def say(msg):
        if progress:
            print(f"[{time.time() - t0:5.0f} s] {msg}", flush=True)

    cache = PC.open_cache(cache_dir)
    all_ids = np.asarray(cache.realisations, dtype=int)
    train_ids, test_ids = split_ids("realisation", quick)
    bp_train, bp_test = split_ids("bp_band", quick)
    if not np.isin(np.concatenate([train_ids, test_ids]), all_ids).all():
        keep = lambda x: np.intersect1d(x, all_ids)  # noqa: E731
        train_ids, test_ids, bp_train, bp_test = map(
            keep, (train_ids, test_ids, bp_train, bp_test)
        )
        print(f"cache holds {all_ids.size} realisations; splits restricted to them")
    used = np.union1d(np.union1d(train_ids, test_ids), np.union1d(bp_train, bp_test))
    digest = C.realisation_digest(train_ids)
    say(f"loading nadir data for {used.size} realisations")
    A = load_nadir(cache, used)
    Ttr, Tte = A.take(train_ids), A.take(test_ids)
    try:
        classes_all = P.read_classes()
        classes = classes_all[test_ids - 1]
    except Exception:  # noqa: BLE001  - no .mat sidecar (the miniature, CI)
        classes = None

    # ---- the fit ------------------------------------------------------------
    say("fitting Ψ_KLu and f_L")
    fits = {w: fit_model(Ttr, w) for w in WEIGHTINGS}
    chosen = SHIPPED_WEIGHTING
    model, det = fits[chosen]
    psi_fit, fl_fit = det["psi"], det["fl"]
    st_train = fl_fit["stats"]
    if not st_train["D_star_min"] > 2.0 * D_FLOOR[0]:
        raise RuntimeError("D* minimum is not above 2 × the floor")

    # ---- f_L form sensitivity (D-space, same Ψ_KLu) ---------------------------
    K_te = K_of(Tte, "nadir", model)
    st_test = closure_stats(Tte, K_te)
    psi_w = A.psi[A.window]
    with np.errstate(divide="ignore", invalid="ignore"):
        h_free = st_train["Gy"] / st_train["G2"]
    h_band = np.broadcast_to(
        st_train["Gy"].sum(0) / st_train["G2"].sum(0), h_free.shape
    )
    forms = {
        "FL(λ) · (s0 + s1 sin ψ) (shipped)": fl_fit["solution"]["h"],
        "FL(λ) · (s0 + s1 sin ψ + s2 cos ψ)": solve_fl(st_train, psi_w, 3)["h"],
        "FL(λ) alone (no ψ scaling)": h_band,
        "free per (sun, band)": h_free,
    }
    fl_forms = {
        k: {
            "rrms_train": closure_rrms(st_train, h),
            "rrms_test": closure_rrms(st_test, h),
        }
        for k, h in forms.items()
    }

    # ---- the ladder, held out -------------------------------------------------
    say("scoring the ladder")
    truth = Tte.rrs
    preds = {GORDON: Tte.gordon}
    for label, spec in (*LADDER, *RECORD):
        preds[label] = spec_rrs(Tte, spec, model)
    scores = {k: summary(v, truth, Tte, classes) for k, v in preds.items()}
    alone = {}
    for term, (add, rem) in ALONE.items():
        alone[term] = {
            "added_alone_to_base": stat(
                spec_rrs(Tte, add, model), truth, Tte.window[None, :, None]
            ),
            "changed_alone_in_final": stat(
                spec_rrs(Tte, rem, model), truth, Tte.window[None, :, None]
            ),
        }
    D_final = rrs_from(
        Tte,
        K_of(Tte, "tt", model),
        fl_term("tt", Tte.psi, Tte.wave, model)[None],
        floor=False,
    )[1]
    Dw = D_final[:, Tte.window]
    with np.errstate(divide="ignore", invalid="ignore"):
        moved = np.abs(np.asarray(Z.denominator_floor_pb24(Dw, *D_FLOOR)) / Dw - 1.0)
    footprint = {
        "fraction_moved_gt_tol": float(np.mean(~(moved <= FOOTPRINT_TOL))),
        "D_model_min": float(np.min(Dw)),
        "fraction_D_nonpositive": float(np.mean(Dw <= 0)),
        "tol": FOOTPRINT_TOL,
    }

    # ---- diagnostics: other KLu/a models, each with its own f_L refit ---------
    say("diagnostics")
    diag_K = {}
    tt_model = dict(model)
    diag_K["TT's Ψ_KLu, f_L refit against it"] = (
        K_of(Ttr, "tt", model),
        K_of(Tte, "tt", model),
    )
    c_iop = fit_psi_iop(Ttr, psi_fit["n_terms"])
    diag_K[NAME_C] = (
        K_iop(Ttr, c_iop),
        K_iop(Tte, c_iop),
    )
    direct = fit_direct(Ttr)
    diag_K[NAME_DIRECT] = (
        K_direct(Ttr, direct),
        K_direct(Tte, direct),
    )
    diagnostics, klu, l23_extras = {}, {}, {}
    node_ts = A.theta_s[A.window]
    K_flat = {
        NAME_C: lambda pc, ts: K_iop_flat(pc, c_iop),
        NAME_DIRECT: lambda pc, ts: K_direct_flat(pc, direct, ts, node_ts),
    }
    for w in WEIGHTINGS:
        if w == chosen:
            continue
        other = fits[w][0]
        name = f"Ψ_KLu(ψ) with “{w}” node averages — not shipped"
        pred = spec_rrs(Tte, SPEC_KLU, other)
        diagnostics[name] = summary(pred, truth, Tte)
        klu[name] = klu_summary(K_of(Tte, "nadir", other), Tte)
    for name, (Ktr, Kte) in diag_K.items():
        f = fit_fl(Ttr, Ktr)
        m2 = dict(tt_model, FL=f["FL"], s=f["s"])
        pred = rrs_from(
            Tte, Kte, fl_term("nadir", Tte.psi, Tte.wave, m2)[None], floor=True
        )[0]
        diagnostics[name] = summary(pred, truth, Tte)
        diagnostics[name]["fl_s"] = list(f["s"])
        if name in K_flat:
            l23_extras[name] = (K_flat[name], m2)
        klu[name] = klu_summary(Kte, Tte)
    ceil = fit_form_ceiling(Ttr)
    name = "least-squares optimum of TT's form through rrs: free Ψ per sun, free f_L"
    K_ceil = ceil["Psi"][None, :, None] / Tte.pc["mu_inf_pb24"]
    diagnostics[name] = summary(
        rrs_from(Tte, K_ceil, ceil["fl"][None], floor=True)[0], truth, Tte
    )
    diagnostics[name]["Psi_per_sun"] = [float(v) for v in ceil["Psi"]]
    diagnostics[name]["fl_range"] = [
        float(np.nanmin(ceil["fl"])),
        float(np.nanmax(ceil["fl"])),
    ]
    klu[name] = klu_summary(K_ceil, Tte)
    klu["Ψ_KLu(ψ) nadir fit / µ∞ — shipped"] = klu_summary(K_te, Tte)
    klu["TT's Ψ_KLu / µ∞(PB24)"] = klu_summary(K_of(Tte, "tt", model), Tte)
    # D2, the ceiling: measured KLu/a, per-band f_L; with 8a's µd and PB24's own.
    for mud, name in (
        ("pb24", "D2 ceiling: measured KLu/a, per-band f_L(λ), 8a's µd"),
        ("data", "D2 ceiling with PB24's own µd"),
    ):
        st = closure_stats(Ttr, Ttr.klu_over_a, mud)
        fl_b = st["Gy"].sum(0) / st["G2"].sum(0)
        pred = rrs_from(Tte, Tte.klu_over_a, fl_b[None, None, :], floor=False, mud=mud)[
            0
        ]
        diagnostics[name] = summary(pred, truth, Tte)
        diagnostics[name]["fl_range"] = [float(fl_b.min()), float(fl_b.max())]

    # ---- the B_p-band split: refit on its training side (Q38) ------------------
    say("the B_p-band split")
    bp_split = None
    if bp_train.size and bp_test.size:
        Btr, Bte = A.take(bp_train), A.take(bp_test)
        bp_model, _ = fit_model(Btr, chosen, psi_fit["n_terms"])
        winb = Bte.window[None, :, None]
        bp_split = {
            "n_train": int(bp_train.size),
            "n_test": int(bp_test.size),
            "coeffs": list(bp_model["coeffs"]),
            "s": list(bp_model["s"]),
            "heldout": {
                GORDON: stat(Bte.gordon, Bte.rrs, winb),
                BASE: stat(spec_rrs(Bte, SPEC_BASE, bp_model), Bte.rrs, winb),
                FINAL: stat(spec_rrs(Bte, SPEC_FINAL, bp_model), Bte.rrs, winb),
                KLU: stat(spec_rrs(Bte, SPEC_KLU, bp_model), Bte.rrs, winb),
            },
            "klu": klu_summary(K_of(Bte, "nadir", bp_model), Bte)["window"],
        }

    # ---- Rrs (Q39) -------------------------------------------------------------
    say("Rrs through the Fresnel transfer")
    tf = transfer_fits(Ttr)
    wint = Tte.window[None, :, None]
    rrs_space = {"nadir_refits": tf, "transfer_alone": {}, "variant": {}}
    for name, (A0, Bc, dw) in transfers(tf).items():
        rrs_space["transfer_alone"][name] = stat(
            apply_transfer(Tte, Tte.rrs, A0, Bc, dw), Tte.Rrs, wint
        )
        rrs_space["variant"][name] = stat(
            apply_transfer(Tte, preds[FINAL], A0, Bc, dw), Tte.Rrs, wint
        )
    A0, Bc, dw = transfers(tf)[RRS_TRANSFER]
    Rrs_final = apply_transfer(Tte, preds[FINAL], A0, Bc, dw)
    Rrs_gordon = apply_transfer(Tte, preds[GORDON], A0, Bc, dw)
    rrs_space["per_theta_s"] = {
        f"{ts:g}": {
            FINAL: stat(Rrs_final[:, k], Tte.Rrs[:, k]),
            GORDON: stat(Rrs_gordon[:, k], Tte.Rrs[:, k]),
        }
        for k, ts in enumerate(Tte.theta_s)
    }

    # ---- per band ---------------------------------------------------------------
    per_band = []
    win = Tte.window
    for w, lam in enumerate(Tte.wave):
        row = [f"{lam:g}"]
        for name in (GORDON, BASE, FINAL, KLU):
            row.append(
                "{:.4f}".format(
                    100 * stat(preds[name][:, win, w], truth[:, win, w])["median_abs"]
                )
            )
        row.append(
            f"{100 * stat(Rrs_final[:, win, w], Tte.Rrs[:, win, w])['median_abs']:.4f}"
        )
        per_band.append(row)
    band_header = ["wave_nm", GORDON, BASE, FINAL, KLU, f"{FINAL}, Rrs"]

    # ---- L23 ----------------------------------------------------------------------
    say("L23")
    l23 = l23_check(model, l23_extras) if with_l23 and not quick else None

    # ---- the gate -----------------------------------------------------------------
    fw = scores[FINAL]["window"]
    l23_key = "ZTT (PB24_nadir), TT's H (V = 15 km) -- the gate"
    gate = {
        "median_abs": fw["median_abs"],
        "median_below": GATE_MEDIAN,
        "median_pass": bool(fw["median_abs"] < GATE_MEDIAN),
        "nonphysical_fraction": fw["nonphysical_fraction"],
        "nonphysical_below": GATE_NONPHYSICAL,
        "nonphysical_pass": bool(fw["nonphysical_fraction"] < GATE_NONPHYSICAL),
        "l23_reference": GATE_L23_REFERENCE,
        "l23_tolerance_pp": GATE_L23_TOLERANCE,
        "l23_heldout_rrms": None if l23 is None else l23[l23_key]["held-out"],
        "l23_pass": None
        if l23 is None
        else bool(l23[l23_key]["held-out"] <= GATE_L23_REFERENCE + GATE_L23_TOLERANCE),
    }
    gate["pass"] = bool(
        gate["median_pass"] and gate["nonphysical_pass"] and gate["l23_pass"]
    )  # an L23 clause that was not run does not pass
    gate["default_by_decision"] = "rt_fit_pb24.md Q45 (JXP, 2026-11-26)"
    klu_gate = scores[KLU]["window"]
    gate["klu_fit_record"] = {
        "median_abs": klu_gate["median_abs"],
        "nonphysical_fraction": klu_gate["nonphysical_fraction"],
        "l23_heldout_rrms": None
        if l23 is None
        else l23["measured-K_Lu fit (9b), TT's H (V = 15 km) -- the gate"]["held-out"],
    }

    tried = {
        str(n): {
            "max_node_misfit": v["max_node_misfit"],
            "cond": v["cond"],
            "coeffs_u_highest_first": list(legendre_to_u_poly(v["p_legendre"])),
        }
        for n, v in psi_fit["tried"].items()
    }
    Psi_nodes = psi_term("nadir", A.psi, model)
    metrics = {
        "azimuth_spread_at_nadir": azimuth_spread(cache, test_ids[:20]),
        "psi_fit": {
            "theta_s": [float(v) for v in A.theta_s],
            "psi_deg": [float(v) for v in A.psi],
            "node_value": [float(v) for v in psi_fit["nodes_all"]],
            "weighting": chosen,
            "weightings": {
                w: {
                    "rrms_train_d_space": fits[w][1]["rrms_train"],
                    "node_value": [float(v) for v in fits[w][1]["psi"]["nodes_all"]],
                    "n_terms": fits[w][1]["psi"]["n_terms"],
                }
                for w in WEIGHTINGS
            },
            "fitted": [float(v) for v in Psi_nodes],
            "tt": [float(v) for v in psi_term("tt", A.psi, model)],
            "tried": tried,
            "node_tol": NODE_TOL,
        },
        "fl_fit": {
            "history": fl_fit["solution"]["history"],
            "forms_d_space_rrms": fl_forms,
            "D_star_min_train": st_train["D_star_min"],
            "D_star_median_train": st_train["D_star_median"],
        },
        "klu_over_a_heldout": klu,
        "heldout_rrs": scores,
        "alone": alone,
        "floor_footprint": footprint,
        "diagnostics": diagnostics,
        "bp_band_split": bp_split,
        "Rrs": rrs_space,
        "l23": l23,
        "gate": gate,
    }
    coefficients = {
        "about": "The measured-K_Lu nadir fit of M8 task 9b (rt_fit_pb24.md "
        "Q34-Q44), angular_model='pb24_klu': a record, used by no variant. "
        "After Q45 the variant 'pb24_nadir' is task 8a's mu_d, mu_inf and H with "
        "TT's own Psi_KLu and f_L, floored; it has no constants of its own and "
        "is scored by the same script (nadir_metrics). The fit recorded here: "
        "Psi_KLu = 1 + F(psi), F a polynomial in u = 1 + cos(psi) "
        "(highest order first), fitted to PB24's measured nadir KLu: a per-sun "
        "average of (KLu/a) * mu_inf(PB24, 8a) (psi_klu_node_weighting), then the "
        "polynomial through the eight window nodes. f_L = FL_AVE(lambda) * "
        "(s0 + s1 sin psi), by "
        "linear closure on nadir rrs with that Psi_KLu, 8a's mu_d and the FF Pbb. "
        "mu_d, mu_inf and H are task 8a's; D floored as task 8b.",
        "variant_pb24_nadir": {
            "mu_d": "pb24 (8a)",
            "mu_inf": "pb24 (8a)",
            "angular_model": "tt",
            "floor": True,
            "bb_over_a_max": Z.NADIR_BB_OVER_A_MAX,
            "theta_v_tol_deg": Z.NADIR_TOL_DEG,
        },
        "n_f_terms": psi_fit["n_terms"],
        "psi_klu_node_weighting": chosen,
        "psi_klu_coeffs_u_highest_first": list(model["coeffs"]),
        "psi_min_deg": Z.PSI_KLU_PB24_NADIR_PSI_MIN,
        "fl_ave_wave_nm": [float(v) for v in A.wave],
        "fl_ave_per_band": [float(v) for v in model["FL"]],
        "fl_ave_decimals": FL_DECIMALS,
        "fl_s_s0_s1": list(model["s"]),
        "d_floor_lo_width": [float(v) for v in D_FLOOR],
        "fit": {
            "split": f"realisations, SPLIT_SEED={P.SPLIT_SEED}, "
            f"test_fraction={P.TEST_FRACTION}",
            "quick": quick,
            "window_max_deg": float(ANGLE_MAX_DEG),
            "n_train_realisations": int(train_ids.size),
            "n_test_realisations": int(test_ids.size),
            "train_digest": digest,
            "n_wave": int(A.wave.size),
            "n_rows_psi": psi_fit["n_rows"],
            "n_rows_psi_dropped": psi_fit["n_bad"],
            "n_rows_fl": st_train["n"],
            "n_rows_fl_dropped": st_train["n_bad"],
            "n_rounds": N_ROUNDS,
            "weighting": "Psi nodes: see psi_klu_node_weighting; polynomial: "
            "nodes weighted by row count; f_L: 1/D* per row (relative in D)",
        },
    }
    pr = probes(A, raw_iops_of(cache, used), model)
    say("done")
    return coefficients, metrics, pr, (band_header, per_band)


# -------------------------------------------------------------- outputs -----
def pct(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return "inf" if np.isinf(v) else f"{100 * v:.{nd}f}"


def outcomes(m) -> list[str]:
    """Each prediction of :data:`PREDICTIONS` beside what was measured."""
    hs, dg = m["heldout_rrs"], m["diagnostics"]
    a, b = hs[RUNG_A]["window"], hs[RUNG_B]["window"]
    cc, dd = dg[NAME_C]["window"], dg[NAME_DIRECT]["window"]
    d2 = dg["D2 ceiling: measured KLu/a, per-band f_L(λ), 8a's µd"]["window"]
    kc = m["klu_over_a_heldout"][NAME_C]["window"]["median_abs_dev"]
    g = hs[RUNG_B]["groups"]
    clear = max(g["bb/a[0, 0.001)"]["median_abs"], g["bb/a[0.001, 0.01)"]["median_abs"])
    yes = lambda ok: "**confirmed**" if ok else "**not confirmed**"  # noqa: E731
    return [
        f"- Rung (a): {pct(a['median_abs'])}% median, "
        f"{pct(a['nonphysical_fraction'])}% non-physical — "
        f"{yes(0.04 <= a['median_abs'] <= 0.05 and a['nonphysical_fraction'] > 0.01)}.",
        f"- Rung (b): {pct(b['median_abs'])}% median (bb/a < 0.01: up to "
        f"{pct(clear)}%) — {yes(0.035 <= b['median_abs'] <= 0.045)}. Fitting "
        "Ψ_KLu to the measured K_Lu makes rrs worse than TT's Ψ_KLu does.",
        f"- Rung (c): {pct(cc['median_abs'])}% median against the D2 ceiling's "
        f"{pct(d2['median_abs'])}%; KLu/a to {pct(kc)}% median |dev| — "
        f"{yes(0.03 <= cc['median_abs'] <= 0.035 and kc <= 0.05)}.",
        f"- Factorised (rung (c)) against the direct fit of KLu/a: "
        f"{pct(cc['median_abs'])}% and {pct(dd['median_abs'])}%, "
        f"{100 * abs(cc['median_abs'] - dd['median_abs']):.2f} pp apart — "
        f"{yes(abs(cc['median_abs'] - dd['median_abs']) < 0.002)}.",
    ]


def domain_lines(m) -> list[str]:
    """The variant inside its stated bb/a domain, beside Gordon and the base."""
    hs = m["heldout_rrs"]
    share = hs[FINAL]["in_domain"]["share_of_window"]
    suns = list(hs[FINAL]["per_theta_s_in_domain"])
    out = [
        f"**Inside the variant's domain** (bb/a ≤ {Z.NADIR_BB_OVER_A_MAX:g}: "
        f"{pct(share, 1)}% of the held-out nadir values), median |err| % in rrs:",
        "",
        "| model | window | q90 | " + " | ".join(f"{t}°" for t in suns) + " |",
        "|---|---|---|" + "---|" * len(suns),
    ]
    for name in (GORDON, BASE, FINAL, KLU):
        d = hs[name]["in_domain"]
        out.append(
            f"| {name} | {pct(d['median_abs'])} | {pct(d['q90_abs'])} | "
            + " | ".join(
                pct(hs[name]["per_theta_s_in_domain"][t]["median_abs"]) for t in suns
            )
            + " |"
        )
    return out


def report(c, m, *, quick: bool) -> str:
    f, g = c["fit"], m["gate"]
    hs = m["heldout_rrs"]
    suns = list(hs[FINAL]["per_theta_s"])
    win_suns = [t for t in suns if float(t) <= ANGLE_MAX_DEG]
    verdict = lambda ok: "–" if ok is None else ("**pass**" if ok else "**FAIL**")  # noqa: E731
    L = [
        "# ZTT (PB24_nadir) — the nadir-only ZTT on PB24 (M8 task 9)",
        "",
        "*Generated by `design/py/fit_ztt_pb24_nadir.py`"
        f"{' --quick' if quick else ''}. Do not edit by hand.*",
        "",
        '**The variant (Q45, task 9c).** `rrs_ZTT(variant="pb24_nadir")`, the '
        "package default, is task 8a's H, µd and µ∞ fitted to PB24 with **TT's own "
        "Ψ_KLu and f_L** and the floor on the denominator: the M7 pattern on "
        "PB24. It is for nadir viewing and "
        f"bb/a ≤ {Z.NADIR_BB_OVER_A_MAX:g} (a `DomainWarning` outside either). The "
        "variant as first built in task 9b, with Ψ_KLu fitted to PB24's measured "
        "K_Lu and f_L by closure (Q35 (b)), failed its gate and is kept here as a "
        'record (`angular_model="pb24_klu"`).',
        "",
        f"Nadir viewing (θv = 0), realisation split: fitted on "
        f"**{f['n_train_realisations']} training realisations** (digest "
        f"`{f['train_digest']}`) at the window suns (θs ≤ {ANGLE_MAX_DEG:g}°), "
        f"scored on its {f['n_test_realisations']} held-out ones; {f['n_wave']} "
        "bands. Errors are relative, in %; medians and q90 are exact and keep "
        "non-physical values (counted as ∞).",
        "",
        "## Predictions (written before the fit, `rt_fit_pb24.md` Q35–Q36)",
        "",
        *[f"- {p}" for p in PREDICTIONS],
        "",
        "## The acceptance gate (Q42, written before the fit)",
        "",
        "For the variant as now defined. It misses the median clause and is the "
        "default by decision (Q45), not by the gate.",
        "",
        "| clause | value | criterion | |",
        "|---|---|---|---|",
        f"| window median, held out, rrs | {pct(g['median_abs'])}% | < "
        f"{pct(g['median_below'])}% | {verdict(g['median_pass'])} |",
        f"| non-physical values | {pct(g['nonphysical_fraction'], 4)}% | < "
        f"{pct(g['nonphysical_below'])}% | {verdict(g['nonphysical_pass'])} |",
        "| L23 held-out rRMS (TT's H) | "
        + (
            "not run"
            if g["l23_heldout_rrms"] is None
            else f"{g['l23_heldout_rrms']:.2f}%"
        )
        + f" | ≤ {g['l23_reference']:.2f} + {g['l23_tolerance_pp']:.1f} pp | "
        f"{verdict(g['l23_pass'])} |",
        "",
        f"**Gate: {'PASS' if g['pass'] else 'FAIL'}.** For the record, the "
        "measured-K_Lu fit of task 9b read "
        f"{pct(g['klu_fit_record']['median_abs'])}% median, "
        f"{pct(g['klu_fit_record']['nonphysical_fraction'], 4)}% non-physical and "
        + (
            "(L23 not run)"
            if g["klu_fit_record"]["l23_heldout_rrms"] is None
            else f"{g['klu_fit_record']['l23_heldout_rrms']:.2f}% on L23"
        )
        + ".",
        "",
        *domain_lines(m),
        "",
        "## Predictions against outcomes",
        "",
        *outcomes(m),
        "",
        "## The measured-K_Lu fit (task 9b; a record, used by no variant)",
        "",
        f"- Ψ_KLu = 1 + F(ψ), {c['n_f_terms']} terms in u = 1 + cos ψ (rule: the "
        f"smallest count whose worst node misfit is ≤ {100 * NODE_TOL:g}%); "
        f"coefficients, highest order first: "
        f"({', '.join(f'{v:.6g}' for v in c['psi_klu_coeffs_u_highest_first'])}).",
        f"- f_L = FL_AVE(λ) · (s0 + s1 sin ψ), s = "
        f"({c['fl_s_s0_s1'][0]:.5f}, {c['fl_s_s0_s1'][1]:.5f}) (TT: {Z.FL_OFFSET}, "
        f"{Z.FL_SIN_COEFF}); FL_AVE(λ) {min(c['fl_ave_per_band']):.4f}–"
        f"{max(c['fl_ave_per_band']):.4f}; {N_ROUNDS} alternating rounds, last "
        f"change {m['fl_fit']['history'][-1]['max_rel_change']:.1e}.",
        f"- Rows: {f['n_rows_psi']:,} for Ψ_KLu ({f['n_rows_psi_dropped']} dropped), "
        f"{f['n_rows_fl']:,} for f_L ({f['n_rows_fl_dropped']} dropped). D* on the "
        f"training rows: min {m['fl_fit']['D_star_min_train']:.3f}, median "
        f"{m['fl_fit']['D_star_median_train']:.1f}.",
        f"- The 13 azimuth nodes at nadir agree to "
        f"{m['azimuth_spread_at_nadir']:.1e} (relative); φ = 0 is used.",
        "",
        "Ψ_KLu at the nadir nodes (ψ = 180° − θs_w). A node is an average of "
        "the measured (KLu/a)·µ∞ over the training rows: “geometric” weights "
        "every row equally; “rrs” weights a row by the share of Equation (12)'s "
        "denominator its K_Lu term carries. Shipped: **"
        + m["psi_fit"]["weighting"]
        + "** (training D-space closure residual after the f_L fit: "
        + ", ".join(
            f"{w}: {pct(v['rrms_train_d_space'], 3)}%"
            for w, v in m["psi_fit"]["weightings"].items()
        )
        + ").",
        "",
        "| θs | ψ | node, geometric | node, rrs-weighted | fitted (shipped) | "
        "TT Eq. (4), (6) |",
        "|---|---|---|---|---|---|",
    ]
    pf = m["psi_fit"]
    for k, ts in enumerate(pf["theta_s"]):
        ext = "" if ts <= ANGLE_MAX_DEG else " *(not fitted)*"
        L.append(
            f"| {ts:g}°{ext} | {pf['psi_deg'][k]:.1f}° | "
            f"{pf['weightings']['geometric']['node_value'][k]:.4f} | "
            f"{pf['weightings']['rrs']['node_value'][k]:.4f} "
            f"| {pf['fitted'][k]:.4f} | {pf['tt'][k]:.4f} |"
        )
    L += [
        "",
        "Order of F(ψ):",
        "",
        "| terms | worst node misfit % | cond |",
        "|---|---|---|",
    ]
    for n, v in pf["tried"].items():
        mark = " **(shipped)**" if int(n) == c["n_f_terms"] else ""
        L.append(f"| {n}{mark} | {pct(v['max_node_misfit'], 3)} | {v['cond']:.2g} |")
    L += [
        "",
        "f_L forms against the same Ψ_KLu (D-space rRMS, %):",
        "",
        "| form | train | held out |",
        "|---|---|---|",
    ]
    for k, v in m["fl_fit"]["forms_d_space_rrms"].items():
        L.append(f"| {k} | {pct(v['rrms_train'], 3)} | {pct(v['rrms_test'], 3)} |")

    L += [
        "",
        "## KLu/a: the model's Ψ_KLu/µ∞ against PB24's measured KLu/a (held out)",
        "",
        "| model | median ratio | median \\|dev\\| % | 16–84% |",
        "|---|---|---|---|",
    ]
    for name, v in m["klu_over_a_heldout"].items():
        w = v["window"]
        L.append(
            f"| {name} | {w['median_ratio']:.4f} | {pct(w['median_abs_dev'])} | "
            f"{w['q16_q84'][0]:.3f}–{w['q16_q84'][1]:.3f} |"
        )
    shipped_klu = m["klu_over_a_heldout"]["Ψ_KLu(ψ) nadir fit / µ∞ — shipped"]
    L += [
        "",
        "The shipped fit by sun and by B_p (median ratio; median |dev| %):",
        "",
        "| " + " | ".join(f"{t}°" for t in suns) + " |",
        "|" + "---|" * len(suns),
        "| "
        + " | ".join(
            f"{shipped_klu['per_theta_s'][t]['median_ratio']:.3f}; "
            f"{pct(shipped_klu['per_theta_s'][t]['median_abs_dev'], 1)}"
            for t in suns
        )
        + " |",
        "",
        "| " + " | ".join(shipped_klu["per_B_p"]) + " |",
        "|" + "---|" * len(shipped_klu["per_B_p"]),
        "| "
        + " | ".join(
            "–"
            if v is None
            else f"{v['median_ratio']:.3f}; {pct(v['median_abs_dev'], 1)}"
            for v in shipped_klu["per_B_p"].values()
        )
        + " |",
    ]

    def ladder_row(name, s, bold=False):
        w = s["window"]
        b = "**" if bold else ""
        return (
            f"| {b}{name}{b} | {pct(w['median_abs'])} | {pct(w['q90_abs'])} | "
            f"{pct(w['rrms_finite'])} | {pct(w['nonphysical_fraction'], 3)} | "
            + " | ".join(pct(s["per_theta_s"][t]["median_abs"]) for t in suns)
            + " |"
        )

    head = (
        "| model | median | q90 | rRMS (finite) | non-physical | "
        + " | ".join(f"{t}°" for t in suns)
        + " |"
    )
    sep = "|---|---|---|---|---|" + "---|" * len(suns)
    L += [
        "",
        "## The nadir ladder, held out, in rrs",
        "",
        f"Window columns pool θs ≤ {ANGLE_MAX_DEG:g}°; the per-sun columns are "
        "medians (80° and 87.75° are extrapolation, never fitted). PB24's H is "
        "the data's at every rung (it equals the shipped table at the nodes), so "
        "H is not a separate rung here.",
        "",
        head,
        sep,
        ladder_row(GORDON, hs[GORDON]),
    ]
    for label, _ in LADDER:
        L.append(ladder_row(label, hs[label], bold=label == FINAL))
    L += [
        "",
        "The measured-K_Lu fit of task 9b, continued from rung (a) (a record):",
        "",
        head,
        sep,
    ]
    for label, _ in RECORD:
        L.append(ladder_row(label, hs[label]))
    L += ["", "Diagnostics (not shipped), same columns:", "", head, sep]
    for name, s in m["diagnostics"].items():
        L.append(ladder_row(name, s))
    L += [
        "",
        "Attribution (window median, %): each term added alone to "
        f"“{BASE}” ({pct(hs[BASE]['window']['median_abs'])}) and changed alone in "
        f"“{FINAL}” ({pct(hs[FINAL]['window']['median_abs'])}): removed, for the "
        "three terms the variant has; swapped in, for the two measured-K_Lu terms "
        "it does not.",
        "",
        "| term | added alone to the base | changed alone in the variant |",
        "|---|---|---|",
    ]
    for term, v in m["alone"].items():
        L.append(
            f"| {term} | {pct(v['added_alone_to_base']['median_abs'])} | "
            f"{pct(v['changed_alone_in_final']['median_abs'])} |"
        )
    fp = m["floor_footprint"]
    L += [
        "",
        f"The floor (lo = {c['d_floor_lo_width'][0]}, width = "
        f"{c['d_floor_lo_width'][1]}) moves {pct(fp['fraction_moved_gt_tol'], 4)}% of "
        f"held-out window values by more than {FOOTPRINT_TOL:g}; the unfloored D is "
        f"≤ 0 on {pct(fp['fraction_D_nonpositive'], 4)}% (minimum "
        f"{fp['D_model_min']:.3g}).",
        "",
        "## Breakdowns (held out, window suns, median |err| %)",
    ]
    cols = (GORDON, BASE, FINAL, KLU)
    for prefix, title in (
        ("bb/a[", "bb/a"),
        ("eta_bb[", "η_bb = bbw/bb"),
        ("B_p[", "B_p"),
        ("class ", "water class"),
    ):
        keys = [k for k in hs[FINAL]["groups"] if k.startswith(prefix)]
        if not keys:
            continue
        L += [
            "",
            f"| {title} | n | " + " | ".join(cols) + " |",
            "|---|---|" + "---|" * len(cols),
        ]
        for k in keys:
            L.append(
                f"| {k} | {hs[FINAL]['groups'][k]['n']:,} | "
                + " | ".join(pct(hs[n]["groups"][k]["median_abs"]) for n in cols)
                + " |"
            )
    bp = m["bp_band_split"] or {
        "n_train": 0,
        "n_test": 0,
        "heldout": {},
        "klu": {"median_ratio": float("nan"), "median_abs_dev": None},
    }
    L += [
        "",
        "## The B_p-band split (Q38): scored on the held-out B_p band",
        "",
        f"{bp['n_train']} training and {bp['n_test']} held-out realisations (the "
        "middle B_p quantile band is held out). The variant has nothing fitted "
        "here (its 8a terms are the realisation split's); the measured-K_Lu fit "
        "is refit on this split's training side, same form and order.",
        "",
        "| model | median | q90 | non-physical |",
        "|---|---|---|---|",
    ]
    for name, v in bp["heldout"].items():
        L.append(
            f"| {name} | {pct(v['median_abs'])} | {pct(v['q90_abs'])} | "
            f"{pct(v['nonphysical_fraction'], 3)} |"
        )
    L.append(
        f"\nKLu/a of the measured-K_Lu fit, held out on this split: median ratio "
        f"{bp['klu']['median_ratio']:.4f}, median |dev| "
        f"{pct(bp['klu']['median_abs_dev'])}%."
    )
    R = m["Rrs"]
    L += [
        "",
        "## In Rrs (Q39): the variant through the Fresnel transfer",
        "",
        "At nadir the transfer is Rrs = A·rrs/(1 − B·rrs), A = A0 (θv-only form) "
        "or A0·t_d/T(0) (downwelling form, with the sample's H). “Transfer alone” "
        "applies it to PB24's own rrs; “variant” to ZTT (PB24_nadir). Held out, "
        "window suns, every bb/a.",
        "",
        "| transfer | A0 | B | transfer alone, median | variant in Rrs, median | q90 |",
        "|---|---|---|---|---|---|",
    ]
    tfs = transfers(R["nadir_refits"])
    for name, (A0, Bc, _) in tfs.items():
        mark = " **(used)**" if name == RRS_TRANSFER else ""
        L.append(
            f"| {name}{mark} | {A0:.5f} | {Bc:.4f} | "
            f"{pct(R['transfer_alone'][name]['median_abs'], 3)} | "
            f"{pct(R['variant'][name]['median_abs'])} | "
            f"{pct(R['variant'][name]['q90_abs'])} |"
        )
    L += [
        "",
        f"Per sun, in Rrs, with “{RRS_TRANSFER}” (θs = 0° is the geometry of a "
        "fully normalised satellite product, Q34):",
        "",
        "| model | " + " | ".join(f"{t}°" for t in suns) + " |",
        "|---|" + "---|" * len(suns),
    ]
    for name in (GORDON, FINAL):
        L.append(
            f"| {name} | "
            + " | ".join(pct(R["per_theta_s"][t][name]["median_abs"]) for t in suns)
            + " |"
        )
    L += ["", "## L23 (never fitted; held-out scenes, rRMS %)", ""]
    if m["l23"] is None:
        L.append("Not run (`--quick`, or the L23 data are absent).")
    else:
        zs = [
            k
            for k in next(iter(m["l23"].values()))
            if k not in ("held-out", "nonphysical_fraction")
        ]
        L += [
            "| model | held out | "
            + " | ".join(f"{z}°" for z in zs)
            + " | non-physical % |",
            "|---|---|" + "---|" * (len(zs) + 1),
        ]
        for name, v in m["l23"].items():
            L.append(
                f"| {name} | {v['held-out']:.2f} | "
                + " | ".join(f"{v[z]:.2f}" for z in zs)
                + f" | {pct(v['nonphysical_fraction'], 3)} |"
            )
    L += [
        "",
        f"Window suns used above: {', '.join(win_suns)}°.",
        "",
        "## Reproducibility",
        "",
        "- `--check` refits and compares every file byte for byte; the tests "
        're-evaluate `nadir_probes.json` through `rrs_ZTT(variant="pb24_nadir")` '
        "without the data and check the shipped constants against "
        "`nadir_coefficients.json`.",
        "",
    ]
    return "\n".join(L)


def to_csv(header, rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def _clean(o):
    """JSON-safe: NumPy scalars/arrays to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def outputs(coefficients, metrics, probes_, bands, *, quick: bool) -> dict[str, str]:
    dump = {"indent": 2, "sort_keys": True}
    return {
        f"{PREFIX}coefficients.json": json.dumps(_clean(coefficients), **dump) + "\n",
        f"{PREFIX}metrics.json": json.dumps(_clean(metrics), **dump) + "\n",
        f"{PREFIX}probes.json": json.dumps(_clean(probes_), **dump) + "\n",
        f"{PREFIX}metrics.md": report(coefficients, metrics, quick=quick),
        f"{PREFIX}per_band.csv": to_csv(*bands),
    }


#: Okabe–Ito, colour-blind-safe, assigned in fixed order (the house palette of
#: ``run_ztt_l23.py``); a model keeps its colour in every panel.
COLORS = {
    GORDON: "#5c5c5c",
    BASE: "#E69F00",
    KLU: "#56B4E9",
    FINAL: "#0072B2",
    "ceiling": "#009E73",
}
FIGURES = (
    f"{PREFIX}fig_terms.png",
    f"{PREFIX}fig_ladder.png",
    f"{PREFIX}fig_breakdowns.png",
)


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


def _plain_log_axis(ax) -> None:
    """Plain numbers (2, 5, 10, …) on a log y axis instead of powers of ten."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def figures(c, m, out: Path) -> None:
    import jax.numpy as jnp
    from run_validation import apply_house_style

    plt = apply_house_style()
    coeffs = tuple(c["psi_klu_coeffs_u_highest_first"])
    pf = m["psi_fit"]
    ts = np.asarray(pf["theta_s"])
    psi_n = np.asarray(pf["psi_deg"])
    nodes = np.asarray(pf["node_value"])
    win = ts <= ANGLE_MAX_DEG

    # (1) the fitted terms.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    psi = np.linspace(Z.PSI_KLU_PB24_NADIR_PSI_MIN, 180.0, 300)
    ax = axes[0]
    ax.plot(
        psi,
        np.asarray(Z.psi_KLu_pb24_nadir(jnp.asarray(psi), coeffs)),
        color=COLORS[KLU],
        lw=2,
        label="measured-K_Lu fit (9b, a record)",
    )
    ax.plot(
        psi,
        np.asarray(Z.psi_KLu(jnp.asarray(psi))),
        color=COLORS[FINAL],
        lw=2.4,
        label="TT Eqs (4), (6): used by ZTT (PB24_nadir)",
    )
    ax.plot(
        psi_n[win],
        nodes[win],
        "o",
        color="#1a1a1a",
        ms=6,
        label="PB24 (KLu/a)·µ∞ node, suns 0–70°",
    )
    ax.plot(
        psi_n[~win],
        nodes[~win],
        "o",
        mfc="white",
        color="#1a1a1a",
        ms=6,
        label="suns 80°, 87.75° (not fitted)",
    )
    ax.set_xlabel("in-water scattering angle ψ (deg); nadir: ψ = 180° − θs_w")
    ax.set_ylabel("Ψ_KLu = K_Lu / K∞")
    ax.set_title("Ψ_KLu from PB24's measured nadir K_Lu", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)
    ax = axes[1]
    rad = np.deg2rad(psi)
    s0, s1 = c["fl_s_s0_s1"]
    ax.plot(
        psi,
        s0 + s1 * np.sin(rad),
        color=COLORS[KLU],
        lw=2,
        label="measured-K_Lu fit (9b)",
    )
    ax.plot(
        psi,
        Z.FL_OFFSET + Z.FL_SIN_COEFF * np.sin(rad),
        color=COLORS[FINAL],
        lw=2.4,
        label="TT Eq. (31): used by the variant",
    )
    ax.axhline(1.0, color="#5c5c5c", lw=0.8, ls=":")
    ax.set_xlabel("ψ (deg)")
    ax.set_ylabel("s(ψ) = s0 + s1 sin ψ")
    ax.set_title("f_L's ψ scaling (× FL_AVE, mean 1)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(
        c["fl_ave_wave_nm"],
        c["fl_ave_per_band"],
        color=COLORS[KLU],
        lw=2,
        label="measured-K_Lu fit (9b)",
    )
    ax.plot(
        Z.FL_AVE_WAVE,
        Z.FL_AVE / Z.FL_AVE.mean(),
        color=COLORS[FINAL],
        lw=2.4,
        label="TT Table A3 (mean 1): used by the variant",
    )
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("FL_AVE(λ)")
    ax.set_title("f_L's spectral shape", loc="left", fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_png(fig, out / FIGURES[0])

    # (2) the ladder per sun.
    hs = m["heldout_rrs"]
    suns = list(hs[FINAL]["per_theta_s"])
    x = np.array([float(t) for t in suns])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ceiling = "D2 ceiling: measured KLu/a, per-band f_L(λ), 8a's µd"
    series = [
        (GORDON, hs[GORDON], COLORS[GORDON], "-"),
        (BASE, hs[BASE], COLORS[BASE], "-"),
        (
            "ZTT (PB24_nadir): 8a terms, TT's Ψ_KLu and f_L",
            hs[FINAL],
            COLORS[FINAL],
            "-",
        ),
        ("measured-K_Lu fit (9b, a record)", hs[KLU], COLORS[KLU], "-"),
        (
            "ceiling: measured K_Lu/a (D2)",
            m["diagnostics"][ceiling],
            COLORS["ceiling"],
            "--",
        ),
    ]
    for label, s, color, ls in series:
        y = [100 * (s["per_theta_s"][t]["median_abs"] or np.nan) for t in suns]
        ax.plot(
            x,
            y,
            "o",
            ls=ls,
            color=color,
            lw=2.4 if "nadir)" in label else 1.6,
            ms=5,
            label=label,
        )
    ax.axvline(75.0, color="#5c5c5c", lw=0.8, ls=":")
    ax.text(
        76, ax.get_ylim()[1] * 0.97, "not fitted", fontsize=8, color="#5c5c5c", va="top"
    )
    ax.set_yscale("log")
    _plain_log_axis(ax)
    ax.set_xlabel("solar zenith θs (deg), nadir view")
    ax.set_ylabel("median |rrs error| (%), held out")
    ax.set_title(
        "Nadir rrs error per sun, PB24 held-out realisations", loc="left", fontsize=10.5
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_png(fig, out / FIGURES[1])

    # (3) breakdowns by bb/a and B_p.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, prefix, title in (
        (axes[0], "bb/a[", "by bb/a"),
        (axes[1], "B_p[", "by particle backscattering ratio B_p"),
    ):
        keys = [k for k in hs[FINAL]["groups"] if k.startswith(prefix)]
        for name, label in (
            (GORDON, GORDON),
            (KLU, "measured-K_Lu fit (9b, a record)"),
            (FINAL, "ZTT (PB24_nadir)"),
        ):
            y = [100 * (hs[name]["groups"][k]["median_abs"] or np.nan) for k in keys]
            ax.plot(
                range(len(keys)),
                y,
                "o-",
                color=COLORS[name],
                lw=2.4 if name == FINAL else 1.6,
                ms=5,
                label=label,
            )
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(
            [k[len(prefix) - 1 :] for k in keys], fontsize=8, rotation=20
        )
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_yscale("log")
        _plain_log_axis(ax)
    limit = [k for k in hs[FINAL]["groups"] if k.startswith("bb/a[")]
    edge = [i for i, k in enumerate(limit) if k.startswith("bb/a[0.5946")]
    if edge:
        axes[0].axvline(edge[0] - 0.5, color="#5c5c5c", lw=0.8, ls=":")
        axes[0].text(
            edge[0] - 0.45,
            axes[0].get_ylim()[0] * 1.15,
            "outside the\nvariant's domain",
            fontsize=8,
            color="#5c5c5c",
            va="bottom",
        )
    axes[0].set_ylabel("median |rrs error| (%), held out, suns 0–70°")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_png(fig, out / FIGURES[2])


# ------------------------------------------------------------------ main -----
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", default=None, help="hyperspectral cache")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK} realisations")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument(
        "--check", action="store_true", help="refit and compare byte for byte"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-l23", action="store_true", help="skip the L23 score")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.quick and args.out_dir == OUT and not args.dry_run:
        parser.error("--quick fits on a subsample; pass --out-dir or --dry-run")
    if args.quick and args.check:
        parser.error("--check compares the shipped files; it cannot be --quick")

    coefficients, metrics, pr, bands = run(
        quick=args.quick,
        cache_dir=args.cache_dir,
        progress=not args.quiet,
        with_l23=not args.no_l23,
    )
    coefficients, metrics = _clean(coefficients), _clean(metrics)
    files = outputs(coefficients, metrics, pr, bands, quick=args.quick)
    print(files[f"{PREFIX}metrics.md"])

    if args.check:
        stale = [
            n for n, text in files.items() if (args.out_dir / n).read_text() != text
        ]
        if not args.no_figures:
            with tempfile.TemporaryDirectory() as tmp:
                figures(coefficients, metrics, Path(tmp))
                for name in FIGURES:
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
    if not args.no_figures:
        figures(coefficients, metrics, args.out_dir)
    print(f"wrote {sorted(files)} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
