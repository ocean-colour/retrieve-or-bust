#!/usr/bin/env python
"""
Fit ZTT's angular terms to PB24 -- M8 task 8b, ``rt_fit_pb24.md`` (Q23–Q25, Q30;
"ZTT (PB24)" completed).

Everything here is on the **window** (θs, θv_air ≤ 70°) of the realisation
split: fitted on its training realisations, scored on its held-out ones, all
451 bands. Deterministic throughout (float64 normal equations summed in chunk
order, in-memory linear solves; no optimiser), so ``--check`` is byte for byte.

(1) **D1, the BRDF-shape test.** ``rrs(θv, Δφ) / rrs(nadir)`` at fixed
    (realisation, θs, λ): PB24 against the 8a variant (TT's Ψ_KLu and f_L on
    the PB24 µd / µ∞ / H / FF Pbb), per θv_air and Δφ, and the final 8b
    variant; the task-6 O25 refit LUT beside them, in its own quantity
    (``Rrs``). It separates the angular *shape* from the nadir level.
(2) **D2, the nadir ceiling.** Ψ_KLu/µ∞ := PB24's own ``KLu/a`` plus a
    per-band ``f_L(λ)`` refit, nadir only, linear (one coefficient per band);
    with 8a's µd (the variant's) and, beside it, PB24's own µd.
(3) **The joint refit, linear in D-space.** With 8a's µd and µ∞ and the exact
    FF Pbb, Equation (12) inverts to a closure denominator per (sample, λ),
    ``D* = [β(ψ)/bb] / (µd · rrs_PB24)``, and

        D* − (a/bb)(1 + |cos θv|/µ∞) − 1/b̃b
            = (a/bb)(|cos θv|/µ∞) · F(ψ)  −  (1/b̃b − 1) · f_L(ψ, λ),

    linear in the coefficients of ``F`` and ``f_L``. ``F(ψ)`` is
    IOP-independent (Q24), a polynomial in cos ψ (Q30) written in
    ``u = 1 + cos ψ`` (no cancellation at 180°; accumulated in the Legendre
    basis of ``t = 2u − 1`` for conditioning and converted exactly);
    ``f_L = FL_AVE(λ) · (s0 + s1 sin ψ + s2 cos ψ)`` with ``FL_AVE`` per band
    (Q25), bilinear, so **two alternating linear passes with a fixed count**.
    Rows are weighted by ``1/D*`` (relative in D; relative in rrs to first
    order: e_rrs = −e_D/(1 + e_D)).

    One streamed pass over the training window accumulates the block normal
    equations (the ``F`` block, the per-band ``f_L`` blocks and their cross
    terms) -- the sufficient statistics of *every* linear fit in that column
    space -- so the alternating passes, the order sensitivity and the
    training residual all run in memory. The held-out pass accumulates the
    same blocks (held-out D-space residuals of every candidate) and scores
    the final variant in ``rrs``.
(4) **A smooth floor on D** (``denominator_floor_pb24``) so the variant is
    finite everywhere; its footprint (window values moved by more than
    :data:`FOOTPRINT_TOL` relative) is reported.

Writes ``design/ztt_pb24/angular_*``: ``angular_coefficients.json`` (the source
of ``PSI_KLU_PB24_COEFFS``, ``FL_AVE_PB24``, ``FL_S_PB24``, ``D_FLOOR_PB24``),
``angular_metrics.{json,md}``, ``angular_probes.json`` (points the tests
re-evaluate through ``rrs_ZTT(variant="pb24")``, no data needed) and figures.

Memory: a worker holds one chunk of :data:`CHUNK` realisations × 832 window
geometries × 451 bands in float64 (≈ 2 GB peak); the default
:data:`DEFAULT_WORKERS` workers start staggered (task 5b's policy). Run the
full fit in its own scope from a plain terminal::

    systemd-run --user --scope -p MemoryMax=80G \\
        python design/py/fit_ztt_pb24_angular.py

    python design/py/fit_ztt_pb24_angular.py --check          # byte for byte
    python design/py/fit_ztt_pb24_angular.py --quick --out-dir /tmp/x
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")  # fits and metrics in float64

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pb24_baseline as RB  # noqa: E402  (histograms, ψ bins, the pool)

from robust.rt import baselines as B  # noqa: E402
from robust.rt import conventions as C  # noqa: E402
from robust.rt import ztt as Z  # noqa: E402
from robust.rt.data import pb24 as P  # noqa: E402
from robust.rt.data import pb24_cache as PC  # noqa: E402

OUT = ROOT / "design" / "ztt_pb24"
PREFIX = "angular_"
O25_LUT = OUT / "o25_refit_lut.npz"

#: Realisations per streamed chunk. Part of the fit's definition: the per-chunk
#: float64 sums are added in chunk order, so a rerun must use the same value.
CHUNK = 10

#: ``--quick``: training and test realisations (the first n of each side).
QUICK = (200, 50)

#: Worker policy as the baseline's (task 5b): ≤ 6, staggered starts.
DEFAULT_WORKERS = 4
MAX_WORKERS = RB.MAX_WORKERS

#: The window: θs and θv_air ≤ 70° (Q21, Q23).
ANGLE_MAX_DEG = P.ANGLE_WINDOW_MAX

#: Terms of F(ψ): shipped, and the sensitivity set (held-out D-space rRMS is
#: reported for each from the same sufficient statistics). Five is TT's
#: order (a quartic), decided before the fit; the rule, written before the
#: numbers: ship five unless its held-out residual exceeds four's (an overfit
#: sign), then four.
N_F_TERMS = 5
F_TERMS_TRIED = (3, 4, 5, 6, 7)
N_F_MAX = max(F_TERMS_TRIED)

#: Alternating passes: a fixed count (deterministic), convergence reported.
N_ROUNDS = 12

#: FL_AVE(λ) is shipped rounded to this many decimals (a 451-value literal).
FL_DECIMALS = 6

#: The floor on D: ``(lo, width)``; PB24's D* minimum is asserted above 2·lo.
D_FLOOR = Z.D_FLOOR_PB24
FOOTPRINT_TOL = 1e-4

#: Held-out tables: bb/a bins (task 6 / 8a edges) and ψ bins (the baseline's).
BBA_EDGES = (0.0, 1e-3, 1e-2, 0.1, 0.5946, 2.0, np.inf)
PSI_EDGES = RB.PSI_EDGES

#: Pooled ψ ranges of the predictions.
PSI_POOLS = {"psi >= 155": (155.0, 180.001), "psi 110.4-134": (110.4, 134.0)}

#: Ψ_KLu is checked for monotonicity and range on this grid (degrees).
PSI_GRID = np.arange(91.0, 180.0 + 0.125, 0.25)
PSI_RANGE_PREDICTED = (0.8, 2.5)

#: Report order of the models (dict order differs between a run and the JSON).
D1_ORDER = (
    "ZTT 8a (TT angular terms), rrs",
    "O25 refit (task 6 LUT), Rrs",
    "ZTT (PB24), rrs",
)
MODEL_ORDER = (
    "ZTT 8a (TT angular terms)",
    "ZTT (PB24), floored",
    "ZTT (PB24), no floor",
)

#: Probe points committed for the tests (seeded).
N_PROBES = 12
PROBE_SEED = P.SPLIT_SEED


# ---------------------------------------------------------------- bases -----
def legendre_basis(t, n: int) -> np.ndarray:
    """``P_0 … P_{n−1}(t)`` by the three-term recurrence, stacked last."""
    t = np.asarray(t, dtype=np.float64)
    cols = [np.ones_like(t)]
    if n > 1:
        cols.append(t.copy())
    for k in range(1, n - 1):
        cols.append(((2 * k + 1) * t * cols[k] - k * cols[k - 1]) / (k + 1))
    return np.stack(cols[:n], axis=-1)


def angular_variable(psi_deg) -> np.ndarray:
    """``t = 2u − 1`` with ``u = 1 + cos ψ``: ψ = 180° is t = −1, 90° is t = 1."""
    return 2.0 * (1.0 + np.cos(np.deg2rad(np.asarray(psi_deg, dtype=np.float64)))) - 1.0


def fl_basis(psi_deg) -> np.ndarray:
    """``(1, sin ψ, cos ψ)`` stacked last."""
    psi = np.deg2rad(np.asarray(psi_deg, dtype=np.float64))
    return np.stack([np.ones_like(psi), np.sin(psi), np.cos(psi)], axis=-1)


def legendre_to_u_poly(p) -> tuple[float, ...]:
    """Legendre coefficients in ``t = 2u − 1`` → monomial coefficients in ``u``,
    highest order first (``jnp.polyval``'s convention, as ``PSI_KLU_PB24_COEFFS``)."""
    from numpy.polynomial import Legendre, Polynomial

    in_t = Legendre(np.asarray(p, dtype=np.float64)).convert(kind=Polynomial)
    in_u = in_t(Polynomial([-1.0, 2.0]))
    return tuple(float(v) for v in in_u.coef[::-1])


# ------------------------------------------------------ normal equations -----
@dataclass
class Blocks:
    """Block normal equations of the D-space fit, summed over rows.

    Columns: the ``n_f`` F terms (shared by every row) and, per band, the
    three f_L terms (present only in that band's rows). ``AtB[w]`` is the
    cross block of band ``w``; the f_L blocks of different bands never meet.
    """

    n_f: int
    n_wave: int
    AtA: np.ndarray = field(init=False)
    AtB: np.ndarray = field(init=False)
    BtB: np.ndarray = field(init=False)
    Aty: np.ndarray = field(init=False)
    Bty: np.ndarray = field(init=False)
    yty: float = 0.0
    n_rows: int = 0

    def __post_init__(self):
        self.AtA = np.zeros((self.n_f, self.n_f))
        self.AtB = np.zeros((self.n_wave, self.n_f, 3))
        self.BtB = np.zeros((self.n_wave, 3, 3))
        self.Aty = np.zeros(self.n_f)
        self.Bty = np.zeros((self.n_wave, 3))

    def add(self, A, Bm, y, w) -> None:
        """Rows ``A (n, W, n_f)``, ``Bm (n, W, 3)``, target ``y (n, W)``, weight
        ``w (n, W)`` (rows with ``w = 0`` drop out)."""
        Aw, Bw, yw = A * w[..., None], Bm * w[..., None], y * w
        self.AtA += np.einsum("nwi,nwj->ij", Aw, Aw)
        self.AtB += np.einsum("nwi,nwk->wik", Aw, Bw)
        self.BtB += np.einsum("nwi,nwk->wik", Bw, Bw)
        self.Aty += np.einsum("nwi,nw->i", Aw, yw)
        self.Bty += np.einsum("nwk,nw->wk", Bw, yw)
        self.yty += float(np.sum(yw * yw))
        self.n_rows += int(np.count_nonzero(w))

    def merge(self, other: Blocks) -> None:
        for name in ("AtA", "AtB", "BtB", "Aty", "Bty"):
            getattr(self, name).__iadd__(getattr(other, name))
        self.yty += other.yty
        self.n_rows += other.n_rows

    def truncated(self, n_f: int) -> Blocks:
        """The same statistics for the first ``n_f`` F terms."""
        out = Blocks(n_f, self.n_wave)
        out.AtA = self.AtA[:n_f, :n_f].copy()
        out.AtB = self.AtB[:, :n_f, :].copy()
        out.BtB = self.BtB.copy()
        out.Aty = self.Aty[:n_f].copy()
        out.Bty = self.Bty.copy()
        out.yty, out.n_rows = self.yty, self.n_rows
        return out

    def residual_ss(self, p, s, FL) -> float:
        """``Σ w² (y − Xθ)²`` for ``θ = (p, FL[w] s[k])`` from the statistics."""
        p, s, FL = (np.asarray(v, dtype=np.float64) for v in (p, s, FL))
        h = FL[:, None] * s[None, :]  # (W, 3)
        ss = self.yty - 2.0 * (p @ self.Aty + np.sum(h * self.Bty))
        ss += p @ self.AtA @ p
        ss += 2.0 * np.einsum("i,wik,wk->", p, self.AtB, h)
        ss += np.einsum("wi,wik,wk->", h, self.BtB, h)
        return float(ss)

    def rrms(self, p, s, FL) -> float:
        """Weighted rRMS in D (a fraction)."""
        return float(np.sqrt(max(self.residual_ss(p, s, FL), 0.0) / self.n_rows))


def solve_pass_a(bl: Blocks, FL):
    """``(p, s)`` given ``FL``: the reduced ``(n_f + 3)`` system. Returns the
    solution and the condition number of the reduced normal matrix."""
    FL = np.asarray(FL, dtype=np.float64)
    N_fs = np.einsum("w,wik->ik", FL, bl.AtB)
    N_ss = np.einsum("w,wkl->kl", FL**2, bl.BtB)
    N = np.block([[bl.AtA, N_fs], [N_fs.T, N_ss]])
    rhs = np.concatenate([bl.Aty, FL @ bl.Bty])
    sol = np.linalg.solve(N, rhs)
    return sol[: bl.n_f], sol[bl.n_f :], float(np.linalg.cond(N))


def solve_pass_b(bl: Blocks, s):
    """``(p, FL)`` given ``s``, by the Schur complement over the per-band
    columns ``B_w s`` (one per band; their block is diagonal)."""
    s = np.asarray(s, dtype=np.float64)
    c = np.einsum("wik,k->wi", bl.AtB, s)  # (W, n_f)
    d = np.einsum("k,wkl,l->w", s, bl.BtB, s)  # (W,)
    e = bl.Bty @ s  # (W,)
    S = bl.AtA - np.einsum("wi,wj,w->ij", c, c, 1.0 / d)
    rhs = bl.Aty - c.T @ (e / d)
    p = np.linalg.solve(S, rhs)
    FL = (e - c @ p) / d
    return p, FL, float(np.linalg.cond(S))


def solve_alternating(bl: Blocks, fl_init, n_rounds: int = N_ROUNDS) -> dict:
    """The bilinear fit by alternating linear passes with a fixed count.

    Each round: pass A (``p``, ``s`` given ``FL``), then pass B (``p``, ``FL``
    given ``s``), then ``FL`` renormalised to mean 1 (the scale moves into
    ``s``). The last round's largest relative change is reported.
    """
    FL = np.asarray(fl_init, dtype=np.float64).copy()
    FL /= FL.mean()
    history, p = [], None
    for _ in range(n_rounds):
        p_a, s, cond_a = solve_pass_a(bl, FL)
        p_b, FL_new, cond_b = solve_pass_b(bl, s)
        scale = float(FL_new.mean())
        FL_new, s = FL_new / scale, s * scale
        change = max(
            float(np.max(np.abs(FL_new / FL - 1.0))),
            0.0 if p is None else float(np.max(np.abs(p_b - p) / np.abs(p).max())),
        )
        p, FL = p_b, FL_new
        history.append(
            {
                "rrms_train": bl.rrms(p, s, FL),
                "max_rel_change": change,
                "cond_pass_a": cond_a,
                "cond_pass_b": cond_b,
            }
        )
    return {"p_legendre": p, "s": s, "FL": FL, "history": history}


# ------------------------------------------------------------- the data -----
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


def chunks_of(ids) -> list[tuple[int, ...]]:
    ids = [int(i) for i in ids]
    return [tuple(ids[i : i + CHUNK]) for i in range(0, len(ids), CHUNK)]


def load_window(ids, cache_dir):
    """The window batch of ``ids`` through the task-3 loader (every band the
    cache holds; the Fresnel check and ``validate`` were run in task 3)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return P.load_batch(
            realisations=list(ids),
            angles="window",
            extras=(),
            drop_zero_rrs=False,
            water_classes=None,
            validate=False,
            check_fresnel=False,
            cache_dir=cache_dir,
        )


@dataclass
class Parts:
    """Equation (12)'s pieces for one window chunk, float64 NumPy, with the
    PB24 µd / µ∞ / H / FF Pbb of task 8a (everything but Ψ_KLu and f_L)."""

    rrs: np.ndarray  # (n, W) PB24
    Rrs: np.ndarray  # (n, W) PB24
    num: np.ndarray  # (n, W) β(ψ)/bb, the exact FF Pbb
    mu_d: np.ndarray  # (n, W) 8a
    mu_inf: np.ndarray  # (n, W) 8a
    a_over_bb: np.ndarray  # (n, W)
    inv_bt: np.ndarray  # (n, W) 1/b̃b
    cos_tv: np.ndarray  # (n,) |cos θv_paper|
    psi: np.ndarray  # (n,)
    theta_s: np.ndarray  # (n,)
    theta_v: np.ndarray  # (n,) above-water
    dphi: np.ndarray  # (n,)
    wave: np.ndarray  # (W,)
    realisation: np.ndarray  # (n,)
    n_geom: int

    @property
    def D_star(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.num / (self.mu_d * self.rrs)

    @property
    def gF(self) -> np.ndarray:
        """The F(ψ) column's factor ``(a/bb)(|cos θv|/µ∞)``."""
        return self.a_over_bb * (self.cos_tv[:, None] / self.mu_inf)

    @property
    def gL(self) -> np.ndarray:
        """The f_L column's factor ``−(1/b̃b − 1)``."""
        return -(self.inv_bt - 1.0)

    @property
    def known(self) -> np.ndarray:
        """``(a/bb)(1 + |cos θv|/µ∞) + 1/b̃b``: the Ψ_KLu = 1, f_L = 0 part."""
        return self.a_over_bb * (1.0 + self.cos_tv[:, None] / self.mu_inf) + self.inv_bt

    def denominator(self, F, fl) -> np.ndarray:
        """D for a given ``F(ψ) (n, 1)`` and ``f_L (n, W)``."""
        return self.known + self.gF * F + self.gL * fl

    def rrs_of(self, D) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.num / (self.mu_d * D)


def parts_of(batch) -> Parts:
    """:class:`Parts` of a loaded window batch, through the shipped functions."""
    import jax.numpy as jnp

    iops, phase, geom = batch.iops, batch.phase_params, batch.geometry
    _, theta_s_air, theta_v_paper, psi = Z.geometry_to_paper_angles(geom)
    psi_b = jnp.asarray(psi)[:, None]
    wave = jnp.asarray(np.asarray(batch.wave, dtype=np.float64))
    bb = iops.bb
    bb_over_a, eta_bb = bb / iops.a, iops.bb_w / bb
    P_bb = Z.P_bb_from_phase(phase, psi_b, use_ff=True)
    num = Z.backward_phase_over_bb(iops, P_bb, psi_b)
    mu_d = Z.mu_d_pb24(
        jnp.asarray(theta_s_air)[:, None],
        bb_over_a,
        eta_bb,
        H=geom.diffuse_fraction,
        wave=wave,
    )
    mu_inf = Z.mu_infinity_pb24(bb_over_a, eta_bb)
    b_tilde = Z.bb_tilde(iops, phase.B_p)
    f64 = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    return Parts(
        rrs=f64(batch.rrs),
        Rrs=f64(batch.Rrs),
        num=f64(num),
        mu_d=f64(mu_d),
        mu_inf=f64(mu_inf),
        a_over_bb=f64(iops.a / bb),
        inv_bt=f64(1.0 / b_tilde),
        cos_tv=np.abs(f64(jnp.cos(jnp.deg2rad(theta_v_paper)))),
        psi=f64(psi),
        theta_s=f64(geom.theta_s),
        theta_v=f64(geom.theta_v),
        dphi=f64(geom.dphi),
        wave=f64(wave),
        realisation=np.asarray(batch.realisation, dtype=int),
        n_geom=int(batch.n_sample // len(np.unique(batch.realisation))),
    )


def tt_terms(pt: Parts):
    """TT's ``F(ψ) (n, 1)`` and ``f_L (n, W)`` (the 8a variant's pair)."""
    import jax.numpy as jnp

    psi_b = jnp.asarray(pt.psi)[:, None]
    F = np.asarray(Z.F_psi(psi_b), dtype=np.float64)
    fl = np.asarray(Z.f_L(psi_b, jnp.asarray(pt.wave)), dtype=np.float64)
    return F, fl


def pb24_terms(pt: Parts, coeffs, FL, s, fl_wave=None):
    """The refit's ``F (n, 1)`` and ``f_L (n, W)`` through the shipped
    evaluators with explicit coefficients (``FL`` on ``fl_wave``, default the
    chunk's bands)."""
    import jax.numpy as jnp

    psi_b = jnp.asarray(pt.psi)[:, None]
    F = np.asarray(Z.F_psi_pb24(psi_b, coeffs), dtype=np.float64)
    fl = np.asarray(
        Z.f_L_pb24(
            psi_b,
            jnp.asarray(pt.wave),
            fl_ave=jnp.asarray(FL),
            fl_wave=jnp.asarray(pt.wave if fl_wave is None else fl_wave),
            s=s,
        ),
        dtype=np.float64,
    )
    return F, fl


def floored(D, lo=None, width=None) -> np.ndarray:
    return np.asarray(Z.denominator_floor_pb24(D, lo, width), dtype=np.float64)


# -------------------------------------------------------------- scoring -----
def hist_of(e) -> np.ndarray:
    """The baseline's log10|e| histogram of a vector of relative errors."""
    return np.bincount(RB.histogram_bin(np.ravel(e)), minlength=RB.N_BIN).astype(
        np.int64
    )


@dataclass
class Score:
    """Streamed error statistics of one model on one side: histograms per
    group, exact float64 sums for rRMS, non-physical counts."""

    groups: dict = field(default_factory=dict)  # name -> (n_bin,) counts
    sum_sq: float = 0.0
    n_finite: int = 0
    n_nonphysical: int = 0
    n: int = 0

    def add(self, e: np.ndarray, pred: np.ndarray, partitions: dict) -> None:
        """``e`` relative errors ``(n, W)``; ``partitions`` name ->
        ``(index, names)``: an integer bin index per sample ``(n,)`` or per
        value ``(n, W)`` into ``names`` (each value in exactly one bin), so
        one ``bincount`` per partition does every group."""
        bins = RB.histogram_bin(e)
        for idx, names in partitions.values():
            idx2 = np.broadcast_to(idx[:, None] if idx.ndim == 1 else idx, e.shape)
            counts = np.bincount(
                (idx2 * RB.N_BIN + bins).ravel(), minlength=len(names) * RB.N_BIN
            ).reshape(len(names), RB.N_BIN)
            for k, name in enumerate(names):
                if name not in self.groups:
                    self.groups[name] = np.zeros(RB.N_BIN, dtype=np.int64)
                self.groups[name] += counts[k]
        fin = np.isfinite(e)
        self.sum_sq += float(np.sum(e[fin] ** 2))
        self.n_finite += int(fin.sum())
        self.n_nonphysical += int(np.sum((pred <= 0) | ~np.isfinite(pred)))
        self.n += int(e.size)

    def merge(self, other: Score) -> None:
        for k, v in other.groups.items():
            if k not in self.groups:
                self.groups[k] = np.zeros(RB.N_BIN, dtype=np.int64)
            self.groups[k] += v
        self.sum_sq += other.sum_sq
        self.n_finite += other.n_finite
        self.n_nonphysical += other.n_nonphysical
        self.n += other.n

    def summary(self) -> dict:
        def stats(h):
            return {
                "n": int(h.sum()),
                "median_abs": RB.hist_quantile(h, 0.5),
                "q90_abs": RB.hist_quantile(h, 0.9),
            }

        return {
            "n": self.n,
            "nonphysical_fraction": self.n_nonphysical / self.n if self.n else None,
            "rrms_finite": float(np.sqrt(self.sum_sq / self.n_finite))
            if self.n_finite
            else None,
            "groups": {k: stats(v) for k, v in self.groups.items()},
        }


def pooled(groups: dict, prefix: str, lo: float, hi: float, edges) -> np.ndarray:
    """Sum the histograms of the bins of ``edges`` inside ``[lo, hi)``."""
    h = np.zeros(RB.N_BIN, dtype=np.int64)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if a >= lo and b <= hi + 1e-9:
            h += groups[f"{prefix}[{a:g}, {b:g})"]
    return h


def binned_index(values, edges) -> np.ndarray:
    """Bin index of ``values`` in ``edges`` (half-open, clipped to the ends)."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def psi_keys() -> list[str]:
    """The ψ-bin group names in edge order (JSON sorts keys; this does not)."""
    return [
        f"psi[{a:g}, {b:g})" for a, b in zip(PSI_EDGES[:-1], PSI_EDGES[1:], strict=True)
    ]


def group_partitions(pt: Parts) -> dict:
    """The breakdown partitions of a chunk: all, ψ bins, θv, θs, bb/a bins."""
    n = pt.psi.size
    tv_n, ts_n = np.unique(pt.theta_v), np.unique(pt.theta_s)
    return {
        "all": (np.zeros(n, dtype=int), ["all"]),
        "psi": (
            binned_index(pt.psi, PSI_EDGES),
            [
                f"psi[{a:g}, {b:g})"
                for a, b in zip(PSI_EDGES[:-1], PSI_EDGES[1:], strict=True)
            ],
        ),
        "theta_v": (
            np.searchsorted(tv_n, pt.theta_v),
            [f"theta_v={v:g}" for v in tv_n],
        ),
        "theta_s": (
            np.searchsorted(ts_n, pt.theta_s),
            [f"theta_s={v:g}" for v in ts_n],
        ),
        "bb/a": (
            binned_index(1.0 / pt.a_over_bb, BBA_EDGES),
            [
                f"bb/a[{a:g}, {b:g})"
                for a, b in zip(BBA_EDGES[:-1], BBA_EDGES[1:], strict=True)
            ],
        ),
    }


def d1_partition(pt: Parts) -> dict:
    """One group per (θv_air, Δφ) node."""
    tv_n, ph_n = np.unique(pt.theta_v), np.unique(pt.dphi)
    idx = np.searchsorted(tv_n, pt.theta_v) * ph_n.size + np.searchsorted(ph_n, pt.dphi)
    names = [f"theta_v={tv:g},dphi={ph:g}" for tv in tv_n for ph in ph_n]
    return {"node": (idx, names)}


# -------------------------------------------------------------------- D1 -----
def shape_ratio_error(pred, truth, pt: Parts) -> np.ndarray:
    """``[pred(g)/pred(nadir)] / [truth(g)/truth(nadir)] − 1`` at fixed
    (realisation, θs, λ); nadir is the (θv = 0, Δφ = 0) node of each sun."""
    n_r = pt.realisation.size // pt.n_geom
    ts_n, tv_n = np.unique(pt.theta_s).size, np.unique(pt.theta_v).size
    ph_n = pt.n_geom // (ts_n * tv_n)
    shape = (n_r, ts_n, tv_n, ph_n, pt.wave.size)
    p5, t5 = pred.reshape(shape), truth.reshape(shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p5 / p5[:, :, :1, :1, :]) / (t5 / t5[:, :, :1, :1, :]) - 1.0
    return r.reshape(pred.shape)


def check_window_layout(pt: Parts) -> None:
    """The loader's C-order (θs, θv, Δφ) per realisation, nadir first."""
    ts_n = np.unique(pt.theta_s)
    tv_n = np.unique(pt.theta_v)
    ph_n = np.unique(pt.dphi)
    g = pt.n_geom
    if g != ts_n.size * tv_n.size * ph_n.size:
        raise RuntimeError("the window is not a full (θs, θv, Δφ) grid")
    expect_ts = np.repeat(ts_n, tv_n.size * ph_n.size)
    expect_tv = np.tile(np.repeat(tv_n, ph_n.size), ts_n.size)
    expect_ph = np.tile(ph_n, ts_n.size * tv_n.size)
    n_r = pt.realisation.size // g
    for arr, exp in (
        (pt.theta_s, expect_ts),
        (pt.theta_v, expect_tv),
        (pt.dphi, expect_ph),
    ):
        if not np.array_equal(arr, np.tile(exp, n_r)):
            raise RuntimeError("geometry order is not realisation-major C-order")
    if tv_n[0] != 0.0 or ph_n[0] != 0.0:
        raise RuntimeError("nadir (θv = 0, Δφ = 0) is not the first node")


def o25_table() -> B.O25Table:
    d = np.load(O25_LUT, allow_pickle=False)
    return B.O25Table(
        theta_s=d["theta_s"],
        theta_v=d["theta_v"],
        dphi=d["dphi"],
        G=d["G"],
        provenance=str(d["provenance"]),
    )


def o25_prediction(batch, table) -> np.ndarray:
    return np.asarray(B.Rrs_o25(batch.iops, None, batch.geometry, coeffs=table))


# ------------------------------------------------------------ the passes -----
def chunk_task(args) -> dict:
    """Worker: one chunk's normal-equation blocks, D1 histograms and, when
    ``model`` is given (the held-out pass), the scores of the final variant."""
    ids, cache_dir, lut_path, model = args
    batch = load_window(ids, cache_dir)
    pt = parts_of(batch)
    check_window_layout(pt)
    W = pt.wave.size
    out = {"n_rows_bad": 0}

    # -- the normal equations ----------------------------------------------
    Ds = pt.D_star
    good = np.isfinite(Ds) & (Ds > 0) & (pt.rrs > 0)
    out["n_rows_bad"] = int((~good).sum())
    w = np.where(good, 1.0 / np.where(good, Ds, 1.0), 0.0)
    y = np.where(good, Ds - pt.known, 0.0)
    A = pt.gF[..., None] * legendre_basis(angular_variable(pt.psi), N_F_MAX)[:, None, :]
    Bm = pt.gL[..., None] * fl_basis(pt.psi)[:, None, :]
    bl = Blocks(N_F_MAX, W)
    bl.add(A, Bm, y, w)
    out["blocks"] = bl
    out["D_star_min"] = float(np.min(Ds[good])) if good.any() else np.inf
    out["D_star_q"] = np.quantile(Ds[good], [0.001, 0.5, 0.999]).tolist()
    del A, Bm

    # -- the 8a variant (TT's Ψ_KLu and f_L): shape (D1) and level ---------
    labels = group_partitions(pt)
    F_tt, fl_tt = tt_terms(pt)
    rrs_tt = pt.rrs_of(pt.denominator(F_tt, fl_tt))
    d1 = {}
    d1_labels = d1_partition(pt)
    sc = Score()
    sc.add(shape_ratio_error(rrs_tt, pt.rrs, pt), rrs_tt, d1_labels)
    d1["ZTT 8a (TT angular terms), rrs"] = sc
    if lut_path is not None:
        Rrs_o25 = o25_prediction(batch, o25_table())
        sc = Score()
        sc.add(shape_ratio_error(Rrs_o25, pt.Rrs, pt), Rrs_o25, d1_labels)
        d1["O25 refit (task 6 LUT), Rrs"] = sc
    scores = {}
    sc = Score()
    sc.add(rrs_tt / pt.rrs - 1.0, rrs_tt, labels)
    scores["ZTT 8a (TT angular terms)"] = sc

    # -- the final variant (held-out pass) ---------------------------------
    if model is not None:
        F_pb, fl_pb = pb24_terms(pt, model["coeffs"], model["FL"], model["s"])
        D_raw = pt.denominator(F_pb, fl_pb)
        D_fl = floored(D_raw, *model["floor"])
        rrs_raw, rrs_fl = pt.rrs_of(D_raw), pt.rrs_of(D_fl)
        with np.errstate(divide="ignore", invalid="ignore"):
            moved = np.abs(D_fl / D_raw - 1.0) > FOOTPRINT_TOL
        out["footprint_moved"] = int(np.sum(moved | ~np.isfinite(D_raw)))
        out["footprint_n"] = int(D_raw.size)
        out["D_model_min"] = float(np.min(D_raw))
        sc = Score()
        sc.add(rrs_fl / pt.rrs - 1.0, rrs_fl, labels)
        scores["ZTT (PB24), floored"] = sc
        sc = Score()
        sc.add(rrs_raw / pt.rrs - 1.0, rrs_raw, labels)
        scores["ZTT (PB24), no floor"] = sc
        sc = Score()
        sc.add(shape_ratio_error(rrs_fl, pt.rrs, pt), rrs_fl, d1_labels)
        d1["ZTT (PB24), rrs"] = sc
    out["d1"], out["scores"] = d1, scores
    return out


def run_pass(ids, cache_dir, lut_path, model, workers: int, label: str, progress):
    """Stream the chunks of ``ids`` through :func:`chunk_task`, summing in
    chunk order (bit-identical for any ``workers``)."""
    tasks = [(c, cache_dir, lut_path, model) for c in chunks_of(ids)]
    total = {"blocks": None, "d1": {}, "scores": {}, "n_rows_bad": 0}
    total.update(
        {
            "D_star_min": np.inf,
            "footprint_moved": 0,
            "footprint_n": 0,
            "D_model_min": np.inf,
            "D_star_q": [],
        }
    )
    t0 = time.time()
    with RB.pool(workers) as ex:
        results = ex.map(chunk_task, tasks) if ex else map(chunk_task, tasks)
        for k, r in enumerate(results, 1):
            if total["blocks"] is None:
                total["blocks"] = r["blocks"]
            else:
                total["blocks"].merge(r["blocks"])
            for key in ("d1", "scores"):
                for name, sc in r[key].items():
                    if name not in total[key]:
                        total[key][name] = Score()
                    total[key][name].merge(sc)
            total["n_rows_bad"] += r["n_rows_bad"]
            total["D_star_min"] = min(total["D_star_min"], r["D_star_min"])
            total["D_star_q"].append(r["D_star_q"])
            if model is not None:
                total["footprint_moved"] += r["footprint_moved"]
                total["footprint_n"] += r["footprint_n"]
                total["D_model_min"] = min(total["D_model_min"], r["D_model_min"])
            if progress and (k % 20 == 0 or k == len(tasks)):
                print(
                    f"  {label}: chunk {k}/{len(tasks)} ({time.time() - t0:.0f} s)",
                    flush=True,
                )
    total["D_star_q"] = np.median(np.asarray(total["D_star_q"]), axis=0).tolist()
    return total


# -------------------------------------------------------------------- D2 -----
IOP_NAMES = ("aw", "aph", "ag", "aNAP", "bph", "bNAP", "bbw", "bbph", "bbNAP")


def nadir_tables(cache, ids, band_idx):
    """Per-(realisation, θs, λ) nadir quantities from the cache tables: the
    closure pieces with 8a's µd, PB24's own µd and KLu/a, and PB24's rrs."""
    import jax.numpy as jnp

    tab = {n: np.asarray(cache.table(n, ids), dtype=np.float64) for n in IOP_NAMES}
    for n in ("KLu", "mu_d", "Esdir_Es_ratio"):
        tab[n] = np.asarray(cache.table(n, ids), dtype=np.float64)
    theta_s = np.asarray(cache.grids["theta_s"], dtype=np.float64)
    wave_all = np.asarray(cache.grids["lambda"], dtype=np.float64)
    rrs = np.asarray(cache.four_d("rrs", ids, theta=0, phi=0), dtype=np.float64)
    bands = np.asarray(band_idx, dtype=int)
    wave = wave_all[bands]
    a = (tab["aw"] + tab["aph"] + tab["ag"] + tab["aNAP"])[:, bands]
    bbp = (tab["bbph"] + tab["bbNAP"])[:, bands]
    bbw = tab["bbw"][:, bands]
    bb = bbw + bbp
    B_p = bbp / (tab["bph"] + tab["bNAP"])[:, bands]
    win = theta_s <= ANGLE_MAX_DEG
    ts = theta_s[win]
    n, nts, W = ids.size, ts.size, wave.size
    psi = 180.0 - np.asarray(Z.in_water_zenith(ts), dtype=np.float64)  # (nts,)
    # The FF Pbb at nadir, as the loader builds the mixture.
    raw = {k: tab[k][:, bands] for k in IOP_NAMES}
    ff_B, ff_w = P._ff_mixture(raw)
    from robust.rt.types import IOPs, PhaseParams

    def rep(x):
        return jnp.asarray(np.repeat(np.asarray(x, dtype=np.float64), nts, axis=0))

    iops = IOPs(a=rep(a), bb_w=rep(bbw), bb_p=rep(bbp))
    phase = PhaseParams(B_p=rep(B_p), ff_B=rep(ff_B), ff_weight=rep(ff_w))
    psi_b = jnp.asarray(np.tile(psi, n))[:, None]
    num = np.asarray(
        Z.backward_phase_over_bb(
            iops, Z.P_bb_from_phase(phase, psi_b, use_ff=True), psi_b
        )
    ).reshape(n, nts, W)
    H = (1.0 - tab["Esdir_Es_ratio"])[:, win][..., bands]
    mu_d_8a = np.asarray(
        Z.mu_d_pb24(
            jnp.asarray(ts)[None, :, None],
            jnp.asarray(bb / a)[:, None, :],
            jnp.asarray(bbw / bb)[:, None, :],
            H=jnp.asarray(H),
        )
    )
    mu_inf_8a = np.asarray(
        Z.mu_infinity_pb24(jnp.asarray(bb / a), jnp.asarray(bbw / bb))
    )
    b_tilde = np.asarray(
        Z.bb_tilde(
            IOPs(a=jnp.asarray(a), bb_w=jnp.asarray(bbw), bb_p=jnp.asarray(bbp)),
            jnp.asarray(B_p),
        )
    )
    return {
        "theta_s": ts,
        "psi": psi,
        "wave": wave,
        "rrs": rrs[:, win][..., bands]
        if rrs.shape[-1] == wave_all.size
        else rrs[:, win],
        "num": num,
        "mu_d_8a": mu_d_8a,
        "mu_d_pb24": tab["mu_d"][:, win][..., bands],
        "mu_inf_8a": mu_inf_8a,  # (n, W)
        "a_over_bb": (a / bb),  # (n, W)
        "inv_bt": 1.0 / b_tilde,  # (n, W)
        "klu_over_a": (tab["KLu"][:, win][..., bands] / a[:, None, :]),  # (n, nts, W)
    }


def nadir_ceiling(train, test) -> dict:
    """D2: per-band f_L(λ) with Ψ_KLu/µ∞ := KLu/a at nadir, fitted on the
    training rows (relative weighting in D), scored held out; with 8a's µd
    (primary) and PB24's own µd."""
    out = {}
    for mud_name in ("mu_d_8a", "mu_d_pb24"):

        def closure(T, mud_name=mud_name):
            with np.errstate(divide="ignore", invalid="ignore"):
                Ds = T["num"] / (T[mud_name] * T["rrs"])
            known = (
                T["a_over_bb"][:, None, :] * (1.0 + T["klu_over_a"])
                + T["inv_bt"][:, None, :]
            )
            gL = -(T["inv_bt"] - 1.0)[:, None, :]
            good = np.isfinite(Ds) & (Ds > 0) & (T["klu_over_a"] > 0) & (T["rrs"] > 0)
            return Ds, known, np.broadcast_to(gL, Ds.shape), good

        Ds, known, gL, good = closure(train)
        w2 = np.where(good, 1.0 / Ds**2, 0.0)
        y = np.where(good, Ds - known, 0.0)
        fl = np.sum(w2 * gL * y, axis=(0, 1)) / np.sum(w2 * gL * gL, axis=(0, 1))
        Ds_t, known_t, gL_t, good_t = closure(test)
        D_model = known_t + gL_t * fl[None, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = Ds_t / D_model - 1.0  # rrs_model / rrs_PB24 − 1
        e = np.where(good_t, e, np.nan)
        per_ts = {}
        for k, t in enumerate(train["theta_s"]):
            v = e[:, k, :]
            per_ts[f"{t:g}"] = {
                "median_abs": float(np.nanmedian(np.abs(v))),
                "q90_abs": float(np.nanpercentile(np.abs(v), 90)),
                "nonphysical_fraction": float(np.mean(~(D_model[:, k, :] > 0))),
            }
        out[mud_name] = {
            "fl_per_band_range": [float(fl.min()), float(fl.max())],
            "fl_per_band_mean": float(fl.mean()),
            "n_train_rows": int(good.sum()),
            "n_test_rows": int(good_t.sum()),
            "heldout_median_abs": float(np.nanmedian(np.abs(e))),
            "heldout_q90_abs": float(np.nanpercentile(np.abs(e), 90)),
            "heldout_nonphysical_fraction": float(np.mean(~(D_model > 0))),
            "per_theta_s": per_ts,
        }
        out[mud_name]["fl_per_band"] = [float(v) for v in fl]
    return out


def nadir_psi_check(test, coeffs) -> dict:
    """The one physical check: the refit Ψ_KLu(180° − θs_w)/µ∞ against PB24's
    KLu/a at nadir, held out, per sun and pooled."""
    import jax.numpy as jnp

    Psi = np.asarray(Z.psi_KLu_pb24(jnp.asarray(test["psi"]), coeffs))  # (nts,)
    Psi_tt = np.asarray(Z.psi_KLu(jnp.asarray(test["psi"])))
    ratio = (Psi[None, :, None] / test["mu_inf_8a"][:, None, :]) / test["klu_over_a"]
    ratio_tt = (Psi_tt[None, :, None] / test["mu_inf_8a"][:, None, :]) / test[
        "klu_over_a"
    ]
    good = test["klu_over_a"] > 0

    def stats(r):
        r = np.where(good, r, np.nan)
        per = {}
        for k, t in enumerate(test["theta_s"]):
            v = r[:, k, :]
            per[f"{t:g}"] = {
                "median_ratio": float(np.nanmedian(v)),
                "median_abs_dev": float(np.nanmedian(np.abs(v - 1.0))),
                "q16_q84": [
                    float(np.nanpercentile(v, 16)),
                    float(np.nanpercentile(v, 84)),
                ],
            }
        return {
            "pooled_median_ratio": float(np.nanmedian(r)),
            "pooled_median_abs_dev": float(np.nanmedian(np.abs(r - 1.0))),
            "per_theta_s": per,
        }

    return {"PB24 refit": stats(ratio), "TT": stats(ratio_tt)}


def psi_shape_check(coeffs) -> dict:
    """Ψ_KLu of the refit on :data:`PSI_GRID`: range and monotonicity."""
    import jax.numpy as jnp

    Psi = np.asarray(Z.psi_KLu_pb24(jnp.asarray(PSI_GRID), coeffs))
    d = np.diff(Psi)
    return {
        "psi_min_deg": float(PSI_GRID[0]),
        "psi_max_deg": float(PSI_GRID[-1]),
        "Psi_min": float(Psi.min()),
        "Psi_max": float(Psi.max()),
        "Psi_at_180": float(Psi[-1]),
        "Psi_at_91": float(Psi[0]),
        "monotone": bool(np.all(d <= 0) or np.all(d >= 0)),
        "within_predicted_range": bool(
            Psi.min() >= PSI_RANGE_PREDICTED[0] and Psi.max() <= PSI_RANGE_PREDICTED[1]
        ),
        "F_zero_crossing_deg": [
            float(PSI_GRID[k]) for k in np.flatnonzero(np.diff(np.sign(Psi - 1.0)) != 0)
        ][:4],
    }


# --------------------------------------------------------------- probes -----
def probes(cache_dir, all_ids, model) -> list[dict]:
    """Seeded window samples with every input and the model's pieces, through
    the shipped evaluators with the fitted numbers."""
    rng = np.random.default_rng(PROBE_SEED)
    n = min(N_PROBES, all_ids.size)
    ids = np.sort(rng.choice(all_ids, n, replace=False))
    batch = load_window(ids, cache_dir)
    pt = parts_of(batch)
    rows = rng.integers(0, pt.realisation.size, n)
    bands = rng.integers(0, pt.wave.size, n)
    F_pb, fl_pb = pb24_terms(pt, model["coeffs"], model["FL"], model["s"])
    D_raw = pt.denominator(F_pb, fl_pb)
    D_fl = floored(D_raw, *model["floor"])
    rrs = pt.rrs_of(D_fl)
    phase, geom = batch.phase_params, batch.geometry
    out = []
    for r, w in zip(rows, bands, strict=True):
        r, w = int(r), int(w)
        out.append(
            {
                "realisation": int(pt.realisation[r]),
                "theta_s": float(pt.theta_s[r]),
                "theta_v": float(pt.theta_v[r]),
                "dphi": float(pt.dphi[r]),
                "wave_nm": float(pt.wave[w]),
                "a": float(batch.iops.a[r, w]),
                "bb_w": float(batch.iops.bb_w[r, w]),
                "bb_p": float(batch.iops.bb_p[r, w]),
                "B_p": float(phase.B_p[r, w]),
                "ff_B": [float(v) for v in np.asarray(phase.ff_B[r, w])],
                "ff_weight": [float(v) for v in np.asarray(phase.ff_weight[r, w])],
                "H": float(geom.diffuse_fraction[r, w]),
                "psi": float(pt.psi[r]),
                "F_pb24": float(F_pb[r, 0]),
                "psi_KLu_pb24": float(1.0 + F_pb[r, 0]),
                "f_L_pb24": float(fl_pb[r, w]),
                "D_no_floor": float(D_raw[r, w]),
                "D_floored": float(D_fl[r, w]),
                "rrs_pb24_variant": float(rrs[r, w]),
                "rrs_PB24": float(pt.rrs[r, w]),
            }
        )
    return out


# ------------------------------------------------------------- outputs -----
def q_pct(v):
    """A percentage cell: ``–`` for none/empty, ``inf`` for an overflow median."""
    if v is None or np.isnan(v):
        return "–"
    return "inf" if np.isinf(v) else f"{100 * v:.2f}"


def report(coefficients, metrics, *, quick: bool) -> str:
    c, m, f = coefficients, metrics, coefficients["fit"]
    L = [
        "# ZTT fitted to PB24 — Ψ_KLu and f_L, jointly (M8 task 8b)",
        "",
        "*Generated by `design/py/fit_ztt_pb24_angular.py`"
        f"{' --quick' if quick else ''}. Do not edit by hand.*",
        "",
        f"Window (θs, θv_air ≤ {ANGLE_MAX_DEG:g}°) of the realisation split: fitted on "
        f"**{f['n_train_realisations']} training realisations** (digest "
        f"`{f['train_digest']}`), scored on its {f['n_test_realisations']} held-out "
        f"ones; {f['n_wave']} bands. Errors are relative, in %; medians and q90 "
        "from the baseline's log10|e| histogram (±half a 0.002 dex bin), rRMS "
        "and counts exact.",
        "",
        "**Predictions (written before the fit, `rt_fit_pb24.md` task 8b):** D1 — "
        "ZTT's shape error alone exceeds 5% (median) at θv_air ≥ 40° and is largest "
        "toward Δφ = 0; O25's stays below 2% everywhere in the window. D2 — ≤ 2.5% "
        "median held out at nadir; above 4% means Equation (12)'s structure "
        "itself limits. Joint refit — window held-out median 4–6%, non-physical "
        "< 1% of values, ψ ≥ 155° ≤ 3%, ψ 110.4–134° < 10%; Ψ_KLu monotone in ψ "
        "and within [0.8, 2.5] over 91–180°; at nadir Ψ_KLu/µ∞ within 5% (median) "
        "of PB24's KLu/a; the floor moves under 1% of window values.",
        "",
        "## D1 — the BRDF-shape test (held out)",
        "",
        "Median |[rrs(θv, Δφ)/rrs(nadir)]_model / [same]_PB24 − 1| in %, per θv_air "
        "(rows) and Δφ (columns); O25 in its own quantity, Rrs.",
        "",
    ]
    d1 = m["D1_shape_heldout"]
    tvs, phs = m["D1_axes"]["theta_v"], m["D1_axes"]["dphi"]
    for name in D1_ORDER:
        if name not in d1:
            continue
        g = d1[name]
        L += [
            f"**{name}**",
            "",
            "| θv_air \\ Δφ | " + " | ".join(f"{p:g}" for p in phs) + " | pooled |",
            "|---|" + "---|" * (len(phs) + 1),
        ]
        for tv in tvs:
            cells = [
                q_pct(g["groups"][f"theta_v={tv:g},dphi={p:g}"]["median_abs"])
                for p in phs
            ]
            L.append(
                f"| {tv:g} | "
                + " | ".join(cells)
                + f" | {q_pct(g['pooled_by_theta_v'][f'{tv:g}'])} |"
            )
        L.append("")
    d2 = m["D2_nadir_ceiling"]
    L += [
        "## D2 — the nadir ceiling (Ψ_KLu/µ∞ := KLu/a, per-band f_L(λ) refit)",
        "",
        "| µd | f_L(λ) range | held-out median | q90 | non-physical | "
        + " | ".join(f"{t}°" for t in d2["mu_d_8a"]["per_theta_s"])
        + " |",
        "|---|---|---|---|---|" + "---|" * len(d2["mu_d_8a"]["per_theta_s"]),
    ]
    for key, label in (
        ("mu_d_8a", "8a's µd (the variant's)"),
        ("mu_d_pb24", "PB24's own µd"),
    ):
        v = d2[key]
        L.append(
            f"| {label} | {v['fl_per_band_range'][0]:.3f}–"
            f"{v['fl_per_band_range'][1]:.3f} "
            f"| {q_pct(v['heldout_median_abs'])} | {q_pct(v['heldout_q90_abs'])} "
            f"| {q_pct(v['heldout_nonphysical_fraction'])} | "
            + " | ".join(q_pct(t["median_abs"]) for t in v["per_theta_s"].values())
            + " |"
        )
    j = m["joint_fit"]
    L += [
        "",
        "## The joint refit, linear in D-space",
        "",
        f"- Rows: {f['n_rows_train']:,} training (window × bands; "
        f"{f['n_rows_bad_train']} dropped as non-finite), {f['n_rows_test']:,} held "
        f"out. D* on the training window: min {f['D_star_min_train']:.3f}, median "
        f"{f['D_star_median_train']:.1f}.",
        f"- F(ψ): {c['n_f_terms']} terms in u = 1 + cos ψ (Legendre-accumulated, "
        f"converted); f_L = FL_AVE(λ) · (s0 + s1 sin ψ + s2 cos ψ), FL_AVE per band "
        f"normalised to mean 1; {N_ROUNDS} alternating rounds, last round's largest "
        f"relative change {j['history'][-1]['max_rel_change']:.1e}; reduced-system "
        f"condition numbers {j['history'][-1]['cond_pass_a']:.2g} (pass A), "
        f"{j['history'][-1]['cond_pass_b']:.2g} (pass B).",
        f"- s = ({', '.join(f'{v:.5f}' for v in c['fl_s_s0_s1_s2'])}); TT: "
        f"({Z.FL_OFFSET}, {Z.FL_SIN_COEFF}, 0). FL_AVE(λ) range "
        f"{min(c['fl_ave_per_band']):.4f}–{max(c['fl_ave_per_band']):.4f} "
        f"(TT's Table A3: 0.981–1.026).",
        "",
        "Order sensitivity (same statistics, same rounds; D-space rRMS, %):",
        "",
        "| F terms | train | held out |",
        "|---|---|---|",
    ]
    for k, v in j["order_sensitivity"].items():
        mark = " **(shipped)**" if int(k) == c["n_f_terms"] else ""
        L.append(
            f"| {k}{mark} | {100 * v['rrms_train']:.3f} | {100 * v['rrms_test']:.3f} |"
        )
    sh = m["psi_shape"]
    nc = m["nadir_psi_check"]
    L += [
        "",
        f"- Ψ_KLu over 91–180°: {sh['Psi_min']:.3f}–{sh['Psi_max']:.3f} "
        f"(at 180°: {sh['Psi_at_180']:.3f}; at 91°: {sh['Psi_at_91']:.3f}); monotone: "
        f"{sh['monotone']}; within [0.8, 2.5]: {sh['within_predicted_range']}. "
        f"TT's: 1.024 at 180°, 0.80 at 134°, crossing 1 at 110.4°.",
        f"- Nadir, held out: Ψ_KLu/µ∞ vs KLu/a, median ratio "
        f"{nc['PB24 refit']['pooled_median_ratio']:.4f} (median |dev| "
        f"{q_pct(nc['PB24 refit']['pooled_median_abs_dev'])}%); TT's pair with 8a's "
        f"µ∞: {nc['TT']['pooled_median_ratio']:.4f} "
        f"({q_pct(nc['TT']['pooled_median_abs_dev'])}%).",
        "",
        "| θs | refit Ψ/µ∞ ÷ KLu/a, median | 16–84% | TT, median |",
        "|---|---|---|---|",
    ]
    for t, v in nc["PB24 refit"]["per_theta_s"].items():
        L.append(
            f"| {t}° | {v['median_ratio']:.4f} | {v['q16_q84'][0]:.3f}–"
            f"{v['q16_q84'][1]:.3f} | "
            f"{nc['TT']['per_theta_s'][t]['median_ratio']:.4f} |"
        )
    L += [
        "",
        "## Held out, in rrs (window)",
        "",
        f"- The floor (lo = {c['d_floor_lo_width'][0]}, width = "
        f"{c['d_floor_lo_width'][1]}): "
        f"{100 * m['floor_footprint']['fraction_moved_gt_tol']:.4f}% of window values "
        f"moved by more than {FOOTPRINT_TOL:g} (prediction: under 1%); the model's "
        f"minimum unfloored D is {m['floor_footprint']['D_model_min']:.3g}.",
        "",
        "| model | median | q90 | rRMS (finite) | non-physical |",
        "|---|---|---|---|---|",
    ]
    hs = m["heldout_rrs"]
    for name in MODEL_ORDER:
        v = hs[name]
        g = v["groups"]["all"]
        L.append(
            f"| {name} | {q_pct(g['median_abs'])} | {q_pct(g['q90_abs'])} | "
            f"{q_pct(v['rrms_finite'])} | {q_pct(v['nonphysical_fraction'])} |"
        )
    final = "ZTT (PB24), floored"
    ref = "ZTT 8a (TT angular terms)"

    def row(name, key):
        return q_pct(hs[name]["groups"][key]["median_abs"])

    L += [
        "",
        "By ψ (median |err| %; pooled ranges of the predictions last):",
        "",
        f"| ψ | {final} | {ref} |",
        "|---|---|---|",
    ]
    for key in psi_keys() + list(PSI_POOLS):
        L.append(f"| {key} | {row(final, key)} | {row(ref, key)} |")
    for prefix, title in (
        ("theta_v=", "θv_air"),
        ("theta_s=", "θs"),
        ("bb/a[", "bb/a"),
    ):
        L += ["", f"By {title}:", "", f"| {title} | {final} | {ref} |", "|---|---|---|"]
        for key in [k for k in hs[final]["groups"] if k.startswith(prefix)]:
            L.append(f"| {key} | {row(final, key)} | {row(ref, key)} |")
    L += [
        "",
        "## Gate",
        "",
        "- `--check` refits and compares every file byte for byte; the tests "
        're-evaluate `angular_probes.json` through `rrs_ZTT(variant="pb24")` '
        "(no data), check the closure D* round trip, the floor's invisibility on "
        "the probes, and the block normal equations against a direct `lstsq` on "
        "the fixture.",
        "",
    ]
    return "\n".join(L)


def outputs(coefficients, metrics, probes_, *, quick: bool) -> dict[str, str]:
    dump = {"indent": 2, "sort_keys": True}
    return {
        f"{PREFIX}coefficients.json": json.dumps(coefficients, **dump) + "\n",
        f"{PREFIX}metrics.json": json.dumps(metrics, **dump) + "\n",
        f"{PREFIX}probes.json": json.dumps(probes_, **dump) + "\n",
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


def figures(coefficients, metrics, out: Path) -> None:
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, m = coefficients, metrics
    # (1) the angular terms: Ψ_KLu(ψ) and s(ψ), refit vs TT; FL_AVE(λ) vs Table A3.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    psi = np.linspace(91.0, 180.0, 300)
    axes[0].plot(
        psi,
        np.asarray(
            Z.psi_KLu_pb24(jnp.asarray(psi), tuple(c["psi_klu_coeffs_u_highest_first"]))
        ),
        label="PB24 refit",
    )
    axes[0].plot(
        psi, np.asarray(Z.psi_KLu(jnp.asarray(psi))), "--", label="TT Eq. (4), (6)"
    )
    nc = m["nadir_psi_check"]["PB24 refit"]["per_theta_s"]
    ts = np.array([float(t) for t in nc])
    psi_n = 180.0 - np.asarray(Z.in_water_zenith(ts))
    Psi_n = np.asarray(
        Z.psi_KLu_pb24(jnp.asarray(psi_n), tuple(c["psi_klu_coeffs_u_highest_first"]))
    )
    axes[0].plot(
        psi_n,
        Psi_n / np.array([nc[t]["median_ratio"] for t in nc]),
        "k.",
        label="nadir: median KLu/a · µ∞ (held out)",
    )
    axes[0].axvline(134.0, color="grey", ls=":")
    sh = m["psi_shape"]
    axes[0].set_ylim(min(0.5, sh["Psi_min"] - 0.1), max(2.6, sh["Psi_max"] + 0.1))
    axes[0].set_xlabel("ψ (deg)")
    axes[0].set_ylabel("Ψ_KLu")
    axes[0].set_title(
        "Ψ_KLu: physical at nadir, a calibration elsewhere", loc="left", fontsize=9
    )
    axes[0].legend(fontsize=7)
    s = c["fl_s_s0_s1_s2"]
    rad = np.deg2rad(psi)
    axes[1].plot(
        psi, s[0] + s[1] * np.sin(rad) + s[2] * np.cos(rad), label="PB24 refit s(ψ)"
    )
    axes[1].plot(
        psi, Z.FL_OFFSET + Z.FL_SIN_COEFF * np.sin(rad), "--", label="TT Eq. (31)"
    )
    axes[1].set_xlabel("ψ (deg)")
    axes[1].set_ylabel("s(ψ) (× FL_AVE, mean 1)")
    axes[1].set_title("f_L's ψ scaling", loc="left", fontsize=9)
    axes[1].legend(fontsize=7)
    wave = np.asarray(c["fl_ave_wave_nm"])
    axes[2].plot(wave, c["fl_ave_per_band"], label="PB24 refit (mean 1)")
    axes[2].plot(
        Z.FL_AVE_WAVE, Z.FL_AVE / Z.FL_AVE.mean(), "--", label="TT Table A3 (mean 1)"
    )
    axes[2].set_xlabel("λ (nm)")
    axes[2].set_ylabel("FL_AVE(λ)")
    axes[2].set_title("f_L's spectral shape", loc="left", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    save_png(fig, out / f"{PREFIX}fig_terms.png")

    # (2) D1: shape error maps.
    d1 = m["D1_shape_heldout"]
    tvs, phs = m["D1_axes"]["theta_v"], m["D1_axes"]["dphi"]
    names = [n for n in D1_ORDER if n in d1]
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.6 * len(names), 4), squeeze=False
    )
    for ax, name in zip(axes[0], names, strict=True):
        g = d1[name]
        grid = np.array(
            [
                [
                    100 * g["groups"][f"theta_v={tv:g},dphi={p:g}"]["median_abs"]
                    for p in phs
                ]
                for tv in tvs
            ]
        )
        im = ax.imshow(
            grid, origin="lower", aspect="auto", vmin=0, vmax=20, cmap="viridis"
        )
        ax.set_xticks(range(len(phs)))
        ax.set_xticklabels([f"{p:g}" for p in phs], fontsize=7, rotation=45)
        ax.set_yticks(range(len(tvs)))
        ax.set_yticklabels([f"{t:g}" for t in tvs], fontsize=7)
        ax.set_xlabel("Δφ (deg)")
        ax.set_ylabel("θv_air (deg)")
        ax.set_title(name, loc="left", fontsize=8)
        fig.colorbar(im, ax=ax, label="median |shape err| (%)")
    fig.tight_layout()
    save_png(fig, out / f"{PREFIX}fig_d1_shape.png")

    # (3) held out in rrs: by ψ bin and by θv, final vs 8a.
    hs = m["heldout_rrs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("ZTT (PB24), floored", "ZTT 8a (TT angular terms)"):
        for ax, prefix in zip(axes, ("psi[", "theta_v="), strict=True):
            keys = (
                psi_keys()
                if prefix == "psi["
                else sorted(
                    (k for k in hs[name]["groups"] if k.startswith(prefix)),
                    key=lambda k: float(k.split("=")[1]),
                )
            )
            vals = [100 * hs[name]["groups"][k]["median_abs"] for k in keys]
            ax.plot(range(len(keys)), vals, "o-", label=name)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(keys, rotation=30, fontsize=7)
    axes[0].set_ylabel("median |rrs err| (%), held out")
    axes[0].set_title("by ψ bin", loc="left", fontsize=9)
    axes[1].set_title("by θv_air", loc="left", fontsize=9)
    axes[0].axhline(7.84, color="grey", ls=":", label="Gordon, task 6 window")
    axes[0].legend(fontsize=7)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.tight_layout()
    save_png(fig, out / f"{PREFIX}fig_heldout.png")


# ------------------------------------------------------------------ main -----
def run(*, quick: bool, cache_dir=None, workers: int, progress=True):
    train_ids, test_ids = split_ids(quick)
    cache = PC.open_cache(cache_dir)
    all_ids = np.asarray(cache.realisations, dtype=int)
    if not np.isin(np.concatenate([train_ids, test_ids]), all_ids).all():
        train_ids = np.intersect1d(train_ids, all_ids)
        test_ids = np.intersect1d(test_ids, all_ids)
        print(f"cache holds {all_ids.size} realisations; split restricted to them")
    digest = C.realisation_digest(train_ids)
    band_idx = np.asarray(cache.four_d_bands, dtype=int)
    wave = np.asarray(cache.grids["lambda"], dtype=np.float64)[band_idx]
    lut = O25_LUT if O25_LUT.is_file() else None
    print(
        f"realisation split, window: fitting on {train_ids.size} training "
        f"realisations (digest {digest[:16]}), scoring on {test_ids.size} held-out; "
        f"{wave.size} bands; {workers} workers",
        flush=True,
    )

    # -- the training pass: sufficient statistics + D1 of the fixed models ----
    tr = run_pass(train_ids, cache_dir, lut, None, workers, "train", progress)
    bl_train: Blocks = tr["blocks"]
    if not tr["D_star_min"] > 2.0 * D_FLOOR[0]:
        raise RuntimeError(f"D* minimum {tr['D_star_min']} is not above 2 × floor")

    # -- the alternating solve, every order ----------------------------------
    fl_init = np.interp(wave, Z.FL_AVE_WAVE, Z.FL_AVE)
    fits = {n: solve_alternating(bl_train.truncated(n), fl_init) for n in F_TERMS_TRIED}
    chosen = N_F_TERMS
    fit = fits[chosen]
    coeffs_u = legendre_to_u_poly(fit["p_legendre"])
    FL = np.round(fit["FL"], FL_DECIMALS)
    s = tuple(float(v) for v in fit["s"])
    model = {"coeffs": coeffs_u, "FL": FL, "s": s, "floor": D_FLOOR}

    # -- the held-out pass -----------------------------------------------------
    te = run_pass(test_ids, cache_dir, lut, model, workers, "test", progress)
    bl_test: Blocks = te["blocks"]
    order = {}
    for n, ft in fits.items():
        order[str(n)] = {
            "rrms_train": bl_train.truncated(n).rrms(
                ft["p_legendre"], ft["s"], ft["FL"]
            ),
            "rrms_test": bl_test.truncated(n).rrms(ft["p_legendre"], ft["s"], ft["FL"]),
            "coeffs_u_highest_first": list(legendre_to_u_poly(ft["p_legendre"])),
            "s": [float(v) for v in ft["s"]],
        }
    # The rule written before the numbers: five unless it overfits (held-out
    # residual above four's), then four. Recorded, and acted on only by rerun.
    overfit = (
        order[str(N_F_TERMS)]["rrms_test"] > order[str(N_F_TERMS - 1)]["rrms_test"]
    )

    # -- D2, the nadir checks, the shape check --------------------------------
    T_train = nadir_tables(cache, train_ids, band_idx)
    T_test = nadir_tables(cache, test_ids, band_idx)
    d2 = nadir_ceiling(T_train, T_test)
    nadir = nadir_psi_check(T_test, coeffs_u)
    shape = psi_shape_check(coeffs_u)

    # -- metrics -------------------------------------------------------------
    tvs = sorted(
        {
            float(k.split("=")[1].split(",")[0])
            for k in tr["d1"]["ZTT 8a (TT angular terms), rrs"].groups
        }
    )
    phs = sorted(
        {
            float(k.split("dphi=")[1])
            for k in tr["d1"]["ZTT 8a (TT angular terms), rrs"].groups
        }
    )

    def d1_with_pools(d):
        out = {}
        for name, sc in d.items():
            summ = sc.summary()
            summ["pooled_by_theta_v"] = {}
            for tv in tvs:
                h = np.zeros(RB.N_BIN, dtype=np.int64)
                for p in phs:
                    h += sc.groups[f"theta_v={tv:g},dphi={p:g}"]
                summ["pooled_by_theta_v"][f"{tv:g}"] = RB.hist_quantile(h, 0.5)
            out[name] = summ
        return out

    heldout = {}
    for name, sc in te["scores"].items():
        summ = sc.summary()
        for pname, (lo, hi) in PSI_POOLS.items():
            h = pooled(sc.groups, "psi", lo, hi, PSI_EDGES)
            summ["groups"][pname] = {
                "n": int(h.sum()),
                "median_abs": RB.hist_quantile(h, 0.5),
                "q90_abs": RB.hist_quantile(h, 0.9),
            }
        heldout[name] = summ

    metrics = {
        "D1_axes": {"theta_v": tvs, "dphi": phs},
        "D1_shape_heldout": d1_with_pools(te["d1"]),
        "D1_shape_train": d1_with_pools(tr["d1"]),
        "D2_nadir_ceiling": d2,
        "joint_fit": {
            "history": fit["history"],
            "order_sensitivity": order,
            "chosen_n_f_terms": chosen,
            "overfit_rule_triggered": bool(overfit),
            "p_legendre": [float(v) for v in fit["p_legendre"]],
        },
        "psi_shape": shape,
        "nadir_psi_check": nadir,
        "floor_footprint": {
            "fraction_moved_gt_tol": te["footprint_moved"] / te["footprint_n"],
            "n_moved": te["footprint_moved"],
            "n": te["footprint_n"],
            "D_model_min": te["D_model_min"],
            "tol": FOOTPRINT_TOL,
        },
        "heldout_rrs": heldout,
        "train_rrs_8a_variant": {k: v.summary() for k, v in tr["scores"].items()},
    }
    coefficients = {
        "about": "ZTT fitted to PB24, the angular terms (M8 task 8b, rt_fit_pb24.md "
        "Q23-Q25, Q30). Psi_KLu = 1 + F(psi), F a polynomial in u = 1 + cos(psi) "
        "(highest order first); f_L = FL_AVE(lambda) * (s0 + s1 sin psi + s2 cos psi), "
        "FL_AVE per band on the 1 nm grid, mean 1, rounded; D floored. Fitted "
        "linearly in D-space on the window with 8a's mu_d, mu_inf, H and the FF Pbb. "
        "Off-nadir Psi_KLu is a calibration through rrs, physical only at nadir.",
        "n_f_terms": chosen,
        "psi_klu_coeffs_u_highest_first": list(coeffs_u),
        "fl_ave_wave_nm": [float(v) for v in wave],
        "fl_ave_per_band": [float(v) for v in FL],
        "fl_ave_decimals": FL_DECIMALS,
        "fl_s_s0_s1_s2": list(s),
        "d_floor_lo_width": [float(v) for v in D_FLOOR],
        "fit": {
            "split": (
                f"realisations, SPLIT_SEED={P.SPLIT_SEED}, "
                f"test_fraction={P.TEST_FRACTION}"
            ),
            "quick": quick,
            "window_max_deg": float(ANGLE_MAX_DEG),
            "n_train_realisations": int(train_ids.size),
            "n_test_realisations": int(test_ids.size),
            "n_all_realisations": int(all_ids.size),
            "train_digest": digest,
            "n_wave": int(wave.size),
            "chunk": CHUNK,
            "n_rounds": N_ROUNDS,
            "f_terms_tried": list(F_TERMS_TRIED),
            "n_rows_train": int(bl_train.n_rows),
            "n_rows_test": int(bl_test.n_rows),
            "n_rows_bad_train": int(tr["n_rows_bad"]),
            "n_rows_bad_test": int(te["n_rows_bad"]),
            "D_star_min_train": float(tr["D_star_min"]),
            "D_star_median_train": float(tr["D_star_q"][1]),
            "o25_lut": None if lut is None else lut.name,
            "weighting": "1/D* per row (relative in D)",
        },
    }
    pr = probes(cache_dir, all_ids, model)
    return coefficients, metrics, pr


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", default=None, help="hyperspectral cache")
    parser.add_argument("--quick", action="store_true", help=f"{QUICK} realisations")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument(
        "--check", action="store_true", help="refit and compare byte for byte"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--allow-more-workers", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.quick and args.out_dir == OUT and not args.dry_run:
        parser.error("--quick fits on a subsample; pass --out-dir or --dry-run")
    if args.quick and args.check:
        parser.error("--check compares the shipped files; it cannot be --quick")
    if args.workers > MAX_WORKERS and not args.allow_more_workers:
        parser.error(
            f"more than {MAX_WORKERS} workers needs --allow-more-workers (task 5b)"
        )

    t0 = time.time()
    coefficients, metrics, pr = run(
        quick=args.quick,
        cache_dir=args.cache_dir,
        workers=args.workers,
        progress=not args.quiet,
    )
    files = outputs(coefficients, metrics, pr, quick=args.quick)
    print(files[f"{PREFIX}metrics.md"])
    print(f"fit done in {time.time() - t0:.0f} s")
    names = (
        f"{PREFIX}fig_terms.png",
        f"{PREFIX}fig_d1_shape.png",
        f"{PREFIX}fig_heldout.png",
    )

    if args.check:
        stale = [
            n for n, text in files.items() if (args.out_dir / n).read_text() != text
        ]
        if not args.no_figures:
            with tempfile.TemporaryDirectory() as tmp:
                figures(coefficients, metrics, Path(tmp))
                for name in names:
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
