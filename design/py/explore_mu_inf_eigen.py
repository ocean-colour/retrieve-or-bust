#!/usr/bin/env python
"""
EXPLORATORY (M7 task 1, Q13): µ∞ as the asymptotic eigenvalue of the RTE.

Not part of the pipeline and not imported by the package. It exists so the
numbers behind Q&A "Fit L23" Q13 in ``claude_prompts/RT/rt_fit_ztt.md`` can be
reproduced.

µ∞ is an IOP: the asymptotic radiance distribution is the eigenvector of the
azimuthally averaged RTE with the smallest attenuation, K∞, and Gershun gives
µ∞ = a / K∞. With L(z, µ) = f(µ) e^{−K z}, on a double-Gauss grid in µ:

    (c I − b P̃ W) f = K diag(µ) f

where P̃ is the azimuth-averaged phase function and W the quadrature weights.
L23's phase function is pure water (β ∝ 1 + f cos²ψ) plus two Fournier–Forand
components (b̃b = 0.010 phytoplankton, 0.018 non-algal; Loisel et al. 2023 §2),
with FF's (n, µ) from HydroLight's Mobley et al. (2002) relation unless a
convention is chosen below. The forward peak is δ-truncated at ``--psic`` deg
(the δ part does not change the radiance distribution), so the kernel's
quadrature converges.

Compared against the profile product's ``asymptotic`` points (where µ∞ *is*
measured) and against TT2017 on random points.

    python design/py/explore_mu_inf_eigen.py [--n 80] [--psic 5] [--sample 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robust.rt import ztt  # noqa: E402
from robust.rt.data import l23 as L  # noqa: E402


def ff(psi, n, m):
    """Fournier–Forand phase function (sr^-1), Mobley et al. (2002) form."""
    nu = (3 - m) / 2
    d = 4 / (3 * (n - 1) ** 2) * np.sin(psi / 2) ** 2
    d180 = 4 / (3 * (n - 1) ** 2)
    s2 = np.sin(psi / 2) ** 2
    t1 = (nu * (1 - d) - (1 - d**nu) + (d * (1 - d**nu) - nu * (1 - d)) / s2) / (
        4 * np.pi * (1 - d) ** 2 * d**nu
    )
    t2 = (1 - d180**nu) / (16 * np.pi * (d180 - 1) * d180**nu)
    return t1 + t2 * (3 * np.cos(psi) ** 2 - 1)


def ff_B(n, m):
    """FF backscatter fraction, closed form."""
    nu = (3 - m) / 2
    d90 = 4 / (3 * (n - 1) ** 2) * 0.5
    return 1 - (1 - d90 ** (nu + 1) - 0.5 * (1 - d90**nu)) / ((1 - d90) * d90**nu)


CONVENTIONS = {
    "mobley2002": lambda B: (lambda m: (1.01 + 0.1542 * (m - 3), m))(
        brentq(lambda m: ff_B(1.01 + 0.1542 * (m - 3), m) - B, 3.0001, 5.0)
    ),
    "slope3.5": lambda B: (brentq(lambda n: ff_B(n, 3.5) - B, 1.0001, 1.5), 3.5),
    "slope4.0": lambda B: (brentq(lambda n: ff_B(n, 4.0) - B, 1.0001, 1.5), 4.0),
    "n1.05": lambda B: (1.05, brentq(lambda m: ff_B(1.05, m) - B, 3.0001, 5.0)),
    "n1.20": lambda B: (1.20, brentq(lambda m: ff_B(1.20, m) - B, 3.0001, 5.0)),
}


def water(f=0.835):
    """Pure-water phase function, β ∝ 1 + f cos²ψ, normalized."""
    return lambda psi: 3 / (4 * np.pi * (3 + f)) * (1 + f * np.cos(psi) ** 2)


def truncate(p, psic):
    """δ-truncate the forward peak inside ``psic`` (rad): returns (p*, F)."""
    if psic <= 0:
        return p, 0.0
    inside = (
        quad(lambda t: p(t) * np.sin(t), 1e-12, psic, limit=400, points=[1e-4, 1e-2])[0]
        * 2
        * np.pi
    )
    flat = p(psic) * 2 * np.pi * (1 - np.cos(psic))
    F = inside - flat
    return (lambda t: np.where(t < psic, p(psic), p(t)) / (1 - F)), F


def grid(N):
    """Double-Gauss nodes and weights in µ on [-1, 1]."""
    x, w = np.polynomial.legendre.leggauss(N // 2)
    up, wu = 0.5 * (x + 1), 0.5 * w
    return np.concatenate([-up[::-1], up]), np.concatenate([wu[::-1], wu])


def kernel(p, mu, wt, nphi=2048):
    """Azimuth-averaged phase function, renormalized to conserve energy."""
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    si = np.sqrt(1 - mu**2)
    cosT = mu[:, None, None] * mu[None, :, None] + si[:, None, None] * si[
        None, :, None
    ] * np.cos(phi)
    K = p(np.arccos(np.clip(cosT, -1, 1))).mean(-1)
    s = (K * wt[None, :]).sum(1) * 2 * np.pi
    return K + np.diag((1 - s) / (2 * np.pi * wt))


def components(N, psic_deg, convention):
    """(kernel, δ-fraction) for water, phytoplankton FF, non-algal FF."""
    mu, wt = grid(N)
    pars = [CONVENTIONS[convention](B) for B in (0.010, 0.018)]
    out = []
    for p in (water(), *(lambda t, q=q: ff(t, *q) for q in pars)):
        ps, F = truncate(p, np.deg2rad(psic_deg))
        out.append((kernel(ps, mu, wt), F))
    return mu, wt, out, pars


def mu_inf(a, bs, mu, wt, comps):
    """µ∞ = a / K∞ for one (scene, λ); ``bs`` = (b_w, b_ph, b_d)."""
    S = (
        sum(b * (1 - F) * K for b, (K, F) in zip(bs, comps, strict=True))
        * 2
        * np.pi
        * wt[None]
    )
    c = a + sum(b * (1 - F) for b, (_, F) in zip(bs, comps, strict=True))
    ev = np.linalg.eigvals((c * np.eye(mu.size) - S) / mu[:, None])
    ev = ev.real[(ev.real > 0) & (np.abs(ev.imag) < 1e-9)]
    return a / ev.min()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="µ∞ eigen exploration (Q13)")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--psic", type=float, default=5.0)
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args(argv)

    from ocpy.hydrolight import loisel23

    e = loisel23.load_ds(1, 0)
    a, b, bnw, bph, bd, bb, bbnw = (
        np.asarray(e[v].data, dtype=float)
        for v in ("a", "b", "bnw", "bph", "bd", "bb", "bbnw")
    )
    bw = b - bnw
    product = L.build_profile_product()
    rng = np.random.default_rng(0)
    asym = np.argwhere(product.flag == L.FLAG_ASYMPTOTIC)
    sel = asym[rng.choice(len(asym), args.sample, replace=False)]
    everywhere = np.argwhere(np.isfinite(a))
    rand = everywhere[rng.choice(len(everywhere), args.sample, replace=False)]

    print(f"N={args.n}, δ-truncation {args.psic}°, {args.sample} points each")
    print("convention  (n, µ) phyto / NAP   eig/profile on asymptotic: 16/50/84%")
    for conv in CONVENTIONS:
        mu, wt, comps, pars = components(args.n, args.psic, conv)
        r = np.array(
            [
                mu_inf(a[i, j], (bw[i, j], bph[i, j], bd[i, j]), mu, wt, comps)
                / product.mu_inf[i, j]
                for i, j in sel
            ]
        )
        q = np.percentile(r, [16, 50, 84])
        shown = " / ".join(f"({n:.3f}, {m:.3f})" for n, m in pars)
        print(f"{conv:10s}  {shown}   {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}")
        if conv == "mobley2002":
            eig = np.array(
                [
                    mu_inf(a[i, j], (bw[i, j], bph[i, j], bd[i, j]), mu, wt, comps)
                    for i, j in rand
                ]
            )
            x = bb[rand[:, 0], rand[:, 1]] / a[rand[:, 0], rand[:, 1]]
            eta = 1 - bbnw[rand[:, 0], rand[:, 1]] / bb[rand[:, 0], rand[:, 1]]
            tt = np.asarray(ztt.mu_infinity_tt2017(x, eta))
            q = np.percentile(tt / eig, [1, 16, 50, 84, 99])
            tt_pct = " ".join(f"{v:.3f}" for v in q)
            print(f"   TT2017/eig on random points, 1/16/50/84/99%: {tt_pct}")
            print(f"   eig µ∞ range on random points: {eig.min():.3f}–{eig.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
