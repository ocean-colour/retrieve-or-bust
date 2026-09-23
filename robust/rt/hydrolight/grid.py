"""
The batch-A ``(bb/a, eta_bb)`` grid, its realizability mask, and the fill set.

**The grid the catalogue asked for does not exist.** Spec run R1 wants ``bb/a``
from 1e-4 to 3 crossed with ``eta_bb = bb_w/bb`` from 0.01 to 0.98, twelve nodes
by eight. Fix those two ratios and the water body is determined:

    ``bb = bb_w / eta_bb``,  ``a = bb / (bb/a)``

and since total absorption cannot fall below pure water's, ``a >= a_w``, so

    **``eta_bb * (bb/a) <= bb_w(lambda) / a_w(lambda)``**

That ceiling is a property of pure water alone. It is 0.498 at 400 nm, 0.346 at
440 and **0.0151 at 550** -- so the reference wavelength matters enormously, and
400 nm is adopted because the realizable region is largest there. Adding the two
physical caps ``a_nw <= 20 m^-1`` (blackwater is about the limit) and
``b_p <= 100 m^-1`` leaves **63 of the 96 nodes**, independent of ``B_p``.

**What rescues the coverage.** ``a_w`` and ``bb_w`` both vary strongly across
330-750 nm -- ``bb_w`` by 25x over 350-750 alone, on a fitted -4.345 power law --
so a water body with flat ``a_nw`` and ``b_p`` traces a *curve* through
``(bb/a, eta_bb)`` rather than sitting at a point. The 63 nodes therefore occupy
about 90 % of the original 12 x 8 box across the 85 bands. The cells that stay
empty are enumerated by :func:`box_occupancy`, not hidden.
"""

from __future__ import annotations

import numpy as np

from .. import conventions as C
from . import deck, vsf

__all__ = [
    "LAMBDA_REF",
    "A_NW_MAX",
    "B_P_MAX",
    "BB_A_NODES",
    "ETA_NODES",
    "water_tables",
    "realize_node",
    "grid_bodies",
    "fill_bodies",
    "box_occupancy",
]

#: Reference wavelength the design coordinates are defined at, nm. 400 because
#: ``bb_w/a_w`` peaks near there and the realizable region is therefore largest.
LAMBDA_REF = 400.0

#: Physical caps on the realised water body. Beyond these the node is not water.
A_NW_MAX = 20.0
B_P_MAX = 100.0

#: The requested design nodes (spec Section 5.2), logarithmic.
BB_A_NODES = np.logspace(-4.0, np.log10(3.0), 12)
ETA_NODES = np.logspace(np.log10(0.01), np.log10(0.98), 8)

#: Power-law slope fitted to the packaged ``bb_w`` over 350-420 nm, used to
#: extend it down to 330 nm. Recomputed by :func:`water_tables`, not assumed.
_BB_W_FIT_BAND = 15


def water_tables(wave=None):
    """Pure-water ``a_w``, ``bb_w`` and ``b_w`` on the delivery grid.

    ``bb_w`` comes from the packaged :data:`robust.rt.conventions.BB_W_L23`
    (350-750 nm) and is extended to 330 nm by **its own** fitted power law
    rather than an assumed -4.32, so the extension is consistent with the table
    the rest of the package uses. ``b_w = 2 bb_w``: pure-water scattering is
    symmetric about 90 degrees, so its backscatter fraction is exactly 0.5.

    Parameters
    ----------
    wave : array_like, optional
        Wavelengths, nm; defaults to :data:`robust.rt.hydrolight.deck.WAVE_HL`.

    Returns
    -------
    tuple of ndarray
        ``(a_w, bb_w, b_w)``.

    Raises
    ------
    ImportError
        If ``ocpy`` is unavailable -- pure-water absorption has no packaged
        fallback here, and inventing one would poison every node.
    """
    from ocpy.water.absorption import a_water

    wave = deck.WAVE_HL if wave is None else np.asarray(wave, dtype=float)
    a_w = np.asarray(a_water(wave), dtype=float)
    slope, intercept = np.polyfit(
        np.log(C.WAVE[:_BB_W_FIT_BAND]), np.log(C.BB_W_L23[:_BB_W_FIT_BAND]), 1
    )
    extended = np.exp(intercept + slope * np.log(wave))
    bb_w = np.where(wave >= C.WAVE[0], np.interp(wave, C.WAVE, C.BB_W_L23), extended)
    return a_w, bb_w, 2.0 * bb_w


def realize_node(bb_a, eta_bb, B_p, a_w_ref, bb_w_ref):
    """Turn one ``(bb/a, eta_bb)`` node into ``(a_nw, b_p)``, or reject it.

    Parameters
    ----------
    bb_a, eta_bb : float
        The design coordinates at :data:`LAMBDA_REF`.
    B_p : float
        Particle backscatter fraction of the VSF design carried.
    a_w_ref, bb_w_ref : float
        Pure-water absorption and backscattering at :data:`LAMBDA_REF`.

    Returns
    -------
    tuple or None
        ``(a_nw, b_p)`` in m^-1, or None when the node is not physical water.
    """
    bb_ref = bb_w_ref / eta_bb
    a_ref = bb_ref / bb_a
    a_nw = a_ref - a_w_ref
    b_p = (bb_ref - bb_w_ref) / B_p
    if a_nw < 0.0 or a_nw > A_NW_MAX or b_p > B_P_MAX or b_p <= 0.0:
        return None
    return float(a_nw), float(b_p)


def _body(wbid, block, a_nw, b_p, design, a_w, bb_w, b_w, provenance):
    """Assemble a spectrally flat ``(a_nw, b_p)`` water body."""
    B_p = design.target_bp
    return deck.WaterBody(
        wbid=wbid,
        block=block,
        a=a_w + a_nw,
        b=b_w + b_p,
        bb=bb_w + B_p * b_p,
        b_p=np.full_like(b_w, b_p),
        vsf_design=design.name,
        scenarios=("S1",),
        want_depth_output=True,
        provenance=provenance,
    )


def grid_bodies(designs=None):
    """The realizable grid nodes, crossed with every tabulated VSF design.

    Parameters
    ----------
    designs : sequence of VSFDesign, optional
        Defaults to the tabulated members of :func:`robust.rt.hydrolight.vsf.designs`.
        Built-ins are excluded here because a node's realizability depends on the
        design's ``B_p``, which a built-in does not declare.

    Returns
    -------
    list of robust.rt.hydrolight.deck.WaterBody
    """
    if designs is None:
        designs = [d for d in vsf.designs() if d.tabulated]
    a_w, bb_w, b_w = water_tables()
    i_ref = int(np.argmin(np.abs(deck.WAVE_HL - LAMBDA_REF)))
    out = []
    for design in designs:
        for i, bb_a in enumerate(BB_A_NODES):
            for j, eta in enumerate(ETA_NODES):
                got = realize_node(bb_a, eta, design.target_bp, a_w[i_ref], bb_w[i_ref])
                if got is None:
                    continue
                a_nw, b_p = got
                out.append(
                    _body(
                        f"A-n{i:02d}e{j:02d}-{design.name}",
                        "A-grid",
                        a_nw,
                        b_p,
                        design,
                        a_w,
                        bb_w,
                        b_w,
                        {
                            "construction": "IOP grid node",
                            "lambda_ref_nm": LAMBDA_REF,
                            "target_bb_over_a": float(bb_a),
                            "target_eta_bb": float(eta),
                            "a_nw_m^-1": a_nw,
                            "b_p_m^-1": b_p,
                            "spectrally_flat": True,
                        },
                    )
                )
    return out


def n_realizable_nodes(B_p=0.012):
    """How many of the 96 nodes exist, at one ``B_p`` -- the spec's 63."""
    a_w, bb_w, _ = water_tables()
    i = int(np.argmin(np.abs(deck.WAVE_HL - LAMBDA_REF)))
    return sum(
        realize_node(x, y, B_p, a_w[i], bb_w[i]) is not None
        for x in BB_A_NODES
        for y in ETA_NODES
    )


def fill_bodies(n=150, seed=20260923, designs=None):
    """A Latin-hypercube fill set drawn directly in ``(a_nw, b_p)``.

    The node grid is defined at one wavelength; away from it the delivered
    ``(bb/a, eta_bb)`` cloud thins in places. These bodies are drawn in the
    *realised* coordinates instead, so they densify what the grid leaves sparse
    without inheriting the grid's reference-wavelength bias.

    Parameters
    ----------
    n : int, optional
        How many, default 150.
    seed : int, optional
        Fixed, so the set is reproducible from the manifest alone.
    designs : sequence of VSFDesign, optional
        As :func:`grid_bodies`.

    Returns
    -------
    list of robust.rt.hydrolight.deck.WaterBody
    """
    if designs is None:
        designs = [d for d in vsf.designs() if d.tabulated]
    a_w, bb_w, b_w = water_tables()
    rng = np.random.default_rng(seed)
    # Latin hypercube in log(a_nw), log(b_p): one sample per stratum, shuffled.
    strata = (np.arange(n) + rng.random(n)) / n
    la = np.log(0.001) + rng.permutation(strata) * (np.log(A_NW_MAX) - np.log(0.001))
    lb = np.log(0.005) + rng.permutation(strata) * (np.log(B_P_MAX) - np.log(0.005))
    out = []
    for k, (a_nw, b_p) in enumerate(zip(np.exp(la), np.exp(lb), strict=True)):
        design = designs[k % len(designs)]
        out.append(
            _body(
                f"A-fill{k:03d}-{design.name}",
                "A-fill",
                float(a_nw),
                float(b_p),
                design,
                a_w,
                bb_w,
                b_w,
                {
                    "construction": "Latin-hypercube fill in (a_nw, b_p)",
                    "seed": seed,
                    "a_nw_m^-1": float(a_nw),
                    "b_p_m^-1": float(b_p),
                    "spectrally_flat": True,
                },
            )
        )
    return out


def box_occupancy(bodies, bins=(12, 8)):
    """Where the delivered ``(bb/a, eta_bb)`` cloud lands in the target box.

    Parameters
    ----------
    bodies : sequence of WaterBody
    bins : tuple, optional
        Log-spaced bin counts over the target box, default the requested 12 x 8.

    Returns
    -------
    dict
        ``{"n_samples", "bb_a_range", "eta_range", "occupied", "n_cells",
        "fraction", "empty_cells"}``. ``empty_cells`` lists ``(i, j)`` index
        pairs with their approximate centres, so a gap is *named*.
    """
    _, bb_w, _ = water_tables()
    bb_a, eta = [], []
    for wb in bodies:
        bb_a.append(np.asarray(wb.bb) / np.asarray(wb.a))
        eta.append(bb_w / np.asarray(wb.bb))
    bb_a = np.concatenate(bb_a)
    eta = np.concatenate(eta)
    lo_x, hi_x = np.log10(BB_A_NODES[0]), np.log10(BB_A_NODES[-1])
    lo_y, hi_y = np.log10(ETA_NODES[0]), np.log10(ETA_NODES[-1])
    h = np.histogram2d(
        np.log10(np.clip(bb_a, BB_A_NODES[0], BB_A_NODES[-1])),
        np.log10(np.clip(eta, ETA_NODES[0], ETA_NODES[-1])),
        bins=bins,
        range=[[lo_x, hi_x], [lo_y, hi_y]],
    )[0]
    empty = []
    for i, j in zip(*np.where(h == 0), strict=True):
        empty.append(
            {
                "i": int(i),
                "j": int(j),
                "bb_over_a": float(10 ** (lo_x + (hi_x - lo_x) * (i + 0.5) / bins[0])),
                "eta_bb": float(10 ** (lo_y + (hi_y - lo_y) * (j + 0.5) / bins[1])),
            }
        )
    return {
        "n_samples": int(bb_a.size),
        "bb_a_range": (float(bb_a.min()), float(bb_a.max())),
        "eta_range": (float(eta.min()), float(eta.max())),
        "occupied": int((h > 0).sum()),
        "n_cells": int(h.size),
        "fraction": float((h > 0).mean()),
        "empty_cells": empty,
    }
