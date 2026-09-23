"""
The designed water-body ensembles: batch B's fill block and batch C's subsets.

Batch B has two blocks (spec Section 6.2). The **L23 block** is
:mod:`robust.rt.hydrolight.l23_recon`; this module builds the **design block** --
1,500 new water bodies drawn by Latin hypercube over ranges deliberately wider
than L23's, each carrying one of the tabulated VSF designs so that the
phase-function axis is crossed with the inelastic physics. That crossing is the
point: it is the one thing no dataset in existence for this project has, and
running batches A and B on disjoint water bodies would have re-created exactly
the hole the campaign was written to close.

Absorption shapes are **not invented**. Normalised ``a_ph`` shapes are drawn
from L23's own library (``aph / aph(440)``), the way PB24 reuses a finite shape
library, so the ensemble is wider than L23 in *magnitude* without being
fictional in *shape*. CDOM and non-algal absorption are exponential with drawn
slopes; particle scattering is a power law with a drawn slope.

Batch C's three subsets (spec Section 7) are drawn **from** the design block, so
that every batch-C water body already has its batch-B partner and the scenario
differences (``S5 - S4`` for CDOM fluorescence, the phi_C series for R4) are
clean.
"""

from __future__ import annotations

import numpy as np

from . import deck, grid, vsf

__all__ = [
    "DESIGN_SEED",
    "design_block",
    "r4_subset",
    "r5_subset",
    "r8_profiles",
]

#: Fixed so the whole ensemble is reproducible from the manifest alone.
DESIGN_SEED = 20260923

#: Draw ranges, each wider than L23's own span (spec Section 6.2).
_RANGES = {
    "a_ph_440": (0.002, 3.0),
    "a_g_440": (0.002, 8.0),
    "S_g": (0.010, 0.022),
    "a_nap_440": (0.001, 2.0),
    "S_nap": (0.006, 0.014),
    "b_p_550": (0.02, 40.0),
    "gamma_p": (-0.5, 2.0),
}


def _lhs(rng, n, k):
    """A Latin hypercube: one sample per stratum in each of ``k`` dimensions."""
    strata = (np.arange(n) + rng.random((k, n))) / n
    return np.stack([rng.permutation(row) for row in strata])


def _aph_shapes(reader):
    """Normalised ``a_ph`` shapes from L23, interpolated onto the delivery grid."""
    raw = reader()
    wave_in = raw["wave"]
    aph = raw["aph"]
    ref = aph[:, int(np.argmin(np.abs(wave_in - 440.0)))]
    keep = ref > 0
    shapes = aph[keep] / ref[keep, None]
    out = np.empty((shapes.shape[0], deck.WAVE_HL.size))
    for i, s in enumerate(shapes):
        out[i] = np.interp(deck.WAVE_HL, wave_in, s)
        out[i][deck.WAVE_HL < wave_in[0]] = s[0]
    return out


def design_block(reader, n=1500, seed=DESIGN_SEED, scenarios=("S1", "S2", "S4")):
    """Batch B's designed water bodies.

    Parameters
    ----------
    reader : callable
        An L23 reader (see :mod:`robust.rt.hydrolight.l23_recon`), used only for
        its ``a_ph`` shape library.
    n : int, optional
        How many, default 1500.
    seed : int, optional
        Fixed; recorded in every body's provenance.
    scenarios : tuple of str, optional

    Returns
    -------
    list of robust.rt.hydrolight.deck.WaterBody
    """
    rng = np.random.default_rng(seed)
    shapes = _aph_shapes(reader)
    keys = list(_RANGES)
    u = _lhs(rng, n, len(keys))
    draws = {}
    for i, key in enumerate(keys):
        lo, hi = _RANGES[key]
        if key == "gamma_p":
            draws[key] = lo + u[i] * (hi - lo)
        else:
            draws[key] = np.exp(np.log(lo) + u[i] * (np.log(hi) - np.log(lo)))
    shape_idx = rng.integers(0, shapes.shape[0], n)

    tabulated = [d for d in vsf.designs() if d.tabulated]
    a_w, bb_w, b_w = grid.water_tables()
    w = deck.WAVE_HL
    out = []
    for k in range(n):
        design = tabulated[k % len(tabulated)]
        a_ph = draws["a_ph_440"][k] * shapes[shape_idx[k]]
        a_g = draws["a_g_440"][k] * np.exp(-draws["S_g"][k] * (w - 440.0))
        a_nap = draws["a_nap_440"][k] * np.exp(-draws["S_nap"][k] * (w - 440.0))
        b_p = draws["b_p_550"][k] * (550.0 / w) ** draws["gamma_p"][k]
        out.append(
            deck.WaterBody(
                wbid=f"B-design-{k:04d}",
                block="B-design",
                a=a_w + a_ph + a_g + a_nap,
                b=b_w + b_p,
                bb=bb_w + design.target_bp * b_p,
                b_p=b_p,
                a_ph=a_ph,
                a_cdom=a_g,
                a_nap=a_nap,
                vsf_design=design.name,
                scenarios=tuple(scenarios),
                want_depth_output=False,
                provenance={
                    "construction": "Latin hypercube over the design ranges",
                    "seed": seed,
                    "draws": {key: float(draws[key][k]) for key in keys},
                    "aph_shape_from_l23_index": int(shape_idx[k]),
                    "ranges": {key: list(_RANGES[key]) for key in keys},
                },
            )
        )
    return out


def _stratified_pick(values, n, rng, weights=None):
    """Pick ``n`` indices spread across ``values``, optionally tail-weighted."""
    order = np.argsort(values)
    if weights is None:
        edges = np.linspace(0, len(order), n + 1).astype(int)
        pairs = zip(edges[:-1], edges[1:], strict=True)
        return np.array([order[a + (b - a) // 2] for a, b in pairs])
    p = np.asarray(weights, dtype=float)[order]
    p = p / p.sum()
    return order[rng.choice(len(order), size=n, replace=False, p=p)]


def r4_subset(bodies, n=100, phi_values=(0.005, 0.01, 0.04, 0.06), seed=DESIGN_SEED):
    """R4: the varied-``phi_C`` series (spec Section 7.1).

    One water body per ``(source body, phi_C)`` pair, spread across the
    ``a_ph(440)`` range. Deliberately small on the scene axis: this measures
    **curvature in phi_C**, which more scenes do not measure better.

    Returns
    -------
    list of WaterBody
    """
    rng = np.random.default_rng(seed + 4)
    i440 = int(np.argmin(np.abs(deck.WAVE_HL - 440.0)))
    aph440 = np.array([b.a_ph[i440] for b in bodies])
    picks = _stratified_pick(aph440, n, rng)
    out = []
    for j, idx in enumerate(picks):
        for phi in phi_values:
            src = bodies[int(idx)]
            out.append(
                _respecialise(
                    src,
                    wbid=f"C-R4-{j:03d}-phi{phi:g}",
                    block="C-R4",
                    scenarios=("S4",),
                    phi_C=phi,
                    extra={"r4_source": src.wbid, "phi_C": phi},
                )
            )
    return out


def r5_subset(bodies, n=500, seed=DESIGN_SEED, tail_power=2.0):
    """R5: the CDOM-fluorescence scenario (spec Section 7.2).

    Stratified on ``a_cdom(440)`` with the **CDOM-rich tail oversampled** --
    ``tail_power`` biases the draw toward high CDOM, which is the sparse-tail
    lesson from the fluorescence head's eutrophic drift. Only ``S5`` runs: the
    ``S4`` partner already exists in batch B, so ``S5 - S4`` isolates the
    process exactly as ``S4 - S2`` isolates chlorophyll fluorescence.

    Returns
    -------
    list of WaterBody
    """
    rng = np.random.default_rng(seed + 5)
    i440 = int(np.argmin(np.abs(deck.WAVE_HL - 440.0)))
    ag440 = np.array([b.a_cdom[i440] for b in bodies])
    picks = _stratified_pick(ag440, n, rng, weights=ag440**tail_power)
    return [
        _respecialise(
            bodies[int(idx)],
            wbid=f"C-R5-{j:03d}",
            block="C-R5",
            scenarios=("S5",),
            extra={"r5_source": bodies[int(idx)].wbid, "tail_power": tail_power},
        )
        for j, idx in enumerate(picks)
    ]


def r8_profiles(bodies, n_base=20, seed=DESIGN_SEED, scenarios=("S1", "S2", "S4")):
    """R8: vertical structure (spec Section 7.3).

    Twenty base water bodies times three structures -- a homogeneous control, a
    subsurface chlorophyll maximum at two depths, and a depth-varying ``phi_C``
    (photoinhibition near the surface). Sixty profiles. This bounds a bias; it
    does not train anything, which is why it stays small.

    Note the tension with batch A, which **requires** homogeneity because the
    asymptotic regime only exists for a homogeneous column. Different runs, not
    a shared setup.

    Returns
    -------
    list of WaterBody
    """
    rng = np.random.default_rng(seed + 8)
    i440 = int(np.argmin(np.abs(deck.WAVE_HL - 440.0)))
    aph440 = np.array([b.a_ph[i440] for b in bodies])
    picks = _stratified_pick(aph440, n_base, rng)
    structures = (
        ("homogeneous", None),
        (
            "scm_20m",
            {
                "kind": "subsurface_chl_max",
                "depth_m": 20.0,
                "gain": 2.5,
                "sigma_m": 8.0,
            },
        ),
        (
            "phiC_photoinhibited",
            {"kind": "phi_C_profile", "surface": 0.008, "deep": 0.030, "scale_m": 15.0},
        ),
    )
    out = []
    for j, idx in enumerate(picks):
        for name, profile in structures:
            src = bodies[int(idx)]
            out.append(
                _respecialise(
                    src,
                    wbid=f"C-R8-{j:02d}-{name}",
                    block="C-R8",
                    scenarios=tuple(scenarios),
                    depth_profile=profile,
                    extra={"r8_source": src.wbid, "structure": name},
                )
            )
    return out


def _respecialise(
    src, *, wbid, block, scenarios, phi_C=None, depth_profile=None, extra=None
):
    """A copy of ``src`` with a new identity and run plan, sharing its optics."""
    provenance = dict(src.provenance)
    provenance.update(extra or {})
    provenance["derived_from"] = src.wbid
    return deck.WaterBody(
        wbid=wbid,
        block=block,
        a=src.a,
        b=src.b,
        bb=src.bb,
        b_p=src.b_p,
        a_ph=src.a_ph,
        a_cdom=src.a_cdom,
        a_nap=src.a_nap,
        vsf_design=src.vsf_design,
        scenarios=scenarios,
        phi_C=src.phi_C if phi_C is None else phi_C,
        depth_profile=depth_profile,
        want_depth_output=False,
        provenance=provenance,
    )
