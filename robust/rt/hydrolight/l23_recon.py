"""
Rebuilding L23's water bodies from their own published IOPs (spec Section 6.2).

**We do not have L23's HydroLight input decks and cannot get them** (Q&A A2).
We do not need them. HydroLight does not care which concentrations generated a
water body; it cares about ``a(lambda)``, ``b(lambda)`` and ``beta_tilde(psi)``.
The L23 netCDFs already on disk carry ``a``, ``bb``, ``bbnw``, ``bnw``, ``aph``
and ``ag`` per scene per wavelength, and ``B_p = bbnw/bnw`` pins the one
Fournier-Forand parameter. So every L23 scene can be re-expressed as a
user-supplied-IOP deck from data we hold, and the batch-B comparability claim
survives A2 intact.

**Three caveats, carried as recorded fields rather than as comments** -- they
travel in every water body's ``provenance`` and therefore into the manifest and
the loader, where a test can assert they are present:

1. ``ff_parameter_inferred`` -- L23's Fournier-Forand parameter is inferred from
   their published ``B_p``, not read from their deck. Fournier-Forand has two
   parameters and a ``B_p`` fixes only one combination of them, so ``mu`` is
   pinned at :data:`robust.rt.hydrolight.vsf.FF_MU_FIXED` and ``n`` is solved
   per wavelength -- and both are recorded.
2. ``sky_matched_at_anchors_only`` -- the sky is matched against
   ``robust/rt/data/ed_l23.npz`` at 0/30/60 degrees, not reproduced from
   Loisel's atmospheric specification.
3. ``optics_not_recipe`` -- "the same water bodies" means the same **optics**.
   That is the only sense that matters for radiative transfer, and it is not the
   sense a Loisel co-author would use.

**Total absorption is carried verbatim.** ``a(lambda)`` is handed to HydroLight
exactly as L23 published it at and above 350 nm, *not* rebuilt from
``a_w + a_ph + a_g + a_nap``. Rebuilding it lets the difference between our
pure-water absorption table and L23's leak into the one quantity radiative
transfer actually uses -- measured at up to **150 %** in the red before this was
fixed, because the non-algal residual went negative and was clipped. The
component split is still reported (metadata item 9), and the most negative
residual is recorded per scene in ``provenance["component_residual_min_m^-1"]``
so the inconsistency is visible rather than absorbed.

**Extrapolation below 350 nm.** L23 stops at 350 nm; the delivery grid starts at
330. The four missing bands are filled by extrapolating each component and
summing, since there is no published total to carry: ``a_ph`` and ``a_nap`` are
held at their 350 nm value (both are flat to
rising there and holding is the conservative choice), ``a_g`` follows its own
fitted exponential slope, ``b_p`` follows its own fitted power law, and pure
water comes from :func:`robust.rt.hydrolight.grid.water_tables`. Every choice is
recorded in ``provenance["uv_extrapolation"]``.
"""

from __future__ import annotations

import numpy as np

from .. import conventions as C
from . import deck, grid, vsf

__all__ = [
    "CAVEATS",
    "infer_ff_n",
    "fixture_reader",
    "release_reader",
    "reconstruct",
]

#: The three caveats every reconstructed body carries. Keys are stable; the
#: loader's tests assert each is present and true.
CAVEATS = (
    "ff_parameter_inferred",
    "sky_matched_at_anchors_only",
    "optics_not_recipe",
)

#: L23's own grid, nm.
_L23_WAVE = C.WAVE


def infer_ff_n(B_p, mu=None, n_lo=1.001, n_hi=1.40, tol=1e-9):
    """Solve ``ff_backscatter_fraction(n, mu) = B_p`` for ``n`` at fixed ``mu``.

    Bisection on a monotone branch, so the answer is deterministic and needs no
    starting guess. ``B_p`` increases with ``n`` at fixed ``mu``.

    Parameters
    ----------
    B_p : array_like
        Target backscatter fraction(s).
    mu : float, optional
        Junge slope to hold fixed; defaults to
        :data:`robust.rt.hydrolight.vsf.FF_MU_FIXED`.
    n_lo, n_hi : float, optional
        Bracket.
    tol : float, optional
        Absolute tolerance on ``n``.

    Returns
    -------
    ndarray
        ``n`` for each ``B_p``.
    """
    mu = vsf.FF_MU_FIXED if mu is None else mu
    target = np.asarray(B_p, dtype=float)
    lo = np.full_like(target, n_lo)
    hi = np.full_like(target, n_hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        val = vsf.ff_backscatter_fraction(mid, mu)
        too_small = val < target
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
        if np.all(hi - lo < tol):
            break
    return 0.5 * (lo + hi)


def fixture_reader(elastic_path, inelastic_path):
    """A reader over the committed 50-scene fixtures -- no ``$OS_COLOR`` needed.

    Returns
    -------
    callable
        ``() -> dict`` of ``a``, ``bb``, ``bbnw``, ``bnw``, ``aph``, ``ag``,
        ``wave`` at zenith 0. The IOPs are bit-identical across L23's three
        zenith files, which is why one zenith suffices.
    """
    from ..data import l23

    read = l23.inelastic_npz_reader(inelastic_path, elastic_path)

    def reader():
        raw = read(0)
        return {
            k: np.asarray(raw[k], dtype=float)
            for k in ("wave", "a", "bb", "bbnw", "bnw", "aph", "ag")
        }

    return reader


def release_reader():
    """A reader over the full 3320-scene L23 release (needs ``$OS_COLOR``)."""
    from ..data import l23

    def reader():
        elastic = l23._read_file(l23.ELASTIC_X, 0)
        inelastic = l23._read_inelastic_file(0)
        out = {
            k: np.asarray(elastic[k], dtype=float)
            for k in ("wave", "a", "bb", "bbnw", "bnw")
        }
        out["aph"] = np.asarray(inelastic["aph"], dtype=float)
        out["ag"] = np.asarray(inelastic["ag"], dtype=float)
        return out

    return reader


def _extend(wave_out, wave_in, values, mode, floor=0.0):
    """Interpolate onto ``wave_out``, extrapolating below ``wave_in[0]``."""
    out = np.interp(wave_out, wave_in, values)
    below = wave_out < wave_in[0]
    if not below.any():
        return out
    if mode == "hold":
        out[below] = values[0]
    elif mode == "exp":
        # fitted exponential slope over the first 20 bands: value = v0 exp(-S dl)
        n = min(20, values.size)
        good = values[:n] > floor
        if good.sum() >= 2:
            slope = np.polyfit(wave_in[:n][good], np.log(values[:n][good]), 1)[0]
            out[below] = values[0] * np.exp(slope * (wave_out[below] - wave_in[0]))
        else:
            out[below] = values[0]
    elif mode == "power":
        n = min(20, values.size)
        slope = np.polyfit(np.log(wave_in[:n]), np.log(values[:n]), 1)[0]
        out[below] = values[0] * (wave_out[below] / wave_in[0]) ** slope
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown extrapolation mode {mode!r}")
    return out


def reconstruct(reader, indices=None, scenarios=("S1", "S2", "S4"), block="B-L23"):
    """Rebuild L23 scenes as :class:`~robust.rt.hydrolight.deck.WaterBody`.

    Parameters
    ----------
    reader : callable
        From :func:`fixture_reader` or :func:`release_reader`.
    indices : sequence of int, optional
        Which L23 scene indices; defaults to all the reader holds.
    scenarios : tuple of str, optional
        Default ``("S1", "S2", "S4")`` -- batch B's three.
    block : str, optional
        Block label recorded on each body.

    Returns
    -------
    list of WaterBody
    """
    raw = reader()
    wave_in = raw["wave"]
    if not np.allclose(wave_in, _L23_WAVE):
        raise ValueError("reader did not return L23's 81-band grid")
    n_scene = raw["a"].shape[0]
    indices = range(n_scene) if indices is None else list(indices)
    wave_out = deck.WAVE_HL
    a_w, bb_w_out, b_w_out = grid.water_tables(wave_out)

    out = []
    for i in indices:
        a = raw["a"][i]
        bbnw = raw["bbnw"][i]
        bnw = raw["bnw"][i]
        aph = raw["aph"][i]
        ag = raw["ag"][i]
        a_w_in = np.interp(wave_in, wave_out, a_w)
        anap = a - a_w_in - aph - ag
        residual = float(np.min(anap))

        B_p_in = bbnw / bnw
        a_ph_o = _extend(wave_out, wave_in, aph, "hold")
        a_g_o = _extend(wave_out, wave_in, ag, "exp")
        a_nap_o = _extend(wave_out, wave_in, np.clip(anap, 0.0, None), "hold")
        b_p_o = _extend(wave_out, wave_in, bnw, "power")
        B_p_o = _extend(wave_out, wave_in, B_p_in, "hold")

        # Total absorption is carried VERBATIM where L23 defines it. Rebuilding
        # it from components would let the difference between our pure-water
        # table and L23's corrupt the one quantity radiative transfer actually
        # uses -- measured at up to 150 % in the red before this was fixed.
        # Below 350 nm, where L23 is silent, the components do the extending.
        a_o = np.interp(wave_out, wave_in, a)
        below = wave_out < wave_in[0]
        a_o[below] = (a_w + a_ph_o + a_g_o + a_nap_o)[below]
        bb_o = bb_w_out + B_p_o * b_p_o
        ff_n = infer_ff_n(B_p_o)

        out.append(
            deck.WaterBody(
                wbid=f"{block}-{i:05d}",
                block=block,
                a=a_o,
                b=b_w_out + b_p_o,
                bb=bb_o,
                b_p=b_p_o,
                a_ph=a_ph_o,
                a_cdom=a_g_o,
                a_nap=a_nap_o,
                vsf_design="l23_ff_per_wavelength",
                scenarios=tuple(scenarios),
                want_depth_output=False,
                provenance={
                    "construction": "reconstructed from L23 published IOPs",
                    "l23_scene_index": int(i),
                    "ff_mu_fixed": vsf.FF_MU_FIXED,
                    "ff_n_at_440": float(
                        ff_n[int(np.argmin(np.abs(wave_out - 440.0)))]
                    ),
                    "B_p_at_440": float(
                        B_p_o[int(np.argmin(np.abs(wave_out - 440.0)))]
                    ),
                    "component_residual_min_m^-1": residual,
                    "uv_extrapolation": {
                        "below_nm": 350.0,
                        "a_total": (
                            "sum of extended components; verbatim at and above 350 nm"
                        ),
                        "a_ph": "hold",
                        "a_cdom": "fitted exponential",
                        "a_nap": "hold",
                        "b_p": "fitted power law",
                        "B_p": "hold",
                    },
                    "caveats": {k: True for k in CAVEATS},
                },
            )
        )
    return out
