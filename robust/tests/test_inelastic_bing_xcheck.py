"""
The formal live-BING cross-check — M2 task 3, coding-plan CQ3a.

The M2 ports are *ports of the fixed BING physics*, and this module is the
contract: :func:`robust.rt.inelastic.raman_factor` against
``bing.rt.rrs.calc_raman_correction_factor`` and ``phi_C * K_fl`` against
``bing.rt.rrs.calc_Rrs_fluorescence`` at **rtol <= 1e-6** on every fixture
sample (150 = 50 scenes x 3 zeniths), both fed byte-identical inputs — the
same excitation nodes, IOPs, and Ed — so any disagreement is the port, not
the data path. Run in float64 (the ``jax_x64`` fixture): the gate has ~9-10
decades of headroom there (measured worst: Raman 4.1e-16, fluorescence
1.1e-13 over *all* wavelengths, tails included), where float32 trapezoid
accumulation alone is ~3e-6.

**The sentinel runs first, and is why this module exists as insurance.** The
whole M2 contract assumes the editable ``bing`` checkout carries the
``inelastic-fixes`` work — above all the ``Lu = Eu/pi`` normalization, without
which BING's fluorescence is ~3x too bright (the assessment's second
headline). A checkout silently rolled back to the pre-fix branch would make
the pin tests *agree with the wrong physics* if our port were then "fixed" to
match. The sentinel prevents that: it evaluates BING on a trivial scene and
compares against the post-fix formula written out here by hand (double-entry
bookkeeping, independent of :mod:`robust.rt.inelastic`), failing loudly with
"predates inelastic-fixes" when the ratio comes out near pi.

Everything here ``importorskip``s on ``bing`` at module level, so GitHub CI —
which has no bing checkout — skips the module while local runs enforce the
pin (the two-tier philosophy of the elastic hash gate).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

bing_rrs = pytest.importorskip(
    "bing.rt.rrs", reason="live BING cross-check needs the bing checkout (CI skips)"
)
bing_raman = pytest.importorskip("bing.rt.raman")

if not all(
    hasattr(bing_rrs, name)
    for name in ("calc_raman_correction_factor", "calc_Rrs_fluorescence")
):  # pragma: no cover - only a stale checkout gets here
    pytest.skip(
        "bing checkout lacks the inelastic functions (predates inelastic-fixes)",
        allow_module_level=True,
    )

# The robust imports must follow the module-level skip: on CI (no bing) the
# module has to stop before importing anything that would still work.
from robust.rt import conventions as C  # noqa: E402
from robust.rt import ed as E  # noqa: E402
from robust.rt import inelastic as I  # noqa: E402
from robust.rt.data.l23 import PHI_C_L23  # noqa: E402
from robust.rt.types import Geometry, IOPs  # noqa: E402

#: The M2 gate (coding plan / design §5): the port must match live fixed-BING
#: to this relative tolerance on shared inputs.
RTOL_GATE = 1e-6


@pytest.fixture()
def batch64(jax_x64, l23_small_inelastic_batch):
    """The fixture batch rebuilt in float64 (inputs + geometry + wave).

    The session batch is float32 (the suite's default regime); the gate runs
    in float64 so that what is measured is the *port*, not float32 trapezoid
    accumulation. Rebuilt per test — cheap (150 x 81) — from NumPy copies so
    no float32 device array leaks through.
    """
    b = l23_small_inelastic_batch
    dtype = jnp.float64
    iops = IOPs(
        a=jnp.asarray(np.asarray(b.iops.a), dtype=dtype),
        bb_w=jnp.asarray(np.asarray(b.iops.bb_w), dtype=dtype),
        bb_p=jnp.asarray(np.asarray(b.iops.bb_p), dtype=dtype),
        a_ph=jnp.asarray(np.asarray(b.iops.a_ph), dtype=dtype),
    )
    geometry = Geometry.nadir(jnp.asarray(b.zenith, dtype=dtype))
    wave = jnp.asarray(np.asarray(b.wave), dtype=dtype)
    return b, iops, geometry, wave


# ------------------------------------------------------------------ sentinel ----


def test_sentinel_bing_carries_the_pi_fix():
    """BING's fluorescence normalization is post-fix — checked double-entry.

    A trivial flat-IOP scene, evaluated by BING and by the documented
    post-fix formula written out here by hand (same trapezoid, then
    ``rrs_F = R_F/pi``, then ``h_C * A*rrs/(1 - B*rrs)``). Measured ratio on
    the fixed checkout: 1.0 to float64 precision. A pre-fix checkout omits
    the ``1/pi`` and lands at ~3.14 — the assertion message names the cure.
    Independent of :mod:`robust.rt.inelastic` by construction: if this
    sentinel and the pin tests below ever disagree, trust the sentinel.
    """
    wave_em = np.array([685.0])
    wave_ex = np.linspace(370.0, 690.0, 65)
    a_em, bb_em = np.array([0.5]), np.array([0.01])
    a_ex, bb_ex = np.full(65, 0.06), np.full(65, 0.01)
    aph_ex = np.full(65, 0.03)
    ed_ex, ed_em = np.ones(65), np.array([1.0])
    phi_C, mu_d, mu_f = 0.02, 0.9, 0.5

    got = np.asarray(
        bing_rrs.calc_Rrs_fluorescence(
            wave_em,
            a_em,
            bb_em,
            a_ex,
            bb_ex,
            aph_ex,
            wave_ex,
            ed_ex,
            ed_em,
            phi_C=phi_C,
            double_gaussian=False,
        )
    ).ravel()

    # The post-fix formula, by hand (Gordon 1979 / S&P98 two-flow + Lu=Eu/pi).
    bb_f = 0.5 * phi_C * aph_ex
    k_ex = (a_ex + bb_ex) / mu_d
    kappa_f = (a_em + bb_em) / mu_f
    integrand = ed_ex * (wave_ex / wave_em[0]) * (bb_f / mu_d) / (k_ex + kappa_f[0])
    r_f = np.trapezoid(integrand, wave_ex) / ed_em[0]
    rrs_f = r_f / np.pi
    h_c = 1.0 / (10.6 * np.sqrt(2.0 * np.pi))  # Gaussian peak value at 685 nm
    expected = h_c * 0.52 * rrs_f / (1.0 - 1.7 * rrs_f)

    ratio = float(got[0] / expected)
    assert ratio == pytest.approx(1.0, rel=1e-6), (
        f"BING fluorescence is x{ratio:.3f} the post-fix value; a ratio near "
        "3.14 means the BING checkout predates inelastic-fixes (missing the "
        "Lu = Eu/pi normalization) — update the editable bing checkout before "
        "trusting any pin below"
    )


# ---------------------------------------------------------------- the pins ----


def test_raman_factor_pins_to_bing(batch64):
    """f_phys == bing's calc_raman_correction_factor, every sample, rtol 1e-6.

    Both sides get the identical single-shift excitation grid, NumPy-interp'd
    excitation IOPs, bing's own b_bR(lambda'), and our Ed ratio (bing takes it
    as an input — the Ed *source* is not under test here; its packaging is
    ``test_ed.py``'s job). Measured worst over 150 x 81: 4.1e-16.
    """
    batch, iops, geometry, wave = batch64
    ours = np.asarray(I.raman_factor(iops, geometry, wave))

    wave_np = np.asarray(wave)
    lam_ex = 1.0 / (1.0 / wave_np + C.RAMAN_SHIFT * 1e-7)
    bb_r = bing_raman.raman_backscattering_coeff(lam_ex)

    for row in range(ours.shape[0]):
        a = np.asarray(iops.a)[row]
        bb = np.asarray(iops.bb)[row]
        reference = bing_rrs.calc_raman_correction_factor(
            a,
            bb,
            np.interp(lam_ex, wave_np, a),
            np.interp(lam_ex, wave_np, bb),
            bb_r,
            np.asarray(E.ratio(float(batch.zenith[row]), jnp.asarray(lam_ex), wave)),
        )
        np.testing.assert_allclose(
            ours[row], reference, rtol=RTOL_GATE, atol=0.0, err_msg=f"row {row}"
        )


def test_fluorescence_pins_to_bing(batch64):
    """phi_C * K_fl == bing's calc_Rrs_fluorescence, every sample, rtol 1e-6.

    Both sides get the identical 65-node excitation quadrature, IOPs, and
    packaged Ed, at the truth's phi_C = 0.02 — where ``K_fl = Rrs_fl(0.02)/
    0.02`` makes the equality exact by construction (record §4.3). All 81
    emission wavelengths compared, Gaussian tails included (atol=0):
    measured worst over 150 x 81 is 1.1e-13. The 'double' emission shape is
    spot-pinned in ``test_inelastic.py``; the gate here is the validated
    default.
    """
    batch, iops, geometry, wave = batch64
    ours = PHI_C_L23 * np.asarray(I.fluorescence_kernel(iops, geometry, wave))

    wave_np = np.asarray(wave)
    wave_ex = np.asarray(I.fl_excitation_grid(dtype=np.float64))

    for row in range(ours.shape[0]):
        a = np.asarray(iops.a)[row]
        bb = np.asarray(iops.bb)[row]
        aph = np.asarray(iops.a_ph)[row]
        zenith = float(batch.zenith[row])
        reference = bing_rrs.calc_Rrs_fluorescence(
            wave_np,
            a,
            bb,
            np.interp(wave_ex, wave_np, a),
            np.interp(wave_ex, wave_np, bb),
            np.interp(wave_ex, wave_np, aph),
            wave_ex,
            np.asarray(E.Ed(zenith, jnp.asarray(wave_ex))),
            np.asarray(E.Ed(zenith, wave)),
            phi_C=PHI_C_L23,
            double_gaussian=False,
        )
        np.testing.assert_allclose(
            ours[row], reference, rtol=RTOL_GATE, atol=0.0, err_msg=f"row {row}"
        )
