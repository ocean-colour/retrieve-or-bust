"""
Tests for :mod:`robust.rt.ztt` (M2 task 3) — the ZTT transcription and its gates.

Three groups, matching the milestone's three gate items.

**(i) Paper reference cases.** The strongest is the worked example in §2.1: "for
θs' = 60°, θs will be 40.3° and ψ will be 139.7° for nadir viewing". Reproducing it
exercises Snell refraction, the scattering-angle formula, *and* the nadir convention
(θv = 180° in the paper, 0° in :class:`robust.rt.types.Geometry`) in one assertion —
the exact chain where a reversed convention yields a wrong-but-plausible BRDF. The
water phase function is checked against its independently quoted 0.23 sr⁻¹, ``f_L``
against the paper's stated natural range, and the TT2017 µ∞ coefficients against
their own tabulated nodes.

**(ii) The gradient gate.** ``jax.grad`` against central finite differences, under
``jax_x64`` and with dtypes pinned on the arrays. This is a hard gate from M2 on:
it is the property the future inversion depends on. M0's notebook §4 measured why
float64 is mandatory — at a 1e-6 tolerance float32 passes at 0 of 33 step sizes.

**(iii) Behaviour, not accuracy.** Standalone rRMS is *reported* in the
implementation record, not asserted here, per the project's unbiased stance. What is
asserted is structural: that ``rrs_ZTT`` depends on solar zenith at all (Gordon
cannot), with the same sign as L23, and that its remaining over-prediction of the
zenith effect stays where it was measured.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import conventions as C
from robust.rt import ztt as Z
from robust.rt.types import Geometry, IOPs, PhaseParams
from robust.tests.conftest import needs_l23, needs_pb24

N = C.N_WAVE

#: A single constant particulate backward phase function, sr⁻¹, used only where a
#: test wants ``P_bb`` held fixed. The real angle-dependent ``Pbb,ST(ψ)`` is
#: :func:`robust.rt.ztt.P_bb_sullivan` and is what ``rrs_ZTT`` uses by default;
#: 0.14 sits mid-range of the 0.12-0.16 sr⁻¹ the synthesis quotes for particles.
P_BB_NOMINAL = 0.14


@contextlib.contextmanager
def jax_x64_enabled():
    """Enable float64 for a block, restoring the prior setting.

    A local context manager rather than the ``jax_x64`` fixture, because these two
    tests need to compare float32 *and* float64 within a single test body.
    """
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def simple_iops(a=0.15, bb_p=0.003, shape=(N,)) -> IOPs:
    return IOPs(
        a=jnp.full(shape, a),
        bb_w=jnp.broadcast_to(jnp.asarray(C.BB_W_L23), shape),
        bb_p=jnp.full(shape, bb_p),
    )


def simple_params(B_p=0.0126) -> PhaseParams:
    return PhaseParams(B_p=jnp.asarray(B_p))


# ------------------------------------------------- (i) paper reference cases --


def test_paper_worked_example_geometry():
    """§2.1: "for θs' = 60°, θs will be 40.3° and ψ will be 139.7° for nadir".

    The single most valuable check in this module. One assertion covers Snell
    refraction, the scattering-angle formula, and the nadir convention.
    """
    theta_s_water = Z.in_water_zenith(jnp.asarray(60.0))
    assert float(theta_s_water) == pytest.approx(40.3, abs=0.05)

    _, _, theta_v_paper, psi = Z.geometry_to_paper_angles(
        Geometry.nadir(jnp.asarray(60.0))
    )
    assert float(theta_v_paper) == 180.0  # the paper's nadir
    assert float(psi) == pytest.approx(139.7, abs=0.05)


def test_snell_round_trips():
    for theta in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        back = Z.above_water_zenith(Z.in_water_zenith(jnp.asarray(theta)))
        assert float(back) == pytest.approx(theta, abs=1e-4)


def test_nadir_scattering_angle_identity():
    """For nadir viewing the paper states ψ = 180° - θs (in-water)."""
    for theta_air in (0.0, 30.0, 60.0):
        ts_water, _, _, psi = Z.geometry_to_paper_angles(
            Geometry.nadir(jnp.asarray(theta_air))
        )
        assert float(psi) == pytest.approx(180.0 - float(ts_water), abs=1e-4)


def test_water_phase_function_matches_the_quoted_value():
    """βw(π)/bbw ≈ 0.23 sr⁻¹ for pure water (synthesis §3.5; Zhang 2009)."""
    assert float(Z.beta_w_over_bb_w(jnp.asarray(180.0))) == pytest.approx(
        0.23, abs=0.01
    )


def test_water_phase_function_is_normalized():
    """Integrating over the backward hemisphere returns 1, by construction.

    ``2π ∫_{90°}^{180°} [βw(ψ)/bbw] sin ψ dψ = 1`` — the closed form is only
    trustworthy if its normalization is, so it is checked numerically.
    """
    psi = np.linspace(90.0, 180.0, 200001)
    shape = np.asarray(Z.beta_w_over_bb_w(jnp.asarray(psi)))
    integral = (
        2 * np.pi * np.trapezoid(shape * np.sin(np.deg2rad(psi)), np.deg2rad(psi))
    )

    assert integral == pytest.approx(1.0, rel=1e-6)


def test_f_L_within_the_papers_natural_range():
    """§2.5: Zaneveld suggested 1.05, with a natural range of 1 to 1.12."""
    psi = jnp.asarray([139.7, 158.1, 180.0])
    values = np.asarray(Z.f_L(psi[:, None], C.canonical_wave()))

    assert values.min() > 1.0
    assert values.max() < 1.12


def test_table_A3_shape_and_ends():
    """Table A3 spans 350-800 nm at 5 nm — 91 values."""
    assert Z.FL_AVE.shape == (91,)
    assert Z.FL_AVE_WAVE[0] == 350.0
    assert Z.FL_AVE_WAVE[-1] == 800.0
    assert 0.97 < Z.FL_AVE.min() and Z.FL_AVE.max() < 1.03


def test_equation_4_is_badly_conditioned_and_the_cost_is_bounded():
    """Pins the one numerically fragile term, and how much it actually costs.

    Equation (4) is a quartic in degrees fitted over a narrow range: at ψ = 180° its
    five terms are -398, +1412, -1866, +1089, -236, cancelling to 0.0239 — a factor
    ~78,000. Float32 therefore loses ~5 significant figures in ``Ψ_KLu``.

    Asserted here rather than merely documented for two reasons: so the *size* of
    the float32 penalty is a checked number rather than a guess, and so anyone
    rewriting the evaluation (Horner, a shifted variable, precomputed powers) sees
    immediately whether they made it better or worse. The paper's other polynomials
    cancel by 1-116x and need no such guard.
    """
    terms = [c * 180.0 ** (4 - i) for i, c in enumerate(Z.FA_COEFFS)]
    cancellation = max(abs(t) for t in terms) / abs(sum(terms))
    assert cancellation > 1e4, f"expected severe cancellation, got {cancellation:.0f}x"

    f32 = float(Z.psi_KLu(jnp.asarray(180.0, dtype=jnp.float32)))
    with jax_x64_enabled():
        f64 = float(Z.psi_KLu(jnp.asarray(180.0, dtype=jnp.float64)))

    assert abs(f32 / f64 - 1) < 1e-4, "float32 penalty grew beyond its recorded size"


def test_full_model_float32_matches_float64_to_1e4(l23_small_batch):
    """The end-to-end consequence of the above: ~5e-5, not more.

    Negligible against ZTT's own 3-5% error, and irrelevant to the gradient gate
    (float64). Pinned so a future change cannot quietly make float32 unusable.
    """
    args = (
        l23_small_batch.iops,
        l23_small_batch.phase_params,
        l23_small_batch.geometry,
        l23_small_batch.wave,
    )
    r32 = np.asarray(Z.rrs_ZTT(*args))
    with jax_x64_enabled():
        r64 = np.asarray(Z.rrs_ZTT(*args))

    assert np.max(np.abs(r32 / r64 - 1)) < 1e-4


def test_psi_KLu_is_near_unity_and_rises_as_psi_falls():
    """``Ψ_KLu = K_Lu/K∞`` should be a modest correction over the useful range."""
    values = [float(Z.psi_KLu(jnp.asarray(p))) for p in (180.0, 158.0, 139.7, 134.0)]

    assert all(1.0 < v < 1.4 for v in values)
    assert values == sorted(values)  # increases as psi decreases


def test_diffuse_fraction_is_a_fraction():
    for theta in (0.0, 20.0, 40.0, 60.0, 75.0):
        H = float(Z.diffuse_fraction(jnp.asarray(theta)))
        assert 0.0 < H < 1.0


def test_mu_d_is_a_plausible_average_cosine():
    """§2.7 notes µd runs ~0.79-0.94 for sun angles 8°-62°."""
    iops = simple_iops()
    bb_over_a = float(jnp.mean(iops.bb / iops.a))
    eta = float(jnp.mean(iops.bb_w / iops.bb))

    for theta in (10.0, 30.0, 60.0):
        value = float(Z.mu_d(jnp.asarray(theta), bb_over_a, eta))
        assert 0.5 < value < 1.0


def test_tt2017_reproduces_its_own_tabulated_nodes():
    """At each tabulated η_bb, the interpolation must return that row exactly.

    Guards the transcription of Twardowski & Tonizzo (2017) Table 1 and the
    log-η interpolation that turns six rows into a surface.
    """
    for eta, (p0, p1, p2) in Z.MU_INF_TT2017_TABLE1.items():
        for bb_over_a in (1e-3, 1e-2, 1e-1):
            expected = p0 + p1 * np.log10(bb_over_a) + p2 * np.log10(bb_over_a) ** 2
            got = float(Z.mu_infinity_tt2017(bb_over_a, eta))
            assert got == pytest.approx(expected, rel=1e-5), f"eta={eta}"


def test_tt2017_mu_infinity_is_physical():
    """0 < µ∞ ≤ 1 across the fitted domain — it is an average cosine."""
    bb_over_a = np.logspace(-4, -1, 25)[:, None]
    eta = np.array([0.0098, 0.05, 0.25, 0.5, 0.9, 0.98])[None, :]

    values = np.asarray(Z.mu_infinity_tt2017(bb_over_a, eta))

    assert values.min() > 0.0
    assert values.max() <= 1.0


# ------------------------------------------------- the unpublished µ∞ gap -----


def test_equation_8_requires_its_missing_coefficients():
    """Equation (8) cannot be evaluated: the paper omits ``m1..m16``.

    The failure is deliberate and loud rather than silently substituted, so no
    downstream result can quietly claim to be the 2018 model.
    """
    with pytest.raises(NotImplementedError, match="m1..m16"):
        Z.mu_infinity(0.01, 0.5)


def test_equation_8_works_once_coefficients_are_supplied():
    """The structure is implemented and ready for the real numbers."""
    coeffs = tuple(0.01 * (i + 1) for i in range(16))

    value = Z.mu_infinity(0.01, 0.5, coeffs)

    assert np.isfinite(float(value))


def test_equation_8_rejects_a_wrong_coefficient_count():
    with pytest.raises(ValueError, match="16 coefficients"):
        Z.mu_infinity(0.01, 0.5, (1.0, 2.0, 3.0))


def test_rrs_rejects_two_conflicting_mu_inf_specifications():
    with pytest.raises(ValueError, match="at most one"):
        Z.rrs_ZTT(
            simple_iops(),
            simple_params(),
            Geometry.nadir(jnp.asarray(30.0)),
            P_bb=P_BB_NOMINAL,
            mu_inf=0.8,
            mu_inf_coeffs=tuple(range(16)),
        )


# ------------------------------------------------------------- (ii) the gate --


def fd_gradient(fn, x, h):
    """Central finite difference of a scalar ``fn`` at ``x`` with step ``h``."""
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


@pytest.mark.parametrize(
    ("name", "step"),
    [("a", 1e-6), ("bb_p", 1e-9), ("B_p", 1e-8), ("theta_s", 1e-3)],
)
def test_gradient_matches_finite_differences(jax_x64, name, step):
    """**The hard gate.** ``jax.grad`` agrees with central differences.

    Run under ``jax_x64`` with the dtype pinned on the arrays: perturbing with a
    Python float would silently compute in float64 regardless and prove nothing
    (M0's notebook §4). Step sizes are scaled to each variable's magnitude — ``a``
    is O(0.1), ``bb_p`` O(1e-3), ``B_p`` O(1e-2) — since one global ``h`` cannot
    suit all four. ``theta_s`` is O(30) and wants the *largest* step of the four:
    sweeping it shows 1e-3 gives 3e-10 relative agreement while 1e-6 degrades to
    1.3e-6 as cancellation takes over.
    """
    dtype = jnp.float64
    base = {
        "a": jnp.asarray(0.15, dtype=dtype),
        "bb_p": jnp.asarray(0.003, dtype=dtype),
        "B_p": jnp.asarray(0.0126, dtype=dtype),
        "theta_s": jnp.asarray(30.0, dtype=dtype),
    }
    wave = jnp.asarray([440.0, 550.0, 660.0], dtype=dtype)
    bb_w = jnp.asarray(C.bb_w(wave), dtype=dtype)

    def scalar(value):
        p = dict(base, **{name: value})
        iops = IOPs(
            a=jnp.full(wave.shape, p["a"], dtype=dtype),
            bb_w=bb_w,
            bb_p=jnp.full(wave.shape, p["bb_p"], dtype=dtype),
        )
        rrs = Z.rrs_ZTT(
            iops,
            PhaseParams(B_p=p["B_p"]),
            Geometry.nadir(p["theta_s"]),
            wave,
            P_bb=P_BB_NOMINAL,
        )
        return jnp.sum(rrs)

    analytic = float(jax.grad(scalar)(base[name]))
    numeric = float(fd_gradient(scalar, base[name], jnp.asarray(step, dtype=dtype)))

    assert analytic == pytest.approx(numeric, rel=1e-6), (
        f"d/d{name}: autodiff {analytic:.10e} vs finite difference {numeric:.10e}"
    )


def test_gradient_returns_a_labelled_IOPs(jax_x64):
    """``jax.grad`` over the container gives per-field derivatives, with signs.

    More absorption lowers reflectance; more particulate backscatter raises it.
    """
    iops = simple_iops(shape=(N,))

    grads = jax.grad(
        lambda x: jnp.sum(
            Z.rrs_ZTT(
                x,
                simple_params(),
                Geometry.nadir(jnp.asarray(30.0)),
                P_bb=P_BB_NOMINAL,
            )
        )
    )(iops)

    assert isinstance(grads, IOPs)
    assert float(grads.a[40]) < 0.0
    assert float(grads.bb_p[40]) > 0.0


# ------------------------------------------------------------ (iii) behaviour -


def test_rrs_is_positive_and_finite():
    rrs = Z.rrs_ZTT(
        simple_iops(),
        simple_params(),
        Geometry.nadir(jnp.asarray(30.0)),
        P_bb=P_BB_NOMINAL,
    )

    assert rrs.shape == (N,)
    assert np.all(np.isfinite(np.asarray(rrs)))
    assert float(jnp.min(rrs)) > 0.0


def mean_rrs_at(zenith, P_bb):
    return float(
        jnp.mean(
            Z.rrs_ZTT(
                simple_iops(),
                simple_params(),
                Geometry.nadir(jnp.asarray(zenith)),
                P_bb=P_bb,
            )
        )
    )


def test_rrs_depends_on_solar_zenith_unlike_gordon():
    """The structural claim that justifies the whole backbone.

    Standard Gordon returns the same answer at every sun angle (a test in
    ``test_baselines.py`` asserts that); ZTT carries a BRDF, so it must not.
    """
    values = [mean_rrs_at(z, P_BB_NOMINAL) for z in (0.0, 30.0, 60.0)]

    assert len(set(values)) == 3


def test_zenith_trend_has_the_same_sign_as_L23():
    """``rrs`` falls with solar zenith, as L23 does.

    This test failed until a transcription bug was found in ``Md_plus``: the
    paper's ``µw`` is the cosine of the **in-water** solar zenith (Equation (13)
    writes ``µw = cos(θs)``, unprimed), while ``H`` and ``P3`` take the primed
    above-water angle. Using the above-water cosine throughout inverted the
    modelled zenith trend and cost ~13 percentage points of rRMS at 60°.

    What exposed it was not this test but the µd check below, which compares
    against an independently quoted number. Worth remembering: a plausible
    *diagnostic* had already convinced me the cause was a missing ``Pbb(ψ)``.
    """
    values = [mean_rrs_at(z, None) for z in (0.0, 30.0, 60.0)]

    assert values[2] < values[1] < values[0]


def test_zenith_trend_magnitude_is_over_predicted():
    """Pins what is still imperfect, so it cannot quietly drift.

    With the ``µw`` fix and the real ``Pbb,ST(ψ)``, the 60°/0° ratio comes out
    ~0.87 against L23's **0.949** — right sign, too strong by roughly a factor
    three in the departure from unity. Recorded as a number rather than described,
    so M3's residual emulator has a stated starting point and any future change
    here is visible.
    """
    ratio = mean_rrs_at(60.0, None) / mean_rrs_at(0.0, None)

    assert 0.83 < ratio < 0.91, f"60/0 ratio {ratio:.4f}"


def test_mu_d_matches_the_papers_quoted_range():
    """§2.7: "for sun angles between 8° and 62°, µd varied only from 0.79 to 0.94".

    **The check that found the ``µw`` bug.** Equation (13)'s ``µd`` reproduces
    those two endpoints only with the in-water cosine; the above-water cosine
    gives 0.573 at 62°, far outside. An independently quoted number caught what a
    physically plausible fitted diagnostic had not.
    """
    for theta_air, expected in ((8.0, 0.94), (62.0, 0.79)):
        mu_w = jnp.cos(jnp.deg2rad(Z.in_water_zenith(jnp.asarray(theta_air))))
        mu_d_eq13 = 1.0 / (0.6 / mu_w + 0.4 / 0.859)
        assert float(mu_d_eq13) == pytest.approx(expected, abs=0.01)


# ------------------------------------------------- Sullivan & Twardowski Pbb --


def test_P_bb_sullivan_reproduces_table_1():
    """The polynomial matches the measured average it was fitted to.

    Sullivan & Twardowski (2009) Table 1 lists ``β̃bp`` at 90-170°; Table 2 gives
    the fourth-order fit, claimed within "<0.5%".
    """
    got = np.asarray(Z.P_bb_sullivan(jnp.asarray(Z.P_BB_ST_ANGLES)))

    np.testing.assert_allclose(got, Z.P_BB_ST_MEAN, atol=0.003)


def test_P_bb_sullivan_uses_the_corrected_a3_exponent():
    """Table 2 prints ``a3 = 8.007E−02``, which cannot be right.

    At ψ = 140° that term alone would contribute ``19600 × 0.08 ≈ 1570`` against a
    tabulated value of 0.137. Refitting Table 1 independently gives
    ``a3 ≈ 7.8e-4``, and the other four published coefficients agree closely, so
    the intended value is ``8.007E−04``. This test pins the corrected exponent and
    the resulting physical magnitude.
    """
    assert Z.P_BB_ST_COEFFS[2] == pytest.approx(8.007e-04)
    assert 0.13 < float(Z.P_bb_sullivan(jnp.asarray(140.0))) < 0.14


def test_P_bb_sullivan_has_the_measured_shape():
    """Falls steeply from 90°, minimum near 140°, then rises toward backscatter."""
    values = np.asarray(
        Z.P_bb_sullivan(jnp.asarray([90.0, 120.0, 140.0, 160.0, 180.0]))
    )

    assert values[0] > values[1] > values[2]  # falling to the minimum
    assert values[2] < values[3] < values[4]  # then rising to 180 deg
    assert float(Z.P_bb_sullivan(jnp.asarray(180.0))) == pytest.approx(0.153, abs=0.005)


def test_rrs_increases_with_the_backward_phase_function():
    """``Pbb(ψ)`` is the numerator of Equation (10): more backward scatter, more rrs."""
    args = (simple_iops(), simple_params(), Geometry.nadir(jnp.asarray(30.0)))

    low = float(jnp.mean(Z.rrs_ZTT(*args, P_bb=0.12)))
    high = float(jnp.mean(Z.rrs_ZTT(*args, P_bb=0.16)))

    assert high > low


def test_rrs_is_jittable_and_batched():
    iops = simple_iops(shape=(7, N))
    geometry = Geometry.nadir(jnp.asarray(np.full(7, 30.0)))
    params = PhaseParams(B_p=jnp.full((7, N), 0.0126))

    jitted = jax.jit(lambda i, p, g: Z.rrs_ZTT(i, p, g, P_bb=P_BB_NOMINAL))
    out = jitted(iops, params, geometry)

    assert out.shape == (7, N)
    assert np.all(np.isfinite(np.asarray(out)))


def test_Rrs_is_rrs_through_the_interface():
    args = (simple_iops(), simple_params(), Geometry.nadir(jnp.asarray(30.0)))

    np.testing.assert_allclose(
        np.asarray(Z.Rrs_ZTT(*args, P_bb=P_BB_NOMINAL)),
        np.asarray(C.rrs_to_Rrs(Z.rrs_ZTT(*args, P_bb=P_BB_NOMINAL))),
        rtol=1e-6,
    )


def test_runs_on_the_cached_l23_fixture(l23_small_batch):
    """Real IOPs, in CI: the model produces finite positive rrs everywhere."""
    rrs = Z.rrs_ZTT(
        l23_small_batch.iops,
        l23_small_batch.phase_params,
        l23_small_batch.geometry,
        l23_small_batch.wave,
        P_bb=P_BB_NOMINAL,
    )

    assert rrs.shape == l23_small_batch.Rrs.shape
    assert np.all(np.isfinite(np.asarray(rrs)))
    assert float(jnp.min(rrs)) > 0.0


# ------------------------ ZTT's internals against HydroLight (M5 task 13) ----
# Three questions, and the answers are more useful than the milestone expected:
# where mu_d stands, why the backbone collapses on PB24, and whether the standing
# Equation-(8) caveat can be closed with this data. (It cannot.)


def test_F_psi_goes_negative_below_the_range_its_paper_fitted():
    """**The dominant cause of the backbone's collapse on PB24.**

    ``F_psi`` is a quartic in the in-water scattering angle, and its own docstring
    records the paper's fitted range: ψ ≳ 134°, ">95% of the angles a polar
    orbiter sees". Below that it is extrapolation, and ``Ψ_KLu = 1 + F(ψ)`` turns
    negative — which flips the sign of the ZTT denominator's leading term and with
    it the whole model. Nadir viewing pins ψ near backscatter, so L23 could never
    have exposed this.
    """
    psi = np.linspace(40.0, 180.0, 1401)
    psi_k = np.asarray(Z.psi_KLu(jnp.asarray(psi)))

    assert np.all(psi_k[psi >= 134.0] > 0.0), "positive throughout the fitted range"

    # It is negative for *everything* below the crossing, not on some interval:
    # the quartic simply has no business being evaluated there.
    negative = psi[psi_k < 0.0]
    assert negative.size > 0
    assert negative.min() == pytest.approx(psi.min())
    assert 105.0 < negative.max() < 115.0, (
        f"psi_KLu crosses zero at {negative.max():.1f} deg; the record quotes "
        "~110, comfortably below the paper's fitted range of psi >~ 134"
    )


@needs_l23
def test_l23_geometry_never_leaves_the_fitted_scattering_range(l23_batch):
    """Why the prototype could not have found this. Nadir pins ψ near 180°."""
    _, _, _, psi = Z.geometry_to_paper_angles(l23_batch.geometry)

    assert float(jnp.min(psi)) > 134.0


@needs_pb24
def test_pb24_leaves_the_fitted_scattering_range_on_much_of_the_window():
    """And why PB24 does: a full BRDF sweeps ψ through 90°, not just backscatter."""
    from robust.rt.data import pb24 as P

    batch = P.load_batch(realisations=4, angles="all")
    _, _, _, psi = Z.geometry_to_paper_angles(batch.geometry)
    psi = np.asarray(psi)
    window = (batch.theta_s <= 70.0) & (batch.theta_v <= 70.0)

    assert psi.min() < 60.0  # the full grid reaches deep forward scattering
    outside = float(np.mean(psi[window] < 134.0))
    assert 0.3 < outside < 0.6, (
        f"{100 * outside:.0f}% of the sanctioned window is outside ZTT's fitted "
        "scattering range; the record quotes ~42%"
    )


@needs_pb24
def test_mu_d_against_pb24s_tabulated_value():
    """**The gate's measurable half.** Pinned so a change to ``mu_d`` announces itself.

    The 2018 paper puts Equation (14)'s error below 1%. Against PB24 the median
    is ~4% and it degrades with solar zenith — ~1.6% at 40°, ~14% at 70-80°. That
    is a real disagreement rather than a defect here: ZTT's ``Md_star`` is itself
    a fit over a narrower IOP range than PB24 spans.
    """
    from robust.rt.data import pb24 as P

    batch = P.load_batch(realisations=25, angles="all", extras=("mu_d",))
    bb_over_a = batch.iops.bb / batch.iops.a
    eta = batch.iops.bb_w / batch.iops.bb
    _, theta_s_air, _, _ = Z.geometry_to_paper_angles(batch.geometry)

    ours = np.asarray(Z.mu_d(jnp.asarray(theta_s_air)[:, None], bb_over_a, eta))
    theirs = np.asarray(batch.aops["mu_d"])
    rel = np.abs(ours / theirs - 1.0)

    assert 0.02 < np.median(rel) < 0.08, f"median {np.median(rel) * 100:.2f}%"

    # and it is worse at large solar zenith, which is the shape of the finding
    at_40 = float(np.median(rel[batch.theta_s == 40.0]))
    at_70 = float(np.median(rel[batch.theta_s == 70.0]))
    assert at_70 > 3 * at_40


@needs_pb24
def test_mu_infinity_cannot_be_refit_from_pb24():
    """**Q17 option 3 is closed, and this is why.**

    µ∞ is the *asymptotic* mean cosine: by definition the light field at depth has
    forgotten the boundary, so µ∞ = a / K∞ with K∞ independent of the solar
    zenith. PB24 tabulates seven K's — and **every one of them varies by ~1.4x
    across solar zenith**, so none of them is K∞. There is therefore no asymptotic
    quantity in this dataset from which µ∞ could be derived, and the standing
    Equation-(8) caveat cannot be closed with it.
    """
    from robust.rt.data import pb24 as P

    batch = P.load_batch(
        realisations=8,
        angles="window",
        geometry_stride=(1, 8, 13),
        extras=("Kd", "Ku", "Ko", "Kod", "Kou", "Knet", "KLu"),
    )
    assert len(set(batch.theta_s.tolist())) >= 6, "need several zeniths to test this"

    for name, values in batch.aops.items():
        if not name.startswith("K"):
            continue
        K = np.asarray(values)
        spreads = []
        for realisation in np.unique(batch.realisation):
            rows = K[batch.realisation == realisation]
            spreads.append(rows.max(axis=0) / np.maximum(rows.min(axis=0), 1e-30))
        spread = float(np.median(np.concatenate(spreads)))

        assert spread > 1.1, (
            f"{name} barely varies with solar zenith ({spread:.3f}) -- if it is "
            "asymptotic after all, mu_infinity could be refit from it and Q17's "
            "option 3 reopens"
        )
