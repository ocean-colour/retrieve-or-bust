"""
Tests for :mod:`robust.rt.inelastic_corr` — M3 task 1, the head machinery.

No trained weights exist yet (task 2), so what this module gates is the
*machinery*, on the properties that must hold regardless of training:

- **Zero-init is the analytic model.** A fresh head's δ is exactly 0
  (zero-initialised output layer), so the corrected forward equals the
  analytic forward — training starts from the physics.
- **The bound binds.** |δ| < ``delta_max`` whatever the parameters.
- **δ_F cannot see φ_C.** Its feature list has no yield column, and the
  composed forward stays exactly linear in φ_C with a (randomised) head
  active — the design §4.4 promise, now enforced against the M3 code.
- **The fallback warns once, then behaves.** Absent weight files:
  ``corrections=None`` degrades to analytic-only behind a single
  ``MissingCorrectionWarning``; ``corrections=False`` is the same numbers,
  silent.
- **Round trip + refusal.** ``save_head``/``load_head`` reproduce the head;
  a file whose feature list mismatches the code is refused, not warned.

The held-out ≤ 5 % gates arrive with task 3, against the *committed*
weights.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from robust.rt import hybrid as H
from robust.rt import inelastic as I
from robust.rt import inelastic_corr as IC
from robust.rt import validation as V
from robust.rt.types import Inelastic

#: One fixed configuration per head kind, small and fast.
RAMAN_CONFIG = IC.HeadConfig("raman", delta_max=1.0)
FL_CONFIG = IC.HeadConfig("fl", delta_max=0.5)


def randomized(head, scale=3.0, seed=7):
    """The head with its parameters replaced by noise — a nonzero δ.

    ``init_head`` is deliberately a zero function, so bound/effect tests
    need parameters that actually fire; scale 3 pushes the tanh well into
    its nonlinear range.
    """
    leaves, treedef = jax.tree_util.tree_flatten(head.params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    noisy = [
        scale * jax.random.normal(k, leaf.shape, leaf.dtype)
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return dataclasses.replace(
        head, params=jax.tree_util.tree_unflatten(treedef, noisy)
    )


def batch_args(batch):
    return (batch.iops, batch.phase_params, batch.geometry, batch.wave)


# ---------------------------------------------------------------- features ----


def test_feature_lists_are_the_designs():
    """§4.5 features, in order — and no φ_C anywhere near δ_F."""
    assert IC.RAMAN_FEATURES == (
        "log10_a_em",
        "log10_bb_em",
        "log10_a_ex",
        "log10_bb_ex",
        "cos_theta_s",
        "wave",
    )
    assert IC.FL_FEATURES == (
        "log10_a_ph440",
        "log10_a_em",
        "log10_bb_em",
        "log10_a_490",
        "cos_theta_s",
        "wave",
    )
    assert not any("phi" in name.lower() for name in IC.FL_FEATURES)


def test_features_shapes_and_finiteness(l23_small_inelastic_batch):
    """(..., n_wave, 6) per head, finite everywhere on the fixture."""
    batch = l23_small_inelastic_batch
    for fn in (IC.features_raman, IC.features_fl):
        x = np.asarray(fn(batch.iops, batch.geometry, batch.wave))
        assert x.shape == (*np.asarray(batch.iops.a).shape, 6)
        assert np.all(np.isfinite(x))


def test_features_fl_requires_aph(l23_small_inelastic_batch):
    """The same physical requirement as the kernel, same loud error."""
    batch = l23_small_inelastic_batch
    iops = dataclasses.replace(batch.iops, a_ph=None)
    with pytest.raises(ValueError, match="a_ph"):
        IC.features_fl(iops, batch.geometry, batch.wave)


# ------------------------------------------------------------------- heads ----


@pytest.mark.parametrize("config", [RAMAN_CONFIG, FL_CONFIG], ids=["raman", "fl"])
def test_fresh_head_is_exactly_zero(config, l23_small_inelastic_batch):
    """Zero-init output layer → δ ≡ 0: training starts from the physics."""
    batch = l23_small_inelastic_batch
    head = IC.init_head(config.kind, config)
    delta = np.asarray(head.delta(batch.iops, batch.geometry, batch.wave))
    assert np.all(delta == 0.0)


@pytest.mark.parametrize("config", [RAMAN_CONFIG, FL_CONFIG], ids=["raman", "fl"])
def test_delta_is_bounded(config, l23_small_inelastic_batch):
    """|δ| < delta_max for arbitrary (noise) parameters — the tanh bound."""
    batch = l23_small_inelastic_batch
    head = randomized(IC.init_head(config.kind, config))
    delta = np.asarray(head.delta(batch.iops, batch.geometry, batch.wave))
    assert np.all(np.isfinite(delta))
    assert np.abs(delta).max() < config.delta_max
    assert np.abs(delta).max() > 0.0  # the noise actually fired


def test_corrected_factor_form(l23_small_inelastic_batch):
    """f_R = 1 + (f_phys − 1)(1 + δ): identity at δ=0, →1 with the increment."""
    batch = l23_small_inelastic_batch
    f_phys = I.raman_factor(batch.iops, batch.geometry, batch.wave)
    assert np.array_equal(
        np.asarray(IC.corrected_raman_factor(jnp.zeros(()), f_phys)),
        np.asarray(f_phys),
    )
    # Where the increment is zero, no delta can move the factor off 1.
    ones = jnp.ones_like(f_phys)
    corrected = IC.corrected_raman_factor(jnp.full((), 0.9), ones)
    assert np.array_equal(np.asarray(corrected), np.asarray(ones))


# ------------------------------------------------------------------ wiring ----


def test_zero_heads_equal_analytic_forward(l23_small_inelastic_batch):
    """forward with fresh heads == forward with corrections=False.

    rtol 1e-6, not exact: (f−1)+1 == f is exact only below 2 (Sterbenz),
    and the fixture's Raman factor stays under 1.4 — but the guarantee
    worth pinning is agreement, not bit-identity.
    """
    batch = l23_small_inelastic_batch
    heads = IC.CorrectionHeads(
        raman=IC.init_head("raman", RAMAN_CONFIG), fl=IC.init_head("fl", FL_CONFIG)
    )
    analytic = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=False,
            check_domain=False,
        )
    )
    corrected = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=heads,
            check_domain=False,
        )
    )
    np.testing.assert_allclose(corrected, analytic, rtol=1e-6)


def test_random_heads_move_the_right_bands(l23_small_inelastic_batch):
    """Each head reaches its own term: Raman across the spectrum, fl at 685."""
    batch = l23_small_inelastic_batch
    wave = np.asarray(batch.wave)
    i550 = int(np.abs(wave - 550.0).argmin())
    i685 = int(np.abs(wave - 685.0).argmin())
    analytic = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=False,
            check_domain=False,
        )
    )

    raman_only_head = IC.CorrectionHeads(
        raman=randomized(IC.init_head("raman", RAMAN_CONFIG))
    )
    moved = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=raman_only_head,
            check_domain=False,
        )
    )
    assert not np.allclose(moved[:, i550], analytic[:, i550], rtol=1e-4)

    fl_only_head = IC.CorrectionHeads(fl=randomized(IC.init_head("fl", FL_CONFIG)))
    moved = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(),
            corrections=fl_only_head,
            check_domain=False,
        )
    )
    assert not np.allclose(moved[:, i685], analytic[:, i685], rtol=1e-4)
    # ... and only the additive line: far from 685 the fl head is inert.
    np.testing.assert_allclose(moved[:, i550], analytic[:, i550], rtol=1e-6)


def test_phi_C_linearity_survives_a_live_fl_head(l23_small_inelastic_batch):
    """The §4.4 promise against the M3 code: δ_F never breaks the yield.

    With a randomised (nonzero) fl head active, the fluorescence term is
    still exactly proportional to φ_C — doubling the yield doubles the
    delta over the raman-only model.
    """
    batch = l23_small_inelastic_batch
    heads = IC.CorrectionHeads(fl=randomized(IC.init_head("fl", FL_CONFIG)))
    base = np.asarray(
        H.forward(
            *batch_args(batch),
            inelastic=Inelastic(fluorescence=False),
            corrections=False,
            check_domain=False,
        )
    )

    def with_phi(phi):
        return np.asarray(
            H.forward(
                *batch_args(batch),
                inelastic=Inelastic(phi_C=jnp.asarray(phi)),
                corrections=heads,
                check_domain=False,
            )
        )

    i685 = int(np.abs(np.asarray(batch.wave) - 685.0).argmin())
    delta_1x = with_phi(0.02)[:, i685] - base[:, i685]
    delta_2x = with_phi(0.04)[:, i685] - base[:, i685]
    np.testing.assert_allclose(delta_2x, 2.0 * delta_1x, rtol=1e-4)


def test_gradients_flow_through_heads(l23_small_inelastic_batch):
    """grad w.r.t. head parameters and w.r.t. inputs, both finite.

    The parameter gradient is what task 2 trains on; the input gradient is
    the inversion's path through the corrected model.
    """
    batch = l23_small_inelastic_batch
    row = jax.tree_util.tree_map(lambda x: x[75:76], batch.iops)
    pp = jax.tree_util.tree_map(lambda x: x[75:76], batch.phase_params)
    geom = jax.tree_util.tree_map(lambda x: x[75:76], batch.geometry)
    heads = IC.CorrectionHeads(
        raman=randomized(IC.init_head("raman", RAMAN_CONFIG)),
        fl=randomized(IC.init_head("fl", FL_CONFIG)),
    )

    def scalar_of_heads(hd):
        return H.rrs_forward(
            row,
            pp,
            geom,
            batch.wave,
            inelastic=Inelastic(),
            corrections=hd,
            check_domain=False,
        ).sum()

    g_heads = jax.grad(scalar_of_heads)(heads)
    for leaf in jax.tree_util.tree_leaves(g_heads):
        assert np.all(np.isfinite(np.asarray(leaf)))

    g_iops = jax.grad(
        lambda io: H.rrs_forward(
            io,
            pp,
            geom,
            batch.wave,
            inelastic=Inelastic(),
            corrections=heads,
            check_domain=False,
        ).sum()
    )(row)
    for leaf in (g_iops.a, g_iops.bb_p, g_iops.a_ph):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_corrected_forward_jits(l23_small_inelastic_batch):
    """The corrected path compiles; eager and compiled agree."""
    batch = l23_small_inelastic_batch
    heads = IC.CorrectionHeads(
        raman=randomized(IC.init_head("raman", RAMAN_CONFIG)),
        fl=randomized(IC.init_head("fl", FL_CONFIG)),
    )

    def f(iops, pp, geom, hd):
        return H.rrs_forward(
            iops,
            pp,
            geom,
            batch.wave,
            "ztt",
            inelastic=Inelastic(),
            corrections=hd,
        )

    eager = np.asarray(f(*batch_args(batch)[:3], heads))
    jitted = np.asarray(jax.jit(f)(*batch_args(batch)[:3], heads))
    np.testing.assert_allclose(jitted, eager, rtol=1e-6)


# ------------------------------------------------------- weights + fallback ----


@pytest.mark.parametrize("config", [RAMAN_CONFIG, FL_CONFIG], ids=["raman", "fl"])
def test_save_load_round_trip(config, tmp_path, l23_small_inelastic_batch):
    """A saved head reloads to the same δ on real inputs."""
    batch = l23_small_inelastic_batch
    head = randomized(IC.init_head(config.kind, config))
    path = tmp_path / "head.npz"
    IC.save_head(head, path)
    loaded = IC.load_head(path)
    assert loaded.config == head.config
    np.testing.assert_array_equal(
        np.asarray(loaded.delta(batch.iops, batch.geometry, batch.wave)),
        np.asarray(head.delta(batch.iops, batch.geometry, batch.wave)),
    )


def test_load_refuses_feature_mismatch(tmp_path):
    """Weights against the wrong feature vector are refused, not warned."""
    head = IC.init_head("raman", RAMAN_CONFIG)
    path = tmp_path / "head.npz"
    IC.save_head(head, path)

    import numpy as _np

    data = dict(_np.load(path, allow_pickle=False))
    data["features"] = _np.asarray(("something", "else"))
    _np.savez(path, **data)
    with pytest.raises(ValueError, match="features"):
        IC.load_head(path)


def test_default_resolution_warns_once_and_is_analytic(l23_small_inelastic_batch):
    """Absent weights: corrections=None == corrections=False, plus one warning.

    ``load_default`` is memoised, so the warning is once per process — the
    cache is cleared here to make the test order-independent, and cleared
    again afterwards so this test does not eat another test's warning.
    Skipped once trained weights are committed (task 2): the fallback path
    then no longer exists to test.
    """
    if IC.DEFAULT_RAMAN_WEIGHTS.exists() and IC.DEFAULT_FL_WEIGHTS.exists():
        pytest.skip("trained weights are committed; the fallback path is gone")
    batch = l23_small_inelastic_batch

    IC.load_default.cache_clear()
    try:
        with pytest.warns(IC.MissingCorrectionWarning, match="ANALYTIC-ONLY"):
            default = np.asarray(
                H.forward(*batch_args(batch), inelastic=Inelastic(), check_domain=False)
            )
        explicit = np.asarray(
            H.forward(
                *batch_args(batch),
                inelastic=Inelastic(),
                corrections=False,
                check_domain=False,
            )
        )
        np.testing.assert_array_equal(default, explicit)
    finally:
        IC.load_default.cache_clear()


def test_elastic_path_never_resolves_corrections(l23_small_inelastic_batch):
    """inelastic=None (and all-off) never touch the loader — no warning.

    The elastic path owes nothing to the ML stack; a warning about missing
    correction weights on a purely elastic call would be noise about
    physics the caller never asked for.
    """
    batch = l23_small_inelastic_batch
    IC.load_default.cache_clear()
    try:
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", IC.MissingCorrectionWarning)
            H.forward(*batch_args(batch), check_domain=False)
            H.forward(
                *batch_args(batch),
                inelastic=Inelastic(raman=False, fluorescence=False),
                check_domain=False,
            )
    finally:
        IC.load_default.cache_clear()


# ------------------------------------------------ the held-out gates (task 3) ----
#
# From here down: the committed weights against the full release — the M3
# acceptance numbers. No train-at-test-time (load_default only); the full-
# release tests skip without $OS_COLOR (CI), while the fixture-level weight
# regression and the FD gradient gate run everywhere the repo does.

from robust.rt.data import l23 as L  # noqa: E402
from robust.tests.conftest import needs_l23_inelastic  # noqa: E402

#: The M3 per-process acceptance bar (coding plan / design DQ6).
GATE = 0.05

#: The committed weights; every test below is meaningless without them.
needs_weights = pytest.mark.skipif(
    not (IC.DEFAULT_RAMAN_WEIGHTS.exists() and IC.DEFAULT_FL_WEIGHTS.exists()),
    reason="committed correction weights missing — run "
    "design/py/train_inelastic_corr.py",
)


@pytest.fixture(scope="module")
def full_release():
    """The full L23 inelastic batch + splits + terms + loaded heads, once.

    Module-scoped: loading six netCDFs and evaluating both analytic terms on
    9960 x 81 costs ~15 s, and three tests share it.
    """
    batch = L.load_inelastic_batch()
    splits = L.make_splits(batch)
    heads = IC.load_default()
    f_phys = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    k_fl = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    return batch, splits, heads, f_phys, k_fl


def median_increment_errors(f_model, f_truth, batch, wave, mask, band):
    """Median (f-1)/(truth-1) - 1 per zenith over a wavelength band.

    Since M4 this delegates to :func:`robust.rt.validation.median_increment_error`
    — the same definition the protocol and ``run_validation.py`` report, so the
    gate here and the number in the M4 table cannot drift apart.
    """
    return V.median_increment_error(
        f_model[mask], f_truth[mask], batch.zenith[mask], band
    )


@needs_weights
@needs_l23_inelastic
def test_heldout_raman_gate(full_release):
    """**The M3 Raman gate**: median |increment error| <= 5 % per zenith.

    Held-out scenes only, 550-700 nm, every zenith *including 0 deg* — the
    line the analytic backbone fails by -38.6 % and delta_R earns. Measured
    at training time: -0.14 / -0.10 / -0.21 % (0/30/60 deg); asserted at the
    gate bar, not at the measured values — the gate is the promise, the
    training log is the achievement. The 490 nm row (analytic +32.6 % at
    60 deg, corrected +1.0 %) rides along at the same bar.
    """
    batch, splits, heads, f_phys, _ = full_release
    wave = np.asarray(batch.wave)
    delta = np.asarray(heads.raman.delta(batch.iops, batch.geometry, batch.wave))
    f_corr = np.asarray(IC.corrected_raman_factor(delta, f_phys))
    truth = np.asarray(batch.truth_raman_factor)

    band = (wave >= 550.0) & (wave <= 700.0)
    for zenith, err in median_increment_errors(
        f_corr, truth, batch, wave, splits.scene_test, band
    ).items():
        assert abs(err) <= GATE, f"zenith {zenith}: 550-700 nm {err:+.4f}"

    at490 = np.abs(wave - 490.0) < 1e-6
    for zenith, err in median_increment_errors(
        f_corr, truth, batch, wave, splits.scene_test, at490
    ).items():
        assert abs(err) <= GATE, f"zenith {zenith}: 490 nm {err:+.4f}"


@needs_weights
@needs_l23_inelastic
def test_heldout_fluorescence_gate(full_release):
    """**The M3 fluorescence gate**: median |685 nm error| <= 5 % per zenith.

    Held-out scenes, phi_C = 0.02 (the truth's). The analytic backbone sits
    at -13.7 % at 60 deg on the full release; measured corrected values are
    +0.08 / +0.07 / +0.10 %.
    """
    batch, splits, heads, _, k_fl = full_release
    wave = np.asarray(batch.wave)
    i685 = int(np.abs(wave - 685.0).argmin())
    delta = np.asarray(heads.fl.delta(batch.iops, batch.geometry, batch.wave))
    model = L.PHI_C_L23 * k_fl * (1.0 + delta)
    truth = np.asarray(batch.truth_fluorescence)

    # The shared protocol metric (robust.rt.validation.peak_ratio_error), so the
    # gate and the M4 table report the same quantity.
    held = splits.scene_test
    for zenith, err in V.peak_ratio_error(
        model[held], truth[held], batch.zenith[held], i685
    ).items():
        assert abs(err) <= GATE, f"zenith {zenith}: 685 nm {err:+.4f}"


@needs_weights
@needs_l23_inelastic
def test_loaded_heads_respect_their_bounds(full_release):
    """|delta| < delta_max on the whole release — the tanh bound, loaded.

    Also records the headroom that matters: training measured max |delta_R|
    = 0.905 against its 1.0 bound, so this doubles as a canary — a retrain
    that saturates the bound shows up here before it shows up as a silently
    clipped correction.
    """
    batch, _, heads, _, _ = full_release
    for head in (heads.raman, heads.fl):
        delta = np.asarray(head.delta(batch.iops, batch.geometry, batch.wave))
        assert np.all(np.isfinite(delta))
        assert np.abs(delta).max() < head.config.delta_max


@needs_weights
def test_committed_weights_regression(l23_small_inelastic_batch):
    """The corrected model on the CI fixture — the weights-integrity pin.

    Runs everywhere (no $OS_COLOR): the fixture mixes train and held-out
    scenes, so this is not the acceptance gate — it is the regression that
    catches a corrupt, stale, or accidentally-reverted weight file. Bands at
    2 %, ~10x the measured ~0.2 % medians, far under the analytic errors
    (-39 %, -14 %) any weight failure would reintroduce.
    """
    batch = l23_small_inelastic_batch
    wave = np.asarray(batch.wave)
    heads = IC.load_default()
    assert heads.raman is not None and heads.fl is not None

    f_phys = np.asarray(I.raman_factor(batch.iops, batch.geometry, batch.wave))
    delta_r = np.asarray(heads.raman.delta(batch.iops, batch.geometry, batch.wave))
    f_corr = np.asarray(IC.corrected_raman_factor(delta_r, f_phys))
    truth_r = np.asarray(batch.truth_raman_factor)
    band = (wave >= 550.0) & (wave <= 700.0)

    k_fl = np.asarray(I.fluorescence_kernel(batch.iops, batch.geometry, batch.wave))
    delta_f = np.asarray(heads.fl.delta(batch.iops, batch.geometry, batch.wave))
    model_fl = L.PHI_C_L23 * k_fl * (1.0 + delta_f)
    truth_fl = np.asarray(batch.truth_fluorescence)
    i685 = int(np.abs(wave - 685.0).argmin())

    for zenith in (0.0, 30.0, 60.0):
        rows = batch.zenith == zenith
        err_r = (
            np.median((f_corr[rows][:, band] - 1.0) / (truth_r[rows][:, band] - 1.0))
            - 1.0
        )
        err_f = np.median(model_fl[rows, i685] / truth_fl[rows, i685]) - 1.0
        assert abs(err_r) <= 0.02, f"zenith {zenith}: Raman {err_r:+.4f}"
        assert abs(err_f) <= 0.02, f"zenith {zenith}: fluorescence {err_f:+.4f}"


@needs_weights
@pytest.mark.parametrize(
    ("name", "step"),
    [
        ("a", 1e-6),
        ("bb_p", 1e-9),
        ("a_ph", 1e-8),
        ("phi_C", 1e-6),
        ("theta_s", 1e-3),
    ],
)
def test_gradient_matches_finite_differences_corrected(
    jax_x64, l23_small_inelastic_batch, name, step
):
    """The M2 FD protocol through the *corrected* forward — heads in the path.

    Same shape as ``test_inelastic.py``'s gate (float64, per-variable steps,
    theta_s at 35 deg — off the piecewise-linear Ed anchors where the
    theta-derivative is one-sided) with ``corrections=load_default()``, so
    the differentiation path the future inversion uses — through both tanh
    heads and their standardisations — is the one pinned.
    """
    from robust.rt.types import Geometry, IOPs, PhaseParams

    batch = l23_small_inelastic_batch
    heads = IC.load_default()
    dtype = jnp.float64
    rows = np.where(batch.zenith == 30.0)[0][:3]

    a0 = jnp.asarray(np.asarray(batch.iops.a)[rows], dtype=dtype)
    bb_w0 = jnp.asarray(np.asarray(batch.iops.bb_w)[rows], dtype=dtype)
    bb_p0 = jnp.asarray(np.asarray(batch.iops.bb_p)[rows], dtype=dtype)
    a_ph0 = jnp.asarray(np.asarray(batch.iops.a_ph)[rows], dtype=dtype)
    B_p0 = jnp.asarray(np.asarray(batch.phase_params.B_p)[rows], dtype=dtype)
    theta0 = jnp.asarray(np.asarray(batch.geometry.theta_s)[rows] + 5.0, dtype=dtype)
    wave = jnp.asarray(np.asarray(batch.wave), dtype=dtype)
    phi0 = jnp.asarray(L.PHI_C_L23, dtype=dtype)

    def scalar(shift):
        offsets = dict.fromkeys(("a", "bb_p", "a_ph", "phi_C", "theta_s"), 0.0)
        offsets[name] = shift
        iops = IOPs(
            a=a0 + offsets["a"],
            bb_w=bb_w0,
            bb_p=bb_p0 + offsets["bb_p"],
            a_ph=a_ph0 + offsets["a_ph"],
        )
        return jnp.mean(
            H.forward(
                iops,
                PhaseParams(B_p=B_p0),
                Geometry.nadir(theta0 + offsets["theta_s"]),
                wave,
                "hybrid",
                inelastic=Inelastic(phi_C=phi0 + offsets["phi_C"]),
                corrections=heads,
                check_domain=False,
            )
        )

    analytic = float(jax.grad(scalar)(jnp.asarray(0.0, dtype=dtype)))
    h = jnp.asarray(step, dtype=dtype)
    numeric = float((scalar(h) - scalar(-h)) / (2.0 * h))

    assert np.isfinite(numeric), f"d/d{name}: step {step:g} left the domain"
    assert analytic == pytest.approx(numeric, rel=1e-6), (
        f"d/d{name}: autodiff {analytic:.10e} vs finite difference {numeric:.10e}"
    )
