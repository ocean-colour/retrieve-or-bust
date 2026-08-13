"""
Residual emulator ΔRrs — the learned half of the hybrid (M3).

A small Flax MLP trained (Optax) on ``Δrrs = rrs_reference − rrs_ZTT``. It learns
only what the analytic backbone misses, which is why it can stay small and why its
extrapolation is bounded — the whole argument for the hybrid over a wholly learned
model.

Four decisions shape this module; each is a measurement or a proof, not a taste.

**1. The correction is relative, not absolute.** The design writes the hybrid as
``Rrs = Rrs_ZTT + ΔRrs``, and it stays additive here — but the network predicts a
dimensionless ``δ(λ)`` and the correction is ``Δrrs = δ · rrs_ZTT``. The reason is
dynamic range: ``rrs`` runs from ~2.5e-2 in the blue to ~6e-6 in the red, so a net
emitting absolute sr⁻¹ would have to span four decades with one set of weights, and
the relative loss (:func:`robust.rt.validation.rrms`) would weight its red-end
errors ~4000× more heavily than its blue-end ones. The *relative* residual, by
contrast, is an O(1) quantity: M2 measured mean **+2.20%**, sd **5.52%** over all
9960 L23 samples. That is what this net predicts, and ``δ`` is directly the number
the design asks to be kept small.

**2. The features are the complete dimensionless state, not a guess.** ``rrs_ZTT``
is *scale-invariant*: multiplying ``(a, bb_w, bb_p)`` by any ``k`` leaves it
unchanged to machine precision (verified to 8.8e-15 at ``k = 10``; a test pins it).
So the backbone sees its inputs only through the ratios, and two numbers span them
— ``u = bb/(a+bb)`` and ``η_bb = bb_w/bb`` invert back to ``(a : bb_w : bb_p)``
exactly. With ``B_p``, the geometry, and λ that is the *whole* input state, so the
emulator is not being starved of anything the backbone knew. Absolute magnitudes are
deliberately absent: radiative transfer in a homogeneous half-space has no absolute
length scale either, so a feature carrying one could only be fitting noise.

**3. λ and ``theta_s`` are first-class because the residual's structure lives
there.** M2 measured a monotone offset in solar zenith (≈ −2%, +2%, +8% at 0/30/60°)
and a spectral hump near 550 nm. A polynomial in λ alone explains 83.9% of the
relative-residual variance at degree 1 and 96.0% at degree 5. **That 84% is the
number an MLP has to beat to justify its nonlinearity**, so the linear model is a
first-class citizen here rather than a footnote: it is this same code with
``hidden=()`` (see :data:`LINEAR_CONFIG`), trained by the same loop on the same
features, which is the only way the comparison is apples-to-apples.

**4. The MLP's accuracy at an unseen solar zenith is not reproducible, and this is
the uncomfortable result of M3.** Everything above concerns *interpolation*, where
the emulator is excellent. Extrapolation in geometry is a different story. Trained on
0°/30° only and asked for the unseen 60°, MLP(16,16) scores ~0.24% in sample and, over
the seeds {23, 1, 7, 101, 2024}, **4.7 / 8.4 / 7.8 / 5.4 / 12.2%** at 60° — a
7.6× spread whose median (7.75%) barely improves on the backbone's 8.09% and whose
worse half is beaten by standard *Gordon* (9.01%). The cause is plain: ``cos θ_s``
spans [0.866, 1.0] in that training set and 60° needs 0.5, so every ``tanh`` unit is
evaluated outside its fitted range, where nothing constrains it and the
initialisation decides the answer.

The **linear** model (``hidden=()``) is the interesting contrast: it gives up a great
deal in sample (2.40%) but lands at **6.16%** at the unseen 60°, stably, beating both
the backbone and Gordon. Nothing about the network is regularising the geometry
direction; the linear model's inability to bend is what saves it.

So M3's honest position: the hybrid's headline result is an *interpolation* result on
the scene split, and geometry extrapolation from L23's three solar zeniths is
unresolved. It needs either early stopping on a geometry-held-out curve, a seed
ensemble, or an architecture whose geometry dependence cannot bend — and it needs
more than one unseen angle to choose between them. Raised for M4; do not let the
scene-split numbers stand in for it.

*How this was nearly got wrong, since the lesson generalises.* A linear skip path
(``δ_raw = W·x + MLP(x)``) appeared to fix the problem — 11.57% without it, 5.40%
with — and was very nearly adopted on the strength of that one comparison. But the
two runs also differed in their Flax parameter *names*, which changes PRNG folding
and hence the initialisation, so architecture and seed moved together. Sweeping seeds
with the architecture fixed showed the skip is no better (median 9.20% vs 7.75%, and
a 25% worst case). Both numbers in that first comparison were real; the inference
from them was not.

Structural choices that follow from the rest of the milestone:

- **Pointwise in λ.** One shared network maps the features *at* a wavelength to
  ``δ`` at that wavelength, rather than a per-λ output head over a fixed 81-band
  grid. This mirrors the backbone — ``rrs_ZTT(λ)`` depends only on the IOPs at λ —
  and it means the emulator is defined on any wavelength grid, including the
  hyperspectral grids M4/M5 will want, with λ as an input it can interpolate in.
- **``tanh``, not ``relu``.** M3's gate is a finite-difference gradient check
  (coding plan M3), and a relu kink makes central differences straddling it wrong by
  O(h) — a real hazard, not a hypothetical. ``tanh`` is smooth, and the target is
  smooth too.
- **A bounded correction, by construction.** ``δ = delta_max · tanh(·)`` cannot
  exceed :attr:`EmulatorConfig.delta_max` no matter how far out of distribution the
  inputs go, so the hybrid can never be driven negative or to a wild value by the
  learned half. The soft penalty on ``|δ|`` then keeps it small *in* distribution;
  the hard cap keeps it sane outside it.
- **The correction starts at exactly zero.** The output layer is zero-initialised,
  so an untrained hybrid *is* the backbone and every reported improvement is an
  improvement over ZTT rather than an artefact of initialisation. (One consequence:
  at step 0 the hidden layers see zero gradient, since the output kernel multiplies
  it; they start learning at step 2. This is the usual zero-init trade and costs one
  step.)
- **Standardisation statistics come from the training split only**, and are stored
  *in* the trained :class:`Emulator`. Fitting them on all the data would leak the
  held-out scenes into the model, quietly, in a way no accuracy number would reveal.

``flax`` and ``optax`` are imported inside the functions that need them, not at
module scope, so the analytic-only path never pays for the ML stack — the same
argument that made :mod:`robust.rt.types` use ``jax.tree_util`` over ``flax.struct``.
The network class is therefore defined inside :func:`_network` rather than at module
scope, which would need Flax at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from . import conventions, validation
from . import ztt as _ztt

__all__ = [  # noqa: RUF022  - grouped by role, not alphabetical
    "FEATURES",
    "EmulatorConfig",
    "LINEAR_CONFIG",
    "DOMAIN_TOL",
    "DomainBreach",
    "Emulator",
    "History",
    "features",
    "fit",
    "fit_l23",
    "fit_pb24",
    "PB24_ENVELOPE",
    "backbone_is_usable",
    "save",
    "load",
    "DEFAULT_WEIGHTS",
    "load_default",
]

#: The feature vector, in order. All are dimensionless or O(100) and are
#: standardised before the first layer using training-split statistics.
#:
#: ``log10_u`` and ``eta_bb`` together span the backbone's entire scale-free IOP
#: state (module docstring, point 2); ``wave_nm`` and ``cos_theta_s`` carry the two
#: structures M2 measured in the residual; ``B_p`` is the phase-function handle.
#: ``cos_theta_v`` and ``cos_dphi`` are **constant in L23** (nadir view, zero
#: azimuth), so their weights are unidentified and they standardise to exactly
#: zero. They are carried anyway so the feature vector is the final one and M5's
#: off-nadir runs need no interface change -- but nothing here can be read as
#: evidence about view geometry.
FEATURES = (
    "log10_u",
    "eta_bb",
    "B_p",
    "wave_nm",
    "cos_theta_s",
    "cos_theta_v",
    "cos_dphi",
)

#: Guard for the standardisation divisor. A feature that is constant over the
#: training split has ``x - mean == 0`` exactly, so this floor turns 0/0 into a
#: clean 0 rather than a NaN.
#:
#: **It is also the domain check's denominator for such a feature**, and that is not
#: a coincidence but the fix for a real bug. Judging a constant feature's excursion
#: against its own *value* let a sensor zenith of 5 deg pass as "in domain" while the
#: standardisation — dividing the same excursion by this floor — produced -3.8e5,
#: saturated every tanh, and collapsed the correction to a flat +0.046 at all 81
#: wavelengths. The check has to measure the excursion in the units the network
#: actually sees, so both divide by the same number.
_STD_FLOOR = 1e-8

#: How far outside its trained range a feature must go before
#: :meth:`Emulator.out_of_domain` reports it, as a fraction of the trained span.
#:
#: **Not zero, and the reason is measured.** The domain is the training split's
#: min/max, so held-out data legitimately *grazes* it: on the full L23 batch the
#: packaged emulator sees ``eta_bb`` reach 3.7e-4 of a span beyond the boundary, for
#: 4 values in a million. Meanwhile a genuinely unsupported input — a 75° sun against
#: a ``cos_theta_s`` floor of 0.5 — sits **48%** of the span beyond, three orders of
#: magnitude further out. A check with no tolerance fires on the first case, which
#: trains the user to silence the warning, and then the second case passes unnoticed.
#: 1% sits between them with ~27x of headroom below and ~48x above.
DOMAIN_TOL = 0.01

#: The solar-zenith range the project treats as **supported**, in degrees — a project
#: decision, not a property of any particular fit.
#:
#: JXP's call (prompt 5, Q7): *"I meant to warn when we extrapolate beyond 60 deg. It
#: should be fine to do anything up to that angle."* L23 provides 0°, 30° and 60°, so
#: 0–60° is the span the reference data covers and inside which interpolation is
#: sanctioned. The domain check therefore judges ``cos_theta_s`` against **this**
#: envelope rather than against the angles a given emulator happened to see: a fit
#: trained on 0°/30° only is *allowed* to be asked for 60° without complaint, even
#: though that is extrapolation for it (and M3 measured that its accuracy there is
#: seed-dependent — the permission is a policy, not a promise).
#:
#: Every other feature is still judged against its trained range, where "outside what
#: I learned" genuinely does mean "unreliable". Pass ``theta_s_limits=None`` to
#: :meth:`Emulator.out_of_domain` to judge the zenith by the trained range too, which
#: is the right question when the subject is a fit's extrapolation rather than the
#: package's supported envelope.
SUPPORTED_THETA_S = (0.0, 60.0)

#: The sanctioned **view**-zenith range. ``None`` by default, meaning "judge
#: ``theta_v`` by what this fit was trained on" -- which for an L23 model is the
#: single value 0, so any off-nadir view is flagged. That is the correct answer
#: for a nadir-only fit and it is why M4 needed no such constant; PB24-trained
#: models set it explicitly (Q14: 0-70 degrees).
SUPPORTED_THETA_V = None


@dataclass(frozen=True)
class Envelope:
    """The angle ranges a *particular* emulator is sanctioned over.

    Until M5 the envelope was one module constant, :data:`SUPPORTED_THETA_S`,
    consulted by every emulator's domain check. That was right while there was
    one model; it stops being right the moment two exist, because widening it for
    a PB24-trained net would silently widen the **shipped L23 net's** envelope
    too -- and a 65-degree query against a model trained to 60 would become "in
    domain", which is exactly the seed-dependent regime M4 measured and warned
    about (record §5.5).

    So the envelope travels **with the weights**. It is a static field of
    :class:`Emulator`, serialised by :func:`save`, and the module constant
    survives only as this class's default.

    A field left ``None`` means "judge that angle by the trained range", which is
    the right question when the subject is a fit's own extrapolation rather than
    a project decision.

    Attributes
    ----------
    theta_s : tuple of float or None
        Sanctioned solar-zenith range, degrees. Defaults to
        :data:`SUPPORTED_THETA_S`.
    theta_v : tuple of float or None
        Sanctioned sensor-zenith range, degrees. Defaults to
        :data:`SUPPORTED_THETA_V` (``None``).
    dphi : tuple of float or None
        Sanctioned relative-azimuth range, degrees. ``None`` by default; note
        that ``cos`` is not monotonic over azimuth, so the bound is computed from
        the interval rather than from its endpoints.
    """

    theta_s: tuple[float, float] | None = SUPPORTED_THETA_S
    theta_v: tuple[float, float] | None = SUPPORTED_THETA_V
    dphi: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Reject a range that is not a range."""
        for name in ("theta_s", "theta_v", "dphi"):
            limits = getattr(self, name)
            if limits is None:
                continue
            lo, hi = limits
            if not lo < hi:
                raise ValueError(
                    f"Envelope.{name}: expected (lo, hi) with lo < hi; got {limits}"
                )
            object.__setattr__(self, name, (float(lo), float(hi)))

    def describe(self) -> str:
        """One line, for a log or an artefact."""
        parts = []
        for name in ("theta_s", "theta_v", "dphi"):
            limits = getattr(self, name)
            span = (
                "trained range"
                if limits is None
                else f"{limits[0]:g}-{limits[1]:g} deg"
            )
            parts.append(f"{name} {span}")
        return "; ".join(parts)


#: The envelope M0-M4 behaved as if it had: the sanctioned solar-zenith span, and
#: the trained range for everything else.
DEFAULT_ENVELOPE = Envelope()

#: Sentinel for "use the emulator's own envelope", so that an explicit ``None``
#: can keep its established meaning of "judge by the trained range".
_OWN_ENVELOPE = object()

#: Guard inside the size penalty's square root, and **not** cosmetic: two
#: deliberate choices collide without it. The output layer is zero-initialised, so
#: ``δ ≡ 0`` at step 0; the penalty is an RMS, so its derivative
#: ``δ_i / (N·sqrt(mean δ²))`` is 0/0 exactly there. Training NaN-ed on the first
#: chunk until this was added. At 1e-24 the floor it puts under the term is 1e-12
#: percent -- unmeasurable -- while the gradient at ``δ = 0`` becomes a clean zero.
_RMS_EPS = 1e-24


@dataclass(frozen=True)
class EmulatorConfig:
    """Architecture and training hyper-parameters.

    Frozen and hashable so it can be a ``meta_field`` of :class:`Emulator` — i.e.
    static under ``jit``, which is what lets ``delta_max`` and ``hidden`` be plain
    Python values inside a traced forward pass.

    Attributes
    ----------
    hidden : tuple of int
        Hidden layer widths. ``()`` means a **linear** model in the standardised
        features — the baseline the MLP has to beat (see :data:`LINEAR_CONFIG`).
        The default is deliberately small: M2 measured that a straight line in λ
        already explains 84% of the residual variance, so if a wide network turns
        out to be necessary, the right response is to suspect the setup rather than
        to widen it.
    delta_max : float
        Hard bound on the relative correction, ``|δ| < delta_max``. 0.5 is ~9× the
        measured residual sd (5.52%), so it never binds on L23 while still making
        the correction bounded out of distribution.
    penalty : float
        Weight on the size term. The loss is
        ``rrms(truth, pred) + penalty * 100*sqrt(mean(δ²))`` — **both terms in
        percent**, so ``penalty`` reads as "percentage points of fit error I am
        willing to pay per percentage point of correction". 0.02 costs ~0.1 pp of
        loss at the residual's natural size: enough to discourage a large
        correction that buys nothing, far too little to suppress a real one.
    learning_rate : float
        Adam step size.
    steps : int
        Number of full-batch gradient steps. Training is **full-batch and
        unshuffled**, so a fit is reproducible from ``seed`` alone with no
        data-order dependence; the training set is only ~0.6 M rows of 7 features,
        which is cheap enough that mini-batching would buy noise, not speed.
    seed : int
        PRNG seed for parameter initialisation — the only stochastic input.
    eval_every : int
        Steps between recorded history points (and held-out evaluations).
    """

    hidden: tuple[int, ...] = (16, 16)
    delta_max: float = 0.5
    penalty: float = 0.02
    learning_rate: float = 3e-3
    steps: int = 3000
    seed: int = 23
    eval_every: int = 100

    def __post_init__(self) -> None:
        """Reject configurations that cannot train, with a reason.

        Raises
        ------
        ValueError
            On a non-positive width, bound, learning rate, or step count.
        """
        if any(h <= 0 for h in self.hidden):
            raise ValueError(
                f"EmulatorConfig.hidden: widths must be > 0; {self.hidden}"
            )
        if self.delta_max <= 0.0:
            raise ValueError(f"EmulatorConfig.delta_max: must be > 0; {self.delta_max}")
        if self.penalty < 0.0:
            raise ValueError(f"EmulatorConfig.penalty: must be >= 0; {self.penalty}")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"EmulatorConfig.learning_rate: must be > 0; {self.learning_rate}"
            )
        if self.steps < 1 or self.eval_every < 1:
            raise ValueError(
                f"EmulatorConfig: steps ({self.steps}) and eval_every "
                f"({self.eval_every}) must be >= 1"
            )


#: The **baseline** configuration: no hidden layers, so ``δ`` is an affine function
#: of the standardised features. This is the number the MLP must beat to earn its
#: nonlinearity (M2 measured 83.9% of the residual variance as linear in λ alone).
#: Same features, same loss, same optimiser — only the depth differs. Measured on the
#: full L23 batch it takes the backbone's 5.95% to **2.57%** (train) and 2.54%
#: (held-out scenes); the default MLP reaches 0.30%, so the nonlinearity earns its
#: place by a factor of ~8.
LINEAR_CONFIG = EmulatorConfig(hidden=())


@dataclass(frozen=True)
class DomainBreach:
    """One feature evaluated outside the range the emulator was trained on.

    A named record rather than a tuple because the *magnitude* matters as much as the
    fact: "0% of values outside" is what a bare fraction prints when four values in a
    million graze the boundary, and it reads as "nothing wrong".

    Attributes
    ----------
    feature : str
        Which of :data:`FEATURES`.
    lo, hi : float
        The trained range for it.
    worst : float
        The most extreme offending value seen.
    fraction : float
        Fraction of the evaluated values lying outside ``[lo, hi]``.
    excess : float
        How far :attr:`worst` lies beyond the range, as a fraction of the trained
        span — the number that distinguishes grazing the boundary from leaving the
        domain. See :data:`DOMAIN_TOL`.
    """

    feature: str
    lo: float
    hi: float
    worst: float
    fraction: float
    excess: float


@dataclass(frozen=True)
class History:
    """Training trace, as host-side NumPy — for the learning curve, not for math.

    Attributes
    ----------
    step : numpy.ndarray
        Step index of each recorded point.
    loss : numpy.ndarray
        Total objective (fit + size penalty), percent.
    fit : numpy.ndarray
        The fit term alone: rRMS of the hybrid on the training split, percent.
    delta_rms : numpy.ndarray
        ``100*sqrt(mean(δ²))`` on the training split, percent — the *magnitude of
        the correction*. The design asks for the hybrid's correction to be small and
        bounded, so this is reported as a first-class number rather than left to be
        inferred from the loss curve.
    eval : dict of str to numpy.ndarray
        Fit-term rRMS on each named held-out mask, same length as :attr:`step`. The
        point of recording these *during* training is that M3's honest question is
        not whether the loss went down but whether the held-out splits followed it.
    """

    step: np.ndarray
    loss: np.ndarray
    fit: np.ndarray
    delta_rms: np.ndarray
    eval: dict[str, np.ndarray] = field(default_factory=dict)


@cache
def _network(config: EmulatorConfig):
    """Build the Flax module: ``len(FEATURES) -> hidden... -> 1``.

    A ``tanh`` MLP, or a bare affine map when ``config.hidden`` is empty (the
    baseline, :data:`LINEAR_CONFIG`). The output layer is zero-initialised so the
    correction starts at exactly zero (module docstring).

    Flax is imported here rather than at module scope, which is why the module class
    is defined inside this function. Memoised on ``config`` so repeated calls return
    the *same* class and instance: Flax names parameters by module path, so the
    structure would match either way, but a stable instance also keeps ``jit``
    caches from being defeated by a fresh, unequal closure constant on every call.
    The cache holds no arrays.

    Parameters
    ----------
    config : EmulatorConfig
        Hashable, which is what makes the memoisation legal.

    Returns
    -------
    flax.linen.Module
        Maps ``(..., len(FEATURES)) -> (..., 1)``.
    """
    from flax import linen as nn

    zero = nn.initializers.zeros

    class ResidualNet(nn.Module):
        """Correction network. See :func:`_network`."""

        hidden: tuple[int, ...]

        @nn.compact
        def __call__(self, x):
            h = x
            for width in self.hidden:
                h = nn.tanh(nn.Dense(width)(h))
            return nn.Dense(1, kernel_init=zero, bias_init=zero)(h)

    return ResidualNet(hidden=tuple(config.hidden))


def features(
    iops,
    phase_params,
    geometry,
    wave: Float[Array, " wave"] | None = None,
) -> Float[Array, "*batch wave feature"]:
    """Raw (un-standardised) feature vectors, one per sample **and wavelength**.

    Takes the same arguments as :func:`robust.rt.forward`, so the emulator can be
    evaluated wherever the forward model can.

    Parameters
    ----------
    iops : robust.rt.types.IOPs
        Absorption and the water/particle backscattering split. Used only through
        the scale-free ratios ``u`` and ``η_bb`` (module docstring, point 2).
    phase_params : robust.rt.types.PhaseParams
        ``B_p``, as a spectrum, a per-sample scalar, or a scalar.
    geometry : robust.rt.types.Geometry
        Solar zenith, sensor zenith, relative azimuth (degrees). ``wind`` is not a
        feature: L23 does not vary it.
    wave : Array, optional
        Wavelengths (nm); defaults to the canonical grid. A *feature*, not just an
        axis — the residual's spectral structure is most of its structure.

    Returns
    -------
    Array
        Shape ``(..., n_wave, len(FEATURES))``, columns in :data:`FEATURES` order.

    Notes
    -----
    Pure JAX and traceable; no validation (that is the loader's job). ``B_p`` is
    read as a spectrum when its trailing axis matches ``n_wave`` and as a per-sample
    scalar otherwise, which is ambiguous only in the degenerate case of a batch with
    exactly ``n_wave`` samples and a per-sample ``B_p``.
    """
    u = iops.u
    shape = u.shape
    wave = conventions.canonical_wave() if wave is None else jnp.asarray(wave)

    B_p = jnp.asarray(phase_params.B_p)
    if B_p.ndim > 0 and B_p.shape[-1] != shape[-1]:
        B_p = B_p[..., None]

    def per_sample(angle):
        """Broadcast a per-sample angle (degrees) up to the spectrum shape."""
        return jnp.broadcast_to(jnp.asarray(angle)[..., None], shape)

    columns = (
        jnp.log10(u),
        iops.bb_w / iops.bb,
        jnp.broadcast_to(B_p, shape),
        jnp.broadcast_to(wave, shape),
        jnp.cos(jnp.deg2rad(per_sample(geometry.theta_s))),
        jnp.cos(jnp.deg2rad(per_sample(geometry.theta_v))),
        jnp.cos(jnp.deg2rad(per_sample(geometry.dphi))),
    )
    return jnp.stack(columns, axis=-1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Emulator:
    """A trained residual emulator: parameters plus its input standardisation.

    A registered pytree whose leaves are the trainable parameters and the
    standardisation vectors, with :attr:`config` static. So ``jit`` treats it as
    data, ``jax.grad`` of the hybrid w.r.t. an :class:`Emulator` gives parameter
    gradients, and — the case that matters for the inversion — differentiating
    w.r.t. the *inputs* while carrying this along works with no special handling.

    The standardisation lives *here*, not in a preprocessing step the caller has to
    remember, because a mismatch between the statistics used at fit time and at call
    time is silent: it produces plausible numbers that are simply wrong.

    Attributes
    ----------
    params : dict
        Flax parameter pytree for the network built by :func:`_network`.
    mean, std : Array
        Per-feature standardisation, shape ``(len(FEATURES),)``, computed on the
        **training split only**.
    config : EmulatorConfig
        Architecture and hyper-parameters. Static (a ``meta_field``).
    envelope : Envelope
        The angle ranges **this** model is sanctioned over, carried with the
        weights rather than read from a module constant — see :class:`Envelope`
        for why that distinction matters once there is more than one model.
    domain : Array or None
        Per-feature ``[min; max]`` over the training split, shape
        ``(2, len(FEATURES))``; ``None`` for a hand-built emulator.

        **This is what makes the extrapolation caveat operational rather than a
        remark in a log.** M3 measured that the emulator's accuracy at an unseen
        solar zenith is seed-dependent (module docstring, point 4), and JXP's
        instruction was that we do not use it at larger angles without warning the
        user. Carrying the training range with the weights is how a caller finds out
        — see :meth:`out_of_domain`, which :func:`robust.rt.hybrid.forward` calls for
        them. Like ``wind`` on :class:`~robust.rt.types.Geometry`, ``None`` here
        contributes no leaves but does change the treedef, so ``jit`` compiles once
        per variant.
    """

    params: dict
    mean: Float[Array, " feature"]
    std: Float[Array, " feature"]
    config: EmulatorConfig = field(metadata={"static": True})
    domain: Float[Array, "2 feature"] | None = None
    envelope: Envelope = field(default=DEFAULT_ENVELOPE, metadata={"static": True})

    def relative_delta(
        self,
        iops,
        phase_params,
        geometry,
        wave: Float[Array, " wave"] | None = None,
    ) -> Float[Array, "*batch wave"]:
        """The dimensionless relative correction ``δ(λ)``, bounded by ``delta_max``.

        Parameters
        ----------
        iops, phase_params, geometry, wave
            As :func:`features`.

        Returns
        -------
        Array
            ``δ``, shape ``(..., n_wave)``. The fractional correction to apply to
            the backbone: ``rrs = rrs_ZTT * (1 + δ)``.
        """
        x = features(iops, phase_params, geometry, wave)
        return _delta(
            _network(self.config), self.params, self._standardise(x), self.config
        )

    def delta_rrs(
        self,
        iops,
        phase_params,
        geometry,
        wave: Float[Array, " wave"] | None = None,
        *,
        rrs_ztt: Float[Array, "*batch wave"] | None = None,
    ) -> Float[Array, "*batch wave"]:
        """The additive correction ``Δrrs = δ · rrs_ZTT`` (sr⁻¹).

        Parameters
        ----------
        iops, phase_params, geometry, wave
            As :func:`features`.
        rrs_ztt : Array, optional
            The backbone, if already computed — the hybrid has it in hand, and
            recomputing ZTT is ~12× the cost of everything else here.

        Returns
        -------
        Array
            ``Δrrs``, sr⁻¹, shape ``(..., n_wave)``.
        """
        if rrs_ztt is None:
            rrs_ztt = _ztt.rrs_ZTT(iops, phase_params, geometry, wave)
        return rrs_ztt * self.relative_delta(iops, phase_params, geometry, wave)

    def out_of_domain(
        self,
        iops,
        phase_params,
        geometry,
        wave: Float[Array, " wave"] | None = None,
        *,
        tol: float = DOMAIN_TOL,
        theta_s_limits=_OWN_ENVELOPE,
    ) -> dict[str, DomainBreach]:
        """Which features are evaluated meaningfully outside the accepted range.

        A **boundary check**: it needs concrete values, so it cannot run under
        ``jit`` (the same rule as ``validate()`` in :mod:`robust.rt.types`).
        :func:`robust.rt.hybrid.forward` calls it on the caller's behalf when the
        inputs are concrete. For the traceable version — the one a fallback policy
        can act on inside ``jit`` — see :meth:`out_of_domain_mask`.

        Parameters
        ----------
        iops, phase_params, geometry, wave
            As :func:`features`.
        tol : float, optional
            Ignore breaches closer than ``tol`` × the trained span; see
            :data:`DOMAIN_TOL` for why this is not zero. ``tol=0.0`` reports any
            excursion at all.
        theta_s_limits : tuple of float, None, or Envelope, optional
            What to judge the **angle** features against instead of the trained
            range. Unset (the default) uses :attr:`envelope`, this model's own
            sanctioned span. ``None`` uses the trained range throughout, which is
            the right question when asking whether *this fit* is extrapolating. A
            ``(lo, hi)`` tuple overrides the solar zenith alone, as before; an
            :class:`Envelope` overrides all three angles.

        Returns
        -------
        dict
            ``{feature_name: DomainBreach}``, one entry per offending feature and
            **empty when everything is in range**, so the caller can test the
            truthiness.

        Raises
        ------
        ValueError
            If this emulator carries no :attr:`domain`.

        Notes
        -----
        Being outside the range is not by itself an error — it is exactly the
        situation M3 measured as unreliable (module docstring, point 4). For the
        solar zenith the relevant feature is ``cos_theta_s``, and note that
        ``cos`` is *decreasing* in the angle: a **larger** zenith gives a
        **smaller** feature value, so extrapolating to a low sun shows up as a
        breach of ``lo``.
        """
        if self.domain is None:
            raise ValueError(
                "Emulator.out_of_domain: this emulator carries no domain (it was "
                "not produced by fit()), so its training range is unknown"
            )
        x = np.asarray(features(iops, phase_params, geometry, wave))
        lo, hi = np.asarray(
            _effective_domain(self.domain, self._envelope_for(theta_s_limits))
        )
        report = {}
        for j, name in enumerate(FEATURES):
            col = x[..., j]
            if not np.isfinite(col).all():
                # A non-finite feature is out of domain by any definition, and saying
                # so beats reporting "nan% of the trained span".
                report[name] = DomainBreach(
                    feature=name,
                    lo=float(lo[j]),
                    hi=float(hi[j]),
                    worst=float("nan"),
                    fraction=float((~np.isfinite(col)).sum()) / col.size,
                    excess=float("inf"),
                )
                continue
            span = float(hi[j] - lo[j])
            scale = span if span > 0.0 else _STD_FLOOR
            excess = max(float(lo[j] - col.min()), float(col.max() - hi[j])) / scale
            if excess <= tol:
                continue
            below, above = col < lo[j], col > hi[j]
            worst = float(col.min()) if below.sum() >= above.sum() else float(col.max())
            report[name] = DomainBreach(
                feature=name,
                lo=float(lo[j]),
                hi=float(hi[j]),
                worst=worst,
                fraction=float(below.sum() + above.sum()) / col.size,
                excess=excess,
            )
        return report

    def out_of_domain_mask(
        self,
        iops,
        phase_params,
        geometry,
        wave: Float[Array, " wave"] | None = None,
        *,
        tol: float = DOMAIN_TOL,
        theta_s_limits=_OWN_ENVELOPE,
    ) -> Float[Array, "*batch"]:
        """Per-sample boolean: is this sample outside the accepted range?

        The **traceable** counterpart of :meth:`out_of_domain`. It reports no detail
        and raises no warning, but it is pure JAX, so a caller can act on it — which
        is what :func:`robust.rt.hybrid.forward`'s ``on_out_of_domain="ztt"`` policy
        does. That distinction matters: a policy implemented with the host-side check
        would silently stop applying under ``jit``, and a model that changes its
        answer when you compile it is worse than one with no policy at all.

        Parameters
        ----------
        iops, phase_params, geometry, wave, tol, theta_s_limits
            As :meth:`out_of_domain`.

        Returns
        -------
        Array
            Boolean, shape ``(...,)`` — the batch shape without the wavelength axis.
            ``True`` where **any** feature at **any** wavelength lies more than
            ``tol`` × the span outside the accepted range.

        Raises
        ------
        ValueError
            If this emulator carries no :attr:`domain`.
        """
        if self.domain is None:
            raise ValueError(
                "Emulator.out_of_domain_mask: this emulator carries no domain (it "
                "was not produced by fit()), so its training range is unknown"
            )
        x = features(iops, phase_params, geometry, wave)
        lo, hi = _effective_domain(self.domain, self._envelope_for(theta_s_limits))
        span = hi - lo
        scale = jnp.where(span > 0.0, span, _STD_FLOOR)
        excess = jnp.maximum(lo - x, x - hi) / scale
        # NOT `excess > tol`: a NaN compares False either way, which would have made
        # this predicate answer "in domain" while out_of_domain answered "out" on the
        # very same input. Negating the in-range test sends NaN to True in both.
        return jnp.any(~(excess <= tol), axis=(-1, -2))

    def _envelope_for(self, theta_s_limits) -> Envelope | None:
        """Resolve the ``theta_s_limits`` argument to an :class:`Envelope`.

        Three states, deliberately distinguishable: unset means *this model's*
        envelope; ``None`` keeps its established meaning of "judge everything by
        the trained range"; a ``(lo, hi)`` tuple keeps its M4 meaning of "the
        solar zenith specifically", which is why the sentinel exists at all --
        without it, "not passed" and "passed None" would be the same call.
        """
        if theta_s_limits is _OWN_ENVELOPE:
            return self.envelope
        if theta_s_limits is None:
            return None
        if isinstance(theta_s_limits, Envelope):
            return theta_s_limits
        return Envelope(theta_s=tuple(theta_s_limits), theta_v=None, dphi=None)

    def _standardise(
        self, x: Float[Array, "... feature"]
    ) -> Float[Array, "... feature"]:
        """Apply the stored per-feature standardisation."""
        return (x - self.mean) / self.std


def _cos_bounds(limits) -> tuple[float, float]:
    """``(min, max)`` of ``cos`` over an angle interval, in degrees.

    Not simply the endpoints: ``cos`` is monotonic on ``[0, 180]`` but the azimuth
    axis is not confined there, so an interval spanning 0 or 180 attains ±1 in its
    interior. Getting this wrong would narrow an envelope silently, which is the
    one direction of error a domain check must not make.
    """
    lo, hi = float(limits[0]), float(limits[1])
    values = [np.cos(np.deg2rad(lo)), np.cos(np.deg2rad(hi))]
    # Interior extrema of cos sit at multiples of 180 degrees.
    k = int(np.ceil(lo / 180.0))
    while k * 180.0 <= hi:
        values.append(np.cos(np.deg2rad(k * 180.0)))
        k += 1
    return float(min(values)), float(max(values))


def _effective_domain(domain, envelope: Envelope | None):
    """The stored domain, with the angle features replaced by the envelope.

    Splits the two meanings the domain check carries: for the IOP and wavelength
    features the trained range is the right bound, because outside it the network is
    genuinely unconstrained; for the **angles** the bound is a *project decision*
    (:class:`Envelope`), so a fit trained on a subset of angles is still allowed to
    be used across the whole sanctioned span.

    Parameters
    ----------
    domain : Array
        ``(2, len(FEATURES))`` of ``[min; max]``.
    envelope : Envelope or None
        Sanctioned angle ranges; ``None`` leaves every trained range alone.

    Returns
    -------
    Array
        The effective ``(2, len(FEATURES))`` bounds.
    """
    domain = jnp.asarray(domain)
    if envelope is None:
        return domain
    for name, feature in (
        ("theta_s", "cos_theta_s"),
        ("theta_v", "cos_theta_v"),
        ("dphi", "cos_dphi"),
    ):
        limits = getattr(envelope, name)
        if limits is None:
            continue
        lo, hi = _cos_bounds(limits)
        j = FEATURES.index(feature)
        domain = (
            domain.at[0, j]
            .set(jnp.asarray(lo, dtype=domain.dtype))
            .at[1, j]
            .set(jnp.asarray(hi, dtype=domain.dtype))
        )
    return domain


def _delta(model, params, x_std, config: EmulatorConfig):
    """Network output squashed to the bounded relative correction ``δ``."""
    raw = model.apply(params, x_std)[..., 0]
    return config.delta_max * jnp.tanh(raw)


def _objective(params, model, x_std, rrs_ztt, rrs_truth, config: EmulatorConfig):
    """Loss = hybrid rRMS + ``penalty`` × correction size, both in percent.

    Returns ``(loss, (fit, delta_rms))``. The fit term is
    :func:`robust.rt.validation.rrms` itself — *the* metric, not a second
    definition of it — which is the point of it being differentiable.
    """
    delta = _delta(model, params, x_std, config)
    fit = validation.rrms(rrs_truth, rrs_ztt * (1.0 + delta))
    delta_rms = 100.0 * jnp.sqrt(jnp.mean(delta**2) + _RMS_EPS)
    return fit + config.penalty * delta_rms, (fit, delta_rms)


def fit(
    iops,
    phase_params,
    geometry,
    wave,
    rrs_truth: Float[Array, "sample wave"],
    *,
    train: np.ndarray,
    eval_masks: dict[str, np.ndarray] | None = None,
    config: EmulatorConfig | None = None,
    rrs_ztt: Float[Array, "sample wave"] | None = None,
    envelope: Envelope = DEFAULT_ENVELOPE,
) -> tuple[Emulator, History]:
    """Train the residual emulator on ``rrs_truth − rrs_ZTT``.

    Parameters
    ----------
    iops, phase_params, geometry, wave
        The full batch, as :func:`features` takes them. Leading axis is the sample.
    rrs_truth : Array
        Reference subsurface reflectance, shape ``(n_sample, n_wave)``. **In
        ``rrs`` space**, not ``Rrs`` — scoring and training both happen there
        (design §6); :func:`fit_l23` does the conversion for an L23 batch.
    train : numpy.ndarray
        Boolean mask over the sample axis: the training split. Required, and
        deliberately not defaulting to "everything" — training on all of it is a
        mistake that costs a milestone's credibility, so it has to be typed out.
    eval_masks : dict of str to numpy.ndarray, optional
        Named held-out masks scored every ``config.eval_every`` steps and recorded
        in the returned :class:`History`.
    config : EmulatorConfig, optional
        Defaults to ``EmulatorConfig()``.
    rrs_ztt : Array, optional
        The backbone on this batch, if already computed.

    Returns
    -------
    Emulator
        The trained emulator, carrying its training-split standardisation.
    History
        Loss, fit, correction magnitude, and the held-out curves.

    Raises
    ------
    ValueError
        If ``train`` is not a boolean mask over the sample axis, or selects nothing.

    Notes
    -----
    Deterministic: full-batch, unshuffled, one seed. Optax and Flax are imported
    here rather than at module scope.
    """
    import optax

    config = EmulatorConfig() if config is None else config
    rrs_truth = jnp.asarray(rrs_truth)
    n_sample = rrs_truth.shape[0]

    train = np.asarray(train)
    if train.shape != (n_sample,) or train.dtype != bool:
        raise ValueError(
            f"fit: train must be a boolean mask of shape ({n_sample},); got "
            f"shape {train.shape}, dtype {train.dtype}"
        )
    if not train.any():
        raise ValueError("fit: train mask selects no samples")

    if rrs_ztt is None:
        rrs_ztt = _ztt.rrs_ZTT(iops, phase_params, geometry, wave)
    x = features(iops, phase_params, geometry, wave)

    # Standardisation from the training split ONLY -- fitting it on everything
    # would leak the held-out scenes into the model. The floor keeps a feature
    # that is constant in L23 (theta_v, dphi) at exactly zero instead of NaN.
    x_train = x[train]
    flat = x_train.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std = jnp.maximum(flat.std(axis=0), _STD_FLOOR)
    # The training range travels with the weights, so a caller asking for a
    # geometry the emulator never saw can be told (Emulator.domain).
    domain = jnp.stack([flat.min(axis=0), flat.max(axis=0)])

    def prepare(mask):
        """Standardised features, backbone, and truth for one split."""
        return ((x[mask] - mean) / std, rrs_ztt[mask], rrs_truth[mask])

    model = _network(config)
    train_data = prepare(train)
    params = model.init(jax.random.key(config.seed), train_data[0])

    tx = optax.adam(config.learning_rate)
    opt_state = tx.init(params)
    grad_fn = jax.value_and_grad(_objective, has_aux=True)

    step_fn = _make_chunk(model, tx, grad_fn, train_data, config)
    eval_fn = _make_eval(model, config)
    score = jax.jit(lambda p: _objective(p, model, *train_data, config))
    eval_data = {
        name: prepare(np.asarray(mask)) for name, mask in (eval_masks or {}).items()
    }

    steps, losses, fits, delta_rmss = [], [], [], []
    curves: dict[str, list[float]] = {name: [] for name in eval_data}

    def record(step: int, params) -> None:
        """Score and record one history point."""
        loss, (fit_term, delta_rms) = score(params)
        steps.append(step)
        losses.append(float(loss))
        fits.append(float(fit_term))
        delta_rmss.append(float(delta_rms))
        for name, data in eval_data.items():
            curves[name].append(float(eval_fn(params, *data)))

    record(0, params)
    done = 0
    while done < config.steps:
        chunk = min(config.eval_every, config.steps - done)
        params, opt_state = step_fn(params, opt_state, chunk)
        done += chunk
        record(done, params)

    history = History(
        step=np.asarray(steps),
        loss=np.asarray(losses),
        fit=np.asarray(fits),
        delta_rms=np.asarray(delta_rmss),
        eval={name: np.asarray(v) for name, v in curves.items()},
    )
    emulator = Emulator(
        params=params,
        mean=mean,
        std=std,
        config=config,
        domain=domain,
        envelope=envelope,
    )
    return emulator, history


def _make_chunk(model, tx, grad_fn, train_data, config: EmulatorConfig):
    """A jitted, ``lax.scan``-ed block of Adam steps.

    Scanning rather than looping in Python matters at this size: the network is
    tiny, so per-step dispatch would otherwise dominate the actual arithmetic.
    ``n_steps`` is static, so each distinct chunk length compiles once (there are
    at most two: ``eval_every`` and the remainder).
    """
    import optax

    def one_step(carry, _):
        params, opt_state = carry
        (_, _aux), grads = grad_fn(params, model, *train_data, config)
        updates, opt_state = tx.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), None

    # n_steps is static: lax.scan's length must be a concrete Python int, and each
    # distinct chunk length compiles once (there are at most two).
    @partial(jax.jit, static_argnums=2)
    def chunk(params, opt_state, n_steps):
        (params, opt_state), _ = jax.lax.scan(
            one_step, (params, opt_state), None, length=n_steps
        )
        return params, opt_state

    return lambda params, opt_state, n: chunk(params, opt_state, int(n))


def _make_eval(model, config: EmulatorConfig):
    """A jitted hybrid-rRMS evaluator for a held-out split."""

    @jax.jit
    def evaluate(params, x_std, rrs_ztt, rrs_truth):
        delta = _delta(model, params, x_std, config)
        return validation.rrms(rrs_truth, rrs_ztt * (1.0 + delta))

    return evaluate


def fit_l23(
    batch,
    splits,
    *,
    config: EmulatorConfig | None = None,
    rrs_ztt: Float[Array, "sample wave"] | None = None,
) -> tuple[Emulator, History]:
    """Train on an L23 batch, held out by scene — the M3 training run.

    Convenience over :func:`fit`: converts the reference ``Rrs`` to ``rrs`` and
    wires the M1 splits in as the training mask and the two held-out curves, so the
    milestone's headline fit is one call and cannot accidentally be trained on the
    wrong split.

    Parameters
    ----------
    batch : robust.rt.data.l23.L23Batch
        As :func:`robust.rt.data.l23.load_batch` returns it.
    splits : robust.rt.data.l23.Splits
        As :func:`robust.rt.data.l23.make_splits` returns it. Trains on
        ``scene_train``; records ``scene_test`` and ``scene_test_60``.
    config : EmulatorConfig, optional
    rrs_ztt : Array, optional
        The backbone on this batch, if already computed.

    Returns
    -------
    Emulator, History

    Notes
    -----
    The recorded 60° curve is ``scene_test_60`` — the held-out scenes **at** 60° —
    and deliberately **not** ``splits.zenith_test``. The latter is every 60° sample,
    ~80% of which belong to training scenes here, so it would read as a held-out
    number while being mostly training error. Nothing about this fit tests zenith
    *extrapolation*: the emulator sees all three angles. For that, train on
    ``splits.zenith_train`` via :func:`fit` and score ``zenith_test`` — which M3 did,
    with an unstable result (module docstring, point 4).
    """
    return fit(
        batch.iops,
        batch.phase_params,
        batch.geometry,
        batch.wave,
        conventions.Rrs_to_rrs(batch.Rrs),
        train=splits.scene_train,
        eval_masks={
            "scene_test": splits.scene_test,
            "scene_test_60": splits.scene_test & splits.zenith_test,
        },
        config=config,
        rrs_ztt=rrs_ztt,
    )


#: Q14's sanctioned angles for a PB24-trained model: 0-70 degrees in both zeniths,
#: with the 80/87.75 shell held out as the extrapolation test.
PB24_ENVELOPE = Envelope(theta_s=(0.0, 70.0), theta_v=(0.0, 70.0))


def backbone_is_usable(rrs_ztt) -> np.ndarray:
    """Per-sample: is the analytic backbone physical across the whole spectrum?

    **Why this exists (M5 tasks 9 and 13, Q17).** ZTT's ``psi_KLu = 1 + F(psi)``
    is a quartic fitted for scattering angles ``psi >~ 134`` degrees (see
    :func:`robust.rt.ztt.F_psi`); it crosses zero at 110.4 and is negative below,
    which flips the sign of the ZTT denominator. Of the non-physical samples,
    **71% have psi below that crossing and only 5% have a non-positive mu_inf** --
    so this is a *scattering-angle* problem, not the ``bb/a`` one an earlier
    version of this docstring claimed. (``mu_infinity_tt2017`` is separately
    outside its own fitted range -- it covers ``bb/a`` up to 0.1 and PB24 reaches
    9.4 over 200 realisations, 20.1 over the release -- but that accounts for a
    twentieth of the effect.) The result either way is a *non-physical* backbone
    -- ``rrs_ZTT <= 0`` on ~22% of PB24's window -- and the
    hybrid is ``rrs_ZTT * (1 + delta)`` with ``|delta| <= 0.5``, so **no bounded
    relative correction can turn a negative backbone into a positive
    reflectance**. Those samples are not hard, they are impossible for this
    functional form.

    JXP's answer to Q17 was to restrict the sanctioned envelope for now and report
    the coverage, refitting µ∞ properly at task 13. This is the predicate that
    restriction is built on: training excludes these samples, so the emulator's
    stored ``domain`` excludes the ``log10_u`` range they occupy, so the ordinary
    domain check flags them at evaluation time with no special case anywhere.

    Parameters
    ----------
    rrs_ztt : array_like
        Backbone prediction, shape ``(n_sample, n_wave)``.

    Returns
    -------
    numpy.ndarray
        Boolean, shape ``(n_sample,)``. A sample counts only if **every** band is
        positive and finite: a spectrum that is negative in the red is not a
        spectrum this model can correct, however good the blue looks.
    """
    values = np.asarray(rrs_ztt)
    return np.all(np.isfinite(values) & (values > 0.0), axis=-1)


def fit_pb24(
    batch,
    splits,
    kind: str = "realisation",
    *,
    config: EmulatorConfig | None = None,
    rrs_ztt: Float[Array, "sample wave"] | None = None,
    envelope: Envelope = PB24_ENVELOPE,
    restrict_to_usable_backbone: bool = True,
) -> tuple[Emulator, History, dict]:
    """Train on a PB24 batch — the M5 training run.

    The PB24 counterpart of :func:`fit_l23`, and different from it in three ways
    that all come from the data rather than from taste:

    1. **The target is the tabulated ``rrs``**, not ``Rrs`` converted through the
       surface map. PB24 ships both, and the nadir map is wrong off-nadir by a
       median 33.6% at ``theta_v = 60`` (record §7.10) -- so training through it
       would fit the emulator to an interface error.
    2. **``theta_v`` and ``dphi`` are live features.** They are already in
       :data:`FEATURES` and were constant in L23, which is exactly why the domain
       check flagged every off-nadir view; PB24 makes them vary.
    3. **Training is restricted to where the backbone is physical** (Q17; see
       :func:`backbone_is_usable`), and to the sanctioned angle window, and the
       excluded share is returned rather than absorbed.

    Parameters
    ----------
    batch : robust.rt.data.pb24.PB24Batch
    splits : robust.rt.data.pb24.Splits
    kind : str, optional
        Which split to train against; the held-out side becomes an eval curve.
    config : EmulatorConfig, optional
    rrs_ztt : Array, optional
        Precomputed backbone, to avoid recomputing it per seed.
    envelope : Envelope, optional
        Sanctioned angles, stored with the weights. Defaults to
        :data:`PB24_ENVELOPE`.
    restrict_to_usable_backbone : bool, optional
        Exclude samples whose backbone is non-physical (default True). ``False``
        trains on everything and is provided so the cost of the restriction can be
        *measured* rather than asserted.

    Returns
    -------
    emulator : Emulator
    history : History
    coverage : dict
        What training saw and what it excluded: ``n_total``, ``n_train``,
        ``n_excluded_backbone``, ``n_excluded_angle``, and ``usable_fraction``.
        Reported so that "the hybrid scores X" is always accompanied by "on this
        share of the data".
    """
    if rrs_ztt is None:
        rrs_ztt = _ztt.rrs_ZTT(
            batch.iops, batch.phase_params, batch.geometry, batch.wave
        )

    lo_s, hi_s = envelope.theta_s if envelope.theta_s else (-np.inf, np.inf)
    lo_v, hi_v = envelope.theta_v if envelope.theta_v else (-np.inf, np.inf)
    in_window = (
        (batch.theta_s >= lo_s)
        & (batch.theta_s <= hi_s)
        & (batch.theta_v >= lo_v)
        & (batch.theta_v <= hi_v)
    )
    usable = (
        backbone_is_usable(rrs_ztt)
        if restrict_to_usable_backbone
        else np.ones(batch.n_sample, dtype=bool)
    )

    train = splits.train(kind) & in_window & usable
    test = splits.test(kind) & in_window & usable
    if not train.any():
        raise ValueError(
            "fit_pb24: the training mask is empty after restricting to the "
            "sanctioned window and a usable backbone"
        )
    if not test.any():
        # The geometry split's test side *is* the shell, so intersecting it with
        # the sanctioned window empties it -- and an empty eval mask records a
        # curve of NaN, which reads as a number and compares False against any
        # gate. `make_splits` raises on an empty side for the same reason.
        raise ValueError(
            f"fit_pb24: the {kind!r} split's held-out side is empty after "
            "restricting to the sanctioned window and a usable backbone. For "
            "'geometry' that is structural -- its test side is the 80/87.75 shell, "
            "which the window excludes by construction; score it directly rather "
            "than as an eval curve."
        )

    coverage = {
        "n_total": int(batch.n_sample),
        "n_train": int(train.sum()),
        "n_excluded_angle": int((~in_window).sum()),
        "n_excluded_backbone": int((in_window & ~usable).sum()),
        "usable_fraction": float((in_window & usable).sum() / batch.n_sample),
    }

    emulator, history = fit(
        batch.iops,
        batch.phase_params,
        batch.geometry,
        batch.wave,
        batch.rrs,
        train=train,
        eval_masks={f"{kind}_test": test},
        config=config,
        rrs_ztt=rrs_ztt,
        envelope=envelope,
    )
    return emulator, history, coverage


def save(emulator: Emulator, path) -> None:
    """Write a trained emulator to a ``.npz``.

    Everything needed to reproduce a prediction goes in the one file: the
    parameters, the standardisation, the training :attr:`~Emulator.domain`, the
    :class:`EmulatorConfig`, **and the feature names**. The last is the important
    one — the weights are meaningless against a different feature vector, and a
    silently-wrong prediction is the failure mode worth designing against, so
    :func:`load` refuses a file whose ``FEATURES`` do not match the current code.

    Parameters
    ----------
    emulator : Emulator
    path : str or pathlib.Path
        Destination. 6.5 KB for the default architecture, which is why the trained
        weights can live in the repo (see :data:`DEFAULT_WEIGHTS`).
    """
    from flax.traverse_util import flatten_dict

    arrays = {
        f"param/{'/'.join(k)}": np.asarray(v)
        for k, v in flatten_dict(emulator.params).items()
    }
    arrays["mean"] = np.asarray(emulator.mean)
    arrays["std"] = np.asarray(emulator.std)
    arrays["features"] = np.asarray(FEATURES)
    if emulator.domain is not None:
        arrays["domain"] = np.asarray(emulator.domain)
    # The envelope goes in the file, not in a constant: two models with different
    # sanctioned spans have to be able to coexist in one process (M5 task 10).
    # NaN encodes "None" -- i.e. judge that angle by the trained range -- because
    # npz has no null and a sentinel angle would be indistinguishable from a real
    # one.
    for name in ("theta_s", "theta_v", "dphi"):
        limits = getattr(emulator.envelope, name)
        arrays[f"envelope/{name}"] = np.asarray(
            (np.nan, np.nan) if limits is None else limits, dtype=float
        )
    cfg = emulator.config
    arrays["config/hidden"] = np.asarray(cfg.hidden, dtype=np.int64)
    for name in (
        "delta_max",
        "penalty",
        "learning_rate",
        "steps",
        "seed",
        "eval_every",
    ):
        arrays[f"config/{name}"] = np.asarray(getattr(cfg, name))
    np.savez(path, **arrays)


def load(path) -> Emulator:
    """Read an emulator written by :func:`save`.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    Emulator

    Raises
    ------
    ValueError
        If the file's feature list differs from the current :data:`FEATURES`. The
        weights would still *run* — the shapes need not have changed — and would
        return plausible nonsense, so this is a refusal rather than a warning.
    """
    from flax.traverse_util import unflatten_dict

    with np.load(path, allow_pickle=False) as data:
        stored = tuple(str(f) for f in data["features"])
        if stored != FEATURES:
            raise ValueError(
                f"emulator.load: {path} was trained on features {stored}, but this "
                f"code uses {FEATURES}. Retrain rather than reinterpret the weights"
            )
        params = unflatten_dict(
            {
                tuple(key.split("/")[1:]): jnp.asarray(data[key])
                for key in data.files
                if key.startswith("param/")
            }
        )
        config = EmulatorConfig(
            hidden=tuple(int(h) for h in data["config/hidden"]),
            delta_max=float(data["config/delta_max"]),
            penalty=float(data["config/penalty"]),
            learning_rate=float(data["config/learning_rate"]),
            steps=int(data["config/steps"]),
            seed=int(data["config/seed"]),
            eval_every=int(data["config/eval_every"]),
        )
        # A file written before M5 task 10 carries no envelope; it gets the
        # default, which is what it was evaluated under, so old weights keep
        # behaving exactly as they did.
        limits = {}
        for name in ("theta_s", "theta_v", "dphi"):
            key = f"envelope/{name}"
            if key not in data.files:
                limits = None
                break
            pair = np.asarray(data[key], dtype=float)
            limits[name] = None if np.isnan(pair).any() else (pair[0], pair[1])
        envelope = DEFAULT_ENVELOPE if limits is None else Envelope(**limits)

        return Emulator(
            params=params,
            mean=jnp.asarray(data["mean"]),
            std=jnp.asarray(data["std"]),
            config=config,
            domain=jnp.asarray(data["domain"]) if "domain" in data.files else None,
            envelope=envelope,
        )


#: The trained weights shipped with the package: MLP(16,16) fit on L23's elastic
#: X=1 scenes, **``scene_train`` split only**, by ``design/py/train_emulator.py``.
#: Committed (6.5 KB) so :func:`robust.rt.hybrid.forward` is a *trained* model out of
#: the box and CI can exercise the real thing, rather than every caller having to
#: reproduce a ~1-minute fit first. Regenerate with that script after any change to
#: :data:`FEATURES`, the architecture, or the loss.
DEFAULT_WEIGHTS = Path(__file__).parent / "files" / "emulator_l23.npz"


@cache
def load_default() -> Emulator:
    """The packaged trained emulator (:data:`DEFAULT_WEIGHTS`), read once.

    Returns
    -------
    Emulator

    Raises
    ------
    FileNotFoundError
        If the weights are missing, with the command that regenerates them.

    Notes
    -----
    Memoised, so the file is read once per process; the returned object is frozen,
    and its arrays are immutable, so sharing it is safe.
    """
    if not DEFAULT_WEIGHTS.exists():
        raise FileNotFoundError(
            f"packaged emulator weights not found at {DEFAULT_WEIGHTS}; regenerate "
            "with `python design/py/train_emulator.py` (needs $OS_COLOR and the L23 "
            "netCDFs)"
        )
    return load(DEFAULT_WEIGHTS)
