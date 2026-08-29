# Design — Inelastic RT Forward Model

*A near-term, buildable engineering plan for adding the two inelastic
processes — Raman scattering by water and chlorophyll-a fluorescence — to
retrieve-or-bust's elastic forward model, keeping the whole map
`Rrs(λ; IOPs, phase, geometry, φ_C)` fast, accurate, and **differentiable**.*

Companion documents: the elastic design
[`design/rt_elastic_model.md`](rt_elastic_model.md) (whose architecture and
contracts this extends), and the assessment that motivates every choice here,
[`context/RT/rt_inelastic_bing_summary.md`](../context/RT/rt_inelastic_bing_summary.md)
(BING's inelastic formulation quantified against the Loisel+2023 scenario
pairs; its two fixable errors have since been fixed in BING — branch
`inelastic-fixes` — so "the analytic terms" below means the *fixed* physics).

Decisions locked in Q&A/Design (`claude_prompts/RT/rt_inelastic_prompts.md`,
2026-08-19/20): **physics + bounded learned corrections**, one head per
process (DQ1); **Raman + Chl fluorescence in v1**, CDOM fluorescence hooks
only (DQ2); interface extensions and the composition law of §3 (DQ3); **φ_C a
differentiable input** with corrections on the φ_C-linear kernel,
single-Gaussian emission (DQ4); **L23 Ed spectra as package data** with a
recorded caveat about solar-model quality (DQ5); the acceptance gate of §6
(DQ6); L23 X1/X2/X4 with the elastic effort's by-scene splits, λ ≥ 400 nm
support, and a HydroLight wishlist (DQ7); a ~1-week M0–M4 prototype under
`robust/rt/` conventions (DQ8).

---

## 1. Goals and non-goals

**Goals.**
- Extend `forward()` so it reproduces the **all-processes-on** ocean
  (HydroLight X4-class truth) — because that is what PACE measures. An
  elastic-only forward model aliases a median 5–20 % (Raman, broadband) and
  ~35 % (fluorescence, at 685 nm) of signal into biased IOPs.
- Keep every property the elastic model earned: **differentiable end-to-end**
  (now including ∂Rrs/∂φ_C), batched/`vmap`-able, validated against
  HydroLight with a protocol and a hard gate.
- **Physics-anchored, not black-box**: analytic Raman and fluorescence terms
  carry the interpretable dependences (Ed spectrum, excitation IOPs, φ_C);
  small bounded corrections absorb only what the two-flow physics provably
  misses.
- Preserve the elastic contract exactly: `inelastic=None` → **bit-identical
  elastic-only output**, so the elastic acceptance gate remains valid.

**Non-goals (v1).**
- CDOM fluorescence (no truth data exist anywhere in hand — §8): interface
  hooks only.
- The inversion; learned priors; operational PACE processing.
- Real-sky/variable atmospheres (Ed handling is L23-tied in v1; §4.2).
- Fixing geometry extrapolation beyond the three L23 zeniths (inherited,
  documented elastic limitation).

---

## 2. Architecture

```
                                    geometry Ω (θ_s; Ed spectrum)
                                              │
  iops (a, a_ph, bb_w, bb_p) ────────┬────────┼──────────────────────┐
  phase_params θ_p ──────────────────┤        │                      │
                                     ▼        ▼                      ▼
                        ┌──────────────────┐  ┌──────────────────┐   ┌──────────────────┐
                        │ ELASTIC HYBRID   │  │ RAMAN            │   │ FLUORESCENCE     │
                        │ Rrs_ZTT + ΔRrs   │  │ f_R = f_phys ×   │   │ Rrs_fl = φ_C ×   │
                        │ (existing, un-   │  │   (1 + δ_R)      │   │   K_fl × (1+δ_F) │
                        │  touched)        │  │ analytic ratio   │   │ analytic kernel  │
                        └────────┬─────────┘  │ + bounded MLP    │   │ + bounded MLP    │
                                 │            └────────┬─────────┘   └────────┬─────────┘
                                 │                     │                      │
                                 └────── × ────────────┘                      │
                                             └────────────── + ──────────────┘
                                                              │
                          Rrs_total(λ) = (Rrs_ZTT + ΔRrs) × f_R(λ)  +  Rrs_fl(λ)
```

- **Raman is multiplicative** — `f_R = 1 + ΔR_raman/R_elastic` computed in the
  two-flow framework, the *self-normalizing* form the assessment validated
  (normalization errors cancel in the ratio; this is why BING's Raman was
  ~right while its additive fluorescence was ×3 off).
- **Fluorescence is additive** — an emission source, not a modification of
  elastic scattering.
- **Two separate correction heads** (DQ1): they see different inputs and fix
  different failure modes — δ_R the high-sun/red two-flow failure (−39 %
  increment error at zenith 0° even with true Ed), δ_F the residual
  zenith/trophic amplitude trends (model/truth ×1.00/0.95/0.86 median by
  zenith, drifting to ~×2–3 at the eutrophic tail).
- Corrections are **bounded** (tanh-scaled, like the elastic ΔRrs) so
  extrapolation degrades toward the analytic physics, not toward network
  free-wheeling.

Everything is JAX; both analytic terms are closed-form algebra plus a
fixed-grid quadrature (trapezoid over the excitation axis), so gradients are
exact through the whole composition, including ∂/∂φ_C.

## 3. Interface and data model

Extensions to the pinned elastic API (all backward compatible):

```python
def forward(iops: IOPs, phase_params: PhaseParams, geometry: Geometry,
            wave: Array, inelastic: Inelastic | None = None) -> Array:
    """Rrs(wave). inelastic=None -> bit-identical elastic-only output."""
```

- **`IOPs` grows `a_ph(λ)`** (phytoplankton absorption), required by the
  fluorescence source term `b_F = φ_C·a_ph`. The elastic path ignores it.
  Callers that already split `a` into components pay nothing; callers with
  only bulk `a` can pass `a_ph=None` and get elastic+Raman (fluorescence
  requires the split — this is a *physical* requirement, not an API whim).
- **`Inelastic` pytree**: `phi_C` (scalar or batched; default 0.02);
  reserved fields `cdom_fl` (None in v1, §8) and `emission_shape`
  (`'single'` default | `'double'`, §4.4). Booleans `raman`, `fluorescence`
  allow enabling each process independently (fluorescence-only requires
  `a_ph`).
- **`Geometry` grows an optional `Ed`** override (`(wave_Ed, Ed)` arrays).
  Default: package L23 spectra interpolated in θ_s (§4.2).
- **Excitation-grid IOPs are internal**: the forward model interpolates the
  supplied IOP spectra onto the Raman excitation grid (λ′ such that
  1/λ′ = 1/λ + 3400 cm⁻¹) and integrates the fluorescence excitation over
  370–690 nm on the native grid. No API surface.
- Wavelength support: **λ ∈ [400, 750] nm official** (Raman excitation for
  λ_em = 400 nm needs IOPs at 352 nm, the practical edge of the L23 grid);
  below 400 nm the model runs but extrapolates excitation IOPs — documented
  caveat, not a gate.

## 4. Components

### 4.1 Reference data and truth channels

L23 (Loisel et al. 2023), already in hand: 3320 IOP scenes × 81 λ
(350–750 nm, 5 nm) × three solar zeniths (0/30/60°) × three inelastic
scenarios — X1 elastic-only, X2 +Raman, X4 +Raman+fluorescence (HydroLight
defaults: Mobley 2012 Raman, **φ_C = 0.02**, single-Gaussian 685 nm emission;
CDOM fluorescence omitted). The scenario pairs give exact per-scene truth:

| channel | truth | trains / gates |
|---|---|---|
| Raman factor | `Rrs_X2 / Rrs_X1` | δ_R head; Raman delta gate |
| fluorescence | `Rrs_X4 − Rrs_X2` | δ_F head; 685 nm peak gate |
| total | `Rrs_X4` | end-to-end gate |

Splits: **identical by-scene train/held-out splits as the elastic effort**,
so elastic and inelastic results compose on the same held-out scenes.

### 4.2 Solar spectrum module (`robust/rt/data/ed_l23.*`)

- Ship the three L23 `Ed(0⁺)(λ)` spectra (scene-independent sky property;
  verified to ~10⁻³ relative) as package data; interpolate in θ_s between
  0–60°; accept the `Geometry.Ed` override.
- The Raman term consumes the ratio `Ed(λ′)/Ed(λ)`; the fluorescence term
  consumes `Ed(λ′)` and `Ed(λ_em)`. Both are first-order dependences — the
  assessment showed a flat-Ed Raman correction is wrong by +60 % (blue) to
  −50 % (red) in increment.
- **Recorded caveat (JXP, DQ5):** the community's current solar-irradiance
  reference models are themselves imperfect ("a poor model of the Sun").
  v1 deliberately inherits whatever solar spectrum HydroLight/L23 used —
  consistency with the truth data trumps absolute solar accuracy for the
  forward model. When the effort moves to real PACE spectra, the Ed source
  (e.g. TSIS-1-era references vs older standards) must be revisited; the
  `Geometry.Ed` override is the seam where that happens.

### 4.3 Analytic Raman term (JAX port of fixed BING)

Physics (all constants HydroLight-consistent, matching the truth): Bartlett
et al. (1998) coefficient `b_R(488) = 2.6e-4 m⁻¹`, `∝ λ′^−5.5` (energy
units); single 3400 cm⁻¹ wavenumber shift (the ~25 nm Walrafen bandwidth is
absorbed by δ_R — at 5 nm resolution the assessment showed the single-shift
error is subdominant); Rayleigh-like phase function (b_b/b = ½);
Sathyendranath & Platt (1998) two-flow terms — first order + both
second-order (RE, ER) — with μ_d = 0.9, μ_u = μ_R = 0.5 and the **true Ed
ratio**; assembled as the *ratio* `f_phys = (R_E + R_Raman)/R_E` applied
multiplicatively to the elastic hybrid.

Known accuracy of this backbone (post-fix BING vs L23, median increment
error 550–700 nm): **+1 % / −4 %** at θ_s = 30°/60°, **−39 % at 0°**, and
+30 % at 490 nm (30–60°). That residual structure — a smooth function of
(θ_s, λ, water clarity) — is exactly δ_R's job.

### 4.4 Analytic fluorescence term (JAX port of fixed BING)

`b_F(λ′) = φ_C · a_ph(λ′)`, isotropic emission (backscatter fraction ½),
excitation integral over 370–690 nm with the `λ′/λ` quanta→energy factor and
per-λ_em attenuation `κ_F(λ)`; two-flow irradiance reflectance converted to
rrs via **L_u = E_u/π** (the normalization the assessment identified and
validated: model/truth at 685 nm = 1.00/0.95/0.86 with it, ~3× without);
standard A·rrs/(1−B·rrs), A = 0.52, B = 1.7.

Emission line: **single Gaussian at 685 nm (σ = 10.6 nm)** — what
L23/HydroLight used, hence what can be validated (DQ4). The 730 nm PS I
shoulder ships as `emission_shape='double'` (0.75/0.25 weights), documented
as *physically better, unvalidatable against L23* — off by default and off
everywhere in v1 training/validation.

**φ_C-linearity by construction (DQ4):** the term is
`Rrs_fl = φ_C · K_fl(IOPs, Ω, λ) · (1 + δ_F)`, with δ_F independent of φ_C.
Training at the truth's φ_C = 0.02 then generalizes to other φ_C exactly to
the extent the RT is φ_C-linear (the only nonlinearity is the (1−B·rrs)
denominator, O(10⁻³) at fluorescence amplitudes). The future inversion gets
an honest, differentiable φ_C handle — the physiology signal (Maritorena's
~1–6 % natural range, Behrenfeld's NPQ) is retrievable rather than baked in.

### 4.5 Correction heads (two small Flax MLPs)

- **δ_R(λ)** — inputs per (scene, λ): `a(λ), bb(λ), a(λ′), bb(λ′), cos θ_s,
  λ` (normalized); output bounded `tanh`-scaled relative correction on
  `(f_phys − 1)`, i.e. `f_R = 1 + (f_phys − 1)(1 + δ_R)`, so the correction
  rescales the Raman *increment* and can never push f_R below 1 by more than
  the bound. Trained on the X2/X1 channel.
- **δ_F(λ)** — inputs per (scene, λ_em): `a_ph(440), a(λ_em), bb(λ_em),
  a(490), cos θ_s, λ_em`; output bounded relative correction on the
  φ_C-linear kernel. Trained on the X4−X2 channel at φ_C = 0.02.
- Size class: the elastic ΔRrs needed 417 parameters for a harder target
  (0.30 % on the full spectrum); these targets are smoother and smaller —
  budget **O(10²–10³) parameters each**, decided by the prototype's
  validation, not in advance.
- Both heads train with the same Optax pipeline/conventions as the elastic
  emulator; weights committed as `robust/rt/files/{raman,fl}_corr_l23.npz`.

### 4.6 Composition, batching, gradients

`Rrs_total = (Rrs_ZTT + ΔRrs) × f_R + Rrs_fl` (§2). Single `jit`-able
callable; batched over scenes; the fluorescence excitation quadrature is a
fixed-size `(n_batch, n_em, n_ex)` contraction — trivially vectorized at L23
scale (3320 × 71 × 65). Gradient correctness w.r.t. *every* input (IOP
spectra, θ_p, θ_s, φ_C) is a gate (§6), checked against central differences
exactly as in the elastic M-milestones.

## 5. Software stack and conventions

Same as elastic: JAX + Flax + Optax, `robust/rt/`, ruff, CI on committed
fixtures (no data mount needed). New modules:

| module | contents |
|---|---|
| `robust/rt/inelastic.py` | analytic Raman + fluorescence terms, composition into `forward()` |
| `robust/rt/inelastic_corr.py` | δ_R/δ_F heads, training entry points |
| `robust/rt/data/ed_l23.npz` | the three L23 Ed(0⁺) spectra |
| `robust/rt/files/{raman,fl}_corr_l23.npz` | trained head weights |
| `robust/rt/data/l23.py` (extend) | X2/X4 loaders + truth-channel builders, reusing the elastic splits |

Committed CI fixture: extend the elastic 50-scene fixture with its X2/X4
counterparts (~doubles a small file; keeps CI running real numbers). The
BING repo's 40-scene fixture (`bing/tests/files/l23_inelastic_fixture.npz`)
is the template.

Cross-check harness: unit tests pin the JAX analytic terms against the fixed
BING implementations (`bing.rt.rrs.calc_raman_correction_factor`,
`calc_Rrs_fluorescence`) at `rtol ≤ 1e-6` on shared inputs — the port must
inherit BING's L23-anchored validation for free before the heads train.

## 6. Validation protocol and acceptance gate (DQ6)

Held-out **by scene** (elastic splits), rrs-space rRMS per the elastic
protocol, reported per zenith. The prototype passes when all of:

1. **Total**: held-out rRMS(Rrs_model vs Rrs_X4) ≤ **0.5 %** at each of
   θ_s = 0°, 30°, 60° (all processes on, φ_C = 0.02).
2. **Raman delta**: median |increment error| of f_R vs X2/X1 ≤ **5 %** over
   550–700 nm at every zenith **including 0°** (the analytic backbone alone
   fails this at 0° by −39 % — this line is what δ_R must earn).
3. **Fluorescence delta**: median |error| of the 685 nm peak vs X4−X2 ≤
   **5 %** at every zenith.
4. **Elastic regression**: `inelastic=None` bit-identical to the current
   elastic hybrid (hash-level test).
5. **Gradients**: central-difference agreement (elastic tolerance) for all
   inputs, now including φ_C.
6. **Speed**: full-batch forward ≤ **2×** the elastic hybrid's runtime.

Also reported (not gated): per-wavelength error spectra; performance vs
trophic state (a_ph(440) deciles) — the known failure axis of the analytic
fluorescence amplitude; behavior at φ_C ≠ 0.02 (linearity check against a
scaled-truth construction, since no varied-φ_C truth exists — §8).

## 7. The ~1-week prototype (DQ8)

| milestone | delivers | gate |
|---|---|---|
| **M0** | API extension (`IOPs.a_ph`, `Inelastic`, `Geometry.Ed`), `inelastic=None` pass-through, scaffold + CI | elastic regression test green |
| **M1** | Ed module (§4.2) + excitation-grid infrastructure + X2/X4 loaders/truth channels on the elastic splits | Ed ratios match L23 spectra; loaders round-trip |
| **M2** | analytic Raman + fluorescence terms in JAX | pinned vs fixed-BING at rtol ≤ 1e-6; reproduces the assessment's error table |
| **M3** | δ_R and δ_F heads trained on X-differences | per-process delta gates (§6.2–3) on held-out scenes |
| **M4** | end-to-end validation + speed + gradients; short report with figures | full gate (§6); numbers regenerated by a committed script |

Then, as for elastic: a coding plan (`design/rt_inelastic_model_coding_plan.md`)
and numbered prompt docs execute this table.

## 8. Beyond v1 — and the HydroLight run wishlist (DQ7)

**CDOM fluorescence (interface hooks shipped in v1, no implementation).**
The third inelastic process; matters in the blue-green in CDOM-rich water.
No truth data exist in L23 (omitted by design) or anywhere in hand. The
`Inelastic.cdom_fl` slot and the additive-term pattern of §4.4 are its
landing zone.

*Update (2026-08-29):* CDOM fluorescence is now designed in a companion
document, [`design/rt_cdom_fluorescence_model.md`](rt_cdom_fluorescence_model.md)
(milestones M5/M6). The paragraph above is retained as the historical
decision record.

**HydroLight run wishlist** — what we would ask of new RT compute, in
priority order:

1. **Denser solar-zenith grid** (e.g. 0–75° in 15° steps, X1/X2/X4): breaks
   the 3-point geometry limitation that both the elastic emulator and the
   inelastic heads inherit; enables a real zenith-interpolation gate.
2. **Varied quantum yield** (e.g. φ_C ∈ {0.005, 0.01, 0.02, 0.04, 0.06} on a
   scene subset): the only direct test of the φ_C-linearity design bet
   (§4.4) and of retrieving φ_C in the eventual inversion.
3. **CDOM fluorescence on/off pairs** (HydroLight's Hawes et al. quantum
   functions) on a CDOM-stratified scene subset: creates the missing truth
   channel for the §8 hook.
4. **Off-nadir viewing geometries**: L23 is nadir-only; PACE is not.
5. **Alternative solar spectra / atmospheres** (e.g. a TSIS-1-based Ed vs
   the HydroLight default): quantifies the DQ5 solar-model concern and how
   it propagates through the Raman/fluorescence terms.
6. **Sub-350 nm output** (or at least excitation-side IOPs to ~330 nm):
   removes the λ < 400 nm extrapolation caveat for UV applications.

**Also beyond v1:** real-sky Ed coupling (the `Geometry.Ed` seam); the PS I
double-Gaussian validation (needs item 3-style dedicated runs or field
spectra); vertically structured Chl/φ_C (L23 is homogeneous; the 685 nm
signal originates shallower than the blue-green — a known, accepted
homogeneity bias).

## 9. Risks and open issues

- **Geometry extrapolation** (inherited): three zeniths cannot certify
  interpolation, let alone extrapolation; the elastic effort already showed
  geometry generalization is fragile. Mitigation: bounded corrections decay
  to the analytic physics; wishlist item 1 is the real fix.
- **φ_C-linearity** holds only to O(10⁻³) within the forward model — but the
  *correction* was trained at 0.02; if the true RT response to φ_C is less
  linear than our model's (no truth to check — wishlist item 2), retrieved
  φ_C inherits a shape bias.
- **Trophic extremes**: δ_F must fix a ×2 amplitude drift at the eutrophic
  tail where scenes are sparse; watch the a_ph(440)-decile diagnostic (§6).
- **Solar spectrum quality** (DQ5, JXP): v1 is internally consistent with
  L23's sun but the community's absolute solar references are suspect;
  revisit at the real-data transition.
- **X4−X2 as "pure fluorescence"** neglects Raman–fluorescence coupling
  (fluoresced photons Raman-scattering, etc.); in HydroLight these
  higher-order paths are present in X4 — they land in δ_F, which is
  acceptable at the 5 % gate but worth remembering if gates tighten.

## 10. References

- Bartlett et al. (1998), *Appl. Opt.* 37, 3324 — Raman coefficient & scaling.
- Sathyendranath & Platt (1998), *Appl. Opt.* 37, 2216 — two-flow transspectral terms.
- Gordon (1979), *Appl. Opt.* 18, 1161 — fluorescence-as-inelastic-scattering, 685 nm Gaussian.
- Maritorena et al. (2000), *Appl. Opt.* 39, 6725 — natural φ range ~1–6 %.
- Behrenfeld et al. (2009), *Biogeosciences* 6, 779 — φ carries physiology (NPQ, iron).
- Loisel et al. (2023), *ESSD* 15, 3711 — the L23 database; X1/X2/X4 scenarios.
- `context/RT/rt_inelastic_bing_summary.md` — the quantitative assessment behind §§4.3–4.5.
