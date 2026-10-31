# Elastic Radiative Transfer (RT)

This prompt doc will guide our efforts to develop elastic radiative transfer (RT) models for ocean color.

## Context

- Read the `context/context_summary.md` file.
- Examine the code for elastic scatterin in this Reposioty
- See the `report_rt_elastic_model.md` report in docs/reports


Read the papers (PDFs) in `context/RT/`. These are:

- `Twardowski_2018_ZTTAnalyticalModel.pdf` : A paper by Michael Twardowski et al. that develops a new analytical model for elastic scattering we refer to as the "ZTT" model.
- `Loisel_et_al_ESSD_2023.pdf` : The paper by Loisel et al. that describes the L23 dataset.

## Code

If you need to run Python use the `ocean14` conda environment. 

## Prompts

### Plan

1. Please read the items in Context above. Our first major goal is to re-derive the 16 coefficients for the ZTT model (instead of using theirs, which
we have not yet obtained).  We will fit to the L23 dataset.  And we will keep the current model which uses
the TT2017 coefficients, but make the L23 coefficients the default.  Before long, however, we will fit to the
PB24 dataset.  I am downloading that now.
Please discuss a plan with me for how to proceed.  Pose questions in the Q&A/Design section below.
Use Opus 5.5.  Log your work.

2. I have answered your first round of questions.  Read my answers
and continue the discussion in the Q&A/Design section below.
Use Opus 5.5.  Log your work.

3. I have answered your second round of questions.  Read my answers
and continue the discussion in the Q&A/Design section below if you
have more questions to ask.  If not, stand by and I'll give you further instructions.
Use Opus 5.5.  Log your work.

4. Ok, now that we are done discussing the design, please:
    - Update the files in `design/` -- `rt_elastic_model.md`, `rt_elastic_implementation.md`, and `rt_model_coding_plan.md` -- to include this new fit of ZTT to L23
    - Write a series of prompts in the "Fit L23" section below to build out the code and perform the fit, as per our plan

Use Opus 5.5.  Log your work.

### Fit L23

*Milestone **M7**, "ZTT (L23 µ∞)". Design: `design/rt_elastic_model.md` §4.6. Plan:
`design/rt_elastic_model_coding_plan.md` → M7. Record: `design/rt_elastic_implementation.md`
§7b. Decisions: Q1–Q12 below. Execute one prompt at a time; each ends with JXP reviewing
before the next.*

**Applies to every prompt below.**
- JXP runs all git. Python in `ocean14`, from the repo root (`robust` is not
  pip-installed). JAX CPU, float32 by default; gradient checks in float64 via the
  `jax_x64` fixture; reference arithmetic in tests in NumPy.
- Each task is **pytest-gated** and lands with its tests. Suites green with and
  without `$OS_COLOR` (missing data skips, never fails; use the existing
  `needs_l23` marker and a small committed fixture where a test needs data).
  `ruff check` and `ruff format --check` clean.
- **The `forward` freeze** (record §8.0) is in force. Appending `None`-default
  container fields and keyword-only arguments is allowed. Any change to pinned
  numbers is a deliberate re-baseline, called out in the log.
- **The hybrid must not move.** Its pinned values and the shipped emulator digest
  stay exactly as they are throughout M7 (Q2a).
- Fit on the **train split** (`l23.make_splits`, `SPLIT_SEED = 23`, by water
  body) and score on held-out scenes. Report per zenith. Every fit is
  deterministic (linear LSQ, no optimizer), and a rerun must be bit-identical.
- Score on everything, then explain (M5 gotcha 1). Write any causal story as a
  prediction and test it before believing it (M5 gotcha 4).
- After each task, update `design/rt_elastic_implementation.md` §7b with what was
  built and measured, and log here under **Logs**.

1. **Task 1 — L23 profiles and the µ∞ / µd product.** Add an opt-in, lazy profile
   reader to `robust/rt/data/l23.py` for `Hydrolight1YY_profile.nc`. Reuse
   `ocpy.hydrolight.loisel23` if it reads these; otherwise open them with xarray
   directly. Each file is ~727 MB, so never load more than the fields asked for.
   From it, build a per-(scene, λ) product:
   (a) µ(z) = (Ed − Eu)/Eo at each depth and zenith;
   (b) **µ∞**, the depth→∞ limit fitted **jointly across the three zeniths**
   (they share µ∞), e.g. µ(τ) = µ∞ + c_θ e^{−τ/τ₀} in optical depth τ = ∫K dz.
   Choose the form by inspecting real profiles first, and record why;
   (c) a quality flag: `asymptotic` (zeniths agree to < 1% at depth),
   `extrapolated`, or `unusable`;
   (d) µd(0⁻) (`md_z` at z = 0) and KLu(0⁻) per zenith.
   Cache the product (it's small) with a provenance header, either outside the
   repo next to the data or as a committed `.npz` if under a few MB. Ask in Q&A
   if unsure. Report coverage of each flag in (log bb/a, log η_bb) as a figure.
   **Gate:** Gershun `(Ed−Eu)/Eo = a/KEd` to 1e-3 (median) at depth; golden rows
   vs the raw netCDF; the product is reproducible bit for bit; on `asymptotic`
   points the extracted µ∞ is zenith-independent to < 1%.
   Use Opus 5.5. Log your work.

2. **Task 2 — Fournier–Forand and L23's phase function.** In `robust/rt/ztt.py`
   (or a small `robust/rt/phase.py` if cleaner): the FF phase function p̃FF(ψ; n, µ)
   and its backscatter ratio B(n, µ) in closed form, in JAX. Use HydroLight's
   convention n = 1.01 + 0.1542(µ − 3) (Mobley et al. 2002), solving for µ from B
   in a differentiable way (e.g. a precomputed monotone table plus `jnp.interp`,
   or implicit differentiation). Then add the two-component particulate backward
   phase function
   Pbb(ψ) = [Σ_k w_k p̃FF(ψ; B_k)] / [Σ_k w_k B_k], with w_k = scattering fraction.
   Append `ff_B` and `ff_weight` to `PhaseParams` (default `None`; component axis
   last; `ff_weight` may be spectral), update the freeze's field-order pin
   deliberately, and have the L23 loader fill them (B = 0.010 for phytoplankton,
   0.018 for non-algal particles; weights from bph, bd). Extend `P_bb_from_phase`
   so FF is used when `ff_B` is set and Sullivan otherwise, **silently** (Q9).
   **Gate:** p̃FF integrates to 1 over the sphere (quadrature, float64); B(µ)
   round-trips 0.010 and 0.018; FF Pbb(180°) for both B values reported against
   Sullivan's 0.153; `ff_B=None` reproduces today's Pbb bit for bit; float64 grad
   vs FD for the new path; every existing test green.
   Use Opus 5.5. Log your work.

3. **Task 3 — The fits.** Write `design/py/fit_ztt_l23.py` (deterministic, with
   `--dry-run`) that writes coefficients and a metrics record to
   `design/ztt_l23/`:
   (a) **µ∞, Eq. (8)**: linear LSQ for m1..m16 on log µ∞ (relative residuals)
   over train-split points flagged `asymptotic`, plus `extrapolated` points with
   a down-weighting you justify. Record the fitted box in (X, L) = (log bb/a,
   log η_bb) for clamping;
   (b) **µd**: M*_d (Eq. 17, 8 coefficients) **jointly with** c(θs′) =
   1 + c₁(1−cos θs′) + c₂(1−cos θs′)² on TT's M⁺_d (Q10b): linear LSQ in log µd
   over all three zeniths;
   (c) **Ψ_KLu diagnostic** (Q11, no refit): KLu(0⁻)/K∞ at ψ = 180°, ~158°,
   ~140°, against TT's 1 + F(ψ), computed with *both* µ∞ variants.
   Report held-out µ∞ and µd errors (median, 16/84/99%) per zenith and as maps in
   (X, L), and set them against TT's quoted 0.19–3.5% (µ∞) and <1% (µd).
   **Gate:** bit-identical rerun; held-out errors reported, not thresholded; the
   fitted coefficients reproduce the script's own metrics when evaluated through
   the functions that task 4 will ship (write those evaluators here, in
   `ztt.py`).
   Use Opus 5.5. Log your work.

4. **Task 4 — Wire the L23 variant in, and pin the hybrid.** In `ztt.py`:
   `MU_INF_L23_COEFFS`, `MD_STAR_L23_COEFFS`, `MD_PLUS_CORR_L23` (with provenance
   docstrings in the house style) and the clamp box; `mu_infinity_l23`, the
   refit µd; keyword-only `variant={"l23","tt2017"}` (default **"l23"**) and
   `mu_inf_model={"l23","l23_closure","tt2017"}` on `rrs_ZTT` and `Rrs_ZTT`.
   Precedence: `mu_inf=` > `mu_inf_coeffs=` > `mu_inf_model=` > `variant`'s
   default. Clamp with a smooth, differentiable clip. **In the same change** (so
   nothing is ever wrong in between): every learned artefact (`emulator_l23.npz`,
   the inelastic/CDOM correction files, any PB24 artefact) records
   `backbone_variant`, defaulting to `"tt2017"` when absent; `hybrid` and the
   emulator's training/feature path pass it to `rrs_ZTT`. Audit every
   `rrs_ZTT` call site (`hybrid.py`, `emulator.py`, `validation.py`,
   `baselines.py`, `design/py/*`) and record each one's variant choice.
   Re-baseline the pinned `forward(mode="ztt")` values deliberately, and flag it
   at the top of the log entry.
   **Gate:** `variant="tt2017"` equals today's rrs bit for bit on the fixture;
   hybrid / `forward(mode="hybrid")` pins and the emulator SHA-256 unchanged;
   the clamp holds at and beyond the box edges with finite gradients; float64
   grad vs FD in a, bb_p, B_p, θs for the L23 variant.
   Use Opus 5.5. Log your work.

5. **Task 5 — The closure µ∞ (Q1).** Eq. (12) is linear in 1/µ∞, so for each L23
   sample solve for the µ∞ that makes ZTT reproduce the reference rrs exactly,
   **using the task-4 L23 µd and the FF Pbb**, so it absorbs as little else as
   possible. Drop or flag samples where the solution is non-physical (≤ 0 or > 1)
   and report how many. Fit Eq. (8) to those values the same way as task 3, add
   the coefficients to `fit_ztt_l23.py`'s outputs, and ship them as
   `mu_inf_model="l23_closure"`. Docstring: **a ceiling, not physics.** Plot
   closure µ∞ against physical µ∞; the gap is the error budget of the other terms.
   **Gate:** reproducible; closure-variant rrs on the train split equals the
   reference within the Eq. (8) fit residual (a self-consistency check).
   Use Opus 5.5. Log your work.

6. **Task 6 — Re-score the backbone: the M7 acceptance gate.** Extend
   `design/py/run_validation.py` (or add `design/py/run_ztt_l23.py`) to score an
   **additive ladder** on train / held-out scenes / each zenith, per λ and per
   B_p bin: ZTT (TT2017 µ∞) → +L23 µd → +L23 µ∞ → +FF Pbb (= "ZTT (L23 µ∞)") →
   closure µ∞, alongside Gordon and O25 refit from the existing ladder. Each
   step's gain is then attributed to the term that earned it. Write
   `design/ztt_l23/metrics.{md,csv}` and figures (dataviz conventions).
   **Acceptance:** "ZTT (L23 µ∞)" beats "ZTT (TT2017 µ∞)" on held-out scenes at
   **each** of 0°, 30°, 60°. If the gate fails, say so plainly and diagnose with
   the ladder; do not tune to pass.
   Use Opus 5.5. Log your work.

7. **Task 7 — Close M7.** An executed explainer notebook
   `notebooks/RT/rt_fit_ztt_l23.ipynb`; complete `design/rt_elastic_implementation.md`
   §7b (status ✅, results, module index, test counts); a review pass over the
   M7 diff (fix or log every finding); and a hand-off listing what comes next.
   That includes: (i) retrain the emulator on the `l23` backbone (Q2b) and
   compare with 0.30%; (ii) PB24 when back at `$OS_COLOR/SD/v5`: re-check
   c(θs′) and Ψ_KLu over ten sun angles, closure-only µ∞ unless profiles exist
   (Q12); (iii) whether the elastic report should gain a section.
   Update the report in `docs/reports/report_rt_elastic_model.md` to include the new results.
   Use Opus 5.5. Log your work.

8. Ok, we are ready to move on to PB24.  I have downloaded their dataset into 
`$OS_COLOR/SD/v5`.  Generate a new prompt doc `rt_fit_pb24.md` in this folder with a series of Tasks:
   - Build new code in the `ocpy` Repo to read the PB24 dataset and prepare it for analysis. I am now on a `pb24` branch in that repo.
   - First assess the RMS of the existing elastic models on the PB24 dataset.
   - Discuss with me how to improve on the elastic backbone
   - Implement the new backbone and assess its performance on the PB24 dataset.

   If you have any opening questions, put them in a Q&A section of the new doc.
   Use Opus 5.5.  Log your work.

### Next (M7 hand-off)

M7 is complete. From record §7b.11, the next steps:

1. **PB24**, once it is back at `$OS_COLOR/SD/v5`:
   - µd's c(θs′) over ten sun angles;
   - Ψ_KLu off-nadir, changed only together with f_L;
   - µ∞ by eigenproblem on PB24's varied phase functions (or closure-only,
     unless PB24 has profiles).
2. **Retrain the emulator** on the surviving backbone after PB24 (Q2b/Q18).
   Re-pin the weights (Q19), and make an "l23" emulator warn when `ff_B` is
   missing (review finding 1b).
3. Revisit the elastic report's §9 once PB24 lands.

## Q&A

### Fit L23

*Questions raised while executing the "Fit L23" prompts. Answer inline under
each **JXP:** line.*

**Q13 (task 1 → task 3, Claude → JXP, *blocking task 3*). Where should µ∞
come from? The profiles alone cannot supply it.**

Task 1's finding (record §7b.5, `design/ztt_l23/fig_profile_coverage.png`): the
three suns agree to < 1% at depth, the only honest sign the light field has gone
asymptotic, on **2.4%** of (scene, λ). All of those are particle-dominated
water (η_bb ≈ 0.03–0.58, median 0.10, against L23's median 0.49). Clear water
never gets there by 50 m, and the approach to µ∞ is non-monotonic (0° falls from
above; 60° falls and then rises), so I found no extrapolation I'd trust. Fitting
Eq. (8)'s 16 coefficients on the 2.4% would leave most of L23's (bb/a, η_bb)
domain to polynomial extrapolation.

What does work: **compute µ∞ exactly from L23's own IOPs.** µ∞ is an IOP, the
smallest-attenuation eigenmode of the azimuthally averaged RTE, and L23 gives us
every input: a, b_w, b_ph, b_d, and the phase functions (water plus FF with
b̃b = 0.010 / 0.018). A discrete-ordinate eigensolve takes about 0.5 ms per point,
so the full release takes minutes (prototype: `design/py/explore_mu_inf_eigen.py`).
Checked against the 2.4% where the profiles *do* measure µ∞:

| FF (n, µ) convention | eigen / profile µ∞, 16 / 50 / 84% |
|---|---|
| **Mobley 2002** (n = 1.01 + 0.1542(µ−3)) | 1.0108 / **1.0141** / 1.0170 |
| fixed slope 3.5 | 1.0108 / 1.0140 / 1.0170 |
| fixed slope 4.0 | 1.0217 / 1.0353 / 1.0494 |
| fixed n = 1.05 | 1.0136 / 1.0203 / 1.0264 |
| fixed n = 1.20 | 1.0161 / 1.0220 / 1.0272 |

The scatter is tight (±0.3%), but there is a **systematic +1.4%** offset. It
doesn't change with quadrature order (40 → 120 streams), δ-truncation angle
(0–10°), or the water depolarization, so it isn't numerical. Mobley-2002 gives
the smallest offset of the five, which is mild support for our Q8 assumption.
The offset isn't from the profiles being short of asymptotic either: the eigen
value sits *above* all three zeniths, including 0°, which approaches from above.
My leading suspect is HydroLight's own discretization (it averages the phase
function over ~10° quads, which smears the forward peak), but that is a
hypothesis I haven't tested.

Options:
(a) **Eigen µ∞ as the fit target everywhere**, with the +1.4% reported as a known
offset against HydroLight.
(b) As (a), but rescaled by the measured offset. This calibrates the eigen µ∞ to
HydroLight, assuming the offset seen in turbid water holds everywhere, which we
can't check in clear water.
(c) Profile-only on the 2.4%. Not viable as the default, in my view.

**I recommend (a) now**, with the offset carried as a stated systematic and the
2.4% kept as a validation set. If the planned HydroLight runs (M6 Q18) can
include a few clear-water IOP sets run very deep, or with HydroLight's
infinitely-deep asymptotic output, that would test (b) properly. Either way the
gain looks real: against eigen µ∞, TT2017's ratio is 1.007 at the median but
**1.116 at the 84th percentile and 1.29 at the 99th**.
**JXP:**  Yes, go with (a)

**Q14 (task 1, Claude → JXP, *not blocking*). Ordering, if Q13 is (a).** The
eigensolver needs the FF phase function, which is task 2. I propose it lands as
a NumPy fitting utility (fitting-only, never on the differentiable path), e.g.
`robust/rt/asymptotic.py`, at the start of task 3 and built on task 2's FF.
Task 1 then closes as built. Its product stays as the validation set and as the
µd(0⁻) / KLu(0⁻) source.
**JXP:**  Use your plan, it is sensible.

**Q15 (task 1, Claude → JXP, *not blocking*). Cache location.** The full
product (12.8 MB) is cached at
`$OS_COLOR/Loisel2023/derived/l23_profile_product_v1.npz`, i.e. next to the
data as the prompt allowed, in a new `derived/` folder. It rebuilds in 7 s.
Is that folder OK?
**JXP:** Yes, that is ok.  Also remind me to rclone it to the `RoB` shared Google Drive folder.

**Q16 (task 3 → tasks 4/6, Claude → JXP, *not blocking task 4*). TT's Ψ_KLu
is ~13% high against L23. Correct it, or measure first?**

The Ψ_KLu diagnostic (record §7b.7) turned up something the plan didn't
expect. At nadir, ZTT needs only the product Ψ_KLu/µ∞ = KLu(0⁻)/a, and L23
measures that directly. TT's 1 + F(ψ) is about **13% high** at all three ψ
(180°, 158°, 140°) once the L23 µ∞ is used, against about 9% with TT2017. So
TT2017's µ∞, which is too high, partly cancels TT's Ψ_KLu error. My prediction,
recorded before task 6 measures it: **the L23 µ∞ alone will lower nadir rrs by
about 2% where a/bb dominates, and could make the backbone worse.**

Options:
(a) Measure first. Task 6's ladder adds a diagnostic rung with KLu(0⁻)/a taken
straight from L23 (an oracle) to size the prize, then we decide.
(b) Add a nadir Ψ_KLu correction now: a multiplicative s(ψ) fitted at the three
ψ, the way c(θs′) was. The ratio is nearly constant in ψ (~1.13), so even a
single scale would capture most of it. It can't be validated off-nadir with
L23, and it ignores the ±10% IOP-driven spread.
(c) As (b), but with an IOP-dependent form, since the measured Ψ_KLu varies
±10% with the IOPs where TT assume < 2%.

**I recommend (a).** It costs one extra rung in task 6 and turns the decision
into a measurement. Task 4 doesn't depend on the answer.
**JXP:** Yes, let's do option (a).

**Q17 (task 3, Claude → JXP, *not blocking*). An output guard I added.** L23
fills only a triangle of the (log bb/a, log η_bb) clamp box, and in the box's
empty corners the fitted µ∞ polynomial reaches −1.1 and 1.6. So both refit
polynomials also carry a smooth output bound: their training range plus a
small margin, kept below 1 for µ∞. It is invisible on L23 (held-out median
error 0.209% with or without it). It matters for inputs outside L23's
distribution, such as an inversion wandering, or PB24 with η_bb down to 4e-6.
OK to keep?
**JXP:** That is fine.

**Q18 (task 4, Claude → JXP, *not blocking*). Two backbones in `forward`
until the emulator is retrained.** Since `rrs_ZTT` now defaults to "l23",
`forward(mode="ztt")` is "ZTT (L23 µ∞)", while `forward(mode="hybrid")` is
still "ZTT (TT2017 µ∞)" plus the shipped δ (pinned, Q2a). That is correct
for each mode, but `hybrid − ztt` is no longer the emulator term, and a reader
comparing the two modes would be comparing two backbones. It ends when the
emulator is retrained on "l23" (Q2b). When should that happen: right after
task 6, or as part of task 7's hand-off? I suggest straight after task 6,
because task 6 also tells us whether the Ψ_KLu correction (Q16) should land
first. Retraining on a backbone that is about to change would waste a training
run.
**JXP:**  Let's hold off.  We are going to start fitting to PB24 soon, and we don't 
want to retrain the emulator until then.

**Q19 (task 4, Claude → JXP, *not blocking*). The shipped weights no longer
regenerate bit for bit, and this predates M7.** `design/py/train_emulator.py`
run from your last commit (`9f3328a`, via `git archive`, so independent of my
changes) writes a file that differs from the committed `emulator_l23.npz`. It
adds the M5 `envelope/*` keys, and the weights differ by up to 0.02. The
current tree gives the identical file, so M7 is not the cause. The elastic
report (§8) still says "bit-identical committed weights". Options: (a) leave
the shipped file (its SHA is pinned and every number is reproducible *from
the file*) and correct the report's wording; (b) regenerate and re-pin. That
would move every hybrid value pin, so I'd rather fold it into the Q2b retrain.
I recommend (a) now.
**JXP:** Go with (a) for now.

**Q20 (task 6, Claude → JXP, *not blocking task 7*). Where next for the
analytic backbone?** The gate passed (2.90% held-out, down from 5.93%), and the
diagnostics say where the rest of the error lives:
- Ψ_KLu can't be corrected alone. L23's measured KLu/a makes things *worse*
  (7.8–9.5%) because TT's f_L is tuned against their Ψ_KLu.
- µ∞ could buy about half the remaining error, but only as an empirical knob
  (the closure, 1.37%).
- O25 refit on L23 is at 0.69%, and the TT2017 hybrid at 0.30%.

Options:
(a) **Stop here for L23.** Ship "ZTT (L23 µ∞)" as the physically fitted
backbone and move the effort to PB24 (µd with ten sun angles, off-nadir ψ),
then retrain the emulator on whichever backbone survives PB24.
(b) **Jointly refit Ψ_KLu and f_L** (Eq. 31's spectral shape and its ψ scaling)
on L23 at nadir. This is empirical calibration of ZTT's two shape factors; it
is likely to approach the closure ceiling, but off-nadir it would be
unvalidated until PB24.
(c) Accept the closure µ∞ as the working backbone for the emulator. That's
the cheapest way to a smaller residual for the network, but it gives up µ∞'s
physical meaning.

I recommend (a). The remaining L23 gains are calibration rather than physics,
and PB24 is where the backbone's real weakness (off-nadir ψ, record §7.16)
gets tested.
**JXP:** Go with (a)

### Design

*Written by Claude (Opus 5.5), 2026-10-30, in response to Plan prompt #1.
Answer inline under each **JXP:** line.*

#### What we are fitting, and against what

Equation (8) is a cubic in `X = log10(bb/a)` whose four coefficients are cubics in
`L = log10(η_bb)`, so `µ∞` is **linear in m1..m16**: given target `µ∞` values the
fit is a single least-squares solve (bit-reproducible, no optimizer).

The question is what the target is. Twardowski & Tonizzo fit Eq. (8) to the
*asymptotic average cosine* computed by HydroLight — a physical quantity, not a
closure knob. **L23 lets us do the same:** the `Hydrolight1YY_profile.nc` files
(which our loader does not read today) carry `Ed_z, Eu_z, Eo_z, KLu_z, KEd_z,
KEu_z, md_z, mu_z` on 61 depths to 50 m. By Gershun, `µ(z) = (Ed − Eu)/Eo = a/K`,
and `µ(z) → µ∞` at depth.

#### Feasibility check (done, scratch script, not committed)

On `Hydrolight100_profile.nc` (θs = 0°), all 3320 × 81 (scene, λ):

| quantity | value |
|---|---|
| L23 `bb/a` | 1.1e-4 – 0.59 (median 0.029; TT fitted 1e-4 – 0.1) |
| L23 `η_bb` | 0.017 – 0.97 (median 0.49; TT fitted 0.0098 – 0.98) |
| `(Ed−Eu)/Eo` vs `a/KEd` at 50 m | agree to 1e-4 (Gershun holds; data are clean) |
| fraction with KEd, KEu, KLu within 1% of each other at 50 m | 63% (0°), 82% (30°), 56% (60°) |
| median \|µ(50 m, 60°)/µ(50 m, 0°) − 1\| | **9.8%** |
| TT2017 µ∞ / L23 µ(50 m, 0°) | median 1.000; 16–84%: 0.975 – 1.035; 99%: 1.16 |
| naive Eq. (8) LSQ on the "converged" 0° points (N = 156k) | MAPE 0.57%, max 92% |

Two lessons. (1) The machinery works and TT2017 is already close (≈ ±3%), so
expect a modest but real gain in the backbone. (2) **50 m is not deep enough to
be asymptotic for a large share of scenes** — µ at 50 m still depends on sun
angle by ~10%, which an asymptotic quantity must not. Extracting µ∞ carefully is
the crux of step 1 below; the 92% max error is that problem showing through.

#### Proposed plan

1. **Extract µ∞ from L23 profiles** (`robust/rt/data/l23.py`: add a profile
   reader + `load_mu_inf()`). Per (scene, λ): use µ(z) = (Ed−Eu)/Eo at all three
   sun angles, keep depths with optical depth K·z above a threshold, and
   estimate µ∞ as the depth→∞ limit (e.g. fit µ(τ) = µ∞ + c·e^{−τ/τ₀} jointly to
   the three zeniths, which share µ∞). Flag each point: *asymptotic* (zeniths
   agree to < 1%), *extrapolated*, or *unusable* (clear-water blue where 50 m is
   only ~1–2 optical depths). Report the coverage in (bb/a, η_bb).
2. **Fit Eq. (8)** by linear LSQ on log-µ (relative residuals) over the train
   split (`SPLIT_SEED = 23`, by water body), scored on held-out scenes. Report
   MAPE vs TT's quoted 0.19–3.5%, residual maps in (X, L), and behaviour at the
   domain edges. Script: `design/py/fit_mu_inf.py`, deterministic.
3. **Wire it in** (`robust/rt/ztt.py`): ship the 16 numbers as a constant
   `MU_INF_L23_COEFFS` (same style as the Table A2 constants, with provenance),
   add `mu_inf_model: {"l23", "tt2017"}` to `rrs_ZTT`/`Rrs_ZTT` with **"l23" the
   default**; `mu_inf_coeffs=` and `mu_inf=` still override. TT2017 stays
   available unchanged.
4. **Re-score the backbone** on L23 (train / held-out scenes / held-out 60°)
   with both µ∞ variants, alongside the existing ladder, and update tests
   (golden values that pinned TT2017 move to an explicit `mu_inf_model="tt2017"`).
5. **Diagnostic, not default — the closure fit.** Because Eq. (12) is linear
   in 1/µ∞, each L23 rrs can be inverted for the µ∞ that makes ZTT exact. Fitting
   Eq. (8) to *those* values tells us how much of the backbone's 5.9% error µ∞
   *could* absorb — but it would soak up f_L, µd and Pbb errors too, so it is a
   ceiling, not a physical µ∞. I recommend reporting it, not shipping it.
6. **PB24 later**: same pipeline once the new download is in hand (see Q6).

#### Questions

**Q1. Target: physical µ∞ (profiles, TT's own method) or closure (rrs)?**
I recommend physical as the shipped default, closure as a reported ceiling
(step 5). Shipping the closure fit would improve rrs more but would stop µ∞
meaning µ∞, and would double-count whatever the emulator later learns.
**JXP:**  Ok, let's go with the physical µ∞.  But do record the closure fit in the
code as well, as yet another option for ZTT.

**Q2. What happens to the hybrid when the default backbone changes?**
The committed emulator (`emulator_l23.npz`) and the inelastic corrections were
trained on residuals of the *TT2017* backbone; swapping µ∞ under them silently
breaks the 0.30% number. Options: (a) store the backbone variant in each
weights file and have `hybrid` pin to it (keeps every published number
reproducible), then retrain on the new backbone as a separate step; (b) retrain
immediately. I recommend (a) now, (b) once the L23 µ∞ is settled.
**JXP:**  Ok, let's go with option (a).

**Q3. Out-of-domain behaviour.** L23 reaches bb/a = 0.59 and η_bb = 0.017; a
16-term polynomial in logs extrapolates badly. Clamp X and L to the fitted range
(differentiable via soft-clip), or let it extrapolate with a `DomainWarning` as
the emulator does? I lean clamp.
**JXP:** Clamp it.

**Q4. Scope: µ∞ only, or also refit Eq. (17) M*_d (8 coefficients) and the
f_L terms?** L23 profiles carry `md_z` at z = 0⁻ — exactly µd — so refitting
M*_d is nearly free, and our transcription of µd already needed one convention
fix. I suggest µ∞ first, M*_d as a quick follow-on, f_L later (it is fitted to
TT's own synthetic set and is the next-largest suspect).
**JXP:** Ok, refit both µ∞ and M*_d.

**Q5. Is it legitimate that µ∞ depends only on (bb/a, η_bb)?** It ignores the
particle phase function, and L23 has only one FF shape (B_p 0.010–0.018), so L23
cannot test that. PB24's wider B_p range can. Fine to accept Eq. (8)'s form for
L23 and revisit with PB24, or do you want a B_p term considered from the start?
**JXP:**  This is a good point.  I would also like to modify this fitted L23
version of ZTT to use the same particle phase function as in L23.

**Q6. Which PB24 are you downloading?** The repo already loads PB24 v5 surface
AOPs (`mu_d`, `KLu`, `Kd`, … on (λ, θs)). µ∞ needs depth profiles (or
deep-water K). If the new download has profiles, step 1 transfers directly; if
not, PB24 can only give a closure-type µ∞ (step 5). Which is it, and where will
it live?
**JXP:**  Where are the PB24 files?

**Q7. Name for the variant in reports.** Proposal: "ZTT (L23 µ∞)" vs "ZTT
(TT2017 µ∞)", reserving plain "ZTT 2018" for the authors' coefficients if they
ever arrive.
**JXP:** Yes, use that.

#### Round 2 — Claude (Opus 5.5), 2026-10-30, responding to JXP's answers

**Settled by your answers.** Physical µ∞ (from L23 profiles) is the default,
and the closure-fit µ∞ ships as a third, non-default option (Q1). The hybrid and
inelastic corrections pin to the backbone variant recorded in their weights
files (Q2a). µ∞ and M*_d inputs are clamped to the fitted range (Q3). Both µ∞
(Eq. 8) and M*_d (Eq. 17) get refit (Q4). Naming: "ZTT (L23 µ∞)" / "ZTT
(TT2017 µ∞)", with "ZTT 2018" reserved for the authors' coefficients (Q7).

**Answer to your Q6 question ("Where are the PB24 files?").** The loader looks
in `$OS_COLOR/SD/v5`, i.e. `/home/xavier/Oceanography/data/Color/SD/v5`
(`robust/rt/data/pb24.py`, `SUBDIR`). The M5 log
(`rt_elastic_coding_prompt_6.md`) records it there on 2026-08-08: 28 GB, 10,001
files (`SD_OLCI_no_R_NNNN.nc`, `SD_hyp_no_R_NNNN.nc`, and
`classes_Rrs_OLCI_v5_20240214.mat`). **It is not on this machine now.**
`$OS_COLOR/SD` doesn't exist, and a search of `/mnt/tank` and `/home/xavier`
for `SD_OLCI_no_R_0001.nc` finds nothing. Put the new download back in
`$OS_COLOR/SD/v5` and the existing loader and tests will pick it up with no
changes. (If the new release has a different layout or version, tell me and I'll
adapt the loader.)

##### New finding 1 — the L23 phase function is two Fournier–Forand components

On your Q5 answer (use L23's own particle phase function): L23 does not use a
single phase function per scene. The component ratios are constant to 1e-6:
`bbph/bph = 0.0100` (phytoplankton) and `bbd/bd = 0.0180` (non-algal particles)
in every scene and at every wavelength. The bulk `B_p = bbnw/bnw`
(0.0103–0.0180) is just the scattering-weighted mix of the two. So L23's exact
particulate backward phase function is

  Pbb(ψ) = βp(ψ)/bbp = [bph · p̃FF(ψ; 0.010) + bd · p̃FF(ψ; 0.018)] / (bbph + bbd),

which is wavelength-dependent through the bph/bd mix. This is the
"Chl-dependent Pbb,FF" case of TT's Figure 3A, done exactly. FF has closed forms
for both p̃(ψ) and B, so this stays analytic and differentiable. Point 7 of the
revised plan covers implementing it; Q8 and Q9 below are the decisions it needs.

##### New finding 2 — ZTT's µd is the larger zenith problem, not µ∞

Same kind of check as round 1, now on µd = Ed/Eod at z = 0⁻ (`md_z`; Ed/Eod
matches it exactly). This compares ZTT's Eq. (14), as transcribed, with L23:

| sun zenith | L23 µd (median) | ZTT/L23: 1% | 16% | **median** | 84% | 99% |
|---|---|---|---|---|---|---|
| 0° | 0.950 | 0.973 | 0.977 | **0.981** | 1.010 | 1.051 |
| 30° | 0.888 | 0.992 | 0.996 | **0.999** | 1.025 | 1.064 |
| 60° | 0.752 | 1.084 | 1.086 | **1.087** | 1.100 | 1.130 |

**At 60°, ZTT's µd is 8.7% too high**, and rrs ∝ 1/µd, so the backbone comes
out ~8% low there. That matches the backbone's 60° degradation (4.3% → 8.1%
rRMS in the elastic report) and is probably most of it. The bias is almost
entirely in the **atmospheric factor M⁺_d(θs′)**, not the IOP factor. The
narrow 1–16% band shows a near-constant offset per zenith, and L23's own
µd(60°)/µd(0°) ratio is 0.787–0.820 across all scenes. That means the
factorization in Eq. (14) holds to about ±2%, but TT's H(θs′, V) and P3 terms
don't describe L23's sky (HydroLight's default atmosphere, 5 m/s wind).
Refitting M*_d alone can't fix that. See Q10.

##### Revised plan

1. **Profile reader** in `robust/rt/data/l23.py` (`load_profiles`, opt-in,
   lazy; the files are 727 MB each) plus a cached per-(scene, λ) product with
   µ(z) = (Ed−Eu)/Eo, µd(0⁻), and KLu(0⁻).
2. **µ∞ extraction**, as planned in round 1: the depth→∞ limit fitted jointly
   across the three zeniths, with per-point quality flags.
3. **Fit µ∞ (Eq. 8)** by linear LSQ in log-µ on the train split, scored on
   held-out scenes, with inputs clamped to the fitted (X, L) box.
4. **Fit µd**: M*_d (Eq. 17, 8 coefficients) jointly with an atmospheric
   correction (Q10), linear LSQ in log space.
5. **Closure µ∞**: invert Eq. (12) for the µ∞ that makes ZTT exact on L23
   (using the FF Pbb and the refit µd, so it absorbs as little else as
   possible), then fit Eq. (8) to those values. This becomes
   `mu_inf_model="l23_closure"`.
6. **One fit script**, `design/py/fit_ztt_l23.py`: deterministic, writes the
   coefficients plus a short metrics/figures record under `design/ztt_l23/`.
7. **Wire into `ztt.py`**: coefficient constants with provenance, `FF`
   phase-function functions, the `variant=` preset (Q9), and clamping.
   Backbone-variant metadata goes in the emulator and correction files (Q2a).
8. **Re-score** the backbone ladder (train / held-out scenes / per zenith) for
   TT2017, L23 and L23-closure, adding the terms one at a time (µd → µ∞ → FF
   Pbb), so each gain is attributed to the term that earned it.
9. **Tests**: today's golden values move to an explicit `variant="tt2017"`.
   New tests check that the fit script reproduces the shipped coefficients,
   that clamping holds, that the FF closed form matches numerical integration,
   and that gradients stay finite.
10. **PB24**: once it's on disk, repeat steps 3–8 where the data allow (Q12).

##### Questions, round 2

**Q8. Fournier–Forand convention.** An FF phase function is fixed by two
numbers, the real index *n* and the PSD slope *µ*. A given B picks out a curve
in (n, µ), not a point. The usual HydroLight choice ("FF with specified bb/b",
Mobley et al. 2002) ties them by *n* = 1.01 + 0.1542(*µ* − 3) and solves for µ
from B. I plan to assume that, since it's what HydroLight does by default, and
to check it two ways: (i) the closure residual should shrink when the FF Pbb
replaces Sullivan, and (ii) the 180° value it gives can be compared with
Sullivan's 0.153. Does the L23 paper (Loisel et al. 2023 §2) state the FF
parameters explicitly? If you have the PDF, putting it in `context/RT/` would
settle this.
**JXP:** I don't believe they state them, but I have put the paper PDF in `context/RT/` for you to check.

**Q9. What "L23 the default" means for the phase function.** The FF Pbb needs
information most callers won't have (the bph/bd split and per-component B). I
propose a preset: `rrs_ZTT(..., variant="l23")` (the default) uses L23 µ∞ +
L23 µd, and uses the FF Pbb **when `PhaseParams` carries FF component
information**, otherwise Sullivan. `variant="tt2017"` reproduces today's
numbers bit for bit. `PhaseParams` would gain optional fields along the lines of
`ff_B` (per-component backscatter ratio) and `ff_weight` (per-component
scattering fraction, spectral), and the L23 loader would fill them. Is it OK
that, without those fields, the default silently falls back to Sullivan? The
alternative is a hard error, or a warning once per session.
**JXP:**  Please do as you describe.  And, yes, it is ok to fall back silently to Sullivan.

**Q10. How to fix µd's atmospheric factor.** L23 has only three sun angles and
one atmosphere, so the 18-coefficient H(θs′, V) can't be refit. Options:
(a) refit M*_d only. Simple, but it leaves the +8.7% at 60° in place.
(b) keep TT's M⁺_d and multiply by a smooth correction c(θs′) = 1 +
c₁(1−cos θs′) + c₂(1−cos θs′)², fitted jointly with M*_d. This uses 2 extra
numbers, fixes all three zeniths, and stays smooth in between, but it's tied to
L23's single atmosphere and visibility.
(c) as (b), but defer it to PB24, which has 10 sun angles.
I recommend (b) now, with the honest caveat that three nodes can't test
interpolation, and revisit with PB24's 10 angles under (c).
**JXP:** Do (b)

**Q11. Should Ψ_KLu (Eqs. 4/6) get a check too?** L23 gives KLu(0⁻) directly,
so Ψ_KLu = KLu(0⁻)/K∞ is measurable at ψ = 180°, 158°, and 140° (nadir view,
three suns). TT's F(ψ) quartic was fitted alongside *their* µ∞, so once µ∞
changes, the two may no longer be consistent. Three ψ values can't support
refitting a quartic, so I propose a **diagnostic only** for now, reported next
to the µ∞ fit, and a refit only with PB24's angular coverage. Agree?
**JXP:** I agree

**Q12. PB24 and µ∞.** As loaded today, PB24 provides near-surface µd, µu,
µtot and seven K's per (λ, θs), but no depth profiles. Its 10 sun angles make
it well suited to µd (Q10c). For µ∞, unless the new download includes profiles
or a deep-water K, PB24 can only give the closure-type µ∞. Does the release
you're fetching include depth profiles?
**JXP:**  We will see, but it might not

#### Round 3 — Claude (Opus 5.5), 2026-10-30: plan locked, no further questions

**Q8 resolved from the paper** (`context/RT/Loisel_et_al_ESSD_2023.pdf`, §2–3).
L23 uses Fournier–Forand (1994) phase functions with b̃b-ph = 0.01
(phytoplankton) and b̃b-d = 0.018 (non-algal particles; Mobley 1994's Petzold
average), held spectrally constant. The phase functions still vary with λ
through bph(λ) and bd(λ). The paper confirms what the data showed in round 2.
The simulations ran with HydroLight v5.0, 5 m/s wind, clear sky, and the
standard atmosphere. As you expected, the FF (n, µ) pair is **not** stated. I
will therefore use HydroLight's own "FF from bb/b" convention (Mobley et al.
2002: *n* = 1.01 + 0.1542(*µ* − 3), solving for *µ* from B), and validate it
two ways: the closure residual must shrink when the FF Pbb replaces Sullivan,
and the FF closed form for B must reproduce 0.010 and 0.018. Every report will
state the convention as an assumption.

**Final decisions:**

| item | decision |
|---|---|
| µ∞ default | physical, from L23 profiles (Eq. 8 refit); `mu_inf_model="l23"` |
| µ∞ options | `"l23_closure"` (inverted from rrs), `"tt2017"` (current) |
| µd | refit M*_d (Eq. 17) **jointly with** c(θs′) = 1 + c₁(1−cos θs′) + c₂(1−cos θs′)² on TT's M⁺_d (Q10b) |
| out of range | clamp log(bb/a) and log η_bb to the fitted box |
| phase function | `variant="l23"`: FF two-component Pbb when `PhaseParams` carries `ff_B`/`ff_weight`, **silent** fallback to Sullivan otherwise; `variant="tt2017"` = today's model, bit for bit |
| hybrid / corrections | pinned to the backbone variant stored in their weights files; retraining is a separate step |
| Ψ_KLu | diagnostic only (L23 nadir: ψ = 180°, 158°, 140°) |
| PB24 | expected at `$OS_COLOR/SD/v5`; µd and Ψ_KLu with 10 sun angles; µ∞ closure-only unless profiles come with it |
| names | "ZTT (L23 µ∞)", "ZTT (TT2017 µ∞)"; "ZTT 2018" reserved |

The ten-step revised plan in Round 2 stands as written. **Standing by for
instructions** (e.g., which steps to execute and on which branch; current
branch is `inelastic-rt`).

## Logs

### 2026-10-30 — Plan prompt #1 (Claude Opus 5.5)

- Read `context/context_summary.md`, `docs/reports/report_rt_elastic_model.md`,
  `robust/rt/ztt.py`, `robust/rt/data/{l23,pb24}.py`, and Twardowski & Tonizzo
  (2018) pp. 1–16 (model, Eq. 8, synthetic-set methods, f_L, validation).
- Found that the L23 `_profile.nc` files (unused so far) carry depth profiles
  of Ed, Eu, Eo and the K's to 50 m — enough to measure µ∞ directly, as TT did
  with HydroLight.
- Ran a throwaway feasibility script (scratchpad, not in repo): Gershun holds to
  1e-4; TT2017 matches L23 µ(50 m) to ≈ ±3% (16–84%); a naive Eq. (8) LSQ gives
  0.57% MAPE but a 92% worst case, because 50 m is not asymptotic for many
  scenes (µ(50 m) still shifts ~10% between 0° and 60° sun).
- Wrote the plan and Q1–Q7 under Q&A/Design. No repository code changed.

### 2026-10-30 — Plan prompt #2 (Claude Opus 5.5)

- Read JXP's answers to Q1–Q7 and recorded the decisions at the top of
  "Round 2" under Q&A/Design.
- Q6: the loader expects PB24 in `$OS_COLOR/SD/v5`. It was there on 2026-08-08
  but is absent now; a filesystem search found no copy.
- Found that L23's particle phase function is two FF components (B = 0.010 for
  phytoplankton, 0.018 for non-algal particles, constant to 1e-6), so ZTT's Pbb
  can be L23's exact mixture.
- Scratch check of µd: ZTT's Eq. (14) vs L23 `md_z(0⁻)` gives a median ratio of
  0.981 / 0.999 / **1.087** at 0° / 30° / 60°, so the error is in the
  atmospheric factor M⁺_d and probably explains most of the backbone's 60°
  error. L23 µd(60)/µd(0) = 0.79–0.82 (separable to ±2%).
- Wrote the revised 10-step plan and Q8–Q12. No repository code changed.

### 2026-10-30 — Plan prompt #3 (Claude Opus 5.5)

- Read JXP's round-2 answers (Q8–Q12). Q8: read the new L23 paper
  (`Loisel_et_al_ESSD_2023.pdf`). It confirms the FF phase functions with
  b̃b = 0.01 / 0.018 and the run conditions (HydroLight v5.0, 5 m/s wind,
  clear sky), but gives no FF (n, µ). Adopted HydroLight's Mobley-2002
  convention as a stated, tested assumption.
- No further questions. Wrote "Round 3" with the final decision table and
  stood by. No repository code changed.

### 2026-10-30 — Plan prompt #4 (Claude Opus 5.5)

- Read the three design docs and the M5/M6 state (`rt_elastic_coding_prompt_7.md`:
  M6 is on Route C; record §7.16 already established that PB24 holds no
  asymptotic K, and that ZTT's µd is 9.2% off at 60° on PB24, which agrees with
  the L23 finding).
- Framed the work as **M7, "ZTT refit to L23"**: independent of M6, and a
  partial Route A (repairing µ∞ and µd with L23's depth profiles).
- Updated `design/rt_elastic_model.md` (new §4.6 and a §8 bullet),
  `design/rt_elastic_model_coding_plan.md` (M6/M7 rows and an M7 section with
  a task/gate table; the prompt's "rt_model_coding_plan.md" was taken to mean
  this file), and `design/rt_elastic_implementation.md` (v0.33, M7 status row,
  new §7b with pre-build measurements and decisions, module-index note).
- Found one constraint the Q&A hadn't covered: the `forward` freeze (§8.0) pins
  `forward(mode="ztt")` by value, so the default switch requires a deliberate
  re-baseline. Task 4 does it explicitly, while the hybrid's pins must not move.
- Wrote prompts 1–7 under "Fit L23". No code changed.

### 2026-10-30 — Fit L23 task 1: profiles and the µ∞ / µd product (Claude Opus 5.5)

**Built.**
- `robust/rt/data/l23.py`, new "depth profiles (M7)" section:
  `load_mu_profiles`, `L23ProfileProduct`, `build_profile_product`,
  `save_` / `load_profile_product`, `default_profile_product_path`, and
  `write_profile_fixture` / `profile_npz_reader`. It reuses
  `ocpy.loisel23.load_ds(..., profile=True)` and reads only the named fields
  and depths.
- Fixture `robust/tests/files/l23_profile_fixture.npz` (556 kB).
- `conftest.py`: `needs_l23_profiles`, `L23_PROFILE_FIXTURE`.
- `robust/tests/test_l23_profiles.py`: 14 tests.
- `design/py/build_l23_profile_product.py` caches the product at
  `$OS_COLOR/Loisel2023/derived/l23_profile_product_v1.npz` and writes
  `design/ztt_l23/profile_product.md` and `fig_profile_coverage.png`.
- `design/py/explore_mu_inf_eigen.py` (exploratory).
- Record §7b.5 written; M7 status row updated.

**Gates.** Gershun to < 1e-3 (median) at 50 m at every zenith ✅. Golden rows
against plain-xarray reads of the netCDF ✅. Bit-reproducible arrays and file
contents ✅. Asymptotic points zenith-independent to < 1% ✅ (true by
construction of the flag, and also checked on the full release).

**What I learned.**
1. Inspecting profiles first (as the prompt asked) overturned the plan's step
   (b). The approach to µ∞ is non-monotonic in sun angle and needs about ≥ 20
   scattering lengths, so the three suns agree at depth on only **2.4%** of
   points, all of them particle-dominated. I didn't fit an extrapolation;
   `extrapolated` became `not_asymptotic`, and `mu_inf` is NaN there.
2. At 50 m, irradiance underflows to 0 or denormals in the UV and far red, so
   the product uses the deepest usable depth (≥ 18 m everywhere).
3. A discrete-ordinate eigensolve of the asymptotic RTE on L23's IOPs plus the
   two FF components gives µ∞ for every point. It matches the 2.4% to ±0.3%,
   with a robust **+1.4%** systematic. Of five FF conventions, Mobley-2002 fits
   best. Against it, TT2017 is ≥ 11.6% high on the worst 16% of points.
   Raised as **Q13** (blocking task 3), with Q14 (ordering) and Q15 (cache
   folder) non-blocking.

Suite: **670 passed, 26 skipped** with the data (the skips are PB24, which is off disk); **628 passed, 68 skipped** with `$OS_COLOR` unset. `ruff check` and `ruff format --check` are clean. No changes to `ztt.py`, the hybrid, or any pinned value.

### 2026-10-30 — Fit L23 task 2: Fournier–Forand and L23's phase function (Claude Opus 5.5)

Read JXP's answers: Q13 (a), eigen µ∞ with the +1.4% as a stated systematic;
Q14, the plan stands; Q15, the cache folder is OK. **Reminder for JXP: rclone
`$OS_COLOR/Loisel2023/derived/l23_profile_product_v1.npz` to the `RoB` shared
Google Drive folder.**

**Built.**
- `robust/rt/phase.py`: FF closed forms, the Mobley-2002 convention, a
  differentiable B → µ inversion (table + Newton, exact implicit gradient),
  and `P_bb_ff` for mixtures.
- `PhaseParams.ff_B` / `ff_weight`, appended with `None` defaults and
  validated; the freeze's field-order pin re-baselined deliberately.
- `ztt.P_bb_from_phase(..., use_ff=False)`.
- The L23 loader fills the fields from `bph` / `bd` (optional reader fields,
  so the hash-pinned fixture is untouched).
- `robust/tests/test_phase.py`: 28 tests. Record §7b.6.

**Deviation, flagged.** The prompt said "FF is used when `ff_B` is set". I
made it **opt-in** (`use_ff=True`) instead. The loader now sets `ff_B` on every
netCDF batch, so automatic use would have moved every full-L23 hybrid number
before task 4 pins the hybrid to its backbone, against the "hybrid must not
move" rule. Task 4's `variant="l23"` passes `use_ff=True`, which reaches the
same end state.

**Gates.**
- p̃FF integrates to 1 (< 1e-6) for B = 0.001–0.035 ✅.
- B → µ → B round trip ≤ 1e-11 (float64) for 0.010 and 0.018 ✅.
- FF Pbb(180°) = **0.1506 / 0.1559** against Sullivan's **0.1529** (reported,
  and pinned) ✅.
- `ff_B=None` (or `use_ff=False`) gives Sullivan bit for bit ✅.
- float64 grad vs FD on the new path ✅.
- Existing tests green ✅: **698 passed, 26 skipped** with the data (the skips are PB24, off disk); **655 passed, 69 skipped** with `$OS_COLOR` unset. `ruff check` and `ruff format --check` are clean.

**Learned.**
1. Differentiating through unrolled Newton steps is not the implicit
   derivative to better than 1e-5. A stop-gradient plus one final step makes
   it exact.
2. The closed-form FF B matches quadrature of p̃ over the backward hemisphere
   to 1e-8, an independent check on the transcription.
3. NumPy `leggauss(4000)` costs seconds and saturates every core under pytest,
   so the quadrature is cached at 1500 nodes.
4. At nadir backscatter, L23's two FF components bracket Sullivan within 2%.

No new questions.

### 2026-10-30 — Fit L23 task 3: the fits (Claude Opus 5.5)

No new answers to read. Q13–Q15 were already applied.

**Built.**
- `robust/rt/asymptotic.py`: the eigen µ∞ (Q13a, Q14).
- `ztt.py` evaluators (`box`, `out_range`, `md_plus_correction`, `mu_d`'s
  refit keywords, `smooth_clip`, `log_inputs`); defaults are unchanged bit for
  bit.
- `design/py/fit_ztt_l23.py` with `--dry-run`, `--check` and
  `--recompute-eigen`.
- `design/ztt_l23/{coefficients,metrics,probes}.json`, `metrics.md`, and two
  residual figures.
- Tests: `test_asymptotic.py` (16) and `test_ztt_l23.py` (11). Record §7b.7.

**Gates.**
- Bit-identical rerun ✅: `--check` identical, eigen recompute identical.
- Held-out errors reported ✅:
  - µ∞ refit: median |err| **0.21%** (TT2017: 1.08%, with 84th/99th
    percentiles of +8.6% / +23.9%);
  - µd refit: **0.34 / 0.22 / 0.46%** at 0° / 30° / 60° (TT: 2.05 / 0.50 /
    **8.75%**).
- Shipped evaluators reproduce the script's own numbers ✅: 5.5e-7 / 1.4e-7
  where no guard acts, and the committed probes reproduce through `ztt` to
  1e-12.
- Suite ✅: **725 passed, 26 skipped** with the data (the skips are PB24, off disk); **681 passed, 70 skipped** with `$OS_COLOR` unset. `ruff check` and `ruff format --check` are clean.

**Learned.**
1. A test caught a wrong claim of mine. The δ-truncation angle is not
   "irrelevant": 5° biased µ∞ by +8e-4. Moved to 2° after a convergence study;
   the Q13 offset is now +1.34%.
2. The discrete-ordinate error grows at low albedo (−7e-4 at ω₀ = 0.1). It is
   pinned against the exact isotropic solution.
3. A rectangular clamp box doesn't protect a polynomial fitted on a triangle of
   data, so I added output guards (Q17).
4. **TT's Ψ_KLu is ~13% high against L23 at all three nadir ψ**, and TT2017's
   µ∞ error partly cancels it. Prediction written down: the L23 µ∞ alone will
   lower nadir rrs by ~2%. Raised as **Q16**, recommending (a): measure in
   task 6 with an oracle KLu/a rung.

### 2026-10-30 — Fit L23 task 4: the L23 variant wired in, the hybrid pinned (Claude Opus 5.5)

> **⚠ Deliberate re-baseline.** `rrs_ZTT` now defaults to `variant="l23"`
> ("ZTT (L23 µ∞)"), so **`forward(mode="ztt")` returns different numbers.** That
> path had no value pin before. A new pin records the new values beside the
> old (`test_env.test_the_backbone_path_produces_pinned_numbers`). The hybrid
> (`mode="hybrid"`, the default) is **unchanged**: its value pin, the elastic
> hash pins and the emulator SHA-256 all pass untouched.

Read JXP's answers: Q16 (a), measure in task 6 with an oracle KLu/a rung;
Q17, keep the output guards.

**Built** (record §7b.8):
- **`ztt.py`:**
  - the L23 constants with provenance;
  - `mu_infinity_l23` and `mu_d_l23`;
  - `variant` (default "l23"), `mu_inf_model` and `mu_d_model` on
    `rrs_ZTT` / `Rrs_ZTT`, with precedence and validation;
  - `"l23_closure"` reserved for task 5.
- **`emulator.py`:** `Emulator.backbone_variant`, a static field. A missing
  key loads as "tt2017", and `save` omits the key for "tt2017", so the shipped
  file and a TT2017 retrain are byte-unchanged.
- **`hybrid.forward`:** the emulator is resolved before the backbone, and the
  hybrid uses the emulator's variant.
- **Call-site audit** (table in §7b.8): `design/py` scripts name "tt2017";
  the inelastic heads and the surface transfer are backbone-independent.

**Gates.**
- `variant="tt2017"` is bit-identical to the committed `ztt.py` ✅ (hash
  pinned).
- Hybrid pins and the emulator SHA are unchanged ✅.
- Finite at and beyond the box ✅ (finite, *not* positive: rrs < 0 at
  bb/a ≈ 25 in both variants, structurally).
- float64 grad vs FD, L23 variant with FF, for a, bb_p, B_p, θs and w_ph ✅.
- Suite ✅: **744 passed, 26 skipped** with the data (the skips are PB24, off
  disk), in three consecutive runs; **700 passed, 70 skipped** with `$OS_COLOR`
  unset. `ruff check` and `ruff format --check` are clean. One earlier run had
  1 failure that did not reproduce. That run printed only its summary, so the
  test isn't known; likely a timing-sensitive test under load. Watch for it.

**Learned.**
1. The audit was worth it: a test assembled the hybrid by hand from
   `rrs_ZTT()`'s default and read 4.77% instead of 0.3%. That is exactly the
   silent mismatch Q2a was about, caught because the default moved.
2. First look on the fixture: the L23 variant's rRMS is 2.48 / 3.53 / 2.73%
   at 0° / 30° / 60° (TT2017: 4.19 / 3.73 / 7.26%), and rrs(60°)/rrs(0°) is
   0.938 against the truth's 0.946 (TT2017: 0.852). 30° is still too strong
   (0.959 vs 0.990), in line with Q16.
3. `forward(mode="ztt")` and `mode="hybrid"` now use different backbones until
   Q2b (raised as **Q18**).
4. A pre-existing reproducibility gap: the committed emulator weights don't
   regenerate bit for bit even at the last commit (raised as **Q19**).

### 2026-10-31 — Fit L23 task 5: the closure µ∞ (Claude Opus 5.5)

Read JXP's answers. Q18: hold the emulator retrain until PB24. Q19 (a): done;
the stale "bit-identical committed weights" line in
`reports/report_rt_elastic_model.md` §8 now carries a dated correction (the
generated `docs/` copy will follow on the next docs build).

**Built** (record §7b.9):
- `ztt.mu_inf_closure`: Eq. (12) inverted for µ∞, with the L23 µd and FF Pbb.
- The closure fit shipped as `MU_INF_L23_CLOSURE_COEFFS` /
  `mu_infinity_l23_closure`, i.e. `mu_inf_model="l23_closure"`. Documented as
  *a ceiling, not physics*; never a default.
- `fit_ztt_l23.py` `closure_stage`, the figure `fig_mu_inf_closure.png`, and
  outputs in `coefficients.json` / `metrics.*` / `probes.json`.
- Tests: 4 new, plus closure probes.

**Gates.**
- Reproducible ✅ (`--check` identical).
- Self-consistency ✅: the closure µ∞ reproduces the reference to float64
  rounding, and every train sample's rrs error is within its µ∞ error.
- Non-physical samples reported: **2 of 806,760** (with the FF Pbb).
- Suite ✅: **748 passed, 26 skipped** with the data (the skips are PB24, off
  disk); **704 passed, 70 skipped** with `$OS_COLOR` unset. `ruff check` and
  `ruff format --check` are clean.

**Learned.**
1. **The ceiling is about half the remaining error.** Held-out rRMS falls from
   2.91 / 2.56 / 3.18% (ZTT (L23 µ∞)) to **1.20 / 1.43 / 1.47%** (closure) at
   0° / 30° / 60°. µ∞ can't take the rest: it depends on zenith (2.8% median
   spread) and on regime (the fit to the closure is 0.9–1.8% vs 0.21% for the
   physical µ∞).
2. **The phase function decides whether µ∞ can carry the error.** With L23's FF
   only 2 closure values exceed 1; with Sullivan's (fixture), 8.4% do.
3. **The other terms' error is structured by regime**: up to +25% in
   particle-dominated water at moderate bb/a, about −5% in clear water, and
   saturation near 0.92 where µ∞ → 1. They cancel in the median (closure/eigen
   ≈ 1.00). That is why the Ψ_KLu ~13% excess (Q16) never showed up as a µ∞
   offset. Task 6's oracle KLu/a rung should separate the two.

No new questions.

### 2026-10-31 — Fit L23 task 6: the ladder and the acceptance gate — **PASS** (Claude Opus 5.5)

No new answers to read (Q16 (a) was applied: the oracle KLu/a step).

**Built:** `design/py/run_ztt_l23.py`, writing `design/ztt_l23/ladder.{md,csv}`,
`ladder_per_wavelength.csv` and two figures; record §7b.10; 2 tests.
**Deviation:** the outputs are named `ladder.*` instead of the prompt's
`metrics.*`, which `fit_ztt_l23.py` already writes.

**Acceptance gate — PASS.** Held-out rRMS, ZTT (L23 µ∞) vs ZTT (TT2017 µ∞):
0° **2.91** vs 4.26, 30° **2.56** vs 4.67, 60° **3.18** vs 8.11%; all scenes
**2.90** vs 5.93%. The TT2017 row reproduces the M4 record exactly.

**Learned.**
1. **Attribution.** µd does the 60° work (−3.6 pp) but hurts 0°/30° alone. The
   L23 µ∞ alone *worsens* TT2017 (+0.89 pp), as predicted in task 3, and is
   worth 1.7 pp inside the full variant. FF Pbb is worth ~0.8 pp, mostly at 30°.
   The terms interact, so both orderings are tabled.
2. **Q16 measured: Ψ_KLu must not be corrected alone.** The oracle (measured
   KLu/a, any depth 0–20 m) gives 7.8–9.5%, against 2.90%. ZTT's K_Lu is an
   effective parameter, ~11–13% above the true one, with f_L tuned to it.
3. What's left is calibration, not physics. The closure ceiling is 1.37%, O25
   refit 0.69%, and the TT2017 hybrid 0.30%. The residual is concentrated at
   530–590 nm. Raised as **Q20** (recommend: stop here for L23, move to PB24).

Suite ✅: **750 passed, 26 skipped** with the data (the skips are PB24, off disk); **705 passed, 71 skipped** with `$OS_COLOR` unset. `ruff check` and `ruff format --check` are clean.

### 2026-10-31 — Fit L23 task 7: closing M7 (Claude Opus 5.5)

Read JXP's answer to Q20: (a), stop the L23 work and move to PB24.

**Done** (record §7b.11):
- **Notebook** `notebooks/RT/rt_fit_ztt_l23.ipynb`: executed, 17 cells, 3
  figures, 0 errors (`nbclient`; `nbconvert` is absent from `ocean14`).
- **Report** `reports/report_rt_elastic_model.md` → v1.1: an
  executive-summary update and a new **§9 "Update (M7)"** with
  `fig_ztt_l23_ladder.png` (drawn by `reports/make_report_figures.py` from
  `ladder.csv`, and added to the docs figure list). The prompt asked for
  `docs/reports/report_rt_elastic_model.md` to be updated; that file is
  *generated* from `reports/` (its banner says so), so I edited the source and
  regenerated the copy with `docs/figures/make_docs_figures.py`.
- **Review pass:** an independent reviewer over the M7 diff found no serious
  correctness bug on the shipped paths. All findings were fixed or logged
  (table in §7b.11):
  - one real silent-broadcast bug (per-sample `ff_weight`), fixed with a
    regression test;
  - the M3 hybrid gate comparing two different backbones, fixed;
  - two tests that couldn't fail, replaced with independent checks;
  - CI never running the FF path on real scenes, fixed with a new 79 kB
    `l23_ff_small.npz` fixture and a value pin;
  - stale or overstated docs, fixed;
  - one design hazard for a future "l23" emulator (1b), logged.
- **Hand-off:** see "Next (M7 hand-off)" above, also in the record.

**Status:** M7 ✅. Coding plan and record updated (record v0.34).

Suite ✅: **752 passed, 26 skipped** with the data (the skips are PB24, off
disk); **707 passed, 71 skipped** with `$OS_COLOR` unset. `ruff check` and
`ruff format --check` are clean. M7 modules: `test_l23_profiles` 14,
`test_phase` 28, `test_asymptotic` 16, `test_ztt_l23` 36.

No new questions.

### 2026-10-31 — Fit L23 task 8: handing off to PB24 (Claude Opus 5.5)

Created `claude_prompts/RT/rt_fit_pb24.md` (milestone M8): context, a
description of the data as downloaded, four tasks (an `ocpy` PB24 reader plus
consolidation caches; a baseline of every elastic model on PB24; a discussion
of how to improve the backbone; implement and assess), and seven opening
questions (Q1–Q7) in its Q&A.

Checked before writing:
- `$OS_COLOR/SD/v5` now holds 5,000 **hyperspectral-only** files (27 GB).
  The OLCI files and the water-class `.mat` that M5 used are absent, so
  `robust.rt.data.pb24`'s defaults and the `needs_pb24` marker still treat
  PB24 as missing.
- The files carry **no depth profiles**, which answers M7 Q12: µ∞ on PB24
  comes from the eigenproblem or a closure fit.
- There is no Raman (`no_R`).
- There is a new field, `Esdir_Es_ratio` (1 − H), directly relevant to µd's
  atmospheric factor.
- `ocpy` is on branch `pb24` (an editable install); `loisel23.py` is the
  pattern for the new reader.

No code written.
