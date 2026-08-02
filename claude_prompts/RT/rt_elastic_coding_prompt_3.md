# Elastic RT Coding — Prompt 3 (M2: ZTT-in-JAX backbone)

## Goals

Implement **Milestone M2**: the **ZTT analytic backbone** in JAX — a differentiable
`Rrs_ZTT(iops, phase_params, geometry, wave)` with the particle phase function
(backward VSF) as an **explicit** input. This is the physically-interpretable half of
the hybrid and our analytical benchmark.

M2 is the first milestone that produces *radiative transfer*. M1 built everything it
consumes; the signature it must fill is already pinned in `ztt.py` and currently
raises `NotImplementedError`.

## Claude

### Skills

Consider `.claude/skills/` — `critical-partner` for checking the equation
transcription (this is the milestone where a sign error is most expensive and least
visible), `code-review`, and `dataviz` for the notebook's figures.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

Two additions M0/M1 settled:

- **Ask in Q&A.** Judgment calls that are JXP's go in the *Q&A* section below as a
  numbered question, phrased so work continues without an answer. Check for answers
  before the next task.
- **Each milestone ships a notebook** (task 4), then a **PR-review pass** (task 5)
  and a **hand-off edit to the next prompt doc** (task 6).

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M2 and the Risks note
  on the placeholder backbone.
- **The ZTT paper** — `context/RT/twardowski2018.pdf` (Twardowski & Tonizzo 2018,
  *Applied Sciences*, "Ocean Color Analytical Model Explicitly Dependent on the
  VSF"). Present on this laptop (3.9 MB) but **gitignored** (`context/*/*.pdf`), so
  it is not in the repo — read it from disk, and quote the equation numbers you
  transcribe so a reader without the PDF can still follow.
- **Synthesis** — `context/RT/rt_elastic_model.md` §2, §3.5 (the phase-function
  axis: `β(π)/bb` ≈ **0.23 sr⁻¹ for pure water** vs **0.12–0.16 for particles**) and
  §4 (the BING `rrs↔Rrs` convention).
- **Implementation record** — `design/rt_elastic_implementation.md` (currently
  **v0.9.1**; §3 is M1 as built, §8 the cross-cutting conventions, §10 CI). At close:
  bump the version, add a **§4 for M2** with *Modules added / Tests / Results /
  Notebook* subsections, flip M2 to ✅ in §1, and refresh the module index.
- **The two notebooks** — `notebooks/RT/rt_elastic_coding_1.ipynb` (M0: JAX,
  autodiff, the float64 argument) and `..._2.ipynb` (M1: `rrs` vs `Rrs`, the
  water/particle split, `B_p`, the splits). **Do not re-explain either**; link.
- **An existing Gordon baseline you can check against** —
  `context/RT/make_rt_elastic_figures.py` and `context/RT/fig_rrms_ladder.csv`.
  See *Status entering M2* for the numbers and the exact rRMS definition.

## Status entering M2

M1 is complete: 117 tests pass (100 + 17 skipped without `$OS_COLOR`), `ruff check`
and `ruff format --check` are clean under `ruff.toml`, and CI runs both on Python
3.12 and 3.14. Build on the following rather than rebuilding it.

### The M1 API, by name

```python
from robust.rt import conventions as C     # A_RRS, B_RRS, RRS_POLE
                                           # Rrs_to_rrs, rrs_to_Rrs
                                           # WAVE, N_WAVE, canonical_wave()
                                           # BB_W_L23, bb_w(wave)
                                           # check_wave, check_iop, check_rrs
from robust.rt.types import IOPs, PhaseParams, Geometry
from robust.rt.data import l23 as L        # load_batch, make_splits, select
                                           # L23Batch, Splits, npz_reader
                                           # ZENITHS, N_SCENES, B_P_EXPECTED
```

- `IOPs(a, bb_w, bb_p)` carries derived `.bb`, `.u = bb/(a+bb)`, `.n_wave`. Use
  `.u` rather than recomputing it.
- `bb_w(wave)` interpolates the embedded L23 table; it is `jit`-safe and
  differentiable, and **clamps** outside 350–750 nm rather than extrapolating.
- `Geometry` is in **degrees** (`theta_s`, `theta_v`, `dphi`, optional `wind`), with
  `Geometry.nadir(theta_s)` for the L23 case. Convert to radians inside `ztt.py`;
  do not change the units at the boundary.
- All three are pytrees via `jax.tree_util.register_dataclass`, so `jax.grad` of a
  scalar of an `IOPs` returns an `IOPs` with labelled per-field derivatives — which
  is what makes the M2 gradient gate readable.
- `load_batch()` returns **9960 samples** (3 zeniths × 3320 scenes) on one flat
  axis; `make_splits` gives boolean masks; `select(batch, mask)` subsets.

### Facts M1 measured that bear directly on M2

| Fact | Consequence for M2 |
|---|---|
| `B_p` is a **spectrum**, shape `(n_sample, 81)`, not a scalar | `Rrs_ZTT` must accept a per-λ `B_p`; do not reduce it to one number |
| `B_p` ∈ **[0.01026, 0.01800]** — a 1.75× slice of the design's 7.5× band | The phase-function axis is *weakly* exercised; do not over-claim from a good fit |
| `bb_w` is **72%** of total `bb` at 400 nm, ~50% at 550, **29%** at 750 | Keep water and particles separate through the ZTT equations — see below |
| `β(π)/bb` ≈ 0.23 (water) vs 0.12–0.16 (particles) | **The reason the split matters here**: ZTT weights the two components with *different* backward-VSF values, so `bb_w` and `bb_p` must enter separately rather than as `bb`. M1 preserving the split is what makes this expressible |
| IOPs are **bit-identical** across the three zenith files | Only geometry differs, so a zenith-dependent term can be tested cleanly against a fixed water body |
| `Rrs(60°)/Rrs(0°)` = **0.949**, `Rrs(30°)/Rrs(0°)` = **0.990** | The geometry signal ZTT should capture, and Gordon cannot: standard Gordon has *no* solar-zenith dependence at all |

### The Gordon baseline already exists — reproduce it, do not re-derive it

`context/RT/make_rt_elastic_figures.py` computed per-wavelength rRMS for standard
Gordon on **Hydrolight100.nc (X=1, Y=0 only)**, with this definition:

```python
rrs  = Rrs_to_rrs(Rrs);  u = bb / (a + bb)
pred = 0.0949 * u + 0.0794 * u**2          # standard Gordon (1988)
rRMS = 100 * sqrt(mean(((pred - rrs) / rrs)**2))   # relative, rrs-space, percent
```

Results (`fig_rrms_ladder.csv`, `std` column):

| λ [nm] | 400 | 450 | 500 | 550 | 600 | 650 | 700 |
|---|---|---|---|---|---|---|---|
| Gordon rRMS [%] | 2.49 | 2.91 | 3.67 | 4.88 | 6.45 | 7.65 | 9.04 |

**Use that rRMS definition verbatim** so every number in this project stays
comparable, and reproduce the table as a consistency check between the new JAX code
and the existing NumPy figure code. A mismatch means one of them is wrong, and it is
much cheaper to find out here than at M4.

### Gotchas carried forward

1. **`pytest` from the repo root** (`robust` may not be pip-installed).
2. **JAX defaults to float32.** The M2 gradient gate *must* run under the `jax_x64`
   fixture. M0's notebook §4 measured why: at a 1e-6 tolerance, float32 meets the
   finite-difference check at **0 of 33** step sizes while float64 meets it at
   **21 of 33**. In float32 the gate would test the step size, not the gradient.
3. **The finite-difference trap** (also M0's notebook §4): pin the dtype on the
   *arrays* (`jnp.asarray(x, dtype=...)`) and assert it. Perturbing with Python
   floats silently computes in float64 and "passes" while proving nothing.
4. **Do not out-precision the reference.** L23 is stored float32; M1's cross-module
   `bb_w` check needs `rtol=1e-5` because of float32 cancellation in the red tail.
   Any comparison against L23 values inherits that limit.
5. **Validators are boundary-only.** `check_*` and `.validate()` read concrete
   values and cannot run inside `jit`. Keep them out of `Rrs_ZTT`.
6. **Test layering that M1 established** (`test_l23.py` is the model): pure logic
   with no data; the **committed fixture** (`files/l23_small.npz`, via
   `L.npz_reader`) so real numbers are exercised in CI; and `needs_l23` only for
   claims that genuinely need all 3320 scenes. Prefer the middle layer — it is what
   makes CI meaningful.

**Open, not blocking:** nothing. Q1–Q3 are answered and closed.

## Prompts

1. Read this doc. Execute the 1st task in the "M2" section below. If you have any
   questions, ask me in the Q&A section below. Use Fable if you can. Log your work.
2. Read this doc. Execute the 2nd task in the "M2" section below. Check my answers
   in Q&A. Use Fable if you can. Log your work.
3. Read this doc. Execute the 3rd task in the "M2" section below. Use Fable if you
   can. Log your work.
4. Read this doc. Execute the 4th task — the notebook. Use Fable if you can. Log
   your work.
5. Read this doc. Execute the 5th task — the PR review. Log your work.
6. Read this doc. Execute the 6th task — the hand-off to prompt 4. Log your work.

## M2

### Tasks

1. **Gordon-in-JAX first — it is not just a fallback.** The coding plan lists a
   Gordon/O25-in-JAX backbone as the *de-risking* option if ZTT proves ambiguous.
   Note that M3's gate is "hybrid **beats standard Gordon**" and M4 scores against
   Gordon/PR05/O25, so **Gordon is a required artifact regardless**. Building it
   first therefore costs nothing and buys an immediately end-to-end path.

   It is a handful of lines (`rrs = G1·u + G2·u²`, `G1 = 0.0949`, `G2 = 0.0794`,
   then `rrs_to_Rrs`). Gate it by **reproducing the rRMS table above** at Y = 0 to
   within rounding, and report the same numbers at 30° and 60° — where Gordon has no
   zenith term, so its rRMS should *worsen* by roughly the measured 5% bias at 60°.
   That number is the benchmark the rest of the prototype has to beat.

   **Where should it live?** The coding plan's package layout has no module for
   comparison models. I suggest `robust/rt/baselines.py` (Gordon now, PR05/O25
   joining at M4) rather than hiding Gordon inside `ztt.py` or `validation.py`.
   Raise it in Q&A if you would rather it went elsewhere.

2. **Transcribe ZTT into JAX.** Implement the ZTT forward relation in
   `robust/rt/ztt.py` as pure JAX functions `Rrs_ZTT(iops, phase_params, geometry,
   wave)`, with the backward VSF / `B_p` entering **explicitly**, and with `bb_w`
   and `bb_p` weighted separately if the equations distinguish them (they should —
   see the `β(π)/bb` row above).

   Document **each transcribed equation with its paper reference** (equation number
   or section). Keep `phase_params` structured so the fuller ZTT backward-VSF
   parameters can replace `B_p` at M5 without changing the signature — `PhaseParams`
   is already built for that (extra optional fields defaulting to `None`).

   Where the paper is ambiguous, say so in the docstring and in Q&A rather than
   guessing silently; an undocumented guess in the backbone is the single most
   expensive thing this milestone could ship, because M3 will train a network to
   compensate for it.

3. As noted in Q4, I have emailed the authors for the coefficients.  I have also downloaded the Twardowski & Tonizzo (2017) paper. Please look for the coefficients in that paper. Use Fable if you can. Log your work.

4. I have downloaded the Sullivan & Twardowski (2009) paper.  Please work on Pbb,ST.  Use Fable if you can. Log your work.

5. **Validate & gate.** Tests in `robust/tests/test_ztt.py`:
   - **(i) Paper reference case.** `Rrs_ZTT` reproduces a value or digitized curve
     quoted in twardowski2018 to a stated tolerance. Say where the number came from.
   - **(ii) Gradient check.** `jax.grad` of `Rrs_ZTT` against central finite
     differences w.r.t. `a`, `bb_p`, `B_p`, and geometry, under `jax_x64`, at a
     stated tolerance and step size. This is a **hard gate** from here on.
   - **(iii) Report, do not gate, accuracy.** Standalone rRMS of `Rrs_ZTT` vs L23
     **per wavelength and per solar zenith**, alongside Gordon from task 1. Per the
     project's unbiased stance this is *reported*, not thresholded — but the
     per-zenith breakdown is the interesting part, since that is where Gordon is
     structurally unable to compete.

   **De-risk.** If a ZTT term resists pinning down, ship task 1's Gordon backbone
   behind the same interface so M3–M4 proceed end-to-end, flag the gap in Q&A and
   the implementation record, and swap true ZTT in later without touching
   `forward`'s signature. Flagged loudly beats quietly approximate.

6. **Notebook.** `notebooks/RT/rt_elastic_coding_3.ipynb` — the M2 explainer,
   following the conventions in the implementation record §2.6/§3.4 and §8:

   - **Executed**, committed with outputs (`jupyter nbconvert --to notebook
     --execute --inplace`), so it reads without a kernel.
   - Data-dependent cells degrade to a message without `$OS_COLOR` (prefer the
     committed fixture via `L.npz_reader`, which needs no mount); bootstrap
     `sys.path` to the repo root.
   - Figures: recessive grid/frame, ink-coloured text, legend **plus** direct
     labels, one hue light→dark for sequential magnitude, and the CVD-checked pair
     `#0072B2`/`#D55E00`. **Render and look at every figure** — that has caught a
     label collision, a truncated axis, and a mis-rendered label so far.
   - Explain the *physics decisions*, not the call signatures: what the backward VSF
     is and why making it explicit is the whole argument for ZTT over Gordon; which
     equations were transcribed and any judgment calls; where ZTT beats Gordon and
     **where it does not**. A figure of rRMS(λ) for Gordon vs ZTT per zenith earns
     its place; so does one showing the gradient check.
   - Be honest about the `B_p` 1.75× slice: a good fit here is not evidence of
     phase-function generalisation.

7. **PR Review.** Fetch the review comments on the open PR and address them. `gh`
   is not authenticated in this environment; the public REST API works:
   `curl -s https://api.github.com/repos/ocean-colour/retrieve-or-bust/pulls/<n>/comments`
   and `.../reviews`. Fix the *class* of each defect rather than the single
   instance, and demonstrate that the fix catches what was reported. Log it.

8. **Finally.** Modify the next prompt doc `rt_elastic_coding_prompt_4.md` (M3:
   emulator + hybrid) given what M2 established. Use Fable if you can. Log your
   work.

### Q&A

**Q4 (M2 task 2, Claude → JXP). The ZTT paper does not publish the coefficients
for its own Equation (8), and that blocks µ∞.**

Equation (8) fits `µ∞(bb/a, η_bb)` — the asymptotic average cosine — as a cubic in
`log(bb/a)` whose four coefficients are each a cubic in `log η_bb`, i.e. sixteen
numbers `m1..m16`. The text says "Coefficients m are provided in Appendix A,
Table A2". **They are not.** Table A2 as printed gives coefficients for Equations
(3) `f`, (4) `fA`, (16) `e`, and (17) `m*_d`, then goes straight from `m*_d,8` into
Table A3. A full-text search of all 30 pages finds `m1` and `m16` nowhere except
inside Equation (8) itself. The paper says "Code for the ZTT model in MATLAB is
available at ioccg.org"; I checked `ioccg.org/groups/software/` and
`ioccg.org/resources/software/` and there is no ZTT/Twardowski/Tonizzo entry.

µ∞ sits in the denominator of Equation (12), so this is not a cosmetic gap — it is
one of the model's five assembled terms.

**What I did.** Everything else is transcribed and verified (see the log).
`mu_infinity` implements Equation (8)'s exact structure but **requires** the
coefficients and raises `NotImplementedError` naming this gap if they are absent. I
deliberately did *not* invent sixteen numbers, because they would silently become
"ZTT" in every M3/M4 comparison thereafter. `rrs_ZTT` also accepts `mu_inf=<value>`
so the rest of the model can be exercised and gradient-checked, with docstrings
stating that a supplied constant is not the paper's parameterization and must not be
reported as ZTT accuracy.

**What I would like you to decide.** Options, roughly in order of my preference:

1. **Email the authors** (mtwardowski@fau.edu, alberto.tonizzo@gmail.com) for
   Table A2's `m` coefficients or the MATLAB code. Cleanest — it gives us the real
   model. You would need to send it; I should not mail strangers on your behalf.
2. **Chase the antecedent.** Equation (8) extends the fit in their reference [40]
   (Twardowski & Tonizzo, the companion paper). If you have that PDF, the µ∞
   parameterization may be published there in usable form — I could not find it in
   the repo's `context/`.
3. **Fit `m1..m16` ourselves** against HydroLight, e.g. from L23. Feasible, but it
   converts the "physics-anchored, not fitted" backbone into a partly-fitted one,
   and I would want that stated loudly in every result rather than buried.
4. **Proceed at M3/M4 with the Gordon baseline as the backbone** (task 1 built it
   for exactly this contingency) and treat ZTT as blocked-pending-coefficients. The
   prototype still lands end-to-end; it just is not ZTT.

Option 1 or 2 keeps the milestone's scientific intent. Until you rule, task 3 can
still gate the *transcribed* terms and the gradient path via `mu_inf=`, and the
implementation record will say plainly that standalone ZTT accuracy is not yet
measurable. — *Not blocking task 3, but it does bound what task 3 can claim.*

>A. I have emailed the authors for the coefficients.  I have also downloaded the Twardowski & Tonizzo (2017) paper. I will ask you to look for the coefficients in that paper. 

## Next

→ `rt_elastic_coding_prompt_4.md` (M3: Emulator + hybrid).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs

### 2026-08-03 (M2 task 4 — the notebook, and a correction to my task-3 diagnosis)

`pytest -q` → **169 passed** (no xfail any more); without `$OS_COLOR`, 149 + 20
skipped. `ruff` clean. Record at v0.13. Notebook:
`notebooks/RT/rt_elastic_coding_3.ipynb`, 20 cells, executed, three figures.

**First, a correction, because I got this wrong in the last entry.** I reported that
the inverted zenith trend was caused by the missing `Pbb(ψ)`, on the strength of a
fitted diagnostic. Then you supplied Sullivan & Twardowski (2009) — thank you — and
with the *real* `Pbb,ST(ψ)` the trend was **still inverted**. So my diagnosis was
wrong.

The actual cause was a transcription bug in `Md_plus`: the paper's `µw` is the cosine
of the **in-water** solar zenith. Equation (13) writes `µw = cos(θs)` unprimed, which
§2 defines as in-water, while `H` and `P3` in the same expression take the *primed*
above-water angle. I had used the above-water cosine throughout.

What caught it was §2.7's quoted range — µd runs 0.79–0.94 for sun 8°–62°. The
above-water cosine gives 0.573 at 62°; the in-water cosine gives 0.792 and 0.936,
reproducing both endpoints. **A quoted constant discriminated where a fitted
diagnostic had actively misled me**: the per-zenith `P_bb` fit came out 0.148 /
0.134 / 0.092, falling monotonically away from backscatter and mostly inside the
literature band, which looked exactly like confirmation. It was the fit absorbing
the µd error into the one free parameter I had given it. Lesson recorded in the
notebook and the record: a parameter fitted against the same data cannot diagnose a
bug in the model that fits it.

**With the fix, M2's result is straightforwardly good.** 60° rRMS went from 21.3% to
**8.09%**, and ZTT now beats standard Gordon at all three zeniths — 4.30 vs 6.02
(0°), 4.70 vs 6.20 (30°), 8.09 vs 9.01 (60°). The per-λ shape is the real story:
Gordon is better in the blue and climbs to 9–13% in the red, while ZTT holds 3–5%
across the range, crossing over near 500–550 nm. The `xfail` I added last time
XPASSed and was replaced by real assertions, which is exactly what `strict=True` is
for.

**Also from the new paper: their published `a3` is a typo.** Sullivan & Twardowski's
Table 2 prints `a3 = 8.007E−02`; at ψ = 140° that term alone would contribute ~1570
against a tabulated 0.137. I refitted their Table 1 independently and got
`a3 ≈ 7.8e-4` while reproducing their other four coefficients closely, so the
intended value is `8.007E−04`. Corrected in the code with the reasoning in the
docstring, and pinned by a test.

**The notebook's own two findings**, both from the gradient-gate figure:

1. **No single finite-difference step clears the gate for all four inputs.** At
   `h = 1e-6` the three IOP-like variables are at 1e-10 or better but `theta_s`
   misses at 1.3e-6; at `h = 1e-3` `theta_s` is superb and the other three fail by
   4e-5 to 7e-3. `theta_s` is O(30), the others O(1e-3)–O(0.1).
2. **An invalid step is not an accurate one.** For `bb_p` (O(3e-3)), steps ≳ 3e-3
   push it negative and the model returns NaN — and my first version of the figure
   let `argmin` pick one of those NaNs as the "best" step, printing `h = 5e-3`. I
   caught it because the annotation contradicted the visible curve. The sweep now
   masks non-finite results and reports how many steps were invalid. Worth carrying
   into M4's protocol.

**Figure work.** Figure 3 initially used a sequential blue ramp for four
*categorical* series — the wrong encoding, and four blues that were genuinely hard
to tell apart. I searched the Okabe-Ito palette for the best 4-subset by worst-case
colour-blind separation and got blue/vermillion/sky/black at ΔE 17.9 (target ≥ 8),
which also keeps the pair used in notebooks 1–2; my first guess, adding green, would
have **failed** at ΔE 5.4 against the blue. Figure 1 lost a set of rotated
per-angle labels that collided with the legend, replaced by one shaded band.

New: `notebooks/RT/rt_elastic_coding_3.ipynb`. Modified: `robust/rt/ztt.py`
(`P_bb_sullivan`, the `µw` fix), `robust/tests/test_ztt.py` (xfail replaced; µd
endpoint test; `Pbb` tests), `design/rt_elastic_implementation.md` (v0.13 — §4.4
rewritten with the true diagnosis, new §4.5). Branch `rt-elastic-prototype` for JXP
to commit. Next: task 5, the PR review.

### 2026-08-03 (M2 task 3 — gates pass; µ∞ unblocked; one honest failure diagnosed)

`pytest -q` → **164 passed + 1 xfailed**; without `$OS_COLOR`, 144 + 20 skipped + 1
xfailed. `ruff` clean. Record at v0.12 with §§4.3–4.4.

**Q4 is largely resolved, using the paper you downloaded.** Twardowski & Tonizzo
(2017), *Optics Express* **25**(15) 18122 — reference [40], the study the 2018 text
says Equation (8) "extended" — publishes µ∞ directly. Its **Table 1** gives
`µ∞ = p0 + p1 log(bb/a) + p2 log²(bb/a)` at six `η_bb` values; I transcribed it and
interpolate the three coefficients in `log η_bb` to recover the 2-D surface
Equation (8) would have given, which also keeps it differentiable.

I rejected its Table 2 (combined quartics) after checking: they reach **µ∞ = 1.35**
at `bb/a = 1e-4`, which is unphysical for an average cosine, and they carry no
`η_bb` dependence. Table 1 stays inside (0.63, 0.98] across all of L23. Worth
noting L23 reaches `bb/a ≈ 0.31` against the fit's 1e-1 upper bound, so the
brightest scenes extrapolate. `mu_infinity` still implements Equation (8) exactly
and requires the sixteen coefficients, so `mu_inf_coeffs=` restores the published
2018 model the moment the authors reply. Everything is labelled **"ZTT with the
TT2017 µ∞"**, never "the 2018 model".

**The gradient gate passes** for `a`, `bb_p`, `B_p`, and `theta_s` — `jax.grad`
against central differences to `rel=1e-6`, under `jax_x64` with dtypes pinned on the
arrays. I scaled the step per variable rather than using one global `h`, since `a`
is O(0.1) and `bb_p` O(1e-3); a single step size would have been testing the step,
not the gradient.

**The most useful result: ZTT beats Gordon in the red, and by a lot.** Per-λ rRMS at
nadir, ZTT vs Gordon: 2.92% vs 9.04% at 700 nm, 3.48% vs 7.65% at 650, 3.96% vs
6.45% at 600. The crossover is near 550 nm; below it Gordon wins (4.61% vs 2.49% at
400). Gordon degrades steadily toward the red while ZTT stays flat at ~3–4% — which
is the analytic backbone doing precisely what it exists for.

**And one honest failure, which I chased rather than reported around.** At 60° ZTT
came out much worse than Gordon (24% vs 4.8%), and worse, `rrs` came out *increasing*
with solar zenith when L23 has it falling. Wrong sign. A model carrying a real BRDF
should not lose to a model with no zenith term at all, so I treated it as a bug in
my transcription until proven otherwise.

It is not. Fitting a single constant `P_bb` per zenith against L23 gives **0.148 at
ψ = 180°, 0.134 at 158°, 0.092 at 139.7°** — falling monotonically away from
backscatter, which is the physical shape of a particulate backward phase function,
and the first two land inside the **0.12–0.16 sr⁻¹** the synthesis quotes for
particles at the angles it quotes. Feed those back in and `rrs` falls with zenith,
with a 60°/0° ratio of **0.934** against L23's **0.949**.

So the geometry path is transcribed correctly; the inverted trend is caused by my
holding `Pbb` constant *in ψ*, when the paper's "constant backward phase function"
means constant across water types — `Pbb,ST(ψ)` is still a function of angle, and
the three zeniths sample ψ = 180°/158°/140°. That distinction decides whether M3 can
build on this, so it was worth the detour.

I encoded it as a **strict `xfail`**: the test asserts the *correct* sign, is
expected to fail today, and under `strict=True` will fail loudly as an XPASS the
moment `Pbb,ST(ψ)` lands — so it prompts its own removal instead of rotting. A test
asserting today's wrong sign would have enshrined the bug; one asserting the right
sign without the marker would have blocked CI. Alongside it, a passing test
documents the fitted-`Pbb` evidence.

**Consequence for the milestone, stated plainly:** ZTT's standalone accuracy cannot
be fairly reported until `Pbb,ST(ψ)` is in hand — the 60° numbers above understate
it badly. §4.4 of the record says so rather than presenting the table as ZTT's
verdict. Two inputs remain outstanding: Equation (8)'s `m1..m16` (you have emailed)
and `Pbb,ST(ψ)` from Sullivan & Twardowski (2009) — **if you can get that paper, I
can close the loop**, and it is the single highest-value thing left for M2.

Sanity values that all landed inside the paper's own stated bands, which is weak
evidence individually but reassuring together: `f_L` ∈ (1.0, 1.12) across ψ and λ
(paper's natural range, Zaneveld's 1.05); `Ψ_KLu` 1.024 at ψ=180° rising to 1.32 at
134°; `µd` inside the quoted 0.79–0.94; the water phase function normalizing to 1
over the backward hemisphere under numerical integration.

New: `robust/tests/test_ztt.py` (28 tests + 1 xfail). Modified: `robust/rt/ztt.py`
(TT2017 µ∞ + docstring), `design/rt_elastic_implementation.md` (v0.12, §§4.3–4.4).
Branch `rt-elastic-prototype` for JXP to commit. Next: task 4, the notebook.

### 2026-08-02 (M2 task 2 — ZTT transcribed; one term blocked by the paper itself)

`pytest -q` → **136 passed**; `ruff check` / `ruff format --check` clean. Record at
v0.11. `robust/rt/ztt.py` now implements the Twardowski & Tonizzo (2018) model, every
function naming its equation. **Nine of the ten terms are done; µ∞ is blocked because
the paper does not publish its own coefficients** — raised as **Q4**.

**The blocker.** Equation (8) fits `µ∞(bb/a, η_bb)` with sixteen coefficients
`m1..m16` and says "Coefficients m are provided in Appendix A, Table A2". They are
not. Table A2 as printed covers Equations (3), (4), (16), (17) and runs from
`m*_d,8` straight into Table A3; a full-text search of all 30 pages finds `m1` and
`m16` only inside Equation (8) itself. The paper points to MATLAB code at ioccg.org;
I checked two plausible URLs there and found no ZTT entry. µ∞ sits in Equation (12)'s
denominator, so it is load-bearing.

I did **not** invent sixteen numbers. They would have silently become "ZTT" in every
M3/M4 comparison thereafter, and the whole point of an analytic backbone is that it
is *not* fitted. `mu_infinity` implements the exact structure and requires the
coefficients, raising with a message that names the gap and the workaround.
`rrs_ZTT` accepts `mu_inf=<value>` so the other nine terms can be exercised and
gradient-checked, with docstrings saying plainly that a supplied constant is not the
paper's parameterization. **The consequence for task 3: it can gate the
transcription and the gradient path, but it must not report a standalone ZTT rRMS.**
Q4 lists the ways out, my preference being to ask the authors.

**Two checks that make me confident about the rest.** I wanted verification that was
independent of my own reading, not just "the code runs".

1. **The paper's own worked example reproduces exactly.** §2.1 says "for θs' = 60°,
   θs will be 40.3° and ψ will be 139.7° for nadir viewing". My code returns
   **40.26°** and **139.74°**. That single check exercises Snell refraction, the
   scattering-angle formula, *and* the nadir convention simultaneously — which
   matters because the paper's zenith angles are **in-water and measured from
   straight down, so nadir viewing is θv = 180°**, the opposite of `Geometry`'s
   `theta_v = 0`. Reversing that would have produced a wrong-but-plausible BRDF, the
   single most expensive error available in this milestone. I put both conversions in
   one function, `geometry_to_paper_angles`, so there is exactly one place to get it
   wrong.
2. **The water phase function matches an independent citation.** The paper defers
   `βw(ψ)` to Zhang et al. (2009) — whose only implementation to hand is the ocpy
   port M1 already found raises "THIS IS NOT SUCCESFULLY CONVERTED YET". Rather than
   stall, I derived it: molecular scattering goes as `1 + f cos²ψ` with
   `f = (1−δ)/(1+δ)`, and normalizing over the backward hemisphere gives
   `βw(ψ)/bbw = (1 + f cos²ψ)/(2π(1 + f/3))`. At ψ = 180° that is **0.2342 sr⁻¹**
   against the **0.23 sr⁻¹** the synthesis quotes for pure water (§3.5, citing Zhang
   2009 and this paper), and the analytic normalization matches numerical integration
   to machine precision. Only the shape is needed — Equation (10) multiplies by
   `bbw`, so the unknown `βw(90°)` cancels. A derivation I can check beats a
   dependency I cannot run.

Other values landed inside the paper's own stated ranges, which is weak but
reassuring evidence: `f_L(180°, 440 nm)` = 1.057 (paper: natural range 1–1.12,
Zaneveld's constant 1.05); `Ψ_KLu` = 1.024 at ψ = 180° rising to 1.315 at 134°;
`H(30°)` = 0.31 (Morel & Prieur assumed 0.4). And end-to-end on the L23 fixture with
*guessed* `P_bb = 0.014` and `µ∞ = 0.75`, `rrs(440)` comes out 0.0111 against L23's
0.0125 — 11% low with two invented inputs, which suggests the structure is sound.

**A reframing worth recording: `Pbb(ψ)` is not a gap, it is the design's M5.** The
paper says four parameters "must be provided from direct measurements or through
some assumptions" (§2.9): `bbp`, `apg`, `Pbb(ψ)`, `b̃bp`. Three already exist in M1's
types — and `B_p` *is* the paper's `b̃bp`. The fourth, `Pbb(ψ)`, is the particulate
backward phase-function shape, supplied by the caller. So **adding `Pbb` to
`PhaseParams` is precisely what the design means by "promote phase_params to the ZTT
backward-VSF parameterization" at M5.** That was abstract before; it is now a
concrete one-field change, which is exactly why `PhaseParams` was built to take extra
optional fields at M1.

Also extracted Table A3 in full (91 values, 350–800 nm at 5 nm) programmatically
rather than by hand, and cross-checked the 5 nm grid has no gaps. The initial text
extraction silently dropped Equations (8)–(10) — pypdf mis-ordered that block — which
I only caught because the equation numbering jumped. PyMuPDF recovered them. Worth
knowing for M5: **do not trust a single PDF text extractor for equations**; the
failure is silent.

One regression fixed: `test_env.py::test_unimplemented_stubs_raise` asserted
`ztt.Rrs_ZTT` raises `NotImplementedError`, which was true at M0 and is now false. I
narrowed it to `forward` (still a stub until M3) and documented why `ztt` left the
list, rather than deleting the test.

New: nothing. Modified: `robust/rt/ztt.py` (implemented), `robust/tests/test_env.py`
(narrowed stub list), `design/rt_elastic_implementation.md` (v0.11, §4.2 with the
per-term table and the blocker). Branch `rt-elastic-prototype` for JXP to commit.
Next: task 3, the gates — bounded by Q4 as noted.

### 2026-08-01 (M2 task 1 — Gordon-in-JAX, and it reproduces the published ladder exactly)

`pytest -q` → **136 passed** (117 M0/M1 + 19 new); without `$OS_COLOR`, **116
passed + 20 skipped**. `ruff check` and `ruff format --check` clean. Record at v0.10
with a new §4.

**The gate passed exactly.** Standard Gordon in JAX reproduces
`context/RT/fig_rrms_ladder.csv` at Y = 0 to better than **1e-5 percentage points**
at all seven wavelengths — 2.4948 / 2.9092 / 3.6714 / 4.8786 / 6.4499 / 7.6535 /
9.0418 %. An independently-written NumPy implementation and this one agree, which
means the rRMS definition, the `Rrs→rrs` conversion, `u`, and the two coefficients
are all consistent between them. That is the single most useful thing this task
produced: every relative accuracy claim from M3 onward is measured against this
baseline, and it is now pinned rather than assumed.

**The per-zenith prediction in this doc was wrong, and the data says so.** I wrote
that Gordon's rRMS "should *worsen* by roughly the measured 5% bias at 60°",
implying error grows with zenith. Measured:

| λ [nm] | 0° | 30° | 60° |
|---|---|---|---|
| 400 | 2.49 | **2.10** | 4.81 |
| 550 | 4.88 | 4.46 | 6.65 |
| 700 | 9.04 | 9.71 | 13.47 |

60° is much the worst everywhere (1.5–2× nadir), but **30° is *better* than nadir in
the blue** — the fixed Gordon coefficients evidently suit ~30° better than 0° below
~550 nm, which makes sense for coefficients derived from RT runs at some nominal
mid-range geometry rather than at nadir. So the honest claim is "60° is the clear
loser", not "error grows with zenith", and I corrected the doc-derived expectation in
the record instead of quietly reporting only the part that matched.

**That has a consequence for how M4 should be read**, which I would rather flag now
than at the gate: the geometry hold-out is *exactly* the angle where Gordon is
weakest. A hybrid win on the 60° split is therefore partly a win against a baseline
evaluated outside its best geometry — real, but not the same as beating it on its
home turf. Recorded in §4.3 and in the test docstring so the M4 write-up cannot
overstate it.

**Design decisions.** Gordon lives in its own `robust/rt/baselines.py` (the coding
plan's layout has no home for comparison models, so this is an addition to it) with
room for PR05/O25 at M4. It takes `forward`'s signature and **ignores**
`phase_params` and `geometry` — not a shortcut but the defining limitation, so M4 can
score every model in one loop. Two tests assert the blindness rather than trusting the
docstring, the sharper one showing Gordon returns *bit-identical* predictions at
0°/30°/60° while the reference `Rrs` does not. That test is only possible because M1
established the IOP fields are identical across the zenith files, so the comparison
isolates geometry perfectly.

**`rrms` landed in `validation.py` now rather than at M4.** The entire value of the
metric is that one definition is shared: the number in this log, the M4 table, and the
synthesis figures have to be the same quantity or none of the comparisons mean
anything. Writing it twice would have been the natural way to get that wrong. It is
pure JAX and differentiable, so M3 can use it directly as a training loss. I read the
exact formula out of the synthesis script rather than reconstructing it from the
prose — `100·sqrt(mean(((pred−truth)/truth)²))`, with `G1`/`G2` *fixed* rather than
fitted, which is what "standard Gordon" means in the gates.

I also checked the reference script's scene mask (`isfinite & rrs > 0`) before
trusting the comparison; it is all-True for L23, so no scenes are silently excluded on
either side. One test guards the reference CSV itself — seven rows, plausible
magnitudes — since a missing or malformed reference would make the gate vacuous
rather than failing.

Q&A got no answer on where baselines should live, so I went with the
`robust/rt/baselines.py` I had proposed in this doc and recorded it as a decision in
§4.2; trivially movable if you want it elsewhere.

New: `robust/rt/baselines.py`, `robust/tests/test_baselines.py`. Modified:
`robust/rt/validation.py` (`rrms`), `robust/rt/__init__.py` (export `baselines`;
status docstring), `design/rt_elastic_implementation.md` (v0.10, new §4). Branch
`rt-elastic-prototype` for JXP to commit. Next: task 2, the ZTT transcription — where
the benchmark above becomes the thing to beat.
