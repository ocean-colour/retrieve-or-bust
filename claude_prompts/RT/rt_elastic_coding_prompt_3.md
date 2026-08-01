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

3. **Validate & gate.** Tests in `robust/tests/test_ztt.py`:
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

4. **Notebook.** `notebooks/RT/rt_elastic_coding_3.ipynb` — the M2 explainer,
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

5. **PR Review.** Fetch the review comments on the open PR and address them. `gh`
   is not authenticated in this environment; the public REST API works:
   `curl -s https://api.github.com/repos/ocean-colour/retrieve-or-bust/pulls/<n>/comments`
   and `.../reviews`. Fix the *class* of each defect rather than the single
   instance, and demonstrate that the fix catches what was reported. Log it.

6. **Finally.** Modify the next prompt doc `rt_elastic_coding_prompt_4.md` (M3:
   emulator + hybrid) given what M2 established. Use Fable if you can. Log your
   work.

### Q&A

## Next

→ `rt_elastic_coding_prompt_4.md` (M3: Emulator + hybrid).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
