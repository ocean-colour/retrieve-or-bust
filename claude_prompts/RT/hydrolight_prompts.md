# HydroLight Runs — the reference data the RT model still needs

This prompt doc collects **every HydroLight run we need to commission** to finish
the `robust.rt` forward model, elastic and inelastic together, and will guide the
work of specifying, commissioning and ingesting them.

It exists because both halves of the model have now hit the same wall from
opposite directions. The elastic effort's M5 established that the analytic
backbone cannot be repaired with the data in hand; the inelastic effort's report
put "commission the geometry runs" at priority 2 and CDOM fluorescence shipped
**unvalidated** for want of a truth channel. Those are not five separate asks.
They are one commissioning campaign, and this document is it.

## Context

Read first, in this order:

- [`design/m5_report.md`](../../design/m5_report.md) — why the elastic model
  stopped, and §3 for the four options and where each now stands. **§3 option 3
  is the crux: µ∞ cannot be refit from any dataset we hold.**
- [`design/prototype_summary.md`](../../design/prototype_summary.md) — the
  validity envelope and its measured coverage (Route C, 2026-09-21).
- [`design/rt_inelastic_model.md`](../../design/rt_inelastic_model.md) §8 — the
  existing six-item wishlist, which this doc supersedes by making it specific.
- [`design/rt_cdom_fluorescence_model.md`](../../design/rt_cdom_fluorescence_model.md)
  §7 — the CDOM run request, already written as a commissioning spec.
- [`reports/report_rt_inelastic_model.md`](../../reports/report_rt_inelastic_model.md)
  §5 and §7 — what the inelastic model may not claim, and the priority order.
- Robert Frouin's comment in
  [`rt_elastic_prompts.md`](rt_elastic_prompts.md) (*Comments / Robert's*) — the
  independent case for exactly this campaign.

## Code

If you need to run Python use the `ocean14` conda environment. Existing loaders
worth reading before designing an ingest:
[`robust/rt/data/l23.py`](../../robust/rt/data/l23.py) and
[`robust/rt/data/pb24.py`](../../robust/rt/data/pb24.py).

---

## What we have, and the hole in the middle

Two HydroLight-derived datasets, and **the axes they cover are disjoint**:

| | **L23** (Loisel+2023) | **PB24** (Pitarch & Brando 2024) |
|---|---|---|
| scenes / realisations | 3320 | 5000 |
| wavelengths | 81, 350–750 nm @ 5 nm | 12 OLCI bands; also 451 bands, 350–800 nm @ 1 nm |
| solar zenith | **3 values** (0/30/60°) | 10 values (0–87.75°) |
| view zenith | **nadir only** | 10 values (0–87.5°) |
| relative azimuth | **0 only** | 13 values (0–180° @ 15°) |
| phase function | **one fixed Fournier–Forand** | `B_p` spans ~6–12× across realisations, FF family |
| **Raman** | **yes** (X=2) | **no** — the files are literally `SD_*_no_R_*` |
| **Chl fluorescence** | **yes** (X=4, at φ_C = 0.02 exactly) | **no** |
| **CDOM fluorescence** | **no** | **no** |
| asymptotic K∞ / µ∞ | no | **no** — seven K's, all varying ~1.4× with θ_s, so all surface K's |

**The hole:** *no dataset we hold has an inelastic process at an off-nadir
geometry, or at any solar zenith other than three, or under more than one phase
function.* The inelastic correction heads are interpolators over three θ_s
anchors and were trained at nadir; the elastic emulator has the geometry but no
inelastic physics to learn from. Crossing those axes is not an optimisation — it
is a measurement nobody has made.

And underneath both: **ZTT's µ∞ and `F(ψ)` are defined against the asymptotic
`K∞`, which neither dataset tabulates.** That single absence is what closed
Route A at M5 and is why run **R1** leads this list.

---

## The runs

Priority order, and each entry says what it unblocks so the list can be cut from
the bottom without guessing. "Scenes" below means water bodies; multiply by
geometries and by scenario (X) for the run count.

### R1 — Asymptotic quantities, and scattering angles below 134°

**The one that unblocks everything else on the elastic side.** Without it Route A
is impossible and the model's off-nadir failure is permanent.

*Why.* `µ∞ = a/K∞` and `F(ψ) = K_Lu/K∞ − 1`, and `K∞` is **θ_s-independent by
definition**. PB24 tabulates seven diffuse-attenuation coefficients and every one
varies by ~1.4× across solar zenith, so none of them is `K∞`; the best proxy
(`a/K_d`) is a *surface* quantity standing in for an asymptotic one, and adopting
it would swap a published parameterization for one whose error we cannot
characterise. Meanwhile `Ψ_KLu(ψ) = 1 + F(ψ)` is a quartic fitted for ψ ≳ 134°
that **crosses zero at 110.4°**, and 42 % of PB24's own sanctioned window sits
below 134°, 16 % below the crossing.

*Spec.*

- **Homogeneous, optically deep water bodies** — the asymptotic regime only
  exists for a homogeneous water column, so this run is deliberately *simpler*
  than L23's stratified scenes.
- **Output the asymptotic radiance distribution and `K∞` directly** if HydroLight
  will report them. We believe it does for homogeneous water bodies, but **that is
  an assumption to check with the operator before the spec is signed off**, not a
  fact this project has verified — and the switch and its version must be recorded
  either way (see *Metadata* below). **Failing that**, output `K_d(z)`, `K_u(z)`, `K_Lu(z)` and `µ̄(z)` on a depth
  grid reaching **≥ 25 optical depths** at every wavelength, so the asymptote can
  be read off the profile and its convergence *shown* rather than assumed.
- **An IOP grid, not a scene ensemble.** Span `bb/a` over **1e-4 to ≥ 3**
  (TT2017 fitted 1e-4–0.1; L23 reaches 0.59; PB24 reaches 20.1) and
  `η_bb = bb_w/bb` over **0.01 to 0.98** (TT2017's own range), logarithmically,
  ~12 × 8 nodes. Two decades beyond the published fit is the point.
- **Geometry down to low ψ**: solar zenith 0–80° and view zenith 0–80° with
  azimuth 0–180°, dense enough to sample **ψ from ≤ 70° to 180°**. ψ ≤ 70 is what
  PB24's window already reaches, and what the current quartic gets catastrophically
  wrong.
- Full 350–750 nm on the L23 5 nm grid.

*Unblocks.* Refitting µ∞ and `F(ψ)` over the domain we actually use; deciding
between repairing ZTT and replacing it; and — either way — a backbone that is
*physical* off-nadir, which is the precondition for every other run here being
worth doing.

### R2 — The VSF *family* varied, not just its parameter

*Why.* This is the one headline gap M5 never reached, and it is Robert's central
point: most analytical models fold phase-function effects into coefficients
derived under a *prescribed* phase function, so they cannot represent independent
variability in its shape. `PhaseParams` was built as a container precisely so it
could carry `beta_tilde_pi` and `backward_slope` — both currently
**uncalibrated**, because nothing we hold constrains them. L23 has one fixed
Fournier–Forand; PB24 varies the FF *parameter* across realisations but stays in
the FF family.

*Spec.*

- **At least three distinct VSF families** at matched `b_bp/b_p`: Fournier–Forand,
  Petzold (average-particle), and a measured/Sullivan–Twardowski-style VSF —
  plus, if available, a two-component (small + large particle) mixture.
- **Matched bulk `B_p` across families**, so a family effect cannot be confounded
  with a backscatter-ratio effect. That pairing *is* the experiment.
- `B_p` spanning the design's nominal ~7× band at minimum; PB24's ~12× if free.
- Shared with R1's geometry grid wherever possible.

*Unblocks.* Calibrating the backward-VSF axis; a held-out-*family* split (which
is a far stronger claim than the held-out-`B_p` split M5 built); and the honest
version of "explicit phase-function dependence", which the package currently
advertises and exercises only weakly.

### R3 — Denser zeniths **and off-nadir views, with X1/X2/X4** — the crossing run

**The single most valuable inelastic run, and the one that closes both reports'
sharpest caveat at once.**

*Why.* The correction heads interpolate over exactly three solar-zenith anchors
and carry **no domain guard**; trained without 60° the Raman head errs by **−74 %**
there — worse than no head at all. The elastic report's item 3 says the same thing
about the emulator from the other side. And every inelastic number we have is
**nadir**: PACE is not nadir, and the Raman and fluorescence source terms depend on
the in-water light field, which is exactly what changes off-nadir.

*Spec.*

- **Solar zenith 0–75° in 15° steps** (6 values, versus today's 3) — enough to
  hold one out and *test* interpolation rather than assert it.
- **Off-nadir views**: view zenith 0–60° and relative azimuth 0–180°, on a grid
  shared with R1/R2 so one geometry setup serves all three.
- **All three scenarios, X=1 / X=2 / X=4**, on the *same* water bodies, so the
  differences `X2 − X1` and `X4 − X2` remain the clean per-process truth channels
  the current model is built on.
- The L23 scene ensemble, or a representative stratified subset of it (see *Cost*
  below) — reusing L23's water bodies makes every existing number directly
  comparable.
- Full 350–750 nm at 5 nm.

*Unblocks.* A real zenith-interpolation gate; domain guards for the heads that are
*measured* rather than guessed; the first off-nadir inelastic truth that has ever
existed for this project; and retiring the elastic report §5 item 3 and the
inelastic report §5 item 1 together.

### R4 — Varied quantum yield φ_C

*Why.* The model's φ_C-linearity is exact **by construction**, and L23 provides
truth at exactly one yield (φ_C = 0.02). Whether the real ocean's fluorescence is
φ_C-linear at the few-percent level is **untested** — and φ_C is the physiology
handle the whole inversion is being built to retrieve, so a shape bias here
propagates straight into the science result.

*Spec.* φ_C ∈ {0.005, 0.01, 0.02, 0.04, 0.06} on a scene subset (a few hundred
scenes is ample), X=4, all three zeniths at minimum — the denser R3 grid if it
runs in the same batch. Full spectral range.

*Unblocks.* Testing φ_C-linearity against truth instead of construction; a
defensible φ_C retrieval.

### R5 — CDOM fluorescence on/off pairs

*Why.* CDOM fluorescence is **shipped and completely unvalidated** — off by
default, analytic-only, its δ_C head untrained and unwired, because *no reference
of any kind exists*: L23 omits the process and BING never implemented it. It is
the only process in the package with zero truth.

*Spec.* Already written as a commissioning spec in
[`design/rt_cdom_fluorescence_model.md`](../../design/rt_cdom_fluorescence_model.md)
§7 — adopt it verbatim. In brief: paired **"X4" vs "X4 + CDOM-fl"** runs on the
same water bodies (so the difference isolates the process, exactly as X4−X2
isolates chlorophyll fluorescence); a CDOM-stratified subset spanning the full
`a_g(440)` range **with the CDOM-rich tail oversampled**; all zeniths; full
350–750 nm; and — non-negotiable — **the exact Hawes quantum-efficiency function,
variant and constants recorded**, since published Hawes constants exist in several
variants and a mismatch would silently re-introduce the error the design took care
to avoid.

*Unblocks.* Training δ_C; scoring the term quantitatively; turning "Hawes-consistent
and plausible" into "validated", or else deleting the term honestly.

### R6 — Sub-350 nm, on the excitation side

*Why.* Raman excitation for a 400 nm emission is at **352.11 nm**, barely inside
L23's 350 nm floor. Below 400 nm emission the excitation leaves the grid entirely,
the terms run on clamped IOPs and the heads never trained there — a measured **13 %
error at 350 nm**. The domain is documented, not enforced.

*Spec.* Output to **330 nm or below**, or at minimum excitation-side IOPs to
~330 nm, on whichever ensemble R3 uses. Cheap if specified up front, impossible to
retrofit.

*Unblocks.* The λ < 400 nm caveat; UV-capable applications.

### R7 — Alternative solar spectra

*Why.* The model is internally consistent with L23's sun, but the community's
absolute solar references are suspect (JXP's DQ5). The concern is currently
**unquantified**, which is the worst state for a concern to be in.

*Spec.* Re-run a modest scene subset under a second solar spectrum (e.g. a
TSIS-1-based Ed against the HydroLight default), X=2 and X=4, all zeniths. A few
hundred scenes is enough to bound it.

*Unblocks.* Turning DQ5 from a worry into a number, and telling us whether the
real-data transition needs an Ed-coupling seam or not.

### R8 — Vertical structure in Chl and φ_C *(stretch)*

*Why.* L23 is vertically homogeneous, but the 685 nm fluorescence signal
originates shallower than the blue-green elastic signal — a known, accepted
homogeneity bias that nothing in hand can quantify. Note the tension with R1,
which *requires* homogeneity; these are different runs, not a shared setup.

*Spec.* A small set of profiles — subsurface chlorophyll maximum at two or three
depths, and a φ_C that varies with depth (photoinhibition near the surface) —
against matched homogeneous controls.

*Unblocks.* Bounding the homogeneity bias. Lowest priority: worth asking for only
if the batch is cheap to extend.

---

## The combined ask

One campaign, three batches, so the geometry setup is paid for once:

| Batch | Runs | Why together |
|---|---|---|
| **A — the backbone** | R1, R2 | Both are IOP-grid runs on homogeneous water over a shared wide geometry grid. Neither needs inelastic physics, so they are the cheapest per answer and they gate everything else. |
| **B — the crossing** | R3, R4, R6 | All three are the L23 ensemble under a denser geometry grid with X1/X2/X4; R4 and R6 are a scene subset and a spectral-range flag on top of R3's setup. |
| **C — CDOM** | R5 | Needs its own scenario (CDOM-fl on/off) and its own stratification, but should share batch B's geometry grid. |

**If only one batch is possible, it is A.** Without a physical backbone off-nadir,
batch B's off-nadir inelastic truth would be measured against a model that cannot
use it. **If only one run is possible, it is R1** — every other item on this list
assumes the backbone question is answerable.

*Cost is the open variable and JXP's call.* PB24 is 5000 realisations × 1300
geometries ≈ 6.5 M spectra, so a campaign on that scale is clearly feasible for
someone; ours is smaller on the scene axis and wider on the scenario axis. Before
committing, we should price batch B at a few sizes (the full 3320 L23 scenes vs a
stratified 500) and find out which axis actually dominates the run time —
geometry, wavelength, or scenario.

## What must be recorded for every run

The metadata contract. Most of these are unrecoverable after the fact, and one of
them (the Hawes variant) has already been identified as a live foot-gun:

1. **HydroLight version and build**, and every non-default switch.
2. **The phase function**: family, parameters, and the tabulated `β̃(ψ)` actually
   used — not just its name.
3. **The Raman scattering coefficient and wavelength-redistribution function**,
   with their source.
4. **The chlorophyll fluorescence quantum efficiency and emission shape** — the
   band centre, width, and whether any PS I shoulder is included.
5. **The CDOM-fluorescence quantum-efficiency function: which published Hawes
   variant, any HydroLight-side modification, all constants.**
6. **The solar spectrum and sky model**, by name and version.
7. **The depth grid and the optical depth reached**, for R1's asymptote.
8. **The output convention** — `Rrs` vs `rrs`, above- vs below-water, and at which
   depth — stated explicitly rather than inferred from magnitudes.
9. **The IOP decomposition**: `a_ph`, `a_g`/`a_cdom`, `a_NAP`, `b_ph`, `b_NAP`
   reported separately, and consistent with `a_ph + a_cdom ≤ a`. The loaders will
   pin this; silent double-counting against `a_dg` conventions elsewhere is a real
   foot-gun.
10. **Whether IOPs are bit-identical across scenarios and geometries** — L23's are,
    and both loaders exploit it; a future release that quietly varies them would
    break us silently.

---

## Prompts

### Setup

1. Read this doc and the six documents in *Context*. Before anything is
   commissioned, have a conversation with me in Q&A/Setup: what you need from me
   to turn this catalogue into a spec someone can run, what you think I have
   wrong, and which runs you would cut first if the budget is a third of this.
   Use Fable if you can. Log your work.

2. I have answered your questions in the Q&A section below. Please review them and
   react accordingly. Ask another round if needed. Use Fable if you can. Log your
   work.

### Spec

1. Write the commissioning document — one self-contained spec per batch, in the
   form whoever runs HydroLight will actually work from, with the metadata
   contract attached to each. Name it `design/hydrolight_runs.md`. Use Fable if
   you can. Log your work.

### Ingest

1. The first batch has arrived. Write the loader, following the conventions of
   `robust/rt/data/l23.py` and `pb24.py` — a `LoadReport` that says what was kept
   and what was dropped, splits, a committed small fixture so CI runs real
   numbers without the data mount, and tests that pin the measured properties of
   the release rather than trusting its documentation. Use Fable if you can. Log
   your work.

## Comments

### Robert's

Quoted in full in [`rt_elastic_prompts.md`](rt_elastic_prompts.md) under
*Comments / Robert's*. The operative sentence for this document:

> My recommendation would be to use a full radiative-transfer solver (most
> naturally HydroLight) as the reference forward model, with particle
> phase-function parameters explicitly varied. For computational retrieval,
> construct a fast differentiable emulator of that solver.

R1 and R2 are that recommendation made specific. Note that he also proposes the
hybrid architecture `Rrs(model) = Rrs(ZTT) + deltaRrs(simulator)` — which is what we
built, and which M5 showed fails off-nadir *because of the ZTT term*, not the
emulator. That is worth putting to him directly.

## Q&A

### Setup

## Logging

Record work in the Logs section below, format:

### \<Date\> (Short summary of the work)

\<Detailed description of the work and what you learned\>

## Logs
