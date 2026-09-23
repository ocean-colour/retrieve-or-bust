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

3. I have answered your second round of questions in the Q&A section below, and 
   went back to address the ones I had missed. Hopefully my answers are consistent. Please review them and
   react accordingly. Ask another round if needed. Use Fable if you can. Log your
   work.

4. Henry is going to need input files to run from.  Generate a prompt in the `Spec` section below to generate them.  Use Fable if you can. Log your work.

### Spec

1. Write the commissioning document — one self-contained spec per batch, in the
   form whoever runs HydroLight will actually work from, with the metadata
   contract attached to each. Name it `design/hydrolight_runs.md`. Use Fable if
   you can. Log your work.

2. **Henry needs input files.** Build the deck generator and produce batch 0's.
   Work from [`design/hydrolight_runs.md`](../../design/hydrolight_runs.md) — it
   settles the conventions, the water bodies, the VSF designs and the metadata
   contract, so this prompt is about *building* them, not re-deciding them. Use
   Fable if you can. Log your work.

   **The ordering problem, and the architecture it forces.** We do not yet know
   HydroLight 6's exact deck syntax — that is §10 Q7, "one example deck and its
   outputs from the working setup". Do **not** block on it. Split the generator
   in two:

   - a **format-neutral intermediate representation** (the "run IR"): one record
     per water body carrying `a(λ)`, `b(λ)`, `bb(λ)`, the tabulated `β̃(ψ, λ)`,
     the depth/bottom specification, the sky and wind parameters, the scenario
     switches and the solar-zenith list — everything physical, in SI, on the
     85-band 330–750 nm grid of spec §2.2. Serialise it as one JSON sidecar plus
     one NPZ of arrays per water body. **All the hard work lives here**, and none
     of it depends on deck syntax;
   - a **thin format adapter** that renders the IR into whatever HydroLight
     actually wants, written last and replaceable in an afternoon once Henry's
     template arrives.

   If the adapter cannot be finished, the IR plus the `β̃(ψ)` and IOP tables are
   *still* shippable — they are the physics, and they are format-light.

   **Where it lives** (A-R2Q3, spec §8.1): a new `robust/rt/hydrolight/` package
   —

   - `grid.py` — the batch-A `(bb/a, η_bb)` grid, the realizability mask of spec
     Appendix A, and the 150-body Latin-hypercube fill set;
   - `vsf.py` — the 27 VSF designs of spec Appendix B: the Fournier–Forand
     closed forms `B_p(n, µ)` and `β̃(ψ; n, µ)`, the 15 two-parameter branches,
     the 10 two-component mixtures, and the Petzold anchor. Emit each as a
     tabulated `β̃(ψ)` on the angular grid HydroLight wants;
   - `ensemble.py` — the batch-B design block (1,500 Latin-hypercube water
     bodies) and the batch-C subsets (R4's 100, R5's 500 CDOM-stratified with the
     tail oversampled, R8's 60 profiles);
   - `l23_recon.py` — the batch-B L23 block: rebuild all 3,320 L23 water bodies
     from their own published `a`, `bb`, `bbnw`, `bnw`, `aph`, `ag`, with the FF
     parameter inferred from `B_p = bbnw/bnw` (spec §6.2), carrying the three
     caveats as recorded fields, not as comments;
   - `deck.py` — the IR, its serialiser, and the format adapter;
   - `manifest.py` — the per-batch manifest of spec §2.7, including the SHA-256
     of every input file.

   **Milestones.**

   - **M0 — the IR and the VSF library.** `vsf.py` complete and tested: the 27
     designs, each reproducing its target `B_p` to < 4e-4, and the Appendix B
     shape table (the 49 % spread at ψ = 1° against 1–3 % beyond 120°)
     regenerated as a test rather than quoted from the spec.
   - **M1 — the grid and the ensembles.** `grid.py` reproducing **63 of 96
     realizable nodes at λ_ref = 400 nm** and the **90 % box occupancy**, with
     the ten empty cells enumerated by the test. `ensemble.py` and `l23_recon.py`
     producing their water-body sets.
   - **M2 — batch 0's 21 water bodies**, chosen and *justified in writing*: the
     10 L23 scenes spanning `a_ph(440)`, the 5 for the R0 convergence sweep
     spanning `bb/a` and trophic state, and the 6 batch-A grid corners. Selection
     must be reproducible from a seed, not hand-picked.
   - **M3 — the in-house pre-check, before Henry runs anything.** Push the
     reconstructed L23 IOPs through our existing `robust.rt.forward` and compare
     against L23's published `Rrs` for the same scenes. This does not validate
     HydroLight — it validates *our reconstruction pipeline*, and it costs no
     compute. If the reconstruction is broken we find it here rather than in the
     pilot's 90 runs. Report the number; a disagreement materially larger than
     the model's own known accuracy on L23 is a bug in `l23_recon.py`.
   - **M4 — the batch 0 delivery.** A tarball for Henry: the run IR, the
     `β̃(ψ)` tables, the IOP tables, the decks if the adapter exists, the
     manifest, and a short `README` stating the conventions of spec §2 in the
     operator's terms and listing the ten questions of spec §10.

   **Tests that must exist**, pinning measured properties rather than
   documentation, in the style of `robust/tests/test_l23.py` and `test_pb24.py`:
   the realizability mask and its count; each VSF design's `B_p`; the
   matched-`B_p` shape table; the L23 round-trip (reconstructed IOPs equal the
   file's IOPs to float tolerance); manifest completeness against the ten items
   of spec §3, with the operator-supplied items 1/3/4/5 explicitly marked absent
   until batch 0 returns; and **byte-identical regeneration** — running the
   generator twice must produce identical files, or the manifest hashes are
   worthless.

   **Do not** generate all ~157,000 decks in this pass. Batch 0 is 21 water
   bodies and 172 runs; build the machinery so the rest is a parameter change,
   and generate the rest only when batch 0 has passed its gates.

   Conventions: `ocean14`, clear docstrings, no git commands. If you hit a
   decision `design/hydrolight_runs.md` does not settle, ask in **Q&A/Decks**
   below rather than inventing one — but prefer to proceed where the spec is
   clear.

3. Please generate a simple HOWTO that I can share with Henry so that he knows what to do and how.  Use Fable if you can. Log your work.

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

Questions from Claude (2026-09-22), after reading this document and the six
Context items — [`design/m5_report.md`](../../design/m5_report.md),
[`design/prototype_summary.md`](../../design/prototype_summary.md),
[`design/rt_inelastic_model.md`](../../design/rt_inelastic_model.md) §8,
[`design/rt_cdom_fluorescence_model.md`](../../design/rt_cdom_fluorescence_model.md) §7,
[`reports/report_rt_inelastic_model.md`](../../reports/report_rt_inelastic_model.md)
§5/§7, and Robert's comment — plus the two existing loaders
([`robust/rt/data/l23.py`](../../robust/rt/data/l23.py),
[`robust/rt/data/pb24.py`](../../robust/rt/data/pb24.py)). Nothing has been
commissioned and no code changed.

Three sections, matching the three things the prompt asks for: **§A** what I need
from you to turn this catalogue into a spec someone can run, **§B** where I think
the catalogue is wrong, **§C** the third-budget campaign. Please answer inline.

---

#### A. What I need from you before a spec can be written

**HQ1 (Who runs it, and on which solver).** Sequoia/Mobley, Loisel's group
(who made L23), Pitarch (who made PB24), or an in-house HydroLight seat? Two
things hang on it. First, **EcoLight and EcoLight-S cannot deliver R1–R3**:
they solve the azimuthally-averaged problem and return no Δφ dependence and no
full `Lu(θ_v, Δφ)`. Every off-nadir/azimuth ask in this document requires full
HydroLight, and if the budget conversation is really a conversation about
EcoLight then the catalogue collapses to R4/R5/R7 and we should know that on
day one. Second, if the operator is Loisel's or Pitarch's group we may be able
to inherit their input decks, which is the difference between HQ2 having an
answer and not.

**A1:** Henry Houskeeper is going to run it on HydroLight; the very latest version

**HQ2 (Do we have L23's HydroLight input decks?).** R3's entire payoff clause
is "reusing L23's water bodies makes every existing number directly
comparable". That needs the 3320 scenes' **inputs** — the `a_ph`/`a_g`/`a_NAP`
and `b_p` specifications, the Fournier–Forand parameters, the sky spec, the
depth grid — not the published netCDFs we already load. Do we have them, can
we get them, or is this question one for whoever runs the campaign? **This
gates the batch-B spec**: if the answer is no, batch B produces a *new*
ensemble, R3's comparability claim comes out of the spec, and the sensible move
is HD4 below (build batch B's water bodies out of batch A's grid instead).

**A2:** No, I do not have those and cannot get them.

**HQ3 (Budget, in units the operator understands, and any deadline).** The
document says cost is your call, and it is, but I cannot price three batches
without an order of magnitude: is the campaign 10³, 10⁵, or 10⁷ HydroLight
runs? Dollars, core-hours, or "one person's quarter" are all fine units. And
does any date bind — a PACE cal/val milestone, a proposal, the paper this work
becomes?

**A3:** . Let's assume 10^5 runs

**HQ4 (Delivery format, precision, mount).** I would ask for PB24's shape —
one netCDF per realisation under a versioned directory, `$OS_COLOR/<name>/v1/`
— plus one manifest file per batch, rather than L23's one-file-per-zenith,
because it makes partial delivery and incremental ingest natural. On
precision: PB24's `rrs` **underflows to exactly 0** at grazing geometries in
float32 and our metric divides by truth (`drop_zero_rrs` exists for this).
I would ask for float64 on the radiometric fields, or failing that an explicit
statement of the floor. Any constraint from the operator's side I should
design around?

**A4:** No contraints;  request what you prefer

**HQ5 (Can the operator drive HydroLight with user-supplied IOPs?).** R1 and
R2 are an `(bb/a, η_bb)` grid with a prescribed tabulated `β̃(ψ)` — which is
not how L23 or PB24 were built (both are driven by component concentrations).
HydroLight's user-supplied-IOP input mode is the clean route to landing on
exact grid nodes and to R2's multiple VSF families. If the operator only ever
drives it with concentrations, both runs have to be re-expressed as
concentration recipes and the spec changes shape substantially. Worth asking
before anything is written.

**A5:** The operator can drive with any input files we give them

**HQ6 (What angular grid will we actually get?).** HydroLight's standard quad
layout is ~10° in θ and 15° in φ — PB24's 10 × 10 × 13 grid is exactly that
structure. R1 asks for geometry "dense enough to sample ψ from ≤ 70° to 180°".
Is a custom quad file on the table, or do we write the spec against the
standard quads and accept the ψ sampling they give? I can check what the
standard quads yield in ψ once I know the version, but I would rather ask than
assume — this is the same class of assumption the document already flags for
`K∞`.

**A6:** Do what you think is best

**HQ7 (Metadata contract: ship decks, not prose).** Items 1–10 of *What must be
recorded* are a contract with a person, and people forget. I would like to
replace most of it with: **deliver every HydroLight input deck, the tabulated
`β̃(ψ)` actually used, the Ed/sky specification, and the version string
alongside the outputs**, plus one machine-readable manifest per batch. That
makes items 1, 2, 6, 7 and 8 automatic and *testable by the loader* rather than
asserted in an email. Only 3, 4 and 5 — the Raman redistribution function and
the two fluorescence quantum-efficiency functions, which live in the code
rather than the deck — would still need prose from the operator. Is that an
acceptable ask to make of them?

**A7:** Yes

**HQ8 (A pilot, written into the spec as a gate).** ~10 runs spanning the
corners of whichever batch goes first, delivered in the **final** format with
the final manifest. We write the loader against it, run the tests, confirm the
output convention (item 8) from real numbers rather than from magnitudes, and
only then sign off the full batch. Cost: days. Value: the difference between
finding a convention mismatch now and finding it after the compute is spent.
Do you want that in the spec?

**A8:** Agreed; let's have a test run first.

**HQ9 (Usable domain, or failure map?).** R1 spans `bb/a` to ≥ 3 and `η_bb` to
0.98. Do we intend the model to be *usable* out there — in which case that
region is training data and needs density — or only to know where it breaks, in
which case a sparse boundary sweep does the job at a fraction of the nodes?
"Two decades beyond the published fit is the point" reads like the second, but
the 12 × 8 logarithmic grid is specified like the first.

**A9:** I'd prefer a usable domain

**HQ10 (Name the switches, not the X's).** R3 asks for X1/X2/X4 and R5 for
"X4 + CDOM-fl". X is L23's labelling, and "X5" is a name we would be inventing.
For a new campaign I would specify the physics switches directly — Raman
on/off, Chl-fluorescence on/off, CDOM-fluorescence on/off — and record the
mapping onto X1/X2/X4 in the manifest, so the loader pins switches rather than
a label. Agreed?

**A10:** Yes

**HQ11 (Ownership and embargo).** If the runs come from Loisel's or Pitarch's
group, is there an embargo, a data-sharing condition, or a co-authorship
expectation? Cheap to settle now, expensive to discover later.

**A11:**  There is no issue here

---

#### B. Where I think the catalogue is wrong

**HD1 — µ∞ probably does not need a HydroLight run at all, and that is the
single point of failure worth removing.** The document's own argument for
excluding PB24 is that `K∞` is θ_s-independent *by definition*. The reason it
is θ_s-independent is that the asymptotic radiance distribution is determined by
the IOPs alone — `(a, b, β̃)` — and has forgotten the incident field entirely.
But that also means `µ∞` is obtainable from an **eigenvalue solve of the
asymptotic RTE** for a given `(a, b, β̃)`: a small in-house computation on the
Preisendorfer/Prieur/Zaneveld line, with published tabulations to check
against. If that works, the half of R1 this document calls "the crux" costs a
week of our own time and no compute, over any `(bb/a, η_bb, β̃)` grid we like —
*including R2's families*, which no run would give us at that density anyway.

This does not kill R1. What survives is the **`F(ψ)` half**, which needs
`K_Lu(θ_v)` from a real solver and genuinely does need R1's geometry grid. But
it changes what R1 is *for*, and it removes the load-bearing assumption the
document already flags — "we believe HydroLight reports `K∞` for homogeneous
water, but that is an assumption to check". I would spend a week testing the
eigenvalue route **before** signing the campaign's largest line item, precisely
so that R1's value does not depend on how that check comes back.

**Do you want that week spent first?**


**A-HD1:** Yes, that makes sense

**HD2 — R1 and R2 are one run, and R2 carries the information nothing else can
give us.** They share water bodies, geometry, wavelengths and outputs; R2 *is*
R1 with a family axis. Running R1 alone is a strict subset, so R2's marginal
cost is a ~3–4× multiplier on batch A, not a second setup. And the family axis
is the only axis in this entire catalogue that **no dataset in existence for us
varies at all**: PB24 varies the Fournier–Forand *parameter*, staying inside one
family, and L23 fixes it. So I would invert the document's tiebreak. "If only
one run is possible, it is R1" becomes: *if only one run is possible it is R1's
geometry grid with R2's families on it*, and the family axis is the **last**
thing cut, not the second. It is also the literal content of Robert's
recommendation.


**A-HD2:** Assume that we will do all the runs 

**HD3 — "matched `B_p` across families" is not straightforwardly
constructible.** Petzold's average-particle VSF is a single measured curve with
a fixed backscatter ratio (~0.018); Fournier–Forand is a one-parameter family
in which `B_p` and the forward slope are *coupled*. To place Petzold at a
different `B_p` you must modify it — at which point it is no longer Petzold and
the family label is doing no work, so a "family effect" and a "modification
effect" are confounded exactly the way the run was designed to avoid. The
honest experiment is either (i) matched `B_p` only over the narrow band where
the natural families genuinely overlap, or (ii) a **two-component (small +
large particle) mixture**, where the mixing ratio sets `B_p` and the component
size distributions set the shape — two genuinely independent knobs, which is
what calibrating `beta_tilde_pi` and `backward_slope` actually requires. The
document lists the mixture as an optional fourth ("plus, if available"). I
would make it the **primary** design and the named families the anchors, rather
than the reverse.

**A-HD3:** Yes, that makes sense

**HD4 — batches A and B as specified re-create the hole this document opens by
describing.** Batch A is an IOP grid, homogeneous, elastic-only. Batch B is the
L23 ensemble under a denser geometry with X1/X2/X4. **Those two share no water
bodies.** So the campaign would deliver, again, two datasets whose axes are
disjoint — and *still* no sample anywhere that has an inelastic process **and** a
varied VSF family. That is the exact failure mode the first page diagnoses in
L23-vs-PB24, reproduced inside one campaign we control.

The fix is cheap: **make batch B's water bodies a subset of batch A's grid
nodes** (or include a block of A's nodes in B's ensemble). Then one block of the
release spans IOPs × geometry × VSF family × scenario simultaneously, every
cross-term is measurable for the first time, and the elastic and inelastic
efforts stop being scored on different oceans. It also makes HQ2's answer stop
mattering.

**A-HD4:** Ok, that's fine

**HD5 — a factual slip in R1, and it makes HD4 cheaper.** R1 describes its
homogeneous water bodies as "deliberately *simpler* than L23's stratified
scenes". **L23 is vertically homogeneous** — R8 says so, and
`design/rt_inelastic_model.md` §8 says so ("L23 is homogeneous; the 685 nm
signal originates shallower than the blue-green — a known, accepted homogeneity
bias"). I think "stratified" has picked up R3's *sampling* sense here. The
consequence is good news: R1's homogeneity requirement is **not** in tension
with reusing L23-like water bodies, so HD4's merge costs nothing physically.
R8's parenthetical tension is real, but it is only about R8's own new profiles.

**A-HD5:** Agreed

**HD6 — R7 does not need HydroLight.** Elastic `Rrs` is a ratio and is close to
independent of the incident spectrum's *shape*; the Raman and fluorescence terms
depend on Ed through `Ed(λ_ex)/Ed(λ_em)` and the absorbed-photon integral, both
of which we can perturb directly by swapping a TSIS-1-based Ed into our own
forward model. That gives the first-order answer in a day or two of in-house
work and **no compute**; a HydroLight run would add only the second-order
coupling, where the altered in-water field feeds back on the source terms. I
would **move R7 out of the campaign entirely** and make it an analysis task —
which also means DQ5 stops being "unquantified" without waiting months for run
time, rather than staying unquantified *because* it is queued behind seven
other runs.

**A-HD6:** Yes, do so

**HD7 — nothing in the catalogue establishes the truth's own error bar, and it
should be item zero.** We gate models at **0.30 %** (elastic) and **0.34 %**
(inelastic) against HydroLight output whose numerical convergence — quad
resolution, depth-grid resolution, and the iteration count on the inelastic
source — is unstated everywhere in this project. If the solver's own
discretisation error at those settings is, say, 0.2 %, then both gates are
partly measuring HydroLight and we have never known it.

A **convergence sub-run** fixes this for almost nothing: five water bodies, each
at two or three settings of each knob, delivered inside the pilot. I would add
it as **R0** and make it a precondition on signing off the batch. It is the
cheapest item in this document and the only one that tells us what our headline
numbers mean.

**A-HD7:** Ok, make that R0

**HD8 — R5's "adopt §7 verbatim" contradicts the batching.**
`rt_cdom_fluorescence_model.md` §7 asks for the full 3,320-scene ensemble and
"all three zeniths"; the combined-ask table says R5 "should share batch B's
geometry grid", which is six zeniths, off-nadir, on a subset. Both cannot hold.
I would rewrite R5 against batch B's grid and B's scene subset, with the
CDOM-rich tail oversampled *within* that subset, and drop the word "verbatim" —
keeping from §7 only the parts that are genuinely non-negotiable (the paired
on/off design and the recorded Hawes variant).

**A-HD8:** Yes, make that rewrite

---

#### C. What I would cut at a third of the budget

Asked directly, so answered directly. **I would not cut runs; I would cut
axes** — and the two axes I would not touch are the ones no data we hold covers
at all (VSF family, and inelastic-at-off-nadir), because those cannot be
recovered by interpolation afterwards. Scene count can.

**Keep, in this order:**

1. **R0** (new, HD7) — the convergence sub-run, inside the pilot. Near-free, and
   it calibrates every number the project has published.
2. **Batch A as a single run: R1's geometry grid carrying R2's families**
   (HD2), trimmed — ~8 × 6 nodes in `(bb/a, η_bb)` rather than 12 × 8, three VSF
   designs (an FF anchor, a measured/Petzold-like anchor, and the two-component
   mixture swept per HD3), the full geometry grid, full 350–750 nm. Asymptotic
   outputs requested; **`K∞` requested but not gating**, because HD1 gives us a
   second route to µ∞.
3. **A minimal R3**: six solar zeniths × a reduced view set (θ_v ∈ {0, 20, 40,
   60}, Δφ ∈ {0, 45, 90, 135, 180}) × X1/X2/X4, on a **stratified ~300 water
   bodies drawn from batch A's grid nodes** (HD4), with the spectral floor at
   330 nm so that **R6 is folded in at essentially zero cost** — which is
   exactly the document's own argument for R6, and it only holds if specified
   now.
4. **R5 folded in** as a fourth scenario flag on a CDOM-stratified ~100 of those
   same ~300 water bodies. Marginal cost once batch B's setup is paid for.
5. **R4 trimmed** to φ_C ∈ {0.005, 0.02, 0.06} on ~50 scenes. Three points
   measure curvature, and curvature is the entire question; five points and a
   few hundred scenes is a training set, which is not what the prompt for R4
   asks for.

**Cut outright:** **R8** (a declared stretch) and **R7** (moved in-house, HD6).

**Cut within runs, in this order:** the scene axis first — L23's 3,320 scenes
are generated from a handful of parameters and are heavily redundant for
correction-head training, so ~300 stratified is my recommendation rather than
something to price both ways; then the IOP-grid density; then the geometry
density. **Never the family axis or the scenario axis.**

**What that third-budget campaign would *not* buy, stated up front so it is not
discovered later:** a φ_C *training* set (only a linearity test); a CDOM tail
dense enough to *train* δ_C at the extreme (enough to score it, not to fit it);
any statement about vertical structure; and a UV claim below ~330 nm.

**A-HC:**  I don't think there is any reason to cut.  We'll find enough time to do the HydroLight runs.

---

**One more thing, for Robert.** The document's own note — that the hybrid
`Rrs(ZTT) + ΔRrs(emulator)` fails off-nadir *because of the ZTT term* — is worth
putting to him with M5's attribution numbers attached (68 % of the non-physical
predictions charged to `Ψ_KLu`, 1 % to µ∞). His recommendation says "possibly as
the physical backbone", not "necessarily", and R1/R2's data is what a **pure**
emulator of HydroLight needs just as much as a repaired ZTT is. Framing batch A
as "the emulator's training set over the domain we actually use, with the ZTT
refit as a bonus" makes its value independent of whether ZTT survives — which,
after M5, seems like the safer bet to write into a spec. **Do you want me to
draft that note to him alongside the spec?**

**A-RF:**  Yes, draft that note

---

### Setup — round 2

Claude, 2026-09-22, after A1–A11. **§B (HD1–HD8), §C and the Robert note came
back blank**, so the disagreements are still open; they are re-put at the end as
single-line decisions with a default each, so you can answer with a word.

First, what A1–A11 settle, and the two places where your answers change my
advice rather than confirm it.

#### What is now decided, and what I will write into the spec

- **A1 → the catalogue is feasible as written.** Henry Houskeeper on the latest
  HydroLight means full HydroLight (HE60-class), not EcoLight: the full
  azimuthally-resolved radiance distribution is available, so R1–R3's off-nadir
  and azimuth axes are on. It also means the operator is a colleague we can ask
  questions of and iterate with, which the pilot (A8) depends on.
- **A4/A6 → I will specify** netCDF4, one file per `(water body, scenario, θ_s)`
  under `$OS_COLOR/<name>/v1/`, **float64 on all radiometric fields** (PB24's
  float32 `rrs` underflows to exactly 0 at grazing geometries and our metric
  divides by truth), a manifest per batch, and **the standard HydroLight quad
  layout — no custom quad file**. Standard quads are what PB24 used, they reach
  ψ ≈ 44° across the full grid and ψ ≈ 65° inside the 0–70° window, which is
  already below the 110.4° zero-crossing and the 134° fit floor that R1 exists to
  probe. A custom quad file would buy resolution we have no use for and cost
  format-compatibility with PB24.
- **A5/A7 → we author the decks, Henry runs them.** That is the cleanest reading
  of "the operator can drive with any input files we give them", and it collapses
  the metadata contract: items 1, 2, 6, 7, 8 become *ours by construction*
  because we wrote the inputs. What we still need back from Henry is items 3, 4
  and 5 — the Raman redistribution function and the two fluorescence
  quantum-efficiency functions — because those live in HydroLight's own data
  files and source, not in our decks. I will ask for those files verbatim.
- **A9 → the 12 × 8 grid stays dense to the edges.** I retract §C's trim to
  8 × 6; a usable domain needs training density out there, and (see below) we can
  afford it.
- **A10 → switches, not X's**, with the mapping recorded in the manifest.
- **A8 → pilot is in**, and R0 (HD7) goes inside it.

#### The cost model in this document is wrong, and it is wrong in our favour

This is the most important consequence of A3, and it changes what the campaign
can be.

The runs section says: *"multiply by geometries and by scenario (X) for the run
count."* **View geometry does not multiply the run count.** One full-HydroLight
run solves the radiative transfer for one water body under one sky and returns
the *entire* radiance distribution — every view zenith and every azimuth at once.
What multiplies is the solar zenith (one sky per run) and the scenario (the
inelastic switches change what is being solved). So:

> **runs = water bodies × solar zeniths × scenarios**

The document's own numbers confirm it. PB24 is quoted as "5000 realisations ×
1300 geometries ≈ 6.5 M spectra"; 1300 = 10 θ_s × 10 θ_v × 13 Δφ, of which only
the 10 θ_s cost anything. PB24 is **50,000 runs**, not 6.5 million — which is
exactly your 10⁵ order, and it means **our budget is about twice the compute that
produced PB24**.

Pricing the *full* catalogue on that model, at the densities the runs sections
actually ask for:

| | water bodies | θ_s | scenarios | runs |
|---|---|---|---|---|
| **Batch A** (R1 + R2) | 96 IOP nodes × 15 VSF designs = 1,440 | 9 (0–80° @10°) | 1 | **12,960** |
| **Batch B** (R3, R6 folded in) | 1,500 | 6 (0–75° @15°) | 3 | **27,000** |
| **R4** (φ_C) | 50 | 6 | 4 extra φ_C | **1,200** |
| **R5** (CDOM-fl) | 150 | 6 | 1 (X4 partner already in B) | **900** |
| **R8** (vertical, stretch) | 30 profiles | 6 | 3 | **540** |
| **R0** (convergence, in the pilot) | 5 | 2 | 2–3 knobs | **~40** |
| | | | | **≈ 42,600 — 43 % of budget** |

Three things follow:

1. **The off-nadir asks are free.** The single most quoted gap in this project —
   "L23 is nadir-only, PACE is not" — costs nothing at all. Every run we
   commission for any reason returns the full BRDF if we ask for the radiance
   file. That is a genuinely surprising result and it should be stated on the
   first page of the spec.
2. **R1 is not the expensive bet the document treats it as.** Batch A is 13 % of
   budget. "If only one run is possible, it is R1" was framed as a
   scarcity argument; at 10⁵ there is no scarcity at that scale, and the real
   scarce resources are **Henry's setup time, disk, and our own deck-generation
   and ingest effort** — not CPU.
3. **§C is moot as posed** — I no longer need to cut anything, and have spent the
   headroom on the two axes that were previously unaffordable: batch B's water
   bodies go from my proposed ~300 to **1,500**, and **the VSF families are
   crossed into batch B** rather than confined to batch A (HD4). The revised
   answer to "what would you cut at a third of the budget" is now: *nothing —
   a third of this budget is roughly the campaign as catalogued.*

**Caveat, stated because the whole sizing rests on it:** the run-count model
above is inferred from how HydroLight solves the problem and from PB24's shape.
It is the first thing to confirm with Henry, and it is question 1 on the email
checklist below. If it is wrong — if his workflow runs one execution per view
direction — the campaign shrinks by ~130× and §C's cut list comes straight back.

#### A2 is not the blocker it looks like

You cannot get L23's input decks — but **we do not need them.** HydroLight does
not care what concentrations generated a water body; it cares about
`a(λ)`, `b(λ)` and `β̃(ψ)`. And A5 says the operator will take any input files we
give, which means the user-supplied-IOP input mode is open to us.

The L23 netCDFs we already load carry `a`, `bb`, `bbnw`, `bnw`, `aph` and `ag`
per scene per wavelength (`robust/rt/data/l23.py`), and `B_p = bbnw/bnw` fixes the
one Fournier–Forand parameter per wavelength. `robust/rt/data/ed_l23.npz` carries
L23's own `Ed(0+)` at each of the three zeniths, so the sky can be matched and
*checked* rather than guessed. So we can rebuild any L23 water body as a
user-supplied-IOP deck from data already on disk, and R3's "reusing L23's water
bodies makes every existing number directly comparable" survives A2 intact.

Three honest caveats: L23's FF parameter is *inferred* from `B_p` rather than
read from their deck; the sky is matched at the three Ed anchors only; and this
reproduces the **optics**, not Loisel's generative recipe — so "the same water
bodies" means "the same IOPs", which is the only sense that matters for RT but
not the sense a Loisel co-author would use. All three are recordable in the
manifest. **R2Q1 below asks whether to do it.**

#### Round-2 questions

Short, and each carries the default I will adopt if you say nothing.

**R2Q1 (Rebuild L23's water bodies from their IOPs).** Do it, as above, for the
batch-B ensemble — so the new release contains a block that is IOP-identical to
L23 and a block that is newly designed to fill the gaps L23 leaves? *Default:
yes, with ~500 of batch B's 1,500 water bodies drawn from L23 and the rest
designed to cover the IOP simplex L23 under-samples.*

**A-R2Q1:**  Use the Default

**R2Q2 (Does the new release supersede L23?).** Once batch B exists — denser
zeniths, full BRDF, three scenarios, varied VSF, sub-350 nm — it is a strictly
better training and validation set than L23 on every axis. Is the intent that it
becomes the **primary** reference, with L23 demoted to a legacy nadir cross-check
(and the published 0.30 %/0.34 % numbers retained as history rather than
extended)? That is a real commitment — it means retraining rather than patching,
and a new set of headline numbers. *Default: yes, supersede.*

**A-R2Q2:**  Yes, supersede

**R2Q3 (Who writes the decks, and where does that code live?).** A5 makes deck
generation our job: ~40,000 HydroLight input decks plus the IOP and `β̃(ψ)` data
files they reference. That is a real piece of engineering and it belongs in the
repo under test, because the loader must later cross-check every delivered file
against the deck that produced it. *Default: a new `robust/rt/hydrolight/`
module — deck writer, the VSF library, the ensemble designer — with tests, and
the generated decks shipped to Henry as a tarball plus a manifest.* Is that
scope you want in `robust/`, or should it live in `design/py/`?

**A-R2Q3:**  Yes put in `robust/`

**R2Q4 (Disk, and how delivery is staged).** At ~1–5 MB per run for the full
radiance distribution plus depth profiles, 42,600 runs is ~50–200 GB. Batch A
wants `K_d(z)`, `K_u(z)`, `K_Lu(z)`, `µ̄(z)` to ≥ 25 optical depths, which is the
heavy part. Is there room on `$OS_COLOR`, and do you want delivery staged
(pilot → batch A → batch B) or all at the end? *Default: staged, batch A first.*

**A-R2Q4:** Hydrolight will run on Henry's machine.  He will have enough space

#### The §B decisions, restated as one-liners

Each is a yes/no. My default is in italics; blank means I proceed with it.

**HD1 (µ∞ without a run).** *Revised by A3.* The eigenvalue route is still worth
having — it gives µ∞ over R2's VSF families at a density no run buys — but the
argument for doing it **first** was to de-risk the campaign's largest line item,
and at 10⁵ runs batch A is no longer that. *Default: ask Henry whether HE60
reports the asymptotic radiance distribution and `K∞` (free, email question 2
below), commission batch A regardless, and do the eigenvalue work in parallel
rather than as a gate.*

**A-HD1:** It is ok if we have more than 10^5 runs.

**HD2/HD3 (VSF design).** *Default: 15 VSF designs — a two-component small+large
mixture as the primary, swept to give `B_p` and backward shape as independent
knobs, with Fournier–Forand and a Petzold-like measured VSF as anchors at the
`B_p` values where they naturally sit. Matched-`B_p`-across-families only where
the families genuinely overlap, and the spec says so rather than pretending the
comparison is clean everywhere.*

**A-HD2/3:** Yes

**HD4 (one ensemble, not two disjoint batches).** *Now forced by the arithmetic
rather than merely advisable: the VSF families are carried into batch B, so one
block of the release spans IOPs × geometry × VSF family × scenario at once.
Default: adopted.*

**A-HD4:** Yes

**HD5 (L23 is homogeneous, not stratified).** Factual correction to R1's prose;
no decision needed. Will be fixed when the spec is written.

**HD6 (R7 out of the campaign, done in-house).** *Default: yes — a day or two of
analysis against a TSIS-1 Ed, no compute, and DQ5 stops being unquantified
without waiting on run time. Note this is now a preference, not a necessity: at
10⁵ we could afford R7 as runs too.*

**A-HD6:** Yes

**HD7 (R0, the convergence sub-run).** *Default: yes, inside the pilot, and a
precondition on signing off batch A. ~40 runs. It is the only thing that tells us
whether our 0.30 % and 0.34 % gates are measuring our model or HydroLight's
discretisation.*

**A-HD7:** Yes

**HD8 (R5 rewritten against batch B's grid).** *Default: yes; "verbatim" drops,
the paired on/off design and the recorded Hawes variant stay.*

**A-HD8:** Yes

**Robert (A-RF).** *Default: yes, I draft a short note to him alongside the spec
— M5's attribution numbers, and the case for framing batch A as the emulator's
training set with the ZTT refit as a bonus rather than the goal.*

**A-RF (round 2):** Yes

#### What I would ask Henry, in the first email

Offered so you can forward or edit it. Ordered by how much of the spec each
answer moves.

1. **Confirm the run-count model.** Does one run return the full radiance
   distribution over all view quads, so that only solar zenith and the inelastic
   switches multiply the run count? *The entire sizing above depends on this.*
2. **Does his build report the asymptotic radiance distribution and `K∞`**
   for homogeneous, optically deep water — and under which switch? If not, we
   fall back to reading the asymptote off a `≥ 25` optical-depth profile, which
   changes the output request (HD1).
3. **Exact version and build string**, and any local modifications.
4. **The Raman and fluorescence data files as shipped in his build** — the
   Raman scattering coefficient and wavelength-redistribution function, the
   chlorophyll-fluorescence quantum efficiency and emission shape, and
   **specifically which published Hawes variant** the CDOM-fluorescence option
   uses, with its constants. These are metadata items 3/4/5 and they are the one
   part of the contract we cannot satisfy from our own decks.
5. **Will he accept user-supplied IOP files** (tabulated `a(λ)`, `b(λ)` and a
   discretised `β̃(ψ)` per water body) rather than concentration-driven decks,
   and is there a preferred format?
6. **What does a run cost him** in wall-clock for an 81-band elastic run and for
   an inelastic one, and how many can run in parallel? This is the number that
   converts 42,600 runs into a delivery date.
7. **One example deck and its outputs from his working setup**, as the template
   we generate 40,000 of.
8. **Sub-350 nm.** Can the band set start at 330 nm, and does his Raman
   implementation handle excitation at the band-set floor by clipping or by
   extrapolation? (This determines whether R6 is genuinely free for λ ≥ 400 nm
   or perturbs those bands slightly.)
9. **Disk and transfer** — what he can produce and how we take delivery.


---

### Setup — round 3

Claude, 2026-09-22. Every question from both rounds is now answered. You asked
whether your answers are consistent: **mostly yes, with three places where two
answers pull in different directions.** They are resolved below with the reading
I will write into the spec — correct me where I have read you wrong. Then six
short questions, all consequences of one thing: **you removed the budget
ceiling**, and that changes the sizing more than anything else in this Q&A.

#### The three inconsistencies, and how I am reading them

**(i) The budget is no longer 10⁵ runs.** A3 said "assume 10⁵". Three later
answers relax it: A-HD2 "assume that we will do all the runs", A-HC "I don't
think there is any reason to cut — we'll find enough time", and A-HD1 (round 2)
"it is ok if we have more than 10⁵ runs". **Reading: there is no compute ceiling;
size the campaign for the science and let Henry's wall-clock be the constraint.**
A3 is superseded. This is the single largest change in this round and §1 below
re-prices on it.

**(ii) "No reason to cut" versus "take R7 out".** A-HC says cut nothing; A-HD6
(both rounds) says yes, move R7 in-house. **Reading: not a contradiction —
R7 leaves the campaign for *speed*, not cost.** Doing it analytically answers DQ5
in a day or two instead of queueing it behind 150,000 runs, and nothing is lost.
**R8, which my §C had cut as a stretch, is back in.** If you did mean R7 to run
as well, say so; at no ceiling it is affordable, it is simply slower than the
alternative.

**(iii) HD1's sequencing was answered twice, differently.** Round 1 asked "do you
want that week spent first?" → *"Yes, that makes sense."* Round 2's default was
the opposite — don't gate on it, commission batch A regardless, do the eigenvalue
work in parallel — and the answer there addressed the budget instead
(*"it is ok if we have more than 10⁵ runs"*), so the sequencing question is
strictly unanswered. **Reading: do the eigenvalue work, but not as a gate.** My
round-1 "first" was justified by scarcity — de-risk the campaign's largest line
item before signing it — and you have now removed the scarcity twice over. Batch
A is 8 % of the campaign below. Asking Henry whether his build reports `K∞` is
free and takes one email; the eigenvalue solve proceeds in parallel and is
valuable regardless, because it gives µ∞ over R2's VSF families at a density no
run buys. **If you specifically want the week spent before anything is
commissioned, say so and I will gate it.**

A fourth, milder one: **A-R2Q1 approved the default of ~500 L23-reconstructed
water bodies, but A-R2Q2 says the new release supersedes L23.** Those sit awkwardly
together — a release that supersedes L23 should not contain only 15 % of it.
R3Q1 below proposes taking all 3,320.

#### 1. What the campaign becomes with no ceiling

Same run-count model as round 2 (**runs = water bodies × solar zeniths ×
scenarios**; view geometry is free), with the three axes that genuinely benefit
from more compute pushed and the rest left where they were:

| | water bodies | θ_s | scenarios | runs |
|---|---|---|---|---|
| **Batch A** (R1 + R2) | 96 IOP nodes × 15 VSF designs = 1,440 | 9 | 1 | 12,960 |
| **Batch B** (R3, R6 folded in) | 3,320 L23-reconstructed + 1,500 designed = 4,820 | 9 | 3 | **130,140** |
| **R4** (φ_C) | 100 | 9 | 4 extra φ_C | 3,600 |
| **R5** (CDOM-fl) | 500, CDOM-stratified | 9 | 1 | 4,500 |
| **R8** (vertical, back in) | 60 profiles | 9 | 3 | 1,620 |
| **R0** (convergence, in the pilot) | 5 | 2 | knob sweep | ~40 |
| | | | | **≈ 152,900 — about 3× PB24** |

At 1 / 3 / 5 minutes per run on 16 cores that is **6.6 / 20 / 33 days** of
wall-clock, and **300–760 GB** delivered at 2–5 MB per run. Henry's throughput and
our disk are now the only real constraints, which is why email questions 6 and 9
matter more than they did.

**But more budget does not mean more of everything, and I want to be explicit
about what I did *not* push**, so the spec does not read as padding:

- **R5 stays a 500-body CDOM-stratified subset**, not all of batch B. CDOM
  fluorescence matters in CDOM-rich water; 500 bodies with the tail oversampled
  is a *better* experiment than 4,820 uniform ones, not a cheaper one.
- **R4 stays ~100 bodies.** It measures curvature in φ_C. More scenes do not
  measure curvature better; more φ_C values would, and it already has five.
- **The IOP grid stays 12 × 8.** It is dense enough for a usable domain (A9) and
  denser nodes buy interpolation accuracy in a space we will fit a smooth model
  over anyway.
- **R8 stays small.** It bounds a bias; it is not a training set.

#### 2. Decisions carried into the spec from rounds 1–2

For the record, so the spec has one place to inherit from: full HydroLight on
Henry's latest build (A1); we author the decks, Henry runs them (A5/A7), with
metadata items 1/2/6/7/8 ours by construction and 3/4/5 requested from his build;
netCDF4 per `(water body, scenario, θ_s)` under a versioned `$OS_COLOR`
directory, float64 radiometry, manifest per batch (A4); standard quad layout, no
custom quad file (A6); physics switches named directly with the X-mapping
recorded (A10); 12 × 8 IOP grid dense to the edges (A9); L23 water bodies rebuilt
from their own IOPs (A-R2Q1); the new release supersedes L23 (A-R2Q2); deck
generator in `robust/rt/hydrolight/` under test (A-R2Q3); pilot first with R0
inside it (A8, A-HD7); 15 VSF designs with a two-component mixture primary
(A-HD2/3); one ensemble, families crossed into batch B (A-HD4); R5 rewritten
against batch B's grid (A-HD8); R7 analytic and in-house (A-HD6); a note to
Robert drafted alongside the spec (A-RF).

#### 3. Round-3 questions

**R3Q1 (All of L23, or 500 of it?).** A-R2Q1 took the default (~500
reconstructed); A-R2Q2 says the release supersedes L23. Those pull apart. *I
recommend all **3,320** L23 water bodies plus ~1,500 designed ones* — it makes
the new release a strict superset of L23 on every axis, which is what
"supersede" has to mean if the old numbers are to be retired rather than merely
set aside. Cost: +~76,000 runs, which is the single largest line item in the
table above and the one place the removed ceiling is actually being spent.

**A-R3Q1:**

**R3Q2 (One zenith grid for everything).** R1 asked 0–80°, R3 asked 0–75° @ 15°.
*I recommend unifying both on **0–80° in 10° steps (9 values)*** — batch A and
batch B then share one geometry design, holding out an intermediate zenith tests
interpolation properly, and the 80° shell stays as a deliberate extrapolation
test (PB24 sanctions 0–70° and holds its 80° shell out the same way).

**A-R3Q2:**

**R3Q3 (Our disk, and what depth output goes where).** A-R2Q4 answers Henry's
disk; ours is still open. We need **300–800 GB** on `$OS_COLOR` to receive this.
*I recommend requesting the full depth profiles (`K_d(z)`, `K_u(z)`, `K_Lu(z)`,
`µ̄(z)` to ≥ 25 optical depths) for **batch A only**, where the asymptote is the
whole point, and surface quantities plus the full radiance distribution for batch
B* — that is most of the difference between 300 GB and 800 GB. Is there room, and
is that split right?

**A-R3Q3:**

**R3Q4 (Staged delivery, so work can start before the campaign finishes).** At
~150,000 runs the delivery is weeks, and the ingest and retraining work does not
need to wait. *I recommend the spec define five deliverables in order — pilot
(incl. R0) → batch A → batch B's L23 block → batch B's designed block → R4/R5/R8
— each self-contained with its own manifest, so the loader and the first
retraining start on the L23 block while the rest runs.*

**A-R3Q4:**

**R3Q5 (R7 and R8 — confirming my reading of (ii)).** R7 leaves the campaign and
is done analytically in-house; R8 is back in at 60 profiles. *Confirm, or tell me
to run R7 as well.*

**A-R3Q5:**

**R3Q6 (HD1 sequencing — confirming my reading of (iii)).** Eigenvalue route runs
in parallel, not as a gate on commissioning. *Confirm, or tell me to gate it.*

**A-R3Q6:**

#### 4. What happens next, on your word

Nothing in this round blocks the spec. If R3Q1–R3Q6 come back as recommended —
or come back blank — I will write `design/hydrolight_runs.md` as **one
self-contained spec per batch** (pilot+R0, A, B, and the B-dependent runs
R4/R5/R8), each with the metadata contract attached, the deck-generation
interface named, and the Henry email checklist folded in as the batch-zero
question list; plus the note to Robert (A-RF).


### Decks

*(For Spec prompt 2. Empty until the deck-generator work raises something
`design/hydrolight_runs.md` does not settle.)*

## Logging

Record work in the Logs section below, format:

### \<Date\> (Short summary of the work)

\<Detailed description of the work and what you learned\>

## Logs

### 2026-09-22 (Setup prompt 1 — read the campaign doc and its six Context items; posed Q&A/Setup round 1)

Executed Setup prompt 1 (model: **Opus 5**, not Fable — the prompt says "use
Fable if you can" and the session model here is Opus; recorded rather than
silently substituted, since every prior log in this effort names Fable).
Nothing commissioned, no code changed, no files touched but this one.

**Read.** This document in full, then the six Context items in the order given:
`design/m5_report.md` (all of it — the negative result, the oracle column, §3's
four options and the two contaminations), `design/prototype_summary.md` (the
envelope and its measured coverage: 25 % of L23 inside both conditions, 3.9 %
of PB24's sanctioned window, backbone physical on 100 % of L23 against 75 % of
PB24), `design/rt_inelastic_model.md` §8 (the six-item wishlist this doc
supersedes) and its §9 risks, `design/rt_cdom_fluorescence_model.md` §7–§9,
`reports/report_rt_inelastic_model.md` §5 and §7, and Robert's comment in
`rt_elastic_prompts.md`. Also read both loaders' module docstrings, which is
where the measured dataset properties live, and confirmed `$OS_COLOR` resolves
(`Loisel2023/`, `SD/` both present).

**Wrote.** Q&A/Setup round 1 — eleven commissioning questions (HQ1–HQ11), eight
disagreements with the catalogue (HD1–HD8), and a concrete third-budget
campaign (§C), plus one question about what to put to Robert.

**What I learned, and the four things that changed my view of the catalogue:**

1. **`K∞`/µ∞ may not need a run at all (HD1).** The document's own reason for
   excluding PB24 — `K∞` is θ_s-independent *by definition* — is the same fact
   that makes µ∞ an eigenvalue problem in `(a, b, β̃)` alone. If that route
   works it removes the campaign's load-bearing assumption ("we believe
   HydroLight reports `K∞`") from the critical path, and it gives µ∞ over R2's
   VSF families at a density no run would buy. Worth a week before signing the
   largest line item.
2. **Batches A and B as specified re-create the hole the document opens by
   describing (HD4).** A is an IOP grid, B is the L23 ensemble; they share no
   water bodies, so the campaign would again produce two datasets with disjoint
   axes and *still* no sample with an inelastic process under a varied VSF.
   Drawing B's water bodies from A's grid nodes fixes it for almost nothing.
   This is the finding I am most confident about.
3. **Nobody has ever stated HydroLight's own numerical error here (HD7).** We
   gate at 0.30 % and 0.34 % against output whose quad resolution, depth grid
   and inelastic-source iteration count are unrecorded in every document in this
   project. A five-water-body convergence sub-run inside the pilot is the
   cheapest item in the catalogue and calibrates every number already published.
   Added as R0.
4. **R7 is not a HydroLight question (HD6).** Elastic `Rrs` is a ratio and
   nearly Ed-shape-independent; the inelastic terms depend on Ed through
   `Ed(λ_ex)/Ed(λ_em)`, which we can perturb in our own forward model. DQ5 can
   stop being "unquantified" in a day or two without waiting on run time.

Two smaller ones: R1 calls L23's scenes "stratified" where
`rt_inelastic_model.md` §8 and R8 both say L23 is vertically **homogeneous**
(HD5 — the slip is in our favour, it makes HD4's merge free); and R5's
instruction to adopt `rt_cdom_fluorescence_model.md` §7 "verbatim" contradicts
the combined-ask table's instruction to share batch B's geometry grid (HD8).

On priorities I disagree with the document's tiebreak. R2 is R1 with a family
axis on the same setup, so "if only one run, R1" understates it: the VSF-family
axis is the only axis **no dataset in existence for us varies at all**, and it
is the one thing that cannot be recovered by interpolation later. My third-budget
campaign therefore cuts *axes*, not runs — scene count first (3320 → ~300),
then IOP-grid density, then geometry density — and never the family or scenario
axes. R8 and R7 come out; R6 folds into batch B as a spectral floor at 330 nm;
R5 folds in as a scenario flag; R4 shrinks to three φ_C values on ~50 scenes,
because measuring curvature is the whole question.

Also proposed two process items not in the document: ship the **HydroLight input
decks** rather than prose for metadata items 1/2/6/7/8, so the loader can test
them (HQ7); and a **~10-run pilot in final format** as a gate before the full
batch (HQ8).

Next: JXP answers Q&A/Setup, then Setup prompt 2 (react, ask a second round if
needed) before any spec is written.

### 2026-09-22 (Setup prompt 2 — reacted to A1–A11; corrected the cost model; posed Q&A/Setup round 2)

Executed Setup prompt 2 (model: **Opus 5**, not Fable — same note as the
previous entry). No code changed, no spec written; only this document edited.

**State of the Q&A.** A1–A11 (§A) answered; **§B (HD1–HD8), §C and the Robert
note came back blank.** Round 2 therefore does two things: acts on §A, and
re-puts every §B/§C item as a one-line decision with an explicit default, so
silence resolves to a stated position rather than to nothing.

**The finding that matters: the cost model in this document is wrong, and wrong
in our favour.** *The runs* section says "multiply by geometries and by scenario
(X) for the run count". **View geometry does not multiply the run count.** One
full-HydroLight run solves one water body under one sky and returns the entire
radiance distribution — every view zenith and azimuth at once. Only solar zenith
and the inelastic switches multiply:

> runs = water bodies × solar zeniths × scenarios

The document's own figures confirm it: PB24 is quoted as "5000 realisations ×
1300 geometries ≈ 6.5 M spectra", and 1300 = 10 θ_s × 10 θ_v × 13 Δφ of which
only the 10 θ_s cost anything — so PB24 is **50,000 runs**, exactly A3's 10⁵
order. Priced on that model the *full* catalogue at the densities the runs
sections ask for is **≈ 42,600 runs, 43 % of budget** (batch A 12,960; batch B
27,000 at 1,500 water bodies; R4 1,200; R5 900; R8 540; R0 ~40).

Three consequences, all of which change the shape of the campaign:

1. **The off-nadir axis is free.** The most-quoted gap in the project — "L23 is
   nadir-only, PACE is not" — costs nothing; any run returns the full BRDF if the
   radiance file is requested. This belongs on page one of the spec.
2. **R1 is not the expensive bet the document treats it as** (13 % of budget), so
   "if only one run is possible, it is R1" is a scarcity argument against a
   scarcity that does not exist at 10⁵. The scarce resources are Henry's setup
   time, disk, and our own deck-generation and ingest effort.
3. **§C is moot as posed.** I no longer cut anything; the headroom went to the
   two axes that were previously unaffordable — batch B's water bodies from my
   proposed ~300 to 1,500, and the VSF families crossed into batch B (HD4).

Flagged in the Q&A rather than asserted: the run-count model is inferred from how
HydroLight solves the problem plus PB24's shape, and it is email question 1 for
Henry. If his workflow is one execution per view direction the campaign shrinks
~130× and §C comes straight back.

**The second finding: A2 is not a blocker.** JXP cannot get L23's HydroLight
input decks — but HydroLight does not need them. It needs `a(λ)`, `b(λ)` and
`β̃(ψ)`, and A5 grants user-supplied-IOP input. The L23 netCDFs we already load
carry `a`, `bb`, `bbnw`, `bnw`, `aph`, `ag` per scene per wavelength, `B_p =
bbnw/bnw` fixes the one Fournier–Forand parameter, and `robust/rt/data/ed_l23.npz`
carries L23's own `Ed(0+)` at all three zeniths so the sky can be matched *and
checked*. So R3's "reusing L23's water bodies" survives A2 — with three recorded
caveats (FF parameter inferred from `B_p` rather than read from their deck; sky
matched at three Ed anchors only; "same water bodies" means same **optics**, not
Loisel's generative recipe).

**Decisions taken from §A and written into the round-2 text:** full HydroLight,
not EcoLight (A1) so R1–R3 are feasible; netCDF4 per `(water body, scenario,
θ_s)` under a versioned `$OS_COLOR` directory, **float64** radiometry (PB24's
float32 `rrs` underflows to 0 at grazing geometries and our metric divides by
truth), manifest per batch (A4); **standard quad layout, no custom quad file** —
standard quads already reach ψ ≈ 44° full-grid and ψ ≈ 65° in the 0–70° window,
below both the 110.4° crossing and the 134° fit floor, and they keep
format-compatibility with PB24 (A6); **we author the decks, Henry runs them**
(A5), which collapses metadata items 1/2/6/7/8 to ours-by-construction and leaves
only 3/4/5 to request from his build; the 12 × 8 IOP grid stays dense to the
edges (A9 — §C's trim to 8 × 6 retracted); switches not X-labels (A10); pilot in,
with R0 inside it (A8).

**Revised my own advice in one place, and said so.** HD1 argued for spending a
week on the asymptotic-RTE eigenvalue route **before** commissioning, to de-risk
the campaign's largest line item. A3 removes that argument — batch A is 13 % of
budget, not a bet — so the default is now: ask Henry whether his build reports
`K∞` (free), commission batch A regardless, and do the eigenvalue work in
parallel rather than as a gate. The route is still worth having, for µ∞ over
R2's VSF families at a density no run buys.

**Also added:** four new questions (rebuild L23 water bodies from their IOPs;
does the new release supersede L23 as the primary reference; where the deck
generator lives — proposed `robust/rt/hydrolight/`, under test, since the loader
must later cross-check deliveries against the decks that produced them; disk and
staged delivery), and a **nine-item first-email checklist for Henry**, ordered by
how much of the spec each answer moves — run-count model, asymptotic output,
version/build, the Raman and fluorescence data files (metadata items 3/4/5, the
one part we cannot satisfy from our own decks, including which Hawes variant),
user-supplied-IOP acceptance, per-run wall clock, one example deck as template,
the 330 nm band floor and how his Raman handles excitation at it, and delivery.

Next: JXP answers round 2 (§B defaults included), then the Spec prompt —
`design/hydrolight_runs.md`, one self-contained spec per batch.

### 2026-09-22 (Setup prompt 3 — reconciled both rounds of answers; re-priced with the budget ceiling removed; posed round 3)

Executed Setup prompt 3 (model: **Opus 5**, not Fable — same note as the two
previous entries). No code changed, no spec written; only this document edited.

**State.** Every question from rounds 1 and 2 is now answered — JXP went back and
filled the §B/§C gaps left blank last time. He asked whether his answers are
consistent. They mostly are; **three pairs pull in different directions**, and
round 3 resolves each with an explicit reading rather than re-asking blind.

1. **The budget ceiling is gone.** A3 said "assume 10⁵ runs"; A-HD2 ("assume that
   we will do all the runs"), A-HC ("no reason to cut — we'll find enough time")
   and A-HD1 round 2 ("ok if we have more than 10⁵") all relax it. Read as: no
   compute ceiling, size for the science, Henry's wall-clock is the constraint.
   **A3 is superseded**, and this is the largest change in the round.
2. **"No reason to cut" vs "take R7 out".** Not a contradiction — R7 leaves the
   campaign for *speed*, not cost (analytic answer in a day or two versus queued
   behind 150,000 runs). **R8, which my §C had cut, is back in.**
3. **HD1 was answered twice, differently.** Round 1: "yes, spend the week first".
   Round 2's default was the opposite (don't gate), and the answer there addressed
   the budget instead — so sequencing is strictly unanswered. Read as: do the
   eigenvalue work, in parallel, not as a gate, because my round-1 "first" was a
   scarcity argument and the scarcity has now been removed twice.
4. Milder: **A-R2Q1 approved ~500 L23-reconstructed water bodies while A-R2Q2
   says the release supersedes L23.** A release that supersedes L23 should not
   contain 15 % of it. R3Q1 proposes all 3,320.

**Re-priced with no ceiling**, same run-count model (runs = water bodies × solar
zeniths × scenarios; view geometry free): batch A 12,960; **batch B 130,140**
(4,820 water bodies = 3,320 L23-reconstructed + 1,500 designed, 9 zeniths, 3
scenarios); R4 3,600; R5 4,500; R8 1,620; R0 ~40 — **≈ 152,900 runs, about 3×
PB24**. That is 6.6 / 20 / 33 days of wall-clock on 16 cores at 1 / 3 / 5 min per
run, and **300–800 GB** delivered. Batch A is now 8 % of the campaign, which is
what retires HD1's sequencing argument.

**Deliberately did not spend the headroom everywhere**, and said so in the Q&A so
the spec does not read as padding: R5 stays a 500-body CDOM-stratified subset
(better experiment than 4,820 uniform, not a cheaper one); R4 stays ~100 bodies
because it measures curvature in φ_C and more scenes do not measure curvature
better; the IOP grid stays 12 × 8; R8 stays small because it bounds a bias rather
than training anything. The three axes that genuinely benefit — L23 completeness,
zenith density, and crossing the VSF families into batch B — are where it went.

**Round-3 questions (six, all short):** all 3,320 L23 water bodies rather than
500 (recommended, and the single place the removed ceiling is actually spent);
unify batch A and B on one zenith grid, 0–80° @ 10° (9 values), holding the 80°
shell as a deliberate extrapolation test the way PB24 does; our own disk on
`$OS_COLOR` — A-R2Q4 answered Henry's, not ours — with a proposed split that
requests ≥ 25-optical-depth profiles for **batch A only** (most of the difference
between 300 and 800 GB); **staged delivery** in five self-contained pieces so the
loader and the first retraining start on batch B's L23 block while the rest runs;
and two confirmations of the readings above (R7/R8, and HD1 sequencing).

**Also consolidated** every decision from rounds 1–2 into one paragraph the spec
can inherit from, so the Spec prompt does not have to re-read three rounds of
Q&A to find them.

Next: the Spec prompt — `design/hydrolight_runs.md`, one self-contained spec per
batch (pilot+R0, A, B, and the B-dependent R4/R5/R8), metadata contract attached
to each, the deck-generation interface named, Henry's question list folded in as
batch zero, plus the note to Robert (A-RF). Round 3 does not block it: if the six
questions come back as recommended, or blank, the spec is written on the defaults.

### 2026-09-22 (Spec prompt 1 — wrote `design/hydrolight_runs.md`, the commissioning specification)

Executed the Spec prompt (model: **Opus 5**, not Fable — same note as the three
previous entries). Round 3's six questions came back blank, which round 3 §4 had
committed to mean "write the spec on the recommended defaults"; that is what was
done. **Wrote `design/hydrolight_runs.md`** (887 lines). No package code changed.

**Shape.** Four self-contained batch specs — batch 0 (pilot + R0, 172 runs),
batch A (R1+R2, 16,659), batch B (R3+R6, 130,140), batch C (R4+R5+R8, 9,720) —
**≈156,700 runs total, 3.1× the compute that produced PB24**, with the metadata
contract attached to each batch and stated per-item as *ours by construction*
(because we author the decks) versus *the operator's*. Plus shared conventions,
the run-count model on page one, a five-stage delivery plan, a ten-question list
for Henry, a departures-from-the-catalogue table, and three appendices — the IOP
grid derivation, the VSF design set, and the drafted note to Robert (A-RF).

**Four things were computed rather than asserted, and two of them changed the
spec.**

1. **33 of R1's 96 IOP-grid nodes are not physical water.** Since `a ≥ a_w`, the
   design coordinates obey `η_bb · (bb/a) ≤ bb_w(λ)/a_w(λ)` — a ceiling that is a
   property of pure water alone. Computed from `conventions.BB_W_L23` and
   `ocpy.water.absorption.a_water`, it is 0.498 at 400 nm, 0.346 at 440 and
   **0.0151 at 550**, so the choice of reference wavelength matters enormously:
   91/96 nodes realizable at 400 nm against 67/96 at 550. Adopted λ_ref = 400 nm;
   adding physical caps (`a_nw ≤ 20`, `b_p ≤ 100` m⁻¹) leaves **63 of 96 nodes**,
   independent of `B_p`. The catalogue's "~12 × 8 nodes" was not achievable as
   written. What rescues it: because `a_w` and `bb_w` vary strongly across
   330–750 nm, each water body traces a *curve* through `(bb/a, η_bb)`, so the 63
   nodes still occupy **86 of the 96 target cells (90 %)**. The ten empty cells
   are named rather than hidden.
2. **Fournier–Forand is a two-parameter family, and I had said otherwise.** HD3
   claimed FF is "a one-parameter family in which `B_p` and the forward slope are
   coupled". FF is derived from `(n, µ)`, and distinct pairs give the same `B_p`
   with different shapes — computed three branches at each of five target `B_p`,
   e.g. `B_p = 0.012` at (1.055, 3.68), (1.095, 3.47) and (1.190, 3.24). This is
   better than round 2's Mie-mixture plan: no new scattering code, and HydroLight
   has FF built in.
3. **But the backward hemisphere barely moves at matched `B_p` — and this
   revises R2's stated purpose.** Evaluating `β̃(ψ)` for those three branches:
   **49 % spread at ψ = 1°, 35 % at 10°, but only 1–3 % beyond 120°.** A
   two-component mixture does better and not much — 5.2 % at 180°, 3.1 % at 135°,
   against 26 % at the forward peak. So at matched bulk `B_p`, nature leaves very
   little freedom in the backscatter hemisphere. R2 will calibrate the
   **forward**-shape axis strongly and the backward axis weakly; "calibrating the
   backward-VSF axis" is not what these runs can deliver at matched `B_p`, and
   the spec says so in §11 and Appendix B rather than discovering it later. The
   design set grew to **27 VSFs** (15 FF branches + 10 mixtures + 2 out-of-family
   anchors) and the honest lever on the backward axis is the out-of-family
   anchors and the designs that deliberately break `B_p` matching.
4. **The campaign re-priced to 156,691 runs** on the round-2 run-count model, up
   slightly from round 3's 152,900 because the IOP grid lost unrealizable nodes
   but gained a 150-body fill set and the VSF axis grew from 16 to 27 designs.
   Batch B is 83 % of it. Wall-clock 6.8 / 20 / 34 days on 16 cores at 1 / 3 / 5
   min per run; 310–780 GB in ~17,400 files.

**The pilot's centrepiece is the L23 reproduction test**, which fell out of the
round-2 finding that A2 is not a blocker. Ten L23 water bodies rebuilt from their
own published IOPs, run at 0/30/60° under S1/S2/S4, compared against the L23
`Rrs` we already load. If they match, **every convention is validated at once** —
sky model, wind, surface, band structure, Raman and fluorescence constants, and
the output convention (metadata item 8, which is thereby *verified* rather than
trusted). Gate: median `|ΔRrs|/Rrs` ≤ 2 % over 400–700 nm at nadir, and the
per-process differences agreeing in sign at every band and within 10 %. The 2 %
allows for the two known approximations — L23's FF parameter inferred from `B_p`,
and the sky matched at three `Ed` anchors only. A failure is diagnostic: it
localises which convention is wrong while it still costs 90 runs to re-run. And
the same check then runs on all 3,320 scenes when batch B lands, for free.

**One deliberate change to an approved decision, recorded rather than made
silently:** round 2 said one netCDF per `(water body, scenario, θ_s)`; the spec
groups the nine solar zeniths inside each file instead, cutting the file count
from ~157,000 to ~17,400 and storing each water body's IOPs once rather than nine
times. Made under A4 ("request what you prefer"), and flagged in §2.1.

**Also settled in the spec:** 330–750 nm at 5 nm, whose 350–750 subset is
bit-compatible with `conventions.WAVE` — which is what makes the L23 comparison
meaningful *and* folds R6 in for free; nine solar zeniths 0–80° @ 10° shared by
every batch, with 80° held out as an extrapolation shell the way PB24 holds
80/87.75°; scenario names S1/S2/S4/S5 with the X-mapping in the manifest; depth
profiles to ≥ 25 optical depths for **batch A only**, which is most of the
difference between 310 GB and 780 GB; and `robust/rt/hydrolight/` as the deck
generator plus `robust/rt/data/hh26.py` as the loader, with a held-out-**VSF
design** split that is a far stronger claim than M5's held-out-`B_p` split.

**The note to Robert (Appendix C)** puts three things to him: that the hybrid he
proposed fails off-nadir because of the ZTT term (68 % of non-physical
predictions charged to `Ψ_KLu` against 1 % to µ∞, and the oracle within 3 % of
the trained hybrid — so the form is the limitation, not the network); that batch
A is deliberately specified so its value does not depend on ZTT surviving; and
the measured caution that phase-function freedom at matched `B_p` appears to be
largely a forward-scattering freedom, with a direct ask for measured VSF sets
that break that.

Next: the Ingest prompt, once batch 0 arrives — `robust/rt/data/hh26.py`,
following `l23.py` and `pb24.py`. Before that, the two in-house tasks the spec
carves out: R7 analytically through the `Geometry.Ed` seam (§8.3) and the `µ∞`
eigenvalue solve (§8.4), neither of which waits on HydroLight.

### 2026-09-23 (Setup prompt 4 — wrote Spec prompt 2, the deck-generator prompt)

Executed Setup prompt 4 (model: **Opus 5**, not Fable — same note as the four
previous entries). Deliverable is a prompt, not code: **Spec prompt 2** now sits
in the Spec section, and an empty **Q&A/Decks** heading was added for it to post
questions into. No package code changed, no decks generated.

**The problem the prompt has to solve.** Henry needs input files, but we do not
yet know HydroLight 6's deck syntax — that is spec §10 Q7, "one example deck and
its outputs from the working setup", and it will not be answered until he
replies. A prompt that says "write the decks" would stall on it. So the prompt
**mandates a two-phase split**:

- a **format-neutral run IR** — one record per water body with `a(λ)`, `b(λ)`,
  `bb(λ)`, the tabulated `β̃(ψ, λ)`, depth/bottom, sky and wind, scenario
  switches and the solar-zenith list, on the 85-band grid — serialised as JSON +
  NPZ. All the physics lives here and none of it depends on deck syntax;
- a **thin format adapter** written last and replaceable in an afternoon once the
  template arrives.

The payoff: if the adapter cannot be finished, the IR plus the `β̃(ψ)` and IOP
tables are still shippable, because they are the physics and they are
format-light. That is the whole reason the prompt is shaped this way rather than
as a single "generate the decks" instruction.

**Structure.** Module layout under `robust/rt/hydrolight/` (A-R2Q3) — `grid.py`,
`vsf.py`, `ensemble.py`, `l23_recon.py`, `deck.py`, `manifest.py` — then five
milestones: M0 the VSF library and IR, M1 the grid and ensembles, M2 batch 0's
21 water bodies chosen reproducibly from a seed rather than hand-picked, M3 an
in-house pre-check, M4 the batch-0 tarball for Henry with a README stating the
conventions in the operator's terms and carrying the ten questions of spec §10.

**M3 is the part I would not have thought to ask for a week ago.** Before Henry
runs anything, push the reconstructed L23 IOPs through our *existing*
`robust.rt.forward` and compare against L23's published `Rrs`. That validates the
**reconstruction pipeline**, not HydroLight, and it costs no compute — so a bug
in `l23_recon.py` surfaces before it consumes the pilot's 90 runs. It is the
cheapest gate in the chain and it sits upstream of the cheapest gate in the spec.

**Tests demanded**, in the style of `test_l23.py`/`test_pb24.py` and pinning
measured properties rather than documentation: the realizability mask and its
count (63 of 96 at λ_ref = 400 nm, 90 % box occupancy, the ten empty cells
enumerated); each VSF design's `B_p` to < 4e-4; the Appendix B matched-`B_p`
shape table **regenerated as a test rather than quoted from the spec** — so if
the FF algebra is wrong, the spec's central claim about the backward hemisphere
fails loudly; the L23 round-trip; manifest completeness against the ten metadata
items with the operator-supplied 1/3/4/5 explicitly marked absent until batch 0
returns; and **byte-identical regeneration**, without which the manifest hashes
are decorative.

**Scoped deliberately narrow:** do *not* generate ~157,000 decks in this pass.
Batch 0 is 21 water bodies and 172 runs; build the machinery so the rest is a
parameter change, and generate the rest only once batch 0 has passed. That
matches the spec's own staging (§9) and keeps the first delivery small enough to
iterate on with Henry.

Next: Spec prompt 2 itself, then the Ingest prompt when batch 0 returns. The two
in-house tasks the spec carves out (R7 through the `Geometry.Ed` seam, §8.3; the
`µ∞` eigenvalue solve, §8.4) remain unblocked and independent of all of this.

### 2026-09-23 (Spec prompt 2 — built the deck generator; batch 0's 172 runs generated; three measured findings revised the spec)

Executed Spec prompt 2 (model: **Opus 5**, not Fable — same note as previous
entries). All five milestones delivered. **712 tests pass** (35 new), `ruff
check` and `ruff format --check` clean. Nothing asked in Q&A/Decks: the spec
settled everything the build needed.

**What landed.** `robust/rt/hydrolight/` — `vsf.py` (the design set and its
deliverability criterion), `grid.py` (the IOP grid, realizability mask, fill
set), `l23_recon.py` (L23's water bodies rebuilt from their own IOPs),
`ensemble.py` (batch B's design block, batch C's three subsets), `deck.py` (the
format-neutral run IR + a deterministic serialiser + the adapter seam),
`manifest.py`. Two scripts: `design/py/make_hydrolight_batch0.py` and
`design/py/hydrolight_precheck.py`. Tests in `robust/tests/test_hydrolight.py`.

**Batch 0 generates to exactly 172 runs** — P1 10 bodies/90 runs, P2 5/70, P3
6/12 — 21 water bodies, 161 files, 880 kB, written to `build/hydrolight/batch0`
and **byte-identical on regeneration** (verified by `diff -r` of two independent
runs). `np.savez` had to be replaced: its zip entries carry wall-clock times, so
two runs of the same generator produce different bytes and every manifest hash
becomes decorative.

**Three findings from building it, two of which the spec had wrong.**

1. **Half of Fournier–Forand's parameter space cannot be shipped as a table.**
   The first `designs()` produced tables whose quadrature `B_p` was out by a
   factor of three. Cause: for `µ` near 3 the forward peak behaves like
   `ψ^−1.93` — the integral converges, but so slowly that at `ψ_min = 1e-5°` the
   table still holds only **64 %** of the scattering. Requiring the delivered
   table to carry the `B_p` it claims (normalisation ≥ 0.995 *and*
   `|B_p(quadrature) − B_p(closed form)| < 4e-4`) forces **`µ ≥ 3.52`**, and the
   consequence is an **uneven** branch count: **one** deliverable design at
   `B_p = 0.004`, six at 0.030. The spec's "5 targets × 3 branches = 15" was not
   achievable; the real set is 17 FF + 9 mixtures + 2 named built-ins = 28
   designs, 26 tabulated.
2. **The spec's central claim about the backscatter hemisphere was wrong by
   about a factor of five, and this is exactly what the prompt's "regenerate the
   shape table as a test rather than quoting it" was written to catch.** Appendix
   B reported 1–3 % backward contrast at matched `B_p` — computed from the
   `(n, µ)` pairs that `deliverable()` later rejected. From the designs we can
   actually ship it is **12–16 % at `β̃(180°)`** and 10 % at 135°, against 31–54 %
   at 1°. So the backward axis is a real measurable effect rather than a rounding
   error, and `beta_tilde_pi`/`backward_slope` *can* be calibrated — the forward
   axis is still two to four times stronger. Appendix B, §11's "what it will not
   settle", the R2 row of the departures table and the note to Robert were all
   rewritten; the note now asks him about the `µ ≥ 3.52` constraint directly,
   which is a better question than the one it replaced. A number quoted into a
   document cannot fail; a test can.
   There is also a shape in the measurement worth keeping: the spread is
   **smallest near 120°** and rises toward both 90° and 180°, because matched
   `B_p` constrains the *integral* over the backward hemisphere rather than its
   shape. The backward-shape information is concentrated at 160–180° and near
   90°, which is where a fit should be weighted.
3. **Reconstructing L23's total absorption from components corrupted it by up to
   150 % in the red.** `a_nap = clip(a − a_w − a_ph − a_g, 0)` goes negative
   wherever our pure-water table differs from L23's, and the clip then leaks into
   `a` — the one quantity radiative transfer actually uses. Fixed by carrying
   `a(λ)` **verbatim** at and above 350 nm and letting the components do the
   extending only below it, where L23 is silent. The most negative residual is
   now recorded per scene in the provenance rather than absorbed. Caught by M3,
   before any HydroLight run — which is what M3 is for.

**M3, the in-house pre-check, on the committed fixtures:** IOPs round-trip to
**1.4e-7** (the fixtures' float32 precision; `a` and `b_p` are exact),
`forward(reconstructed)` agrees with `forward(loader IOPs)` to **4.6e-7**, and
the rRMS against L23's published X=1 `Rrs` is **0.2576 %** by either route —
identical, which is the point. The reconstruction preserves the water.

**Spec updated** rather than left stale: totals 156,691 → **156,124** (batch A
1,788 bodies / 16,092 runs, since the grid crosses the 26 *tabulated* designs —
the two built-ins declare no `B_p`, so a node's realizability cannot be
evaluated for them); box occupancy 90 % → **92 %** measured, with eight empty
cells rather than ten; Appendix A and B now point at the code that reproduces
them; §8.1 records that the generator exists and why the adapter does not.

**The adapter is deliberately unwritten.** `deck.render_deck` raises with a
message naming spec §10 Q7. A deck that is *nearly* right is worse than none —
it would run, and quietly answer a different question. What ships instead is the
run IR (JSON + deterministic NPZ), the `β̃(ψ)` tables, the IOP tables, a
human-readable *provisional* description per run that is labelled in three
places as not being a deck, and a `README.txt` stating the conventions in the
operator's terms with the ten questions appended.

**Deliberately not done:** the other ~156,000 decks. Batch 0 is 21 water bodies;
the rest is a parameter change once batch 0 passes its gates, per spec §9.

Next: send batch 0 to Henry with the ten questions; the Ingest prompt writes
`robust/rt/data/hh26.py` when it returns. The two in-house tasks remain
unblocked — R7 through the `Geometry.Ed` seam (§8.3) and the `µ∞` eigenvalue
solve (§8.4).

### 2026-09-23 (Spec prompt 3 — wrote the operator HOWTO, and packaged batch 0)

Executed Spec prompt 3 (model: **Opus 5**, not Fable — same note as previous
entries). Wrote **`design/hydrolight_howto.md`** (221 lines), published it as a
shareable page at
<https://claude.ai/code/artifact/9599486c-c0a9-4d38-8767-016396ea02c1>, and added
a reproducible tarball to the batch-0 generator. No package logic changed.

**The framing decision, and it is the substance of this deliverable.** The
obvious HOWTO would explain how to run 156,000 HydroLight runs. That would be
the wrong document, because **Henry cannot run anything yet** — we do not know
his deck syntax (spec §10 Q7), and one unanswered question (Q1, whether one run
returns the whole radiance distribution) could change the campaign size by 130×.
So the HOWTO is organised around **three stages with "we are here" on stage 0**,
and its single loudest instruction is *do not run anything yet*. Stage 0 is four
questions and about thirty minutes of his time; stage 1 is the 172-run pilot;
stage 2 is the campaign, included only so he can price his time.

That reordering matters: a document that opened with the conventions would invite
him to start, and the first thing he produced would be unusable.

**The four blocking questions**, separated from the six that do not block:
the run-count model; one example deck plus whether he accepts user-supplied IOP
files; his Raman and fluorescence data files verbatim (including **which Hawes
variant**, which is the one that would be silently wrong rather than loudly
wrong); and his version string. The other six — per-run cost, `K∞` availability,
the 330 nm floor, sky model, disk, and "anything here you would do differently" —
are listed as useful but non-blocking so he can answer at leisure.

**Register.** Written for a colleague who runs HydroLight and we do not: it says
so, and question 10 explicitly invites him to overrule our conventions. Every
request that could look arbitrary carries its reason in one clause — float64
because our metric divides by `rrs` and float32 underflows it to zero at grazing
geometries; echo the IOPs back because that is how we verify the deck we wrote is
the deck that ran; 172 runs because finding a convention mismatch after 156,000
is the expensive way.

**Also done:** `make_hydrolight_batch0.py --tar` now writes
`HH26_batch0.tar.gz` (161 files, 134 kB) with fixed member mtime/uid/gid/mode
and `gzip` mtime zeroed, so **the archive is byte-identical on regeneration** —
verified by regenerating to the same path and comparing SHA-256. Python's
`tarfile` and `gzip` both stamp wall-clock times by default, the same defect
`deck.write_arrays` had to fix for the NPZ container.

**The published page** exists because Henry does not have the repository, so a
link is more shareable than a file path. Same content as the markdown; the
markdown stays the source of truth in `design/` alongside the spec it condenses.

Next: send Henry the link and the tarball. The Ingest prompt writes
`robust/rt/data/hh26.py` when batch 0 returns. The two in-house tasks remain
unblocked — R7 through the `Geometry.Ed` seam (spec §8.3) and the `µ∞`
eigenvalue solve (§8.4), neither of which waits on any of this.
