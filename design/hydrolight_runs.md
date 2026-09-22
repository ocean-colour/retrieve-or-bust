# HydroLight Runs — Commissioning Specification

*The reference data the `robust.rt` forward model still needs: four batches,
≈157,000 HydroLight runs, one release.*

**Date:** 2026-09-22. **Authors:** J. Xavier Prochaska and Claude.
**Operator:** Henry Houskeeper (full HydroLight, latest build).
**Status:** specification — nothing commissioned. Decisions locked in
Q&A/Setup rounds 1–3 of
[`claude_prompts/RT/hydrolight_prompts.md`](../claude_prompts/RT/hydrolight_prompts.md)
(2026-09-22).

**Audience.** §§1–3 and §§4–7 are written for the operator and are meant to be
excerpted: **each batch section is self-contained**, with its own run table,
inputs, outputs, metadata contract and acceptance criteria. §§8–11 and the
appendices are for us.

**The catalogue this implements** is
[`claude_prompts/RT/hydrolight_prompts.md`](../claude_prompts/RT/hydrolight_prompts.md)
(runs R0–R8). This document supersedes it wherever the two differ; §11 lists
every place they do.

---

## 0. The campaign in one table

| Batch | Runs | Contents | Blocks |
|---|---|---|---|
| **0 — pilot** | **172** | L23 reproduction test, R0 convergence sweep, grid corners | everything; nothing else starts until it passes |
| **A — backbone** | **16,659** | R1 (asymptotic quantities, low scattering angles) + R2 (VSF designs) | the elastic backbone off-nadir; the emulator's IOP×geometry×VSF training set |
| **B — crossing** | **130,140** | R3 (dense zeniths × off-nadir × X1/X2/X4) + R6 (330 nm floor) | every inelastic caveat; supersedes L23 |
| **C — dependents** | **9,720** | R4 (φ_C), R5 (CDOM fluorescence), R8 (vertical structure) | φ_C linearity; δ_C training; the homogeneity bias |
| | **≈156,700** | ≈3.1× the compute that produced PB24 | |

Wall-clock at 1 / 3 / 5 minutes per run on 16 cores: **6.8 / 20 / 34 days**.
Delivered volume at 2–5 MB per run: **310–780 GB**, in **≈17,400 files**.

**Not commissioned:** R7 (alternative solar spectra) — answered analytically
in-house instead, §8.3. See §11.

---

## 1. The run-count model — read this first

**One full-HydroLight run returns the entire upwelling radiance distribution.**
Every view zenith and every relative azimuth come out of a single solve. What
multiplies the run count is the *sky* (one solar zenith per run) and the
*scenario* (the inelastic switches change what is being solved):

> **runs = water bodies × solar zeniths × scenarios**

This is the single most consequential fact in the campaign, and it is why the
off-nadir coverage this project has lacked since M1 — *"L23 is nadir-only, PACE
is not"* — **costs nothing extra**. It is also the reason the numbers above are
affordable at all.

The arithmetic is confirmed by PB24: quoted as "5000 realisations × 1300
geometries ≈ 6.5 M spectra", where 1300 = 10 θ_s × 10 θ_v × 13 Δφ. Only the ten
solar zeniths cost anything, so PB24 is **50,000 runs**, not 6.5 million.

**This model is question 1 for the operator (§10).** If the working setup
requires one execution per view direction, the campaign is ~130× larger than
specified and must be re-scoped before anything is committed.

---

## 2. Conventions shared by every batch

### 2.1 Release name, directory layout, file format

- **Release name:** `HH26` (operator surname + year). Placeholder pending the
  operator's preference; whatever is chosen is frozen before batch 0 ships.
- **Mount:** `$OS_COLOR/HH26/v1/`, one subdirectory per batch:
  `pilot/`, `batchA/`, `batchB/`, `batchC/`.
- **One netCDF4 file per `(water body, scenario)`, containing all nine solar
  zeniths.** This groups the solar-zenith axis inside the file rather than
  across files, which stores each water body's IOPs once instead of nine times
  and cuts the file count from ~157,000 to **~17,400**. *(This is a change from
  the round-2 statement of one file per `(water body, scenario, θ_s)`, made
  under A4 "request what you prefer"; the reason is file count and IOP
  duplication, and it is recorded here rather than made silently.)*
- **Naming:** `HH26_<batch>_<wbid>_<scenario>.nc`, e.g.
  `HH26_B_L23-02871_S4.nc`. `<wbid>` is the water-body identifier from the
  manifest; L23-reconstructed bodies carry their L23 scene index.
- **Precision: float64 on every radiometric field.** PB24's float32 `rrs`
  underflows to exactly zero at grazing geometries, and our metric divides by
  truth. If float64 is impossible, the representable floor must be stated in
  the manifest.

### 2.2 Wavelength grid

**330–750 nm, 5 nm bands, 85 bands**, band centres at 330, 335, …, 750 nm.

The 350–750 nm subset is **bit-compatible with L23's 81-band grid**
(`robust.rt.conventions.WAVE`), which is what makes the L23 reproduction test of
§4 meaningful. The extension to 330 nm is R6: it supplies Raman excitation for
emission bands below 400 nm, which L23's 350 nm floor truncates (a measured 13 %
error at 350 nm today). **It is nearly free if specified now and impossible to
retrofit.**

One thing to confirm with the operator (§10 Q8): how the Raman implementation
treats excitation that falls below the band-set floor — clipping or
extrapolation. This determines whether the 330 nm floor perturbs the ≥ 400 nm
bands at all. For emission at 400 nm the 3400 cm⁻¹ excitation sits at 352 nm,
inside the old grid, so the expected perturbation is confined to the finite
width of the Raman redistribution function.

### 2.3 Solar zenith grid, and view geometry

- **Solar zenith: θ_s ∈ {0, 10, 20, 30, 40, 50, 60, 70, 80}° — nine values**,
  shared by every batch so one geometry design serves the whole campaign.
  - 0–70° is the **sanctioned window**. 80° is retained as a deliberate
    extrapolation shell, exactly as PB24 holds its 80/87.75° shell out.
  - Nine values with 10° spacing let an intermediate zenith be **held out and
    tested**, which three anchors could never do. This is what retires the
    inelastic report's §5 item 1 (the Raman head errs by −74 % at an unseen
    60°) and the elastic report's §5 item 3 together.
- **View geometry: the standard HydroLight quad layout** (≈10° in polar angle,
  15° in azimuth), no custom quad file. Request the **full upwelling radiance
  distribution** at every quad. Standard quads are what PB24 used, they reach
  ψ ≈ 65° inside the 0–70° window and ψ ≈ 44° over the full grid — already well
  below the 134° floor of ZTT's fitted `F(ψ)` and its 110.4° zero-crossing,
  which is the whole point of R1. A custom quad file would buy resolution we
  have no use for and cost format-compatibility with PB24.
- At θ_s = 0 the problem is azimuthally symmetric; the redundant azimuths are
  kept rather than special-cased, so every file has the same shape.

### 2.4 Scenario switches

Named by switch, not by L23's X-label, with the mapping recorded in the manifest
(A10):

| Scenario | Raman | Chl fluorescence | CDOM fluorescence | L23 equivalent |
|---|---|---|---|---|
| **S1** | off | off | off | X = 1 |
| **S2** | **on** | off | off | X = 2 |
| **S4** | **on** | **on** | off | X = 4 |
| **S5** | **on** | **on** | **on** | *(none — new)* |

`S2 − S1` and `S4 − S2` remain the clean per-process truth channels the current
model is built on; `S5 − S4` becomes the CDOM-fluorescence channel that has
never existed anywhere.

Chlorophyll fluorescence quantum yield is **φ_C = 0.02** everywhere except R4.

### 2.5 Water column, surface, and sky

- **Homogeneous, optically deep, no bottom**, everywhere except R8. L23 is also
  vertically homogeneous, so this is *not* a simplification relative to it —
  the catalogue's description of L23 as "stratified" was an error (§11).
- **Sky:** HydroLight's semi-empirical clear-sky model, parameters fixed across
  the campaign and recorded. **Wind speed 5 m s⁻¹**, fixed.
- **The sky is validated, not assumed.** `robust/rt/data/ed_l23.npz` holds
  L23's own `Ed(0+)` spectra at θ_s = 0/30/60°. Batch 0 compares the delivered
  `Ed(0+)` against them at those three anchors; a mismatch there is a sky-model
  mismatch and is far cheaper to find in 172 runs than in 157,000.

### 2.6 Outputs required from every run

Per run, on the 85-band grid, at every solar zenith in the file:

1. `Rrs(λ, θ_v, Δφ)` **above water** and `rrs(λ, θ_v, Δφ)` **just below the
   surface** — both, with the convention stated explicitly in the manifest
   rather than inferred from magnitudes.
2. `Lu(0⁻, λ, θ_v, Δφ)` — the full upwelling radiance distribution.
3. `Ed(0⁺, λ)`, `Ed(0⁻, λ)`, `Eu(0⁻, λ)`, `Lu(0⁺, λ, θ_v, Δφ)`.
4. Surface diffuse-attenuation coefficients: `K_d`, `K_u`, `K_Lu`, and the mean
   cosines `µ_d`, `µ_u`, `µ_tot`.
5. **The IOPs as HydroLight received them** — `a(λ)`, `b(λ)`, `bb(λ)`, and the
   tabulated `β̃(ψ, λ)` — echoed back into the output file. This is not
   redundancy: it is how the loader verifies that the deck we wrote is the deck
   that ran.
6. **Batch A only:** depth profiles `K_d(z)`, `K_u(z)`, `K_Lu(z)`, `µ̄(z)` to
   **≥ 25 optical depths**, and the **asymptotic radiance distribution and
   `K∞`** if the build reports them (§10 Q2). Requesting depth profiles for
   batch A only is most of the difference between 310 GB and 780 GB delivered.

### 2.7 The manifest

One JSON file per batch, `HH26_<batch>_manifest.json`, carrying:

- release name, version, batch, creation date;
- **HydroLight version and build string**, and every non-default switch;
- the deck-generator commit hash from `robust/rt/hydrolight/`;
- the shared conventions of §2.2–§2.5 as literal values (band centres, quad
  layout, θ_s list, sky parameters, wind speed, scenario→switch map);
- one record per run: water-body id, scenario, θ_s list, **SHA-256 of the input
  deck** and of the output file;
- the operator-supplied physics constants — metadata items 3, 4 and 5 of §3.

The manifest is the file the loader's tests read. Anything not in it is not
recorded.

---

## 3. The metadata contract

Ten items. **Most are ours by construction**, because we author the decks (A5,
A7) — which is exactly why this campaign can promise what previous ones only
asked for. The three that are not ours are the ones that have already been
identified as live foot-guns.

| # | Item | Supplied by |
|---|---|---|
| 1 | HydroLight version and build, and every non-default switch | **operator** (one string) |
| 2 | The phase function: family, parameters, **and the tabulated `β̃(ψ)` actually used** | **us** (in the deck; echoed in output per §2.6.5) |
| 3 | **The Raman scattering coefficient and wavelength-redistribution function**, with source | **operator** — lives in HydroLight's data files, not our decks |
| 4 | **The chlorophyll-fluorescence quantum efficiency and emission shape** — band centre, width, any PS I shoulder | **operator** |
| 5 | **The CDOM-fluorescence quantum-efficiency function: which published Hawes variant, any HydroLight-side modification, all constants** | **operator** |
| 6 | The solar spectrum and sky model, by name and version | us (deck) + operator (model version) |
| 7 | The depth grid and the optical depth reached | us (deck), verified in output |
| 8 | The output convention — `Rrs` vs `rrs`, above vs below water, at which depth | **operator**, stated explicitly; verified in batch 0 |
| 9 | The IOP decomposition — `a_ph`, `a_cdom`, `a_NAP`, `b_ph`, `b_NAP` reported separately and consistent with `a_ph + a_cdom ≤ a` | us (deck), echoed in output |
| 10 | Whether IOPs are bit-identical across scenarios and solar zeniths | us by construction; **pinned by a loader test** |

**Items 3, 4 and 5 are requested as files, verbatim, not as prose.** Published
Hawes constants exist in several variants (fulvic vs humic; HydroLight's own
default), and a mismatch between the truth's constants and our kernel's would
silently re-introduce the error the CDOM design took care to avoid.

**This table is attached to every batch below.** For batches that do not
exercise an item (batch A runs no inelastic physics, so items 3–5 are moot) the
batch section says so.

---

## 4. Batch 0 — the pilot, and R0

**172 runs. Nothing else is commissioned until this passes.**

### 4.1 Purpose

Three questions, none of which can be answered after the compute is spent:

1. **Do our decks reproduce L23?** If yes, every convention — sky model, wind,
   surface, band structure, Raman and fluorescence constants, output convention
   — is validated at once, against 3,320 published spectra we already hold.
2. **What is HydroLight's own numerical error here?** (R0.) This project gates
   models at **0.30 %** and **0.34 %** rRMS against HydroLight output whose quad
   resolution, depth-grid resolution and inelastic-source iteration count have
   never been recorded. If the solver's discretisation error at the chosen
   settings is comparable, both gates are partly measuring HydroLight and we
   have never known it.
3. **Does the delivery format work?** Final format, final manifest, ingested by
   the real loader.

### 4.2 The runs

| | water bodies | θ_s | scenarios | configs | runs |
|---|---|---|---|---|---|
| **P1 — L23 reproduction** | 10 L23 scenes, spanning the `a_ph(440)` range | 0, 30, 60° | S1, S2, S4 | baseline | **90** |
| **P2 — R0 convergence** | 5, spanning `bb/a` and trophic state | 0, 60° | S4 | **7** | **70** |
| **P3 — grid corners** | 6, at the corners of batch A's realizable grid | 0, 80° | S1 | baseline | **12** |
| | | | | | **172** |

**P1 water bodies** are reconstructed from L23's own published IOPs — see §6.2.
The comparison is against the L23 `Rrs` we already load for those same scene
indices.

**P2's seven configurations** are the baseline plus, one at a time, a coarser
and a finer setting of each of: **quad resolution**, **depth-grid resolution**,
and **inelastic-source iteration count**. The operator chooses the alternative
settings; what matters is that each knob is moved in both directions from
whatever the campaign's baseline will be.

**P3** exercises the user-supplied-IOP input path at the extremes (§10 Q5) and
the 80° grazing shell.

### 4.3 Acceptance

- **P1 — the convention gate.** Median `|ΔRrs|/Rrs` over 400–700 nm at nadir,
  against L23's published `Rrs` for the same scenes, **≤ 2 %**; and the
  per-process differences `S2 − S1` and `S4 − S2` agree with L23's `X2 − X1` and
  `X4 − X2` in sign at every band and within **10 %** in magnitude.
  *A failure here is diagnostic, not fatal* — it localises which convention is
  wrong (sky, wind, band structure, Raman constants, output convention) while it
  still costs 90 runs to re-run. The batch does not proceed until it is
  understood.
  *Expect residuals:* L23's Fournier–Forand parameter is **inferred** from their
  published `B_p`, not read from their deck, and the sky is matched at three Ed
  anchors only (§6.2). The 2 % figure is chosen to accommodate that; a much
  smaller residual is good news and a much larger one is information.
- **P2 — the error-bar gate.** Report the spread in `Rrs` across the seven
  configurations. **If the spread at the chosen baseline settings exceeds
  0.1 %**, the project's 0.30 % and 0.34 % headline gates must be restated
  against it, and this document's acceptance criteria elsewhere revised. This is
  a *measurement*, not a pass/fail: there is no setting at which we refuse to
  proceed, only a number we are obliged to publish.
- **P3 — the path gate.** All twelve runs complete, return physical `rrs > 0` at
  every band, and the echoed IOPs (§2.6.5) match the deck bit-for-bit.
- **Format gate.** The batch loads through `robust/rt/data/hh26.py` with a
  `LoadReport` and zero hand-editing, and every metadata item of §3 is present.

### 4.4 Metadata contract

**All ten items of §3 apply**, and batch 0 is where items 1, 3, 4, 5 and 8 are
collected for the first time. Items 3–5 must arrive as the operator's actual
data files. Item 8 (output convention) is *verified* here against L23 rather
than trusted.

---

## 5. Batch A — the backbone (R1 + R2)

**16,659 runs. 1,851 water bodies × 9 solar zeniths × 1 scenario (S1).**

### 5.1 Purpose, and what it unblocks

`µ∞ = a/K∞` and `F(ψ) = K_Lu/K∞ − 1` are defined against the **asymptotic**
`K∞`, which no dataset we hold tabulates — PB24's seven diffuse-attenuation
coefficients all vary ~1.4× across solar zenith, so none of them is `K∞`. That
single absence closed Route A at M5. Meanwhile `Ψ_KLu(ψ) = 1 + F(ψ)` is a
quartic fitted for ψ ≳ 134° that crosses zero at **110.4°**, and 42 % of PB24's
sanctioned window sits below 134°, 16 % below the crossing — which is why 22.3 %
of ZTT's predicted `rrs` on PB24 are **zero or negative**.

Batch A unblocks: refitting `µ∞` and `F(ψ)` over the domain we actually use;
deciding between repairing ZTT and replacing it; and — either way — **the
emulator's training set over IOPs × geometry × VSF**, which is what a pure
differentiable emulator of HydroLight needs whether or not ZTT survives. Framing
it that way is deliberate: it makes batch A's value independent of a bet M5 has
already partly lost (Appendix C).

### 5.2 Water bodies

**Design coordinates** `(bb/a, η_bb)` at **λ_ref = 400 nm**, where the
realizable region is largest, with `η_bb = bb_w/bb`:

- `bb/a`: 12 nodes, logarithmic, **1 × 10⁻⁴ to 3** (TT2017 fitted 1e-4–0.1; L23
  reaches 0.59; PB24 reaches 20.1).
- `η_bb`: 8 nodes, logarithmic, **0.01 to 0.98** (TT2017's own range).

**Only 63 of those 96 nodes exist in real water**, and this is a physical fact
rather than a budget decision — see **Appendix A**. Each surviving node is
realised as a spectrally flat non-water absorption `a_nw` and particle
scattering `b_p`:

> `bb(λ_ref) = bb_w(λ_ref)/η_bb` · `a_nw = bb(λ_ref)/(bb/a) − a_w(λ_ref)` ·
> `b_p = (bb(λ_ref) − bb_w(λ_ref))/B_p`

so that `a(λ) = a_w(λ) + a_nw` and `b(λ) = b_w(λ) + b_p`. Because `a_w` and
`bb_w` both vary strongly across 330–750 nm, **each water body traces a curve
through `(bb/a, η_bb)` across the 85 bands**: the 63 nodes deliver coverage of
**90 % of the full 12 × 8 target box**, not 63 isolated points. The ten
unreachable cells are named in Appendix A.

- **63 realizable nodes × 27 VSF designs = 1,701 water bodies** (Appendix B).
- **+ 150 fill bodies**, drawn directly in `(a_nw, b_p)` by Latin hypercube, to
  densify regions the node grid leaves sparse at wavelengths away from λ_ref.
- **= 1,851 water bodies.**

Homogeneous, optically deep, no bottom.

### 5.3 Geometry, spectral, scenarios

Nine solar zeniths and the standard quad layout (§2.3); 330–750 nm at 5 nm
(§2.2); **S1 only** — batch A carries no inelastic physics, which is what makes
it the cheapest answer per run in the campaign.

### 5.4 Outputs

Everything in §2.6, **including item 6**: depth profiles `K_d(z)`, `K_u(z)`,
`K_Lu(z)`, `µ̄(z)` to **≥ 25 optical depths** at every band, and the
**asymptotic radiance distribution and `K∞`** if the build reports them.

**If `K∞` is not reported directly, the run is still worth doing**: the
asymptote is read off the profile and its convergence *shown* rather than
assumed, which is why the ≥ 25 optical-depth requirement is not negotiable. This
is also why batch A is not gated on the answer to §10 Q2 — and why we are
computing `µ∞` independently by eigenvalue solve in parallel (§8.4).

### 5.5 Acceptance

- Every run returns physical `rrs > 0` at every band and every quad, or the
  exceptions are enumerated in the `LoadReport`.
- The depth profiles converge: `K_Lu(z)` is flat to within 1 % over the last two
  optical depths for at least 95 % of runs. **Report the other 5 %** rather than
  dropping them.
- Echoed IOPs match the decks bit-for-bit.
- `ψ` coverage reaches **≤ 70°** over the delivered quads, verified rather than
  assumed.

### 5.6 Metadata contract

**Items 1, 2, 6, 7, 8, 9, 10 of §3 apply.** Items 3, 4 and 5 (the Raman and
fluorescence constants) are **not exercised by this batch** — S1 runs no
inelastic physics — but must still be recorded once, in batch 0, because batches
B and C depend on them.

Item 7 (depth grid and optical depth reached) is **load-bearing here** in a way
it is not elsewhere: it is the evidence that the asymptote was reached.

---

## 6. Batch B — the crossing (R3 + R6)

**130,140 runs. 4,820 water bodies × 9 solar zeniths × 3 scenarios.**
**This is 83 % of the campaign, and it is the batch that supersedes L23.**

### 6.1 Purpose, and what it unblocks

Every inelastic number this project holds is **nadir, at one of three solar
zeniths, under one fixed phase function**. The correction heads interpolate over
three anchors and carry no domain guard; trained without 60° the Raman head errs
by **−74 %** there — worse than no head at all. PACE is not nadir, and the Raman
and fluorescence source terms depend on the in-water light field, which is
exactly what changes off-nadir.

Batch B delivers, for the first time anywhere in this project: nine solar
zeniths (so one can be held out and interpolation *tested*), the full BRDF,
three scenarios on the *same* water bodies (so `S2 − S1` and `S4 − S2` stay clean
per-process truth channels), varied VSF designs crossed with inelastic physics,
and a 330 nm floor.

**It supersedes L23** (A-R2Q2): once it exists, retraining happens against it and
the published 0.30 % / 0.34 % numbers become history rather than something to
extend.

### 6.2 Water bodies — two blocks

**Block B-L23 — all 3,320 L23 water bodies, reconstructed.**

We do not have L23's HydroLight input decks and cannot get them (A2). **We do not
need them.** HydroLight does not care which concentrations generated a water
body; it cares about `a(λ)`, `b(λ)` and `β̃(ψ)`. The L23 netCDFs we already load
carry `a`, `bb`, `bbnw`, `bnw`, `aph` and `ag` per scene per wavelength, and
`B_p = bbnw/bnw` fixes the one Fournier–Forand parameter. So each L23 scene is
rebuilt as a user-supplied-IOP deck from data already on disk.

Three caveats, recorded in the manifest and repeated wherever the comparison is
used:

1. L23's FF parameter is **inferred from `B_p`**, not read from their deck.
2. The sky is **matched at three Ed anchors** (0/30/60°), not reproduced from
   their atmospheric specification.
3. "The same water bodies" means **the same optics**, not Loisel's generative
   recipe. That is the only sense that matters for radiative transfer, and it is
   not the sense a Loisel co-author would use.

Batch 0's P1 gate is what turns these three caveats into a measured number.

**Block B-design — 1,500 new water bodies.** Latin-hypercube draws over
`a_ph(440)`, `a_g(440)`, `S_g`, `a_NAP(440)`, `S_NAP`, `b_p(550)` and the
particle scattering slope, with ranges **deliberately wider than L23's**, and one
of the 27 VSF designs assigned round-robin so the phase-function axis is crossed
with the inelastic physics. This block is what makes the release a superset of
L23 rather than a re-run of it.

Homogeneous, optically deep, no bottom. φ_C = 0.02.

### 6.3 Geometry, spectral, scenarios

Nine solar zeniths and the standard quad layout (§2.3); **330–750 nm at 5 nm**
(§2.2 — R6 is folded in here and is nearly free); **S1, S2 and S4 on every water
body**.

### 6.4 Outputs

Everything in §2.6 **except item 6** — no depth profiles. Surface quantities
plus the full radiance distribution. This is the choice that keeps the campaign
near 310 GB rather than 780 GB.

### 6.5 Acceptance

- **IOPs bit-identical across the three scenarios** for a given water body
  (metadata item 10) — pinned by a loader test, because L23's are and both
  existing loaders exploit it.
- `S2 − S1 > 0` at every band where Raman can contribute, and `S4 − S2` is
  positive and peaked near 685 nm.
- The B-L23 block reproduces L23's `Rrs` at θ_s ∈ {0, 30, 60}, nadir, to the
  tolerance established in batch 0 — **on all 3,320 scenes, not the 10 of the
  pilot.** This is the campaign's single largest verification and it costs
  nothing extra.
- Every run physical, or enumerated.

### 6.6 Metadata contract

**All ten items of §3 apply.** Items 3 and 4 (Raman redistribution; chlorophyll
fluorescence efficiency and emission shape) are **load-bearing here**: they are
the constants our analytic kernels must match, and `S4 − S2` is meaningless
without knowing which emission shape produced it. Item 5 is not exercised (no
CDOM fluorescence in S1/S2/S4).

---

## 7. Batch C — the dependents (R4, R5, R8)

**9,720 runs.** All three reuse batch B's water bodies and geometry, which is
why they are a batch rather than three campaigns.

### 7.1 R4 — varied quantum yield φ_C — 3,600 runs

*Why.* The model's φ_C-linearity is exact **by construction**, and L23 provides
truth at exactly one yield. Whether the real ocean is φ_C-linear at the
few-percent level is untested — and φ_C is the physiology handle the whole
inversion is being built to retrieve, so a shape bias propagates straight into
the science result.

*Spec.* **100 water bodies** drawn from block B-design across `a_ph(440)`
deciles, at **φ_C ∈ {0.005, 0.01, 0.04, 0.06}** — four values; 0.02 already
exists in batch B's S4. Nine solar zeniths, scenario S4, full spectral range.
`100 × 4 × 9 = 3,600`.

*Sized deliberately small on the scene axis.* This measures **curvature in
φ_C**, and curvature is measured by more φ_C values, not by more scenes. A few
hundred scenes would be a training set, which is not the question.

*Acceptance.* `Rrs(φ_C)` monotone increasing at 685 nm for every water body;
the departure from linearity reported as a number, per band and per `a_ph`
decile.

### 7.2 R5 — CDOM fluorescence on/off pairs — 4,500 runs

*Why.* CDOM fluorescence is **shipped and completely unvalidated** — off by
default, analytic-only, its δ_C head untrained and unwired, because no reference
of any kind exists. It is the only process in the package with zero truth.

*Spec.* **500 water bodies** drawn from batch B, **stratified on `a_g(440)`
with the CDOM-rich tail deliberately oversampled** (the sparse-tail lesson from
δ_F's eutrophic drift). Nine solar zeniths, **scenario S5 only** — the S4
partner already exists in batch B, so the pair difference `S5 − S4` isolates the
process exactly as `S4 − S2` isolates chlorophyll fluorescence.
`500 × 1 × 9 = 4,500`.

This **supersedes** `design/rt_cdom_fluorescence_model.md` §7, which asked for
the full 3,320-scene ensemble at three zeniths. That request and the instruction
to share the geometry grid could not both be honoured; what survives from §7 is
what was actually non-negotiable — the paired on/off design and the recorded
Hawes variant.

*Non-negotiable:* **metadata item 5.** Published Hawes constants exist in several
variants, and a mismatch would silently re-introduce the error the CDOM design
took care to avoid.

*Acceptance.* `S5 − S4 > 0` in the blue-green for every water body, monotone in
`a_g(440)`, and zero (to numerical precision) where `a_g(440) → 0`.

### 7.3 R8 — vertical structure — 1,620 runs

*Why.* Every other water body in this campaign is vertically homogeneous, but
the 685 nm fluorescence signal originates shallower than the blue-green elastic
signal — a known, accepted homogeneity bias that nothing in hand can quantify.

*Spec.* **20 base water bodies × 3 vertical structures = 60 profiles**: a
homogeneous control, a subsurface chlorophyll maximum at two depths, and a φ_C
that varies with depth (photoinhibition near the surface). Nine solar zeniths,
S1/S2/S4. `60 × 3 × 9 = 1,620`.

*Note the tension with batch A*, which **requires** homogeneity because the
asymptotic regime only exists for a homogeneous column. These are different
runs, not a shared setup.

*Acceptance.* The homogeneous controls reproduce their batch-B counterparts
bit-for-bit where the water body is shared.

### 7.4 Metadata contract for batch C

**All ten items of §3 apply.** Item 5 is load-bearing for R5 and is the reason
R5 exists as a separate scenario. Item 4 is load-bearing for R4 (the emission
shape must be the same one batch B used, or the φ_C series is not a series).
Item 7 (depth grid) is load-bearing for R8 in a way it is not for R4 or R5.

---

## 8. What we build, and what we do not commission

### 8.1 `robust/rt/hydrolight/` — the deck generator (A-R2Q3)

The operator will run any input files we give them (A5), which makes deck
generation **our** engineering, not theirs: ~157,000 decks plus the IOP tables
and tabulated `β̃(ψ)` files they reference. It lives in the package, under test,
because the loader must later cross-check every delivered file against the deck
that produced it.

Contents: the VSF library (Appendix B), the IOP-grid and ensemble designers
(§5.2, §6.2), the L23 reconstruction, the deck writer, and the manifest writer.
Tests pin the realizability mask, the VSF `B_p` values, and the round-trip
deck → manifest → loader.

**This is what collapses the metadata contract**: items 1, 2, 6, 7, 8, 9 and 10
become ours by construction and *testable*, instead of asserted in an email.

### 8.2 `robust/rt/data/hh26.py` — the loader

Written against batch 0 before batch A is committed, following the conventions
of [`l23.py`](../robust/rt/data/l23.py) and [`pb24.py`](../robust/rt/data/pb24.py):
a `LoadReport` saying what was kept and what was dropped, splits (by water body,
by VSF design, by solar zenith, by block), a committed small fixture so CI runs
real numbers without the data mount, and tests that pin the **measured**
properties of the release rather than trusting its documentation.

The held-out-**VSF-design** split is new and is a far stronger claim than the
held-out-`B_p` split M5 built.

### 8.3 R7 is not commissioned — it is answered analytically

Elastic `Rrs` is a ratio and is close to independent of the incident spectrum's
*shape*; the Raman and fluorescence terms depend on `Ed` through
`Ed(λ_ex)/Ed(λ_em)` and the absorbed-photon integral, both of which can be
perturbed directly by swapping a TSIS-1-based `Ed` into our own forward model
through the existing `Geometry.Ed` seam. That gives the first-order answer in a
day or two with **no compute**; HydroLight would add only the second-order
coupling where the altered in-water field feeds back on the source terms.

R7 leaves the campaign for **speed, not cost** (A-HD6): it turns DQ5 from
"unquantified" into a number now, rather than in a month behind 157,000 runs.

### 8.4 `µ∞` by eigenvalue solve, in parallel

The asymptotic radiance distribution is determined by the IOPs alone — which is
*why* `K∞` is θ_s-independent, the very fact that ruled PB24 out. So `µ∞` is
obtainable from an eigenvalue solve of the asymptotic RTE for a given
`(a, b, β̃)`, over any grid we like, including all 27 VSF designs at a density no
run buys.

This runs **in parallel with commissioning, not as a gate on it** (A-HD1, round
3 reading). The round-1 argument for doing it first was to de-risk the
campaign's largest line item; batch A is 11 % of the campaign, so that argument
no longer holds. Its value now is redundancy: batch A's `µ∞` no longer depends
on the answer to §10 Q2.

---

## 9. Delivery, acceptance, and what can start when

Five self-contained deliverables, each with its own manifest, so ingest and
retraining begin long before the campaign finishes.

| # | Deliverable | Runs | Gate | What starts on arrival |
|---|---|---|---|---|
| 1 | **Batch 0** | 172 | §4.3 — all four gates | the loader; the conventions are frozen |
| 2 | **Batch A** | 16,659 | §5.5 | `F(ψ)` and `µ∞` refit; the emulator's IOP×VSF training |
| 3 | **Batch B, block B-L23** | 89,640 | §6.5 | retraining the inelastic heads; the zenith-interpolation gate |
| 4 | **Batch B, block B-design** | 40,500 | §6.5 | the held-out-VSF-design split |
| 5 | **Batch C** | 9,720 | §7.1–7.3 | φ_C linearity; δ_C training; the homogeneity bound |

**Nothing after deliverable 1 is committed until batch 0 passes.** That is the
whole point of a pilot, and it is the cheapest insurance in this document.

---

## 10. Questions for the operator — batch zero

Ordered by how much of this specification each answer moves.

1. **Confirm the run-count model of §1.** Does one run return the full radiance
   distribution over all view quads, so that only solar zenith and the inelastic
   switches multiply the run count? *The entire campaign sizing depends on this.*
2. **Does the build report the asymptotic radiance distribution and `K∞`** for
   homogeneous, optically deep water — and under which switch? If not we read the
   asymptote off the ≥ 25 optical-depth profile (§5.4), which we are requesting
   regardless.
3. **Exact version and build string**, and any local modifications.
4. **The Raman and fluorescence data files as shipped** — metadata items 3, 4
   and 5, verbatim, including **which published Hawes variant** the
   CDOM-fluorescence option uses and its constants. This is the one part of the
   contract we cannot satisfy from our own decks.
5. **Will user-supplied IOP files be accepted** — tabulated `a(λ)`, `b(λ)` and a
   discretised `β̃(ψ)` per water body — and in what format?
6. **What does a run cost** in wall-clock, for an 81-band elastic run and for an
   inelastic one, and how many run in parallel? This is the number that turns
   157,000 runs into a date.
7. **One example deck and its outputs** from the working setup, as the template
   we generate 157,000 of.
8. **Sub-350 nm.** Can the band set start at 330 nm, and does the Raman
   implementation handle excitation at the band-set floor by clipping or by
   extrapolation (§2.2)?
9. **Sky and surface.** Which sky model, and is a fixed 5 m s⁻¹ wind acceptable?
10. **Disk and transfer** — what can be produced and how we take delivery.

---

## 11. What this campaign will and will not settle, and where it departs from the catalogue

### Departures from `hydrolight_prompts.md`

| | The catalogue said | This spec says | Why |
|---|---|---|---|
| Cost model | "multiply by geometries **and** by scenario for the run count" | runs = water bodies × **solar zeniths** × scenarios | §1. View geometry is free; the catalogue's own PB24 figures confirm it |
| R1 water bodies | "simpler than L23's **stratified** scenes" | L23 is **homogeneous**; no simplification is involved | `rt_inelastic_model.md` §8 and R8 both say so |
| R1 grid | "~12 × 8 nodes" over the stated ranges | **63 of 96 nodes**, plus a 150-body fill set; 90 % box coverage via the spectral sweep | Appendix A — 33 nodes are not physical water |
| R2 | "three distinct VSF **families** at matched `B_p`" | 27 designs; **the forward-shape axis is the strong one** | Appendix B — at matched `B_p` the backward hemisphere barely moves |
| R3 water bodies | "the L23 scene ensemble" (needs their decks) | **reconstructed from L23's published IOPs** | A2; §6.2 — the decks are unobtainable and unnecessary |
| R5 | "adopt §7 verbatim" | rewritten against batch B's grid and subset | A-HD8; §7.2 — §7 and the batching contradicted each other |
| R7 | a run | **analytic, in-house** | A-HD6; §8.3 |
| R0 | *(absent)* | **added**, inside the pilot | A-HD7; §4 — nothing established the truth's own error bar |
| Priority | "if only one run is possible, it is R1" | moot — no ceiling; the VSF axis is the last thing cut, not the second | A-HD2, A-HC |

### What it will settle

The off-nadir failure of the elastic backbone; whether ZTT can be repaired or
must be replaced; a real zenith-interpolation gate and **measured** domain guards
for the correction heads; the first off-nadir inelastic truth that has ever
existed for this project; φ_C-linearity against truth instead of construction;
δ_C trained and CDOM fluorescence scored rather than merely "plausible"; the
λ < 400 nm caveat; a held-out-VSF-design split; and — through R0 — what the
project's own 0.30 % and 0.34 % gates actually mean.

### What it will not settle

- **The backward-VSF axis, fully.** Appendix B measures why: at matched bulk
  `B_p`, `β̃(ψ)` in the backscatter hemisphere moves by only a few percent across
  every family we can construct. `PhaseParams.beta_tilde_pi` and
  `backward_slope` will be better constrained than today — but the honest claim
  will be about the **forward** shape, which moves by 25–50 % at matched `B_p`.
- **Real skies and atmospheres.** One sky model, one wind speed, clear sky.
- **Anything below 330 nm**, or above 750 nm.
- **Bottom effects, or optically shallow water.** Deep water only.
- **Vertical structure beyond a bound.** R8 is 60 profiles; it bounds a bias, it
  does not train anything.

---

## Appendix A — the `(bb/a, η_bb)` grid, and why 33 nodes do not exist

Given `η_bb = bb_w/bb` and a target `bb/a`, the water body is fully determined at
the reference wavelength:

> `bb = bb_w/η_bb`, `a = bb/(bb/a)`, so `a = bb_w/(η_bb · (bb/a))`

Since total absorption cannot fall below pure water's, `a ≥ a_w`, and therefore

> **`η_bb · (bb/a) ≤ bb_w(λ)/a_w(λ)`**

That ceiling is a property of pure water alone. Computed from
`robust.rt.conventions.BB_W_L23` and `ocpy.water.absorption.a_water`:

| λ (nm) | `a_w` (m⁻¹) | `bb_w` (m⁻¹) | `bb_w/a_w` | realizable nodes of 96 |
|---|---|---|---|---|
| 400 | 0.0066 | 0.003305 | **0.4984** | **91** |
| 440 | 0.0063 | 0.002196 | 0.3458 | 89 |
| 550 | 0.0565 | 0.000856 | 0.0151 | 67 |

**λ_ref = 400 nm is adopted** because the realizable region is largest there.

Adding the two physical caps `a_nw ≤ 20 m⁻¹` (blackwater is about the limit) and
`b_p ≤ 100 m⁻¹` leaves **63 of 96 nodes**, independently of `B_p`. The dropped
nodes are the low-`bb/a` column — which would need absorption above 20 m⁻¹ — and
the two cells at high `bb/a` *and* high `η_bb`, which are mutually exclusive by
construction: water-dominated backscatter pins `bb` low, and `a ≥ a_w` then pins
`bb/a` low.

Because `a_w` and `bb_w` vary strongly across 330–750 nm (`bb_w` by 25× over
350–750 alone, slope −4.345), each water body traces a curve through
`(bb/a, η_bb)`. Over the 63 nodes × 5 `B_p` values × 85 bands the delivered
samples span `bb/a` from 1.9 × 10⁻⁵ to 4.0 and `η_bb` from 0.0007 to 0.991, and
occupy **86 of the 96 cells (90 %)** of the original target box. Ten cells remain
empty and are listed by the generator's test, not hidden.

*Reproduce:* `robust/rt/hydrolight/grid.py` (to be written), or the derivation
above from the two packaged water tables.

---

## Appendix B — the VSF design set, and what it can and cannot vary

**27 designs**, all deliverable to HydroLight as a tabulated `β̃(ψ)`:

| group | count | construction |
|---|---|---|
| **Fournier–Forand, two-parameter sweep** | **15** | 5 target `B_p` × 3 distinct `(n, µ)` branches |
| **Two-component mixtures** | **10** | 5 target `B_p` × 2 recipes, mixing a large/forward-peaked component with a small/high-backscatter one |
| **Out-of-family anchors** | **2** | Petzold average-particle (`B_p ≈ 0.0183`); one measured Sullivan–Twardowski-style VSF if the operator has one tabulated |

### FF is a two-parameter family, and that matters

Fournier–Forand is usually *used* as a one-parameter family indexed by `B_p`, but
it is derived from two — the real refractive index `n` and the Junge slope `µ` —
so **distinct `(n, µ)` pairs give the same `B_p` with different shapes**.
Computed solutions, `|ΔB_p| < 4 × 10⁻⁴`:

| target `B_p` | branch A | branch B | branch C |
|---|---|---|---|
| 0.004 | n=1.080, µ=3.26 | n=1.115, µ=3.17 | n=1.230, µ=3.07 |
| 0.007 | n=1.020, µ=3.86 | n=1.100, µ=3.31 | n=1.190, µ=3.15 |
| 0.012 | n=1.055, µ=3.68 | n=1.095, µ=3.47 | n=1.190, µ=3.24 |
| 0.020 | n=1.075, µ=3.73 | n=1.150, µ=3.45 | n=1.210, µ=3.33 |
| 0.030 | n=1.040, µ=4.08 | n=1.150, µ=3.59 | n=1.165, µ=3.55 |

This is better than the round-2 plan (which proposed Mie-computed two-component
mixtures as the primary design) because it needs no new scattering code and
HydroLight has FF built in.

### But the backward hemisphere barely moves — measured, not assumed

`β̃(ψ)` for the three branches at matched `B_p = 0.012`, relative to branch B:

| ψ | 1° | 10° | 45° | 90° | 120° | 135° | 160° | 180° |
|---|---|---|---|---|---|---|---|---|
| branch A / B | **1.147** | **1.023** | 0.927 | 0.966 | 1.004 | 1.019 | 1.032 | 1.034 |
| branch C / B | **0.771** | **0.758** | 0.967 | 1.005 | 1.000 | 0.995 | 0.991 | 0.990 |

A **49 % spread at ψ = 1°** and **35 % at 10°**, against **1–3 % everywhere
beyond 120°**. A two-component mixture does better but not much: at matched
`B_p = 0.012` it moves `β̃(180°)` by 5.2 % and `β̃(135°)` by 3.1 %, while moving
the forward peak by 26 %.

**The conclusion, stated plainly because it revises R2's premise.** At matched
bulk `B_p`, nature leaves very little freedom in the backscatter hemisphere. R2
will therefore calibrate the **forward**-shape axis strongly and the backward
axis weakly — which is still a real and valuable experiment, since the forward
peak drives multiple scattering and propagates into `Rrs`, but it is not the
"calibrating the backward-VSF axis" the catalogue promised. The out-of-family
anchors and the designs that deliberately **break** `B_p` matching are what bound
the backward axis, by contrast rather than at matched `B_p`.

*Reproduce:* the FF backscatter fraction and phase function, `B_p(n, µ)` and
`β̃(ψ; n, µ)`, are closed-form (Fournier & Forand 1994; Mobley, *Light and
Water*); the tables above are computed from them.

---

## Appendix C — note to Robert Frouin (A-RF)

*Drafted for JXP to send or edit.*

> Your recommendation — use HydroLight as the reference forward model with
> particle phase-function parameters explicitly varied, and build a fast
> differentiable emulator of it — is what we are now commissioning. The
> specification is ≈157,000 runs in four batches; the phase-function axis is 27
> distinct tabulated VSFs crossed with a 63-node IOP grid and, for the first
> time, crossed with the inelastic processes as well.
>
> Two findings from the last milestone are worth putting to you directly,
> because both bear on the hybrid architecture you proposed.
>
> **First, `Rrs(model) = Rrs(ZTT) + ΔRrs(emulator)` is what we built, and it
> fails off-nadir because of the ZTT term, not the emulator.** Measured on PB24:
> 22.3 % of ZTT's predicted `rrs` are zero or negative, and the attribution is
> **68 % to `Ψ_KLu`** — whose quartic `F(ψ)` is fitted only for ψ ≳ 134° and
> crosses zero at 110.4°, while 42 % of PB24's sanctioned window sits below 134°
> — against **1 % to µ∞**. The correction is a *bounded relative* one,
> `|δ| ≤ 0.5`, so no correction of any size can repair a backbone that has gone
> negative. An oracle correction chosen with the truth in hand and clipped to the
> same bound scores 5324 % where the trained hybrid scores 5485 %: the network is
> within 3 % of the best its functional form permits. **The form is the
> limitation, not the emulator.**
>
> **Second, µ∞ cannot be refit from either dataset we hold.** `µ∞ = a/K∞` and
> `F(ψ) = K_Lu/K∞ − 1` are defined against the asymptotic `K∞`, which is
> θ_s-independent by definition; PB24 tabulates seven diffuse-attenuation
> coefficients and all seven vary ~1.4× across solar zenith, so none of them is
> `K∞`. That is what the new runs are for.
>
> Which raises the question we would like your view on. Your phrasing was ZTT
> "possibly as the physical backbone of a hybrid emulator". Given the above, we
> are specifying batch A so that its value does **not** depend on ZTT surviving:
> it is an IOP × geometry × VSF training set for a **pure** differentiable
> emulator of HydroLight, with the ZTT refit as a bonus rather than the goal.
> Does that match your intent — and would you still put ZTT at the centre, given
> that the interpretable backbone is precisely the part that breaks off-nadir?
>
> One measured caution on the phase-function axis, since it is your central
> point. At matched bulk `B_p`, we find `β̃(ψ)` in the backscatter hemisphere
> moves by only a few percent across every family we can construct — 1–3 % beyond
> 120° for Fournier–Forand at matched `B_p`, ~5 % for two-component mixtures —
> while the forward peak moves by 25–50 %. So "independent variability in
> phase-function shape" appears to be largely a *forward*-scattering freedom once
> `B_p` is fixed. If you know of measured VSF sets that break that, they would
> change the design of batch A, and we would rather hear it now than after the
> runs.

---

## Appendix D — provenance

Decisions locked in Q&A/Setup of
[`claude_prompts/RT/hydrolight_prompts.md`](../claude_prompts/RT/hydrolight_prompts.md),
2026-09-22: full HydroLight on the operator's latest build (A1); no L23 input
decks available (A2); no compute ceiling (A3, superseded by A-HD2, A-HC, A-HD1
round 2); format and precision at our discretion (A4); user-supplied inputs
accepted (A5); quad layout at our discretion (A6); decks shipped in place of
prose for the metadata contract (A7); pilot first (A8); a **usable** domain, not
a failure map (A9); switches named rather than X-labels (A10); no embargo (A11);
`µ∞` eigenvalue route in parallel (A-HD1); the VSF design axis is the last thing
cut (A-HD2); shape and `B_p` as independent knobs (A-HD3); one ensemble, families
crossed into batch B (A-HD4); L23 is homogeneous (A-HD5); R7 in-house (A-HD6);
R0 added (A-HD7); R5 rewritten (A-HD8); nothing cut (A-HC); L23 water bodies
rebuilt from their IOPs (A-R2Q1); the release supersedes L23 (A-R2Q2); deck
generator in `robust/` (A-R2Q3); note to Robert drafted (A-RF). Round-3
questions R3Q1–R3Q6 returned blank and were resolved on the recommended
defaults, as round 3 §4 committed.

Sources for the computed tables: `robust.rt.conventions.BB_W_L23` (pure-water
backscattering, 350–750 nm; extended to 330 nm by its own fitted −4.345 power
law), `ocpy.water.absorption.a_water` (pure-water absorption), and the
Fournier–Forand closed forms.
