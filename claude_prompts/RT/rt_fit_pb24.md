# Fitting ZTT to PB24 — milestone M8

This prompt doc guides the next step after M7 ("ZTT (L23 µ∞)",
`rt_fit_ztt.md`): take the elastic backbone to **PB24**, the Pitarch & Brando
HydroLight ensemble. PB24 varies the two axes L23 holds fixed: the particle
phase function (B_p spans ~6×) and the full viewing geometry (10 sun zeniths ×
10 view zeniths × 13 azimuths).

## Context

- `claude_prompts/RT/rt_fit_ztt.md`: M7's decisions (Q1–Q20), its hand-off
  ("Next (M7 hand-off)"), and its logs.
- `design/rt_elastic_implementation.md`: §7 (M5: what PB24 revealed. ZTT's
  Ψ_KLu quartic fails below ψ ≈ 134° and changes sign at 110.4°; ZTT's rrs is
  ≤ 0 on 22% of PB24's window; PB24 has no asymptotic K) and §7b (M7).
- `docs/reports/report_rt_elastic_model.md` §9 (M7 summary) and
  `design/m5_report.md`.
- `robust/rt/data/pb24.py`: the M5 loader. It reads per-realisation netCDFs
  with xarray directly. It was built for the **OLCI** files and the
  water-class sidecar, which this download does not include (see "The data as
  downloaded").

The two papers on the PB24 dataset and OC25 are in `context/RT/`:
- `pitarch2025a.pdf` : The paper by Pitarch and Brando that describes the PB24 dataset.
- `pitarch2025.pdf` : The paper by Pitarch+ that describes the OC25 model and its performance

## The data as downloaded (checked 2026-10-31)

`$OS_COLOR/SD/v5`: **5,000 files, 27 GB, all `SD_hyp_no_R_NNNN.nc`** (5.8 MB
each). Each file is one realisation:

- **Spectral grid:** 451 bands, 350–800 nm at 1 nm.
- **Geometry:** `theta_s` (10) × `theta` (10) × `phi` (13), i.e. 1,300
  geometries per realisation.
- **Constituents:** `C`, `N`, `Y` (labels; record §7.2 says they are not the
  generators).
- **IOP components on λ:** `aw`, `aph`, `ag`, `aNAP`, `bw`, `bph`, `bNAP`,
  `bbw`, `bbph`, `bbNAP`.
- **Reflectance on (λ, φ, θ, θs):** `Rrs`, `rrs` (with underwater zenith
  angles; see its `comment`) and `Q`.
- **Surface AOPs on (λ, θs):** `Kd`, `Ku`, `Ko`, `Kod`, `Kou`, `Knet`,
  `KLu`, `mu_d`, `mu_u`, `mu_tot`, `R`.
- **New relative to the M5 loader's `RAW_FIELDS`:** `Esdir_Es_ratio` on
  (λ, θs), the above-surface direct-to-total downwelling irradiance. That is
  1 − H, the diffuse fraction ZTT's µd needs. On L23, TT's fitted H(θs′, V) was
  the term that put µd 8.7% high at 60°.
- **`no_R`:** no Raman, so these runs are elastic.
- **No depth profiles.** This settles M7's Q12: on PB24, µ∞ can come only from
  the asymptotic eigenproblem on PB24's IOPs (as M7 did for L23), or from a
  closure fit.
- **Absent compared with M5:** the 5,000 `SD_OLCI_no_R_NNNN.nc` files and
  `classes_Rrs_OLCI_v5_20240214.mat`. So `robust.rt.data.pb24`'s default
  `resolution="olci"`, its water-class reader, and the suite's `needs_pb24`
  marker (which checks for `SD_OLCI_no_R_0001.nc`) all still treat PB24 as
  missing. The 26 PB24 tests keep skipping.

> **Addendum (2026-10-31, Claude Fable 5.1).** The OLCI set has since been
> fetched but not unpacked: `~/Downloads/SD_OLCI.zip` (809 MB) holds
> `v5/SD_OLCI_no_R_0001–5000.nc` and `v5/classes_Rrs_OLCI_v5_20240214.mat`,
> 1.35 GB unpacked. `$OS_COLOR/SD/v5` is still hyperspectral-only, so the
> statements above about `needs_pb24` and the skipping tests hold until it is
> unzipped (Q8). **Update, later the same day:** unpacked to `$OS_COLOR/v5/`,
> one level above where the loader looks; a move is proposed in Q16.

## Code

- Python in the `ocean14` conda environment, from the retrieve-or-bust repo
  root (`robust` is not pip-installed).
- `ocpy` is an editable install from `/mnt/tank/Oceanography/python/ocpy`,
  branch **`pb24`**. Its L23 reader `ocpy/hydrolight/loisel23.py` is the model
  for the new PB24 reader: `$OS_COLOR`-resolved paths, plain xarray, and
  docstrings that say where the data comes from.
- **JXP runs all git, in both repos.**
- Working agreements as in M7: pytest-gated tasks, suites green with and
  without the data, ruff clean in retrieve-or-bust, the `forward` freeze, and
  the hybrid pinned to its own backbone (Q2a of M7).

## Prompts

### Plan

1. Read this doc and the items in Context.  Read my first round of answers to the opening questions in Q&A/Plan.
   Then ask me additional questions if you have any.
   Provide recommendations for these, if you do ask any.
   After we are done, I will have you modify the Tasks below.
   Also be aware that we also wish to recalculate the OC25 model on PB24.  
   We will then compare with 
   Use Fable if you can. Log your work.

2. I have answered the next round of questions in Q&A/Plan.  Read my answers and continue the discussion 
   in the Q&A/Plan section below if you have any more questions to ask.
   Provide recommendations for these, if you do ask any.
   After we are done, I will have you modify the Tasks below.
   We will then compare with Use Fable if you can. Log your work.

3. I have answered the next round of questions in Q&A/Plan.  Read my answers and 
   modify the Tasks below as needed.  
   Use Fable if you can. Log your work.

### Tasks

*Each task ends with a log entry and JXP reviewing before the next.*

*Revised 2026-10-31 (Claude Fable 5.1) after Q1–Q17. The original four tasks
are now eight: old task 1 → tasks 1–4, old task 2 → tasks 5–6, old task 3 →
task 7, old task 4 → task 8. Task numbers quoted inside the Q&A above refer to
the original list.*

**Applies to every task.**
- Python in `ocean14` from the repo root; JXP runs all git in both repos;
  `ocpy` work goes on its `pb24` branch.
- Pytest-gated, and the suites are green **with and without** the data
  (`needs_pb24` plus a small committed fixture; missing data skips, never
  fails). `ruff check` and `ruff format --check` clean in retrieve-or-bust.
- The `forward` freeze (record §8.0) is in force. Appending `None`-default
  container fields and keyword-only arguments is allowed. Any pinned number
  that moves is a deliberate re-baseline, called out at the top of the log
  entry. **L23 pins and the shipped emulator digest do not move at all**; the
  hybrid stays pinned to its own backbone (M7 Q2a).
- **Directions.** PB24's `rrs(theta)` is at the in-water angle, `Rrs(theta)`
  at the above-water angle (Q9). From task 3 on, a PB24 sample is an
  above-water geometry, `Geometry.theta_v` is the above-water sensor zenith,
  and the window is θs ≤ 70°, θv_air ≤ 70°.
- Fits are deterministic (linear LSQ, no optimiser); a rerun is bit-identical;
  every fit is on the train side of a saved split and scored held out.
- Score on everything, then explain (M5 gotcha 1); write any causal story as
  a prediction and test it before believing it (M5 gotcha 4).
- After each task: update `design/rt_elastic_implementation.md` with a new
  **§7c (M8)** subsection for what was built and measured, and log here under
  **Logs**.

1. **Task 1 — The `ocpy` reader.** `ocpy/hydrolight/pb24.py` (name per Q2),
   in the style of `loisel23.py`, netCDF only (Q3):
   - path resolution from `$OS_COLOR` (`SD/v5`), with a warning when unset;
   - `resolution={"hyp", "olci"}`, default `"hyp"` (Q1, Q8);
   - `load_ds(index, resolution)` for one realisation, plain xarray;
   - `available(resolution)`: the realisation indices on disk;
   - `grids(resolution)`: the shared `lambda`, `theta_s`, `theta`, `phi`,
     read from one file and asserted identical on every later load;
   - a module docstring that records what the files are (Pitarch & Brando
     2025, Zenodo 10.5281/zenodo.11637178), the variable list, and the two
     direction conventions: `rrs`/`Q` at in-water `theta`, `Rrs` at
     above-water `theta`, φ = 0 towards the sun.
   The class `.mat` stays in `robust` (Q8). Tests in `ocpy/tests/test_pb24.py`
   skip without the data.
   **Gate:** golden values (a few `rrs`, `Rrs`, `aph`, `Esdir_Es_ratio`
   entries at named indices) against a direct `xarray` read; grids identical
   across a sample of files in both resolutions; 5000 realisations found in
   each resolution; the `ocpy` suite green with and without `$OS_COLOR`.
   Use Opus 5.5. Log your work.

2. **Task 2 — The consolidation caches, in `robust`.** A module
   `robust/rt/data/pb24_cache.py` and a builder `design/py/build_pb24_cache.py`
   that open the 5,000 hyperspectral files **once** through the task-1 reader
   and write, under `$OS_COLOR/SD/derived/pb24_v1/` (Q13):
   - (a) the **per-realisation table**: `C`, `N`, `Y`, the ten IOP components
     on λ, the eleven (λ, θs) AOPs, and `Esdir_Es_ratio` (λ, θs), one float32
     `.npy` per field (≈ 1.2 GB in all);
   - (b) the **full 4-D** `rrs`, `Rrs` and `Q`, shape
     `(5000, 10 θs, 10 θ, 13 φ, 451 λ)`, one float32 `.npy` per field,
     written and read memory-mapped (11.7 GB each);
   - a JSON sidecar with the realisation ids, the grids, the field list and
     dtypes, the source file names and sizes, the reader version, and no
     timestamp, so a rebuild is byte-identical;
   - readers that return memmaps and slice by realisation, angle and band, so
     nothing downstream opens a netCDF again.
   float32 is stored exactly as read (no float64 round trip), which is what
   makes bit-identity a property rather than a tolerance. The OLCI set is
   small enough to read per file and is **not** cached.
   **Gate:** every table field and a random 2% of the 4-D arrays
   bit-identical to per-file reads (plus all of realisations 1, 2500 and
   5000 in full); a second build is byte-identical, sidecar included; the
   builder reports sizes and wall time; tests skip without the data and run
   on a 3-realisation miniature of the cache committed as a fixture.
   Use Opus 5.5. Log your work.

3. **Task 3 — `robust.rt.data.pb24` rebuilt on the caches, in above-water
   geometry.** Rewrite the loader on top of task 2 (the `ocpy` reader behind
   it, as `l23` wraps `loisel23`), keeping `PB24Batch`, `LoadReport`,
   `select`, `make_splits` and their roles, with these changes (Q9, Q10, Q14):
   - **A sample is an above-water geometry** `(realisation, θs, θv_air, Δφ)`.
     `Rrs` is read at its node; `rrs` and `Q` are interpolated to
     θ_w = asin(sin θv_air / 1.34), linearly in θ between the in-water nodes.
     The interpolation error is *measured*, not assumed: linear against
     cubic, and the Fresnel check below.
   - **Angle sets:** `"window"` (θs ≤ 70°, θv_air ≤ 70°), `"shell"` (its
     complement in air), `"all"`, and `"in_water_only"` (the in-water nodes
     50–87.5°, un-interpolated, with `Rrs` absent), for ZTT physics only.
     `ANGLE_WINDOW_MAX` keeps its value and gains the air-angle meaning.
   - **`Geometry.diffuse_fraction`**: the `None`-default per-sample spectrum
     is appended to `Geometry` here (allowed under the freeze; the field-order
     pin is re-baselined deliberately) and filled from `1 − Esdir_Es_ratio`
     at the sample's θs (Q10a). Nothing consumes it until task 4, so every
     model is unchanged by this task.
   - **`PhaseParams.ff_B` / `ff_weight`** filled per realisation from
     `bbph/bph`, `bbNAP/bNAP` and `(bph, bNAP)`, so the L23 variant's FF path
     runs on PB24's own phase functions (Q4); `B_p` is the bulk spectrum as
     before.
   - `read_classes` from the `.mat`, as M5.
   - **Splits over all 5000 realisations**, built once by
     `design/py/make_pb24_splits.py` and saved to
     `design/ztt_pb24/splits.json` by realisation id: `SPLIT_SEED = 23`,
     `TEST_FRACTION = 0.2`, B_p band quantiles (0.4, 0.6) with the
     chlorophyll confound reported, and the geometry split in air angles
     (held out: θv_air ≥ 80° or θs ≥ 80°). `make_splits` loads the file; a
     batch whose realisations are not all in the file raises.
   - `needs_pb24` checks the hyperspectral file; the committed fixture
     (`PB24_SMALL_FIXTURE`) is regenerated under the new sample definition,
     a deliberate re-baseline of every M5 test pinned to in-water samples.
     Tests that assert data properties (B_p span, zero-`rrs` location, Gershun)
     keep asserting them.
   **Gate:** golden values against raw reads, including an interpolated
   `rrs` recomputed by hand; the **Fresnel check** on every realisation
   loaded, `Rrs(θ_air)/rrs(θ_w)` normalised at nadir within 5% of
   T(θ_air)/T(0) for θ_air ≤ 70° (distribution reported); window and shell
   counts asserted; splits disjoint, non-empty on each side, and identical on
   reload; suite green with and without the data.
   Use Opus 5.5. Log your work.

4. **Task 4 — Conventions: the view angle, the diffuse fraction, the surface
   transfer.** Three changes to the model code, each bit-identical at nadir
   so no L23 pin moves (Q9a, Q9c, Q10):
   - **`geometry_to_paper_angles` refracts `theta_v`** (above-water → in-water)
     before forming ψ and the paper's θv; `Geometry.theta_v`'s docstring
     says "above-water". Every PB24 pin that encoded in-water view angles is
     re-baselined and listed in the log.
   - **`Geometry.diffuse_fraction`** (`None`-default). `Md_plus` and `mu_d`
     take `H=` and use it in place of `diffuse_fraction(θs′, V)` when given;
     `rrs_ZTT` passes the geometry's field through. `None` is TT's H, bit for
     bit.
   - **A Fresnel-based surface transfer.** `conventions.fresnel_transmittance`
     (unpolarised, n = 1.34) and a `FresnelTransfer` with
     `Rrs = A₀ T(θv)/T(0) · rrs / (1 − B rrs)`, two numbers fitted on the
     realisation split's training rows by `design/py/fit_surface_fresnel.py`
     (deterministic, `--check`), shipped as constants with provenance, and
     accepted wherever `SurfaceTransfer` is today. `default_transfer()` and
     `surface_pb24.npz` stay loadable, documented as superseded.
   - Dated correction notes in record §7.10 and §7.12 (the pairing, the
     "7.2×" claim, O25's M5 numbers), not rewrites.
   **Gate:** L23 value pins, the `mode="ztt"` pin and the emulator SHA-256
   unchanged; `theta_v = 0` bit-identical before and after; float64 gradient
   vs FD through the refraction and through `H`; the Fresnel transfer's
   held-out PB24 error (window and shell, per θv) reported beside M5's table
   evaluated on the *correctly paired* data; a test that fails if the
   transfer is ever fitted on held-out rows.
   Use Opus 5.5. Log your work.

5. **Task 5 — O25: reproduce the published LUTs, and score the published
   model.** (Q11 i–ii, Q17a.)
   - `baselines.o25_published_table(path=None)` reads the four
     `Data/G{0,1}{w,p}.txt` files from the O25 checkout (`$O25_DIR`, default
     `../O25` beside the repo) into an `O25Table` on PB24's nodes (13 azimuth
     blocks × 10 θs × 10 θv). Nothing from that GPL-3.0 repository is
     committed; tests skip when it is absent.
   - `fit_o25_table` gains the paper's recipe as options: unweighted,
     `rrs_max=0.06` applied in `Rrs`, every node fitted on all samples at that
     node.
   - `design/py/reproduce_o25.py`: the OLCI set, all 5000 realisations, all
     1300 geometries, the paper's recipe; writes the refit LUT and a CSV of
     relative differences to the published G's per node and per coefficient,
     with summary quantiles in and outside the window.
   - **Prediction, written before the fit:** the four G's agree with the
     published files to better than 1% at every in-window node, with the
     residual traceable to float32 storage and the `Rrs ≤ 0.06` cut. A larger
     disagreement means our zenith or azimuth convention differs from theirs,
     and is diagnosed (by azimuth mirror and by zenith axis swap) before
     anything else is scored.
   **Gate:** reproducible; the comparison CSV committed; the prediction's
   outcome stated plainly either way; `Rrs_o25` with the published table
   evaluated at a published node reproduces the LUT's own polynomial to
   float64 rounding.
   Use Opus 5.5. Log your work.

6. **Task 6 — Baseline: every elastic model we have, scored on PB24.**
   `design/py/run_pb24_baseline.py`, writing to `design/ztt_pb24/`
   (CSV + markdown + figures, dataviz conventions), on the saved splits, all
   5000 realisations at every in-window geometry and all 451 bands, with
   `--quick` for development and the sample named in every artefact (Q13).
   Models:
   - standard Gordon;
   - **O25 published** (true IOPs in, published LUT, labelled "calibrated on
     all 5000"), and **O25 form, refit on PB24** on each split's training
     rows, relatively weighted as M5 (the paper's unweighted objective
     reported once beside it);
   - ZTT (TT2017 µ∞), ZTT (L23 µ∞), the L23 closure variant; each with
     Sullivan and with PB24's own FF phase function where the variant allows;
   - the L23-trained hybrid, with and without its out-of-domain fallback
     (751–800 nm is out of domain by construction and reported as such).
   **Spaces and cuts (Q12):** each model in its native space (`rrs` at θ_w
   for Gordon and ZTT; `Rrs` at θ_air for O25), plus one common table in `rrs`
   through the task-4 transfer with the transfer's own error beside it.
   Cuts: the window, the shell, ψ ≥ 134°, the in-water-only set (ZTT and
   Gordon only), and everything; per λ, θs, θv_air, ψ bin, B_p bin and water
   class; held out and train for each split kind.
   **Diagnostics**, also on PB24's ten sun angles:
   - **µd**, three ways against PB24's `mu_d` per λ and θs: TT's H (V = 15
     and 40 km), the data H, and the data H with M7's c(θs′);
   - **Ψ_KLu/µ∞ against `KLu/a`.** PB24's `KLu` is for nadir-viewing radiance,
     so this is nadir only, ψ = 180° − θs_w over ten suns (180° to 131.8°),
     with both µ∞ variants and the M7 oracle rung;
   - **eigen µ∞** (Q15) on every (realisation, λ) with PB24's per-realisation
     FF pair (kernels on a B grid, interpolation error measured), cached
     under `derived/`; the spread of µ∞ at fixed (bb/a, η_bb) across B_p, and
     each Eq. (8) variant against it;
   - the ψ distribution in the window and the **share of non-physical rrs**
     per model, which tests the Q9d predictions (ψ_min ≈ 91°; far below 22%);
   - `bb_w` from the files against `conventions.bb_w`.
   **Gate:** reproducible (a rerun is byte-identical); every number traced to
   a committed CSV; headers derived from the same labels as the values (M5
   task 6); nothing thresholded. This task measures.
   Use Opus 5.5. Log your work.

7. **Task 7 — Discuss how to improve the elastic backbone.** No code. From
   task 6's numbers, a design discussion in Q&A/Design covering at least:
   - (i) µd's atmospheric factor: the shipped default H(θs′, λ) fitted on
     PB24's `Esdir_Es_ratio` (Q10b) behind the per-sample seam, and whether
     `M*_d` needs refitting beyond bb/a = 0.1;
   - (ii) µ∞ from the eigenproblem on PB24's IOPs and phase functions, with
     or without the B_p dependence Eq. (8) lacks (M7 Q5): refit Eq. (8), add
     a B term, or a small differentiable emulator of the eigen µ∞;
   - (iii) a Ψ_KLu valid over the window's ψ range (≈ 91°–180°), refitted
     **jointly with f_L**; M7 showed changing Ψ_KLu alone breaks the balance,
     and PB24 measures `KLu` only at nadir, so off-nadir Ψ_KLu is constrained
     through `rrs` alone;
   - (iv) whether ZTT's structure is worth keeping off-nadir, against an
     alternative backbone with an explicit VSF, judged on task 6's numbers
     and on the published O25's;
   - (v) what this means for the emulator retrain (M7 Q2b/Q18), including
     review finding 1b (an "l23" emulator should warn when `ff_B` is absent).
   Each option gets a prediction written down before any test (M5 gotcha 4).
   Pose numbered questions; JXP chooses.
   Use Opus 5.5. Log your work.

8. **Task 8 — Implement the chosen backbone and assess it on PB24.** Scope
   per task 7's decisions, following M7's pattern:
   - a deterministic fit script `design/py/fit_ztt_pb24.py` with `--check`,
     fitting on the realisation split's training rows;
   - coefficients shipped as constants with provenance, including the
     PB24-fitted default H(θs′, λ);
   - a new `variant` on `rrs_ZTT` (e.g. `"pb24"`), leaving `"l23"` and
     `"tt2017"` bit for bit;
   - an additive ladder with order-independent attribution, run through
     task 6's script so the numbers are comparable.
   **Acceptance (proposed; confirm in task 7):** the new backbone beats
   ZTT (L23 µ∞) on PB24 held-out realisations, inside the window *and* on
   everything; the share of non-physical rrs falls; L23 numbers are reported
   too, so a PB24 gain that costs L23 shows. Then a notebook
   (`notebooks/RT/rt_fit_ztt_pb24.ipynb`), a review pass over the M8 diff, the
   record's §7c completed, a report update (`reports/`, regenerated into
   `docs/`), and a hand-off that names the emulator retrain.
   Use Opus 5.5. Log your work.


## Q&A

### Plan

*Opening questions from Claude (Opus 5.5), 2026-10-31. Answer inline under
each **JXP:** line.*

**Q1. Hyperspectral only, or the OLCI files too?** M5 trained and scored on
the OLCI set (12 bands, 400–753 nm). This download is hyperspectral only (451
bands, 350–800 nm), without the water-class `.mat`. I suggest hyperspectral as
primary: it is what PACE wants. Band subsets can be drawn from it, and the
consolidation layer makes that cheap. The cost: M5's OLCI numbers and the
class-based reporting can't be reproduced unless the OLCI files and the
`.mat` are fetched too. Fetch them, or retire the OLCI path?
**JXP:** I have fetched them.

**Q2. Where does `ocpy` stop and `robust` start?** My proposal mirrors L23:
- **`ocpy`** holds the format-level reader and the consolidated caches (no
  splits, no model types, no JAX).
- **`robust.rt.data.pb24`** keeps batches, splits, the angle window and the
  `LoadReport`, rebuilt on top of `ocpy`.

Module name: `ocpy/hydrolight/pitarch24.py` (by author and year, as
`loisel23.py`) or `pb24.py`? And should the cache-building code live in `ocpy`
(reusable by others) or in `robust`?
**JXP:** Your split is good.  Use `pb24.py`.  Put the cache-building code in `robust`.

**Q3. Cache format and location.** The full 4-D `rrs` is ~12 GB per field,
so caches are subsets. I propose:
- `$OS_COLOR/SD/derived/` next to the data, as M7 did for L23.
- **npz** for the IOP/AOP table (simple, bit-reproducible, ~0.2 GB).
- **zarr** (chunked by realisation) for the angle/band reflectance subsets, so
  one can read a slab without loading 12 GB.

zarr would be a new dependency for `ocpy`. Fine, or prefer netCDF/npz
throughout? Rclone to `RoB` as before?
**JXP:** For `ocpy`, only work on the NetCDF files.  Put the cache-building code in `robust`.
Defer the `rclone` to `RoB` for now. 

**Q4. PB24's phase functions — I need the paper.** The µ∞ eigenproblem (and
any explicit-VSF backbone) needs PB24's particle phase functions exactly. For
L23 they were Fournier–Forand with fixed b̃b per component. M5 measured PB24's
`bbph/bph` varying 30× across realisations (0.0010–0.0358) and `bbNAP/bNAP`
between 0.010 and 0.020. Are these FF, with which (n, µ) convention, or
another family? Please put the PB24 dataset paper (Pitarch & Brando) in
`context/RT/`, and the O25 paper if not already there.
**JXP:**  I have added both papers and name them in Context.

**Q5. Spectral grid for scoring.** Our canonical grid is L23's 350–750 nm at
5 nm (81 bands); PB24 runs to 800 nm at 1 nm. Score on PB24's own 451 bands
(ZTT's f_L table covers 350–800), on the canonical 81 for comparability with
M7, or both?
**JXP:** Use PB24's bands

**Q6. Geometry scope.** M5 sanctioned 0–70° in both zeniths (M5 Q14) and held
the 80/87.75° shell out as an extrapolation test. Keep that window for M8's
gates? Should a ψ ≥ 134° subset be a first-class reporting cut, given that's
where ZTT's Ψ_KLu is fitted?
**JXP:** Yes, keep that window, and do a ψ ≥ 134° subset as a first-class reporting cut.

**Q7. Splits.** M5 used three: realisation, B_p band, and geometry. A
reminder from M5's gotchas: refit per split, and a seed doesn't name a
partition when the realisation set changes. Keep all three, with task 2 fixing
the partitions once (saved by realisation id) so every later task scores on
identical sets?
**JXP:** Yes, do so.

#### Round 2 — Claude (Fable 5.1), 2026-10-31, after JXP's answers to Q1–Q7

**Settled by your answers.** Hyperspectral is primary and PB24's own 451
bands are the scoring grid (Q1, Q5); the OLCI set is fetched (Q1; it is still
zipped, Q8). `ocpy/hydrolight/pb24.py` reads the netCDFs and nothing else; the
consolidation caches and their builders live in `robust` (Q2, Q3). rclone to
`RoB` is deferred (Q3). Both papers are in `context/RT/` and read (Q4). The
0–70° window stays and ψ ≥ 134° is a first-class cut (Q6). Task 2 fixes all
three partitions once, by realisation id (Q7). Also noted: O25 is to be
recalculated on PB24 (Q11).

##### Read, and measured before asking

*From the two papers.*

- **Phase functions (closes Q4).** PB24 is Fournier–Forand throughout,
  selected from the backscatter ratio with Mobley et al. (2002)'s relation.
  That is the convention M7 *assumed* for L23 (M7 Q8); for PB24 it is stated
  (Pitarch & Brando 2025 §1.1, Table 1). B_ph ~ N(µ, σ) with
  µ = 0.002 + 0.008·exp(−0.56 log₁₀C), floored at 0.001; B_NAP ~ U(0.01, 0.02);
  both constant in λ within a realisation. So `robust.rt.asymptotic` transfers
  with a per-realisation (B_ph, B_NAP) pair in place of L23's fixed one.
- **The sky.** HydroLight 5.1.2; HCNRAD sky radiances; RADTRANX irradiances
  with **40 km visibility** (ZTT's `DEFAULT_VISIBILITY_KM` is 15); US Navy
  aerosol (air mass 5, RH 80%, 2.5 cm water); ozone 354.9 DU; wind 5 m/s;
  T = 20 °C, S = 35; infinitely deep. The K's are finite differences between
  0 and 1 cm. Azimuth φ = 0 is towards the sun.
- **Water.** Absorption from WOPP shifted to S = 35; scattering from Zhang &
  Hu (2021). To be checked against `conventions.bb_w` before any fit.
- **O25 as published** (Pitarch et al. 2025 §2.2): the four G coefficients
  are LUTs over (θs, θv, Δφ) on the full 10 × 10 × 13 grid, fitted by
  **unweighted** LSQ to the **OLCI** set, 12 bands × 5000 realisations =
  60,000 points per geometry, restricted to Rrs ≤ 0.06 sr⁻¹. Around it sit
  anw(560) from χ (a cubic, Table 1), γ from a log band ratio (eq. 8,
  Table 3), and a QAA-style inversion that makes the BRDF correction. Code:
  github.com/jaipipor/O25 (stand-alone), HyperCP, ThoMaS. The published O25
  was calibrated on all 5000 realisations, so on any PB24 split it has seen
  the test set.

*From the files (scratch scripts, not committed).*

1. **`rrs` and `Rrs` are tabulated in different directions, and M5 paired
   them wrongly.** The `rrs` variable's `comment` says its zenith angles are
   underwater and that Snell's law must be applied to align it with `Rrs`.
   That is HydroLight's layout: the in-water quads carry `theta` as the
   in-water propagation angle, the above-water quads carry the same nominal
   `theta` in air, and in-water directions beyond the critical angle (48.3°
   for n = 1.34, i.e. the nodes 50–87.5°) reach no above-water direction at
   all. M5 concluded the opposite (record §7.10: "`rrs × Q = R` to 0.00%
   settles it: same direction"). That test cannot settle it: `Q` is tabulated
   per in-water direction too, so `rrs·Q = (Lu/Ed)(Eu/Lu) = R` is an identity
   in any direction. Measured instead, on three realisations, two bands and
   two (θs, φ): `Rrs(θ_air)/rrs(θ_w)` normalised at nadir, with `rrs`
   interpolated to θ_w = asin(sin θ_air / 1.34), against the flat-surface
   Fresnel transmittance. One slice (560 nm, θs = 60°, φ = 135°; files
   1 / 777 / 4321):

   | θ_air | 10° | 30° | 50° | 60° | 70° |
   |---|---|---|---|---|---|
   | Fresnel T(θ)/T(0) | 1.000 | 0.999 | 0.986 | 0.959 | 0.883 |
   | **refracted pairing** | 1.006 / 0.998 / 1.005 | 0.994 / 1.002 / 0.998 | 0.967 / 0.971 / 0.969 | 0.935 / 0.937 / 0.933 | 0.864 / 0.860 / 0.856 |
   | nominal pairing (M5) | 0.994 / 1.007 / 0.998 | 0.898 / 0.988 / 0.943 | 0.714 / 0.829 / 0.828 | 0.591 / 0.681 / 0.740 | 0.466 / 0.501 / 0.630 |

   With the refracted pairing the ratio is IOP-independent to 1% and within
   3% of Fresnel out to 70° (the other slices agree; at 80° and 87.5° it
   departs, where a 5 m/s rough surface is not flat and the interpolation
   between the 40° and 50° nodes is coarse). With the nominal pairing it
   varies by 35% across three realisations at 70°. The pairing is Snell's and
   the ratio is almost entirely surface physics. Consequences in Q9.

2. **TT's diffuse fraction does not describe PB24's sky.** `1 − Esdir_Es_ratio`
   against ZTT's `diffuse_fraction` (Equation 16), realisation 1 (the field is
   realisation-independent, as an above-surface quantity should be):

   | θs | 0° | 30° | 60° | 70° | 80° |
   |---|---|---|---|---|---|
   | PB24, 443 nm | 0.197 | 0.222 | 0.349 | 0.470 | 0.743 |
   | PB24, 550 nm | 0.124 | 0.140 | 0.219 | 0.294 | 0.487 |
   | PB24, 700 nm | 0.084 | 0.094 | 0.147 | 0.197 | 0.326 |
   | TT H, V = 15 km (ZTT default) | 0.272 | 0.307 | 0.451 | 0.576 | 0.800 |
   | TT H, V = 40 km (PB24's sky) | 0.276 | 0.306 | 0.456 | 0.522 | 0.636 |

   TT's H is spectrally flat and about twice PB24's at 550 nm; PB24's falls
   2.3× from 443 to 700 nm, as a Rayleigh-dominated sky must. M7's c(θs′)
   absorbed the same mismatch on L23 (under V = 15 km). See Q10.

3. **Machine.** 24 cores, 755 GB RAM, 8.4 TB free on the data volume. The
   caches can be generous (Q13).

##### Questions, round 2

**Q8. Unpack the OLCI set?** It is in `~/Downloads/SD_OLCI.zip`, not in
`$OS_COLOR/SD/v5`. The zip carries the `v5/` prefix, so
`unzip -n ~/Downloads/SD_OLCI.zip -d $OS_COLOR/SD` puts the 5,000 OLCI files
and the class `.mat` where M5's loader and `needs_pb24` expect them (1.35 GB),
and the 26 skipping tests run again with no code change. May I do that at the
start of task 1, or will you? Two follow-ons, where I recommend "yes" to both:
the `ocpy` reader takes `resolution={"hyp", "olci"}` with "hyp" the default;
and the `.mat` class reader stays in `robust` (Q3: `ocpy` reads netCDF only).
**JXP:** I have unpacked it into `$OS_COLOR/SD/v5`.

**Q9. Adopt Snell pairing, and express the geometry in above-water terms.**
Finding 1 touches four things; I recommend all four.
- (a) **The convention.** `Geometry.theta_v` is documented as "sensor zenith";
  `ztt.geometry_to_paper_angles` uses it as the *in-water* angle (no
  refraction), which is what M5's loader fed it from the `rrs` grid. Define
  `theta_v` as the **above-water** sensor zenith and refract it inside
  `geometry_to_paper_angles`, as `theta_s` already is. Bit-identical at nadir,
  so no L23 pin moves; M5's PB24-only pins move and are re-baselined
  deliberately, under the freeze's rule.
- (b) **The loader.** A sample becomes an above-water geometry: `Rrs` read at
  its node, `rrs` interpolated to θ_w (the 40° and 50° in-water nodes bracket
  air angles 59.5–90°, so the interpolation is local; its error is measured in
  task 1 against the Fresnel check above). The in-water nodes 50–87.5° stay
  available as an `"in_water_only"` angle set for ZTT physics; they are not
  remote-sensing geometries.
- (c) **The surface transfer.** M5's 1300-cell table (`surface_pb24.npz`,
  record §7.10, "7.2× better") was fitted across mismatched directions; the
  fall of A from 0.53 to 0.34 is Fresnel transmittance plus refraction, not a
  property of the water. Replace it with the physical form
  `Rrs = A(θv) rrs / (1 − B rrs)`, `A(θv) = A₀ T(θv)/T(0)`, one or two fitted
  numbers measured against PB24; keep the table loadable for the record.
  O25's M5 numbers (6.5%) were scored through the table and are redone.
- (d) **The window.** Q6's 0–70° becomes θs ≤ 70° and θv_air ≤ 70° (θ_w ≤
  44.5°); the shell is 80–87.5° in air. Predictions, written before task 2
  measures them: the minimum ψ in that window is ≈ 91° (44° on the in-water
  grid), so Ψ_KLu's sign flip below 110.4° stays in scope but on a far smaller
  share than M5's 16%, and ZTT's non-physical share falls well below 22%.
Task 1 gates (b) by the Fresnel check (refracted ratio within 5% of T(θ)/T(0)
at θ_air ≤ 70° on every realisation it loads). Record §7.10 and §7.12 get a
dated correction note, not a rewrite.

**JXP:** Ok, proceed with all 4

**Q10. µd's atmospheric factor: H from the data, or refit?** Finding 2 says
`M⁺_d` has been fed the wrong H on both datasets. Options: (a) make the diffuse
fraction an **input**, H(λ, θs′), through a `None`-default `Geometry` field
(appending is allowed under the freeze; `Geometry.Ed` is the precedent), filled
by the PB24 loader from `Esdir_Es_ratio`, with TT's H(θs′, V) as the fallback;
(b) refit a spectral H(θs′, λ) on PB24 and ship it as the default, which ties
the backbone to one sky; (c) both: the seam from (a), a PB24-fitted default from
(b). I recommend (c). Task 2's µd diagnostic then becomes a three-way
comparison per λ and θs: PB24 `mu_d` against ZTT µd with TT's H, with the data
H, and with the data H plus M7's c(θs′). Prediction: the data H removes most of
the θs trend; what remains is `M*_d`'s bb/a extrapolation (record §7.16,
−1.02 at bb/a > 1.36). Is a per-sample H input acceptable for the forward API?
For PACE the atmospheric correction supplies it.

**JXP:** Ok, do both.  And, a per-sample H input is acceptable for the forward API.

**Q11. "Recalculate O25 on PB24": which of these, compared with what?** Your
note ends mid-sentence ("We will then compare with"). Three readings:
- (i) **Reproduce the published G LUTs** from the OLCI set with the paper's
  recipe (all 1300 geometries, all 5000 realisations, unweighted, Rrs ≤ 0.06)
  and compare with the published LUTs. This validates the reader, the azimuth
  convention and the fitter in one shot, before anything else is scored.
- (ii) **Score the published O25** (true IOPs in, Rrs out, published LUTs) on
  PB24 as the benchmark, labelled "calibrated on all 5000"; and our refit on
  each split's train set, as M5 did.
- (iii) **Implement the full BRDF correction** (χ → anw(560), γ, inversion,
  normalisation to θs = θv = 0) and compare corrected Rrs with PB24's. Only
  needed if the comparison is between BRDF corrections, not forward models.
I recommend (i) and (ii) in task 2, (iii) deferred. Both need the published
LUTs: please clone `https://github.com/jaipipor/O25` into `context/RT/O25/`
(or allow me to fetch it). And which comparison did you mean: O25 against the
new backbone, against the hybrid, or against our refit?

**JXP:** Yes, do (i) and (ii).  I have put the Repo in `Oceanography/python/O25`.

**Q12. Which space, and which cuts, for task 2's tables?** With Q9: each model
in its native space (ZTT and Gordon in `rrs` at θ_w; O25 in `Rrs` at θ_air),
plus one common table in `rrs` through the Q9c transfer, with the transfer's
own PB24 error reported beside it. Cuts: the Q9d window, the shell, the
ψ ≥ 134° subset, and the in-water-only set reported separately; per λ, θs,
θv, ψ bin and B_p bin as written. One note: the L23 emulator's `wave_nm`
feature stops at 750 nm, so 751–800 nm is out of domain for the hybrid by
construction and is reported as such. OK?

**JXP:** Yes, that is ok.

**Q13. Cache format and scope.** Q3 placed the builders; the format is still
open. Recommend: no zarr; one float32 `.npy` per field, memory-mappable, under
`$OS_COLOR/SD/derived/`, with a JSON sidecar (realisation ids, grids, field
list, source file names and sizes, version). float32 is stored as read, so a
rebuild is bit-identical by construction. Scope: (a) the per-realisation
table, with IOPs, labels, the eleven (λ, θs) AOPs and `Esdir_Es_ratio`, is
≈ 1.2 GB (the 0.1–0.3 GB above was low); (b) given 8.4 TB free and 755 GB RAM,
cache the **full 4-D** `rrs`, `Rrs` and `Q` for all 5000 realisations
(11.7 GB each, 35 GB total) rather than subsets, so every later subset is a
slice and the 5000 files are opened once. OK to spend 35 GB? And for task 2:
all 5000 realisations at every in-window geometry and band (≈ 1.8 × 10⁹
samples per model, minutes per model on 24 cores), or a stated subsample?

**JXP:** Ok, that is fine for now

**Q14. Splits.** Over all 5000 realisations, saved to
`design/ztt_pb24/splits.json` by id; `SPLIT_SEED = 23` and `TEST_FRACTION =
0.2` as M5; B_p band quantiles (0.4, 0.6) with the chlorophyll confound
reported; the geometry split redefined in air angles (hold out θv_air ≥ 80°
and θs ≥ 80°). OK?

**JXP:** Those splits look good

**Q15. Eigen µ∞ on PB24 already in task 2?** The eigensolver needs the FF
components per realisation, which PB24 states exactly. Cost ≈ 2.3 M solves,
about 30 core-minutes, less with kernels built on a B grid and interpolated.
I'd run it in task 2 as a diagnostic so that task 3 (ii) starts from numbers:
the spread of µ∞ at fixed (bb/a, η_bb) across B_p decides whether Eq. (8)'s
two-variable form can survive. Prediction: several per cent across
B_p 0.001–0.036, well above the 0.21% fit error on L23. Fine to add?

**JXP:** Yes, add that.

#### Round 3 — Claude (Fable 5.1), 2026-10-31, after JXP's answers to Q8–Q15

**Settled by your answers.** Snell pairing, the above-water `theta_v`
convention, a Fresnel-based surface transfer and the window in air angles
(Q9, all four). The diffuse fraction becomes a per-sample input to the forward
API with a PB24-fitted default (Q10c). O25: reproduce the published LUTs from
the OLCI set and score the published model as the benchmark; the full BRDF
correction is deferred (Q11 i–ii). Native-space scoring plus one common table
(Q12). Memory-mapped float32 `.npy` caches, the full 4-D reflectances
included (Q13). Splits over all 5000 realisations, geometry split in air
angles (Q14). The eigen µ∞ runs in task 2 as a diagnostic (Q15).

**Checked after your answers.**

- **The OLCI set is on disk, but one level up.** `$OS_COLOR/v5/` holds the
  5,000 `SD_OLCI_no_R_NNNN.nc` and the class `.mat` (862 MB). The loader,
  the `needs_pb24` marker and the hyperspectral set expect `$OS_COLOR/SD/v5/`,
  which still holds only the hyperspectral files. See Q16.
- **The O25 checkout** (`~/Oceanography/python/O25`, commit 9978fad) carries
  the four G LUTs as `Data/G{0,1}{w,p}.txt`, 130 rows × 10 columns each: 13
  azimuth blocks (0–180° by 15°) of 10 θs rows × 10 θv columns, on PB24's own
  nodes (the code labels θs = 87.5 where the files say 87.75). Its χ cubic
  and γ coefficients match the paper's Tables 1 and 3 exactly, and its water
  table is WOPP absorption plus Zhang scattering at 20 °C / 35 PSU, the same
  inputs PB24 used, so no water-IOP convention question remains: the PB24
  files carry `aw`, `bw`, `bbw` per realisation and those are what every model
  will be fed. The repository is **GPL-3.0**; retrieve-or-bust is BSD-3. See
  Q17.
- **The OLCI files** carry the same variables as the hyperspectral ones,
  including `Esdir_Es_ratio`, on the 12 OLCI bands 400–753 nm.

**Defaults I will take unless you object** (no question needed):

- Task 2 scores all 5000 realisations at every in-window geometry and band,
  with a `--quick` subsample for development. Every artefact names its sample.
- The diffuse-fraction seam is a `None`-default `Geometry` field,
  `diffuse_fraction`, shaped like a spectrum per sample; `None` falls back to
  TT's H(θs′, V).
- Reproducing O25 (Q11 i) follows the paper's recipe to the letter: the OLCI
  set, all 5000 realisations, each of the 1300 geometries fitted separately,
  unweighted least squares, samples with Rrs > 0.06 sr⁻¹ dropped. Prediction,
  written before the fit: the four G's agree with the published files to
  better than 1% at every in-window node, and the residual differences trace
  to float32 storage and the Rrs cut. A larger disagreement means our azimuth
  or zenith convention differs from theirs, which is exactly what this step
  is for.
- Scoring the published O25 uses the true IOPs from the files and the
  published LUT, in `Rrs`, and is labelled "calibrated on all 5000" in every
  table. The split-wise refit keeps M5's relatively-weighted objective and
  is labelled as such; the paper's unweighted objective is reported beside
  it once, so the choice of objective is visible.

##### Questions, round 3

**Q16. Move the OLCI set into `$OS_COLOR/SD/v5`?** The zip unpacked to
`$OS_COLOR/v5/` (it carries a `v5/` prefix). Moving its 5,001 files down one
level, `mv $OS_COLOR/v5/* $OS_COLOR/SD/v5/ && rmdir $OS_COLOR/v5`, puts both
resolutions and the class sidecar where M5's loader, `needs_pb24` and the new
`ocpy` reader (one directory for the release, Q2) expect them. I recommend
that over teaching the readers two locations. May I do the move at the start
of task 1, or will you?

**JXP:** I moved them

**Q17. The published O25 LUTs are GPL-3.0: reference them, or vendor them?**
Options: (a) `robust` reads the four text files from the checkout, at a path
given by `$O25_DIR` (default `../O25` beside the repo), with tests that skip
when it is absent, and only the *comparison* numbers (our refit against the
published values) are committed; (b) copy the four files into
`robust/rt/files/` with their licence, which puts GPL material in a BSD-3
repository; (c) commit only our own refit LUT, which is our work on open data,
and keep the published LUT as an external reference. I recommend (a) now and
(c) for anything the package ships, so nothing GPL enters the repository and
the published model stays scorable on any machine with the checkout. OK?

**JXP:** Go with (a) for now

## Logs

### 2026-10-31 — Plan: doc created (Claude Opus 5.5)

Created from `rt_fit_ztt.md` task 8.
- **Checked the download:** 5,000 hyperspectral files (27 GB), no OLCI files,
  no class sidecar, no depth profiles, no Raman. There is a new
  `Esdir_Es_ratio` field.
- **Checked `ocpy`:** branch `pb24`, editable install; `loisel23.py` is the
  pattern to follow.
- **Wrote:** four tasks (the `ocpy` reader plus consolidation, the PB24
  baseline, the backbone discussion, implement and assess) and seven opening
  questions. No code written.

### 2026-10-31 — Plan prompt #1, round 2 (Claude Fable 5.1)

- Read: this doc and JXP's Q1–Q7 answers; `rt_fit_ztt.md` (Q1–Q20, the
  hand-off, the logs); record §7 and §7b; report §9; `m5_report.md`;
  `robust/rt/data/pb24.py`, `baselines.py` (O25), `ztt.py` (geometry, F_psi,
  f_L, H), `asymptotic.py`, `phase.py`; `ocpy/hydrolight/loisel23.py`; and
  both Pitarch papers in full.
- Found the OLCI set zipped in `~/Downloads/SD_OLCI.zip`, not in
  `$OS_COLOR/SD/v5` (addendum under "The data as downloaded"; Q8).
- Two scratch measurements, not committed: PB24's `rrs` is tabulated at
  in-water angles and pairs with `Rrs` through Snell, which contradicts record
  §7.10 and affects M5's surface transfer, its O25 numbers and the window
  (Q9); TT's diffuse fraction is about twice PB24's and spectrally flat where
  PB24's is not (Q10).
- From the papers: PB24's phase functions are Fournier–Forand through
  Mobley 2002 with per-realisation B_ph and B_NAP (closes Q4); the sky has
  40 km visibility; O25's calibration recipe and code location (Q11).
- Wrote Round 2 (Q8–Q15) under Q&A/Plan, each with a recommendation. No
  repository code changed; `ocpy` untouched.

### 2026-10-31 — Plan prompt #2, round 3 (Claude Fable 5.1)

- Read JXP's answers to Q8–Q15: all recommendations accepted (Q9 all four,
  Q10 both, Q11 i and ii with the O25 repo at `~/Oceanography/python/O25`,
  Q12–Q15 yes).
- Checked the disk: the OLCI set unpacked to `$OS_COLOR/v5/`, not
  `$OS_COLOR/SD/v5/` (addendum updated; Q16).
- Read the O25 checkout: LUT layout 13 azimuth blocks × 10 θs × 10 θv; χ and
  γ coefficients match the paper; water table matches PB24's inputs; licence
  GPL-3.0 against our BSD-3 (Q17).
- Wrote Round 3: the settled list, four defaults I will take unless told
  otherwise (full-sample scoring with `--quick`, the `Geometry.diffuse_fraction`
  seam, the paper-exact O25 recipe with a written prediction, labelling of
  the published and refit O25), and Q16–Q17. No repository code changed;
  `ocpy` untouched.

### 2026-10-31 — Plan prompt #3: the Tasks revised (Claude Fable 5.1)

- Read JXP's answers to Q16 (the OLCI set and class sidecar are now in
  `$OS_COLOR/SD/v5`, verified: 5,000 + 5,000 files and the `.mat`) and Q17
  (option (a): the published O25 LUTs are read from the checkout, nothing
  GPL is committed).
- Rewrote **Tasks** from four to eight, folding in Q1–Q17: the `ocpy`
  reader (1); the memory-mapped caches in `robust` (2); the loader rebuilt in
  above-water geometry with Snell pairing, the diffuse-fraction and FF
  fields, and splits over all 5000 (3); the view-angle convention, the
  `Geometry.diffuse_fraction` seam and a Fresnel-based surface transfer (4);
  the O25 reproduction and published-model scoring with a written prediction
  (5); the baseline with native-space scoring, the cuts, and the µd, Ψ_KLu,
  eigen-µ∞ and non-physical-share diagnostics (6); the discussion (7); the
  implementation (8). Added an "Applies to every task" preamble and a note
  mapping old task numbers to new ones. Model lines now read "Use Fable if
  you can", as asked.
- Two corrections to the original task text while revising: PB24's `KLu` is
  for nadir-viewing radiance, so the Ψ_KLu check is nadir-only over ten suns
  (not "off-nadir"); and the window's ψ range is ≈ 91°–180° under Q9d, not
  "≥ 44°".
- No repository code changed; `ocpy` untouched. Ready to start task 1 on
  instruction.

### 2026-10-31 — Task 1: the ocpy reader (Claude Opus 5.5)

- Read: this doc in full; `ocpy/hydrolight/loisel23.py`; the skip patterns
  in `ocpy/tests/test_spectra.py` and `test_pangaea.py`; one hyperspectral
  and one OLCI file opened with xarray (variable names, dims, dtypes,
  attributes) before coding.
- **Built** (ocpy, branch `pb24`, untracked new files):
  `ocpy/hydrolight/pb24.py` with `pb24_path` (`$OS_COLOR/SD/v5`, warning
  when unset), `RESOLUTIONS`, `file_path`, `available`, `grids`,
  `check_grids`, `load_ds(index, resolution="hyp", engine="netcdf4",
  verify_grids=True)`, `READER_VERSION = 1`, `GRID_NAMES`,
  `N_REALISATIONS`; module docstring with the files, the variables and dims,
  the direction conventions and the citation. Tests:
  `ocpy/tests/test_pb24.py` (18 tests; 3 data-free).
- **Decisions the task left open.** (i) `load_ds` is **eager**
  (`xarray.load_dataset`, file closed), as `loisel23`. (ii) Engine
  **`netcdf4`**, not `h5netcdf`: identical data, but `h5netcdf` mangles the
  "º" in six attributes into surrogate escapes and is ~2× slower (0.13 vs
  0.056 s per hyp file); `engine=` stays a keyword. (iii) The grid check is
  on by default and compares dtype and values exactly; `verify_grids=False`
  skips it. (iv) Grids are cached per process from the lowest index on disk.
  (v) No `ruff format` on ocpy: ocpy has no ruff config and 68 of its 85
  files would be reformatted, so the new files follow ocpy's existing style
  (single quotes); `ruff check` is clean on both.
- **Golden values** (float32; reader = direct `xarray.open_dataset` =
  pasted, bit for bit), (res, realisation, λ, θs, θ, φ) → rrs, Rrs, aph,
  Esdir_Es_ratio:
  - hyp 1, 443, 30, 10, 90 → 0.0071871569, 0.0037851033, 0.0881650001,
    0.7778172493;
  - hyp 2500, 560, 60, 40, 135 → 0.0865792707, 0.0447451025, 0.0140349995,
    0.7885347009;
  - hyp 5000, 700, 0, 0, 0 → 6.9509493e-05, 3.6750615e-05, 6.30e-05,
    0.9163848162;
  - olci 1, 443, 30, 10, 90 → 0.0071869213, 0.0037860486, 0.0881650001,
    0.7773049474;
  - olci 4321, 560, 60, 40, 135 → 0.1077755392, 0.0589836724, 0.0081909997,
    0.7880697250.
  Also: every variable of realisation 2500 equals a direct read, both
  resolutions.
- **Measured.** 5000 realisations per resolution (1–5000, contiguous).
  Grids identical across 10 sampled files per resolution (test) and across
  all 10,000 files (one-off sweep, 39 s). No NaNs in realisation 1.
- **Suite** (`python -m pytest ocpy/tests -q`, ocpy root):
  with the data `4 failed, 168 passed, 7 skipped, 12 warnings in 22.23s`;
  with `env -u OS_COLOR` `4 failed, 145 passed, 30 skipped, 14 warnings in
  5.90s`. The 4 failures are pre-existing and unrelated, the same in both
  runs: `test_plot_oc_scene.py::TestDetectFileType` (three, detection
  returns `'UNKNOWN'`) and `TestLoadMODISL2Scene::test_wavelength_matching`.
  Not touched. `test_pb24.py` alone: 18 passed.
- **Surprises in the files.** (1) The OLCI files are separate HydroLight
  runs on the same realisations, not a band subset of the hyp files: `C`,
  `N`, `Y` and the IOPs at OLCI bands match bit for bit (realisation 1717),
  but `rrs`/`Rrs` differ up to 0.12%, `Kd` 0.03%, `Esdir_Es_ratio` up to
  2.8% (400 nm, θs = 87.75°); created 2024-02-07 (OLCI) vs 2024-03-08 (hyp).
  A hyp slice at OLCI bands will not reproduce O25's calibration set exactly
  (task 5 should use the OLCI files, as planned). (2) The `theta` and `phi`
  attributes name the radiance *propagation* direction ("zero towards
  zenith", "zero towards the sun"); recorded in the docstring. (3) `C`, `N`,
  `Y` sit on a length-1 `scalar` dim, and the 4-D fields are stored
  `(lambda, phi, theta, theta_s)`. (4) `lambda` is a Python keyword, so
  `ds['lambda']` only. Nothing contradicts the task doc.
- Record §7c (new) and §7c.1 written in `design/rt_elastic_implementation.md`.
- **No git command was run** in either repository (beyond read-only
  `git status`/`git log`).

### 2026-11-02 — Task 2: the consolidation caches (Claude Opus 5.5)

- Read: this doc in full (Task 2, Q13, the Task 1 log); `ocpy/hydrolight/pb24.py`
  and `ocpy/tests/test_pb24.py`; the precedents `design/py/build_l23_profile_product.py`,
  `robust/rt/data/l23.py` (derived-product path, JSON provenance, atomic save) and
  `robust/tests/conftest.py`. Opened realisation 1 through the reader and listed every
  variable's dims and dtype before coding: all 28 fields named in Task 2 exist, float32,
  with the dims the doc states (no discrepancy).
- **Built** (retrieve-or-bust, untracked new files): `robust/rt/data/pb24_cache.py` —
  `CACHE_VERSION = 1`, `default_cache_dir()` (`pb24_path`'s parent / `derived/pb24_v1`),
  field constants (`SCALAR_FIELDS`, `IOP_FIELDS`, `AOP_FIELDS` incl. `Esdir_Es_ratio`,
  `TABLE_FIELDS`, `FOUR_D_FIELDS`, `SOURCE_DIMS`, `CACHE_DIMS`),
  `build(cache_dir, realisations, resolution, workers, progress, bands)` →
  `BuildSummary`, `open_cache(cache_dir)` → `PB24Cache` (`array`, `table`, `four_d`,
  `four_d_at`, `realisation_index`, `angle_index`, `band_index`), `read_sidecar`,
  `verify(...)` → `VerifyReport`. `design/py/build_pb24_cache.py` (`--cache-dir`,
  `--workers`, `--realisations`, `--band-step`, `--skip-build`, `--check`,
  `--rebuild-check`, `--pack`). `design/ztt_pb24/cache_build.md`. Fixture
  `robust/tests/files/pb24_cache_small.zip` (one file; 29 members, 2,871,457
  bytes unpacked, 2,190,735 packed; see the 2026-11-03 addendum). Tests
  `robust/tests/test_pb24_cache.py` (25). Conftest: `needs_pb24_hyp`,
  `needs_pb24_cache`, `PB24_CACHE_FIXTURE`, `PB24_CACHE_FIXTURE_BANDS`; `needs_pb24`
  untouched (task 3 retargets it).
- **Decisions the task left open.** (i) Cache order: λ last everywhere;
  `(λ, θs)` fields stored `(real, θs, λ)` to match the 4-D `(real, θs, θ, φ, λ)`; the
  length-1 `scalar` dim dropped (`C`, `N`, `Y` are `(n,)`). (ii) Rows in ascending
  realisation id (input ids are sorted and de-duplicated). (iii) **Writes are streamed,
  not memory-mapped**: the first full build used `np.lib.format.open_memmap` as
  suggested; reading finished in 114 s, then `msync` of the dirty pages on the ZFS
  volume crawled (~0.5 MB/s after 20 min) and I stopped it. Each row is now appended
  with plain `write()` after the `np.save` header (`_NpyStream`), fsynced; bytes
  identical to the memmap-written fixture (checked with `cmp`); readers still memmap.
  (iv) Atomicity: arrays to `*.npy.partial`, old sidecar deleted, arrays renamed,
  sidecar written last; `open_cache` refuses a directory without a sidecar; a crashed
  build deletes its partials (test). (v) Workers: ordered `Pool.imap`, `fork` context
  (works from stdin/unguarded scripts), `spawn` when JAX is already imported (fork
  under JAX warns of deadlock); builder default 8. (vi) `verify` compares raw 32-bit
  words (also `np.array_equal(equal_nan=True)`); its 4-D sample is per (seed, id,
  field), so it does not depend on workers or on the id set; seed 24. "A random 2% of
  the 4-D arrays" read as 2% of the values of every realisation's three arrays, on
  top of 1, 2500, 5000 in full. (vii) Fixture band subset: every 10th band (indices
  0, 10, …, 450 → 350–800 nm by 10, 46 bands), recorded in the sidecar as
  `four_d_bands`; `four_d(bands=)` takes full-grid indices and maps them, so the
  fixture and the full cache run the same path. (viii) Sidecar has no absolute paths
  (source file names and sizes only) and no worker count, so it is portable and
  worker-independent. (ix) The version-mismatch test runs on a copy of the fixture
  (no data needed). (x) `angle_index`/`band_index` take exact grid values only.
- **Measured.** Full cache `$OS_COLOR/SD/derived/pb24_v1/`: 29 files,
  **36,351,031,568 bytes** (3 × 11,726,000,128 4-D; 12 × 90,200,128; 10 × 9,020,128;
  3 × 20,128; sidecar 367,984). Build **286.9 s** (8 workers). `--check`:
  "5000 realisations re-read through the ocpy reader; 293,165,000 table values
  compared (all of them); 181,061,166 of 8,794,500,000 4-D values compared (2.06%:
  realisations [1, 2500, 5000] in full, a random 2% of the rest, seed 24); all
  bit-identical (51.1 s)." `--rebuild-check` (4 workers, into
  `$OS_COLOR/SD/derived/.pb24_v1_rebuild_check`, then deleted): second build 268.6 s,
  "rebuild-check: byte-identical (29 files)", sidecar included (SHA-256 and `filecmp`).
  Fixture: `--check` and `--rebuild-check` pass.
- **Suite** (`python -m pytest robust/tests -q`, repo root): with the data
  `1 failed, 796 passed, 5 skipped, 2 warnings in 228.57s (0:03:48)`; with
  `env -u OS_COLOR` `721 passed, 81 skipped, 4 warnings in 162.75s (0:02:42)`. The one
  failure is pre-existing and unrelated, `test_emulator.py::
  test_excluding_an_unusable_backbone_also_narrows_the_geometry` (`needs_pb24`, M5
  OLCI loader; fails alone too: `0.433 > 5 × 0.156`); it runs now only because the
  OLCI files are in place, and its in-water geometry is what task 3 re-baselines. Not
  touched. `test_pb24_cache.py`: 24 passed with the data.
- **Ruff.** `ruff check robust/ design/` and `ruff format --check robust/ design/`
  clean (CI lints `robust/`). `ruff check .` over the whole repo reports 302
  pre-existing errors and 22 unformatted files, all in files this task did not touch
  (`context/RT/*.py`, `notebooks/`, `docs/`, `reports/py/`, `proposals/`, `setup.py`).
- **Surprises.** The `open_memmap`/ZFS `msync` stall above. `np.ascontiguousarray`
  promotes 0-d to 1-d (bit me on `C`, `N`, `Y`). The `(λ, θs)` AOP block is 1.08 GB, the
  table ≈ 1.17 GB, as Q13 estimated. Nothing in the files contradicts the doc.
- Record §7c.2 written in `design/rt_elastic_implementation.md` (plus module-index lines).
- **No state-changing git command was run** (read-only `git status`/`git log`/`git diff`
  only).

### 2026-11-03 — Task 2 addendum: the fixture as one zip (Claude Fable 5.1)

- JXP declined to commit 29 fixture files. Chosen option: pack the miniature
  into one deterministic zip, `robust/tests/files/pb24_cache_small.zip`
  (2,190,735 bytes; 2,871,457 unpacked), and unpack it into a temporary
  directory once per test session.
- **Built.** `pb24_cache.pack(cache_dir, zip_path)` (sorted members, fixed
  1980-01-01 timestamps, deflate, so packing is byte-reproducible) and
  `pb24_cache.unpack(zip_path, dest)` (refuses nested member paths). Builder
  flag `--pack ZIP`. Conftest: `PB24_CACHE_FIXTURE` now names the zip; new
  session fixture `pb24_cache_dir` unpacks it; `pb24_cache_small/` deleted.
  Tests use the unpacked directory; one new test,
  `test_pack_is_deterministic_and_round_trips`, checks pack twice → identical
  bytes → equal to the committed zip → unpack restores every file.
- **Measured.** Rebuilding the miniature through the CLI
  (`--realisations 1 993 2500 --band-step 10 --pack`) reproduces the
  committed zip byte for byte. `test_pb24_cache.py`: 25 passed with the
  data; 15 passed, 10 skipped without `$OS_COLOR`. `ruff check` and
  `ruff format --check` clean on `robust/` and `design/`. Full suite with the
  data: `1 failed, 797 passed, 5 skipped` (the pre-existing `test_emulator`
  failure, for task 3); without `$OS_COLOR`: `722 passed, 81 skipped`.
- Record §7c.2, `cache_build.md` and the Task 2 log entry above updated to
  name the zip. No state-changing git command was run.

### 2026-11-03 — Task 3: the loader in above-water geometry (Claude Opus 5.5)

- **Re-baselined pins** (each deliberate; L23 pins and the emulator digest did
  not move; no model code or model number changed):
  - `test_env.py` `FROZEN_CONTAINERS["Geometry"]` gains `diffuse_fraction`
    (appended, `None` default, consumed by nothing until task 4).
  - **Fixture:** `robust/tests/files/pb24_small.npz` (471 kB, M5 OLCI raw
    fields) **deleted**; the task-2 miniature `pb24_cache_small.zip`
    (2,190,735 B, realisations 1, 993, 2500, 4-D at every 10th band) feeds the
    real loader through `cache_dir=`. PB24 fixtures now total 2.19 MB (were
    2.66 MB). The OLCI path has no fixture: a test serves the miniature's
    values through a fake per-file `reader` in the netCDF layout and gets the
    cache path's batch bit for bit.
  - `test_pb24.py` (rewritten, 60 tests): bands 12 → 46 (fixture); window 832
    and shell 468 per realisation, unchanged in number but now in air
    (θv max 70 air = θ_w 44.53); new in-water-only count 650. **Zero `rrs`:**
    M5's 2 zeros in the shell (993, 753 nm, θs 80, θ 87.5) → **0 in every air
    set**; 3 in `"in_water_only"` (993, θs 80, 87.5 in water, 730/750/800 nm);
    the drop-cost pin 22 → 3 × 46 − 3. Golden values now via the ocpy reader,
    incl. a hand-interpolated `rrs`. Splits: masks from the file (993 is
    realisation-test; none of 1/993/2500 is in the B_p band, so that split
    *raises* on the fixture, asserted); bp_band/disjointness tests run on a
    tmp file built by `build_splits` over the three; corr(log B_p, log C)
    −0.49 (600, OLCI) → −0.469 (5000, hyp), B_p span > 5 → 29.9× from the
    file; the seed-sweep test is gone (the seed is no longer an argument),
    replaced by reload identity and a byte-identical rebuild from the cache;
    the npz-snapshot and `npz_reader` tests are gone with the npz.
  - `test_conventions.py::test_the_fitted_transfer_beats_the_nadir_constants_on_held_out_data`:
    the **shipped M5 table now loses** to Lee's nadir constants at θv = 60
    (16.0% vs 6.8%; it encodes the old pairing, Q9c). Re-baselined to a table
    fitted on the batch's own train rows (`fit_transfer`): 6.8% → 1.3%,
    5.25× (assert ≥ 4×), nadir 0.8% (< 5%), plus an assertion that the shipped
    table is > 2× worse than the constants.
  - `test_baselines.py::test_the_three_d_table_beats_the_zenith_only_refit_on_held_out_pb24`:
    scored through the shipped table the 3-D table loses (17.45% → 19.50%);
    through `fit_transfer` on the train rows 7.31% → 4.34%, 1.68× (M5 1.67×;
    threshold 1.4 kept).
  - `test_baselines.py::test_the_nadir_transfer_would_have_hidden_the_difference`:
    own-fit transfer; gains 1.68× (aware) vs 1.25× (nadir). Pins
    `nadir_gain < 1.2` → `< 1.35` and `aware > 1.3 × nadir` → `> 1.2 ×`.
  - `test_emulator.py::test_excluding_an_unusable_backbone_also_narrows_the_geometry`
    (the pre-existing failure) — **rewritten** to what the data shows: the
    excluded share is 24% and rises monotonically with θv_air, 13 / 15 / 26 /
    44% at 0 / 20 / 40 / 60, i.e. 3.3×, not 5× (13% of nadir samples are
    excluded too: water, not geometry). Asserts monotone + > 2.5×, and the
    cause directly: 94% excluded at ψ < 110, 14% at ψ ≥ 134. Cell-structure
    assertion unchanged.
  - **Inputs changed, pins unchanged:** the 7 other `test_emulator.py`
    PB24 tests, the 2 in `test_baselines.py` and the 1 in
    `test_conventions.py` now pass `resolution="olci"` explicitly (M5's set),
    `realisations=pb24_mixed_realisations(n)` (first-n ids no longer reach
    both sides of the saved splits: ids 1–21 are all realisation-train) and
    the `needs_pb24_olci` marker. The 3 `test_ztt.py` PB24 tests run on the
    hyp default unchanged: ψ is identical to M5's (42% of the window below
    134°, 16% below 110.4°) because `geometry_to_paper_angles` still reads
    θv as in-water — **task 4's** re-baseline; µd median 4.57% (40°: 1.54%,
    70°: 14.0%).
- Read: this doc in full (data, Code, preamble, tasks 3–4, Q4, Q9, Q10, Q14,
  the task 1–2 logs); `pb24.py` (M5), `pb24_cache.py`, `test_pb24.py`,
  `test_pb24_cache.py`, `conftest.py`, `types.py` (`Geometry`, `PhaseParams`),
  `l23.py`'s FF fill, the PB24 tests in `test_ztt/conventions/baselines/
  emulator/validation.py`, `emulator.fit_pb24`, `ztt.P_bb_from_phase`, the four
  M5 scripts, `ocpy/hydrolight/pb24.py`. Checked the files: `theta` nodes 0–80
  by 10, 87.5 (same for `rrs`/`Rrs`); θ_w of the air nodes 0, 7.45, 14.79,
  21.91, 28.67, 34.87, 40.26, 44.53, 47.30, 48.21.
- **Built.** `robust/rt/data/pb24.py` rewritten (cache for hyp, ocpy
  `load_ds` per file for OLCI, one assembly path): `theta_air_to_water`,
  `interp_in_water(field4d, theta_nodes, theta_w, axis=-3)`, `ANGLE_MODES`
  incl. `"in_water_only"`, `FresnelReport` on every `LoadReport`,
  `interpolation_error` → `InterpolationReport`, `build_splits`/
  `write_splits`/`read_splits`, `make_splits(batch, kinds, path)`,
  `PB24Batch.theta_w`, `IOPs.a_ph`, `ff_B`/`ff_weight`,
  `Geometry.diffuse_fraction`; `types.Geometry.diffuse_fraction` (+ a [0, 1]
  check in `validate`). `design/py/make_pb24_splits.py` (`--check`) →
  `design/ztt_pb24/splits.json` (176,936 B) + `splits.md`;
  `design/py/check_pb24_geometry.py` → `design/ztt_pb24/geometry_check.md`.
  Conftest: `needs_pb24` (hyp file + cache sidecar), `needs_pb24_olci`,
  `pb24_mixed_realisations`. Docstring notes in `emulator.fit_pb24` (the
  "33.6%" is 6.8% correctly paired) and `data/__init__.py`. Record §7c.3.
- **Decisions.** (i) The Fresnel check runs inside `load_batch` on every
  realisation *read* (all 1300 geometries, all θs, Δφ, bands; nadir excluded
  as 0 by construction), streamed into histograms (quantiles to 5e-5,
  extremes exact); it **warns**, never raises. (ii) `"in_water_only"` covers
  every θs (10 × 5 × 13); `Rrs` is all NaN (validate skips an all-NaN `Rrs`).
  (iii) M5 consumer tests stay on OLCI; data-property tests use the hyp
  default. (iv) The cubic reference uses nodes 0–80 (`Q` is `inf` at 87.5
  where `rrs` underflows). (v) `bNAP = 0` (16 realisations, 15 at all bands):
  NAP weight 0, `ff_B` from the realisation's other bands else 0.015.
  (vi) `B_P_EXPECTED` widened (0.0009, 0.04) → (0.0009, 0.045): hyp bulk B_p
  reaches 0.0430. (vii) Removed `RAW_FIELDS`, `write_fixture`, `npz_reader`;
  `make_splits` lost `seed`/`test_fraction`/`bp_quantiles` (the file holds
  them); `confound_reference` still draws in memory. (viii) The splits file
  stores per-realisation mean B_p (451 bands) so band membership never
  depends on a batch's bands. (ix) `fit_transfer` kept for M5's scripts.
- **Measured — Fresnel check, all 5000, 451 bands**
  (`python design/py/check_pb24_geometry.py`, 135 s), deviation of
  (Rrs/rrs at nadir-normalised) / (T/T0) − 1, 293,150,000 values per node:

  | θv_air | T/T0 | q0 | q0.01 | q0.5 | q0.99 | q1 |
  |---|---|---|---|---|---|---|
  | 10 | 1.0000 | −1.07% | −0.57% | +0.18% | +1.14% | +1.56% |
  | 20 | 0.9998 | −0.85% | −0.72% | −0.22% | +0.25% | +0.51% |
  | 30 | 0.9989 | −1.60% | −1.03% | −0.11% | +0.52% | +0.82% |
  | 40 | 0.9957 | −1.34% | −1.19% | −0.81% | −0.27% | −0.04% |
  | 50 | 0.9862 | −3.45% | −2.40% | −1.63% | −1.06% | −0.79% |
  | 60 | 0.9592 | −3.79% | −3.14% | −2.16% | −1.23% | −0.92% |
  | 70 | 0.8833 | **−5.30%** | −4.34% | −3.34% | −1.19% | −0.24% |
  | 80 | 0.6638 | +4.40% | +6.16% | +8.84% | +16.89% | +20.49% |
  | 87.5 | 0.2422 | +107% | +119% | +138% | +179% | +207% |

  **Within 5% on 4,999 of 5,000**; 5 of 2.05 × 10⁹ window values exceed, all
  realisation **4492** (θv_air 70, 356–376 nm, θs 70/80, nadir rrs ≈ 3e-6),
  worst −5.30% (pinned by a test). Per θs, max |dev| 4.51–5.30%. Fixture
  (3 realisations, 46 bands): worst −4.60%.
- **Measured — linear vs cubic** (250 realisations, seed 23, θs ≤ 70),
  |lin/cub − 1| for `rrs`, median / q0.99 / max: 10° 0.12/0.41/0.53%,
  20° 0.16/0.80/1.08%, 30° 0.12/0.69/0.96%, 40° 0.10/0.52/0.71%,
  50° 0.28/1.55/2.05%, 60° 0.04/0.22/0.31%, 70° 0.38/2.11/**2.94%**,
  80° 0.31/1.63/2.26%, 87.5° 0.23/1.21/1.66%. `Q`: max 0.90%.
- **Measured — counts and data.** Window 832, shell 468, all 1300,
  in-water-only 650 per realisation. Zero `rrs`: 36,680 values, 1350
  realisations, θs ≥ 40, **all at the in-water node 87.5** → none in any air
  set. Splits: realisation 1000/4000, B_p band [0.01322, 0.01522]
  1000/4000, 194 realisations in both test sets; geometry rule θv_air ≥ 80 or
  θs ≥ 80 (468/832 per realisation). Confound: corr(log B_p, log C) = −0.469;
  B_p-band test/train median C 0.913, N 1.308, Y 1.053, bb_p 1.168 against a
  random-split reference C [0.78, 1.13], N [0.85, 1.21], Y [0.92, 1.05],
  bb_p [0.83, 1.10] — N and bb_p are outside their bands (the band
  holds out NAP-richer, higher-bb_p water), C and Y are not. Gershun a/Knet = mu_tot on
  the fixture: median 0.9997.
- **Suite** (`python -m pytest robust/tests -q -p no:cacheprovider`): with
  the data `814 passed, 5 skipped, 2 warnings in 274.87s (0:04:34)`; with
  `env -u OS_COLOR` `737 passed, 82 skipped, 4 warnings in 178.29s
  (0:02:58)`. The 5 skips with data are the pre-existing hash-anchor/solar/
  fallback ones. `test_pb24.py` alone: 60 passed; 52 passed, 8 skipped
  without the data. `ruff check robust/ design/`: "All checks passed!";
  `ruff format --check robust/ design/`: "62 files already formatted".
- **Scripts.** `cross_dataset.py`, `run_pb24_validation.py`,
  `train_emulator_pb24.py`, `fit_surface.py` all import
  (`PYTHONPATH=design/py:. python -c "import ..."`). Not rerun. Their
  assumptions the new definition breaks: they now default to hyp (451 bands)
  where they were written for OLCI (pass `resolution="olci"` to reproduce
  M5's setting; `run_pb24_validation.py`'s per-wavelength CSV is OLCI-only);
  their first-n realisations are mapped onto the saved splits (n ≤ 21 has no
  realisation-test side); their numbers are on M5's pairing.
  **`fit_surface.py` must not be rerun before task 4**: it would overwrite
  `robust/rt/files/surface_pb24.npz` (the shipped table `default_transfer`
  serves, pinned in `test_conventions.py`) with an air-paired fit.
- **Surprises.** (1) `Q = inf` exactly where `rrs = 0` (Eu/Lu with Lu = 0),
  all at the in-water 87.5 node. (2) The strict gate fails on one
  realisation (4492) by 0.3 pp at 5 values: the task doc's "within 5% on every
  realisation" holds on 4,999. (3) `bNAP = bbNAP = 0` in 16 realisations.
  (4) Hyp `bbph/bph` reaches 0.0502 and bulk B_p 0.0430 (M5's OLCI sample
  of 600: 0.0358). (5) The shipped M5 surface table is worse than no table
  on correctly paired data (Q9c confirmed); the nadir map's error at
  θv = 60 is 6.8%, not M5's 33.6%. (6) ψ in the window is unchanged until
  task 4 refracts θv. (7) Filling `ff_B` changes nothing measurable in the
  backbone-usability share (24.5% with FF and with Sullivan, 20 realisations).
- **No state-changing git command was run** (read-only `git status` /
  `git diff` only).
