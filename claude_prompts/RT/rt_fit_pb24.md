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

1. Read this doc and the items in Context. Answer nothing yet; put your
   questions in Q&A/Plan. Then wait.
   Use Opus 5.5. Log your work.

### Tasks

*Each task ends with a log entry and JXP reviewing before the next.*

1. **Task 1 — A PB24 reader in `ocpy`** (branch `pb24`). A module
   `ocpy/hydrolight/pitarch24.py` (name per Q2), in the style of
   `loisel23.py`:
   - path resolution from `$OS_COLOR` (`SD/v5`);
   - `load_ds(index)` for one realisation;
   - a list of available realisations;
   - the shared grids (`lambda`, `theta_s`, `theta`, `phi`), asserted
     identical across files on load.

   Above that, a **consolidation layer**, so analysis never opens 5,000 files
   twice:
   - (a) a per-realisation table of the λ-dependent IOPs, constituent labels,
     and the (λ, θs) surface AOPs including `Esdir_Es_ratio`. About 0.1–0.3 GB,
     built once.
   - (b) a builder for **geometry/band subsets** of `rrs` / `Rrs` / `Q`
     (e.g. a band list × an angle window), because the full 4-D reflectance
     for all realisations is ~12 GB per field.

   Both caches go next to the data (format per Q3), carry a provenance
   header, and rebuild bit-identically. Tests go in `ocpy/tests/` and must
   skip without the data. Then point `robust.rt.data.pb24` at the `ocpy`
   reader (as `robust.rt.data.l23` wraps `ocpy.hydrolight.loisel23`), make
   hyperspectral the default resolution (per Q1), and fix `needs_pb24`. All
   without changing what the M5 tests assert, except where the data itself
   changed.
   **Gate:**
   - golden values against raw netCDF reads;
   - consolidated arrays bit-identical to per-file reads;
   - a rebuild is bit-identical;
   - the robust PB24 tests run (not skip) and pass.

   Use Opus 5.5. Log your work.

2. **Task 2 — Baseline: every elastic model we have, scored on PB24.** A
   script in `design/py/` (e.g. `run_pb24_baseline.py`), writing to
   `design/ztt_pb24/`. Score on PB24's held-out split(s) (M5's realisation and
   B_p-band splits; geometry split reported), in rrs space, per λ, per θs,
   per θv, per ψ bin and per B_p bin. The models:
   - standard Gordon;
   - the O25 form refit on PB24 (geometry-indexed, as M5 required);
   - ZTT (TT2017 µ∞), ZTT (L23 µ∞) and the L23 closure variant;
   - the L23-trained hybrid, with and without its out-of-domain fallback.

   Report every model on **everything**, and separately inside M5's sanctioned
   window and within ZTT's fitted ψ ≥ 134°. Also re-run M7's term diagnostics
   on PB24's ten sun angles:
   - µd against PB24's `mu_d`, with TT's H vs `1 − Esdir_Es_ratio`;
   - Ψ_KLu/µ∞ vs PB24's `KLu/a`, now off-nadir;
   - the share of non-physical (≤ 0) rrs per model.

   **Gate:** reproducible (a rerun is byte-identical), and every number traced
   to a committed CSV. Nothing is thresholded. This task measures.
   Use Opus 5.5. Log your work.

3. **Task 3 — Discuss how to improve the elastic backbone.** No code. From
   task 2's numbers, write a design discussion in Q&A/Design. It should cover
   at least:
   - (i) µd's atmospheric factor from PB24's own `Esdir_Es_ratio` and ten sun
     angles;
   - (ii) µ∞ from the eigenproblem on PB24's IOPs and phase functions (Q4),
     with or without the B_p / phase-function dependence that Eq. (8) lacks
     (M7 Q5);
   - (iii) a Ψ_KLu valid over PB24's ψ range (≥ 44°), refitted **jointly with
     f_L**. M7 showed that changing Ψ_KLu alone breaks the balance;
   - (iv) whether ZTT's structure is worth keeping off-nadir, against an
     alternative backbone with an explicit VSF;
   - (v) what this means for the emulator retrain (M7 Q2b/Q18).

   Each option gets a prediction written down before any test (M5 gotcha 4).
   Pose numbered questions; JXP chooses.
   Use Opus 5.5. Log your work.

4. **Task 4 — Implement the chosen backbone and assess it on PB24.** Scope per
   task 3's decisions, following M7's pattern:
   - a deterministic fit script with `--check`;
   - coefficients shipped as constants with provenance;
   - a new `variant` on `rrs_ZTT`, leaving `"l23"` and `"tt2017"` bit for bit;
   - an additive ladder with attribution, run through task 2's script so
     the numbers are comparable.

   **Acceptance (proposed; confirm in task 3):** the new backbone beats
   ZTT (L23 µ∞) on PB24 held-out realisations, within the sanctioned window
   *and* on everything; the share of non-physical rrs falls; L23 numbers are
   reported too, so a PB24 gain that costs L23 shows. Then a notebook, a
   review pass, a record section, a report update, and a hand-off.
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
**JXP:**

**Q2. Where does `ocpy` stop and `robust` start?** My proposal mirrors L23:
- **`ocpy`** holds the format-level reader and the consolidated caches (no
  splits, no model types, no JAX).
- **`robust.rt.data.pb24`** keeps batches, splits, the angle window and the
  `LoadReport`, rebuilt on top of `ocpy`.

Module name: `ocpy/hydrolight/pitarch24.py` (by author and year, as
`loisel23.py`) or `pb24.py`? And should the cache-building code live in `ocpy`
(reusable by others) or in `robust`?
**JXP:**

**Q3. Cache format and location.** The full 4-D `rrs` is ~12 GB per field,
so caches are subsets. I propose:
- `$OS_COLOR/SD/derived/` next to the data, as M7 did for L23.
- **npz** for the IOP/AOP table (simple, bit-reproducible, ~0.2 GB).
- **zarr** (chunked by realisation) for the angle/band reflectance subsets, so
  one can read a slab without loading 12 GB.

zarr would be a new dependency for `ocpy`. Fine, or prefer netCDF/npz
throughout? Rclone to `RoB` as before?
**JXP:**

**Q4. PB24's phase functions — I need the paper.** The µ∞ eigenproblem (and
any explicit-VSF backbone) needs PB24's particle phase functions exactly. For
L23 they were Fournier–Forand with fixed b̃b per component. M5 measured PB24's
`bbph/bph` varying 30× across realisations (0.0010–0.0358) and `bbNAP/bNAP`
between 0.010 and 0.020. Are these FF, with which (n, µ) convention, or
another family? Please put the PB24 dataset paper (Pitarch & Brando) in
`context/RT/`, and the O25 paper if not already there.
**JXP:**

**Q5. Spectral grid for scoring.** Our canonical grid is L23's 350–750 nm at
5 nm (81 bands); PB24 runs to 800 nm at 1 nm. Score on PB24's own 451 bands
(ZTT's f_L table covers 350–800), on the canonical 81 for comparability with
M7, or both?
**JXP:**

**Q6. Geometry scope.** M5 sanctioned 0–70° in both zeniths (M5 Q14) and held
the 80/87.75° shell out as an extrapolation test. Keep that window for M8's
gates? Should a ψ ≥ 134° subset be a first-class reporting cut, given that's
where ZTT's Ψ_KLu is fitted?
**JXP:**

**Q7. Splits.** M5 used three: realisation, B_p band, and geometry. A
reminder from M5's gotchas: refit per split, and a seed doesn't name a
partition when the realisation set changes. Keep all three, with task 2 fixing
the partitions once (saved by realisation id) so every later task scores on
identical sets?
**JXP:**

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
