# Coding Plan — Elastic RT Forward Model

*Staged, milestone-gated implementation plan for the differentiable elastic-RT forward
model specified in [`design/rt_elastic_model.md`](rt_elastic_model.md).*

This is the build plan for the **Week-1 prototype** and the immediate track beyond it.
It turns the design into concrete milestones, each with tasks, deliverables, and a
**pytest acceptance gate**.

## Ground rules (from Q&A/Coding)

- **Execution (CQ1).** Claude implements now on a **branch** (suggest
  `rt-elastic-prototype`). Each milestone is a **reviewable commit/PR**; **JXP runs all
  git** and reviews (per `CLAUDE.md`). Claude does not run git.
- **Structure (CQ2).** Milestone-gated **M0–M5**, fine-grained (task-level) for the
  Week-1 milestones (M0–M4), coarser for M5.
- **ZTT (CQ3).** The paper is in hand: `context/RT/twardowski2018.pdf` (Twardowski &
  Tonizzo 2018, *Applied Sciences* — "Ocean Color Analytical Model Explicitly Dependent
  on the VSF"). M2 transcribes its equations. A Gordon/O25-in-JAX backbone is kept as a
  fallback to de-risk (see Risks).
- **Testing/conventions (CQ4).** Every milestone is **gated by `pytest`** (incl. the
  gradient-correctness check). Follow **BING conventions**: tests live in
  `robust/tests/` as `test_*.py` with a `conftest.py` and a `files/` fixtures dir;
  ruff formatting; light `jaxtyping` on public signatures.
- **Environment (CQ5).** **CPU-only** for now. Add `jax`, `flax`, `optax` (CPU) and
  update `requirements.txt`; packages may be added to `ocean14`.
- **Validation split (CQ6).** Held-out generalization = (a) seeded **random 20% of
  scenes** and (b) **hold out one solar-zenith angle** (train 0°/30°, test 60°).

## Package layout (mirrors BING)

```
robust/rt/
  __init__.py         # exports forward(), public types
  types.py            # IOPs, PhaseParams, Geometry pytrees (jaxtyping)
  conventions.py      # A=0.52,B=1.7; wavelength grid; bb_w model; asserts
  data/l23.py         # L23 loader (via ocpy.hydrolight.loisel23) → arrays
  ztt.py              # Rrs_ZTT analytic backbone (JAX), phase fn explicit
  emulator.py         # Flax MLP ΔRrs + Optax training
  hybrid.py           # forward(): Rrs_ZTT + ΔRrs; mode flag
  validation.py       # rRMS/speed/grad protocol; comparison vs Gordon/PR05/O25
robust/tests/
  conftest.py, files/, test_*.py   # one test module per rt module
design/py/            # figure & validation-run scripts (non-package)
```

---

## Milestones at a glance

| M | Goal | pytest acceptance gate |
|---|------|------------------------|
| **M0** | Env + scaffold | jax(CPU) imports; `robust.rt` imports; `pytest` collects; trivial smoke test green |
| **M1** | Data + conventions | L23 loader shapes/ranges; `Rrs↔rrs` round-trip; `B_p` in range; golden-value vs known L23 |
| **M2** | ZTT-in-JAX backbone | ZTT reproduces paper reference values (tol); grad vs finite-diff; standalone rRMS reported |
| **M3** | Emulator + hybrid | hybrid **beats standard Gordon** rRMS (train, 3 zeniths); grad check on `forward`; throughput logged |
| **M4** | Validation (**prototype done**) | hybrid beats Gordon on **both held-out splits** + passes grad gate; metrics table + figures |
| **M5** | Beyond week 1 | (coarse) HydroLight PF/BRDF runs; ZTT backward-VSF params; API frozen |

---

## M0 — Environment & scaffold

**Tasks.**
- Add `jax`, `flax`, `optax` (CPU wheels) to `requirements.txt`; install into `ocean14`
  (or a note if a dedicated env is preferred later).
- Create the `robust/rt/` package skeleton and `robust/tests/` with `conftest.py` and an
  empty `files/`.
- Confirm `robust` is importable (setup.py already defines the package).

**Deliverable.** Importable `robust.rt` (stubs) + green test collection.
**Gate.** `pytest -q` runs; `test_env.py`: `import jax; jax.numpy` works on CPU,
`from robust import rt` succeeds.

## M1 — Data & conventions

**Tasks.**
- `conventions.py`: `A_RRS=0.52, B_RRS=1.7`; `Rrs_to_rrs`/`rrs_to_Rrs`; canonical
  wavelength grid (L23 350–750, 81 bands); pure-water `bb_w(λ)`; load-time asserts.
- `types.py`: `IOPs(a, bb_w, bb_p)`, `PhaseParams(B_p, ...)`, `Geometry(theta_s,
  theta_v, dphi, wind)` as JAX pytrees with `jaxtyping` shapes.
- `data/l23.py`: load the **elastic** set via `ocpy.hydrolight.loisel23.load_ds(1, Y)`
  for `Y∈{0,30,60}` (`Hydrolight1{Y:02d}.nc`); assemble `(IOPs, Geometry, Rrs)` batches;
  compute `B_p = bbnw / bnw`; expose the seeded splits (CQ6).

**Deliverable.** A one-call L23 batch loader returning JAX arrays.
**Gate.** `test_conventions.py` (round-trip `Rrs→rrs→Rrs` to 1e-6; asserts fire on bad
input); `test_l23.py` (shapes `(3320, 81)`; `a, bb ≥ 0`; `B_p` within ~[0.004, 0.03];
one **golden-value** row cross-checked against the raw netCDF).

## M2 — ZTT analytic backbone (JAX)

**Tasks.**
- Read `context/RT/twardowski2018.pdf`; transcribe the ZTT forward relation (the VSF/
  backward-VSF-explicit `Rrs` model) into `ztt.py` as pure JAX functions
  `Rrs_ZTT(iops, phase_params, geometry, wave)`.
- Wire the explicit phase-function input (`B_p` for now; structured so the ZTT
  backward-VSF parameters slot in later).

**Deliverable.** Differentiable `Rrs_ZTT`.
**Gate.** `test_ztt.py`: (i) reproduces a **paper reference case** (a value/curve quoted
in twardowski2018) to stated tolerance; (ii) **`jax.grad` vs central finite differences**
agree (tol) w.r.t. `a, bb_p, B_p, geometry`; (iii) standalone rRMS vs L23 **reported**
(logged, not gated to a number — per the "unbiased" stance).

## M3 — Residual emulator + hybrid

**Tasks.**
- `emulator.py`: a **small Flax MLP** `ΔRrs`, inputs e.g. `(u or (ω_bw, ω_bp), B_p,
  geometry, λ)`; train with **Optax** on `Rrs_L23 − Rrs_ZTT`, **relatively weighted**
  (the BING lesson: unweighted lets red-λ terms run away); L2/​size regularization to
  keep the residual small.
- `hybrid.py`: `forward()` = `Rrs_ZTT + ΔRrs`, with a `mode ∈ {ztt, emulator, hybrid}`
  flag so all three §6-options compare on identical data.

**Deliverable.** Trained differentiable `forward()`.
**Gate.** `test_hybrid.py`: hybrid **beats standard Gordon** rRMS on the **train** split
at all three solar zeniths; `jax.grad` finite-diff check on the full `forward`; a
throughput number (scenes·λ/s, batched) is recorded (must not collapse vs ZTT alone).

## M4 — Validation — *Week-1 prototype complete*

**Tasks.**
- `validation.py`: the design §6 protocol — rRMS (rrs-space, relatively weighted) **per
  λ, per solar-zenith, per `B_p` bin**; **held-out** metrics on the CQ6 splits; speed;
  the gradient-correctness gate; all **alongside Gordon, PR05, O25** on the same splits.
- `design/py/run_validation.py`: regenerates the metrics table + a couple of figures.

**Deliverable + Definition of Done.** The prototype is "done" when the M4 gate passes.
**Gate (acceptance).** Hybrid **beats standard Gordon on BOTH held-out splits** (random
20% scenes; unseen 60° zenith) **and** passes the gradient-correctness gate; metrics
table + figures produced and committed.

## M5 — Beyond Week 1 (coarse; forward-model track)

- Commission **HydroLight** runs that vary the **particle phase function** and **sensor
  zenith/azimuth** (the axes L23 fixes); retrain/extend the emulator; re-run §6 with
  held-out phase-function shapes. Add **PB24** as a multi-angular cross-comparison.
- Promote `phase_params` from `B_p` to the **ZTT backward-VSF** parameterization.
- Freeze the `forward` API as the shared engine for training-data generation and (the
  separately designed) inversion.

**Detailed 2026-08-08, and one bullet above is now out of date.** M5's reference data is
**PB24** (`$OS_COLOR/SD/v5`), which turned out to vary the particle backscatter ratio per
realisation (~30×) as well as the geometry — so the *held-out phase-function split* is
constructible without commissioning HydroLight runs, and commissioning drops to a stretch
item that answers only the across-VSF-family question. PB24 also showed that the Lee-2002
`Rrs ↔ rrs` map is nadir-only (45.7% median error at θv = 60°), adding a
**geometry-aware surface transfer** task that this sketch did not anticipate.

The task sequence (3–12), with a test gating each, lives in
[`claude_prompts/RT/rt_elastic_coding_prompt_6.md`](../claude_prompts/RT/rt_elastic_coding_prompt_6.md)
§M5; the record of what was decided and why is
[`rt_elastic_implementation.md`](rt_elastic_implementation.md) §7.

**Gate (acceptance) — provisional, pending JXP's answer to Q15.** Hybrid retrained on PB24
**beats O25 refit on PB24** on the held-out-realisation split **and** the held-out-`B_p`
split; the geometry split (train 0–70°, test 80–87.75°) is **reported**, not gated, on the
same reasoning that took M4's zenith split out of its gate.

For that gate to mean anything, O25 must first be given a **geometry-indexed coefficient
table**: `fit_o25` as built groups by solar zenith only, which off-nadir handicaps the
benchmark rather than flattering it (record §7.6, finding 2). Beating a rival we crippled
is not a result.

---

## Testing strategy

- **Framework/layout:** `pytest`, tests in `robust/tests/` (BING layout), `conftest.py`
  with shared fixtures (a small cached L23 batch under `files/`), CPU-deterministic
  (fixed seeds; `jax.config` float64 where needed for the FD check).
- **First-class gates:** the **gradient-correctness** test (grad vs finite difference)
  and **golden-value** tests (numbers pinned against the raw L23 netCDF and the ZTT
  paper) recur across milestones.
- **No blind targets:** accuracy gates are *relative* ("beats standard Gordon on the
  held-out splits"), consistent with the project's unbiased-uncertainty stance; absolute
  rRMS/latency are reported, not thresholded.

## Requirements / dependency changes

Add to `requirements.txt` (CPU): `jax`, `flax`, `optax`, plus `jaxtyping`. `xarray`,
`ocpy` (L23 loader), and `pytest` are already present.

## Risks & de-risking

- **ZTT transcription (M2).** If a term in twardowski2018 is ambiguous or slow to pin
  down, fall back to an **O25/Gordon-in-JAX backbone** so M3–M4 proceed end-to-end;
  swap true ZTT back in without changing the `forward` signature. Flagged, not assumed.
- **Emulator overfitting (M3).** Small network + regularization + the held-out gate
  (M4) guard against it; the residual is small by construction.
- **JAX on `ocean14` (M0).** CPU wheels only; if the install perturbs `ocean14`, move to
  a dedicated `rt-jax` env (CQ5 leaves this open).
- **L23 is nadir-view, fixed FF (M1–M4).** Only solar-zenith geometry and `B_p` vary in
  hand; full BRDF and phase-function-shape variation wait for M5's HydroLight runs —
  the prototype's scope is honest about this.

## Definition of done (Week-1 prototype)

**M4 gate passed:** a differentiable JAX hybrid `Rrs = Rrs(ZTT) + ΔRrs` that beats
standard Gordon on both held-out splits, passes the gradient-correctness gate, runs fast
on a full L23 batch, and ships with a validation table + figures — all under
`robust/rt/` with `pytest`-green tests, on a branch for JXP to review and merge.
