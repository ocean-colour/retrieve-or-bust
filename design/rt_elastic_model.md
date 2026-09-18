# Design — Elastic RT Forward Model

*A near-term, buildable engineering plan for retrieve-or-bust's elastic
radiative-transfer forward model: a fast, accurate, **differentiable** map
`Rrs(λ; a, bb, phase function, geometry)`.*

Companion to the synthesis + roadmap in
[`context/RT/rt_elastic_model.md`](../context/RT/rt_elastic_model.md), which motivates
the physics and the model lineage. This document is the **implementation plan**.

Decisions locked in Q&A/Design (see `claude_prompts/rt_elastic_prompts.md`): near-term
buildable scope with a **~1-week first prototype** (DQ1); **HydroLight reference**, with
a large set of **Loisel+2023 (L23) outputs already in hand** (DQ2); **JAX**, with **ZTT
implemented analytically in-framework** so the hybrid is differentiable end-to-end
(DQ3); a **simple explicit phase-function parameter** chosen here (DQ4, §4.2);
**forward-model only, but differentiable** — inversion out of scope but interface-ready
(DQ5); acceptance = **fast + accurate + differentiable**, protocol-based (DQ6, §6).

---

## 1. Goals and non-goals

**Goals.**
- A single callable `forward(iops, phase_params, geometry) → Rrs(λ)` that is **fast**
  (vectorized/batched), **accurate** (validated against the RT reference), and
  **differentiable** (JAX; gradients w.r.t. all inputs).
- **Physics-anchored, not black-box**: the hybrid `Rrs = Rrs(ZTT) + ΔRrs(emulator)`,
  where ZTT carries interpretable scaling, geometry, and an *explicit* phase-function
  dependence, and a small emulator learns only the residual (multiple-scattering and
  phase-function effects the analytic backbone misses).
- **Elastic only** (no Raman / fluorescence).
- Geometry and **phase-function shape** are first-class inputs from day one — even
  where the current reference data hold them fixed, the API carries them.

**Non-goals (this doc).**
- The **inversion** (IOP retrieval) — deferred to a companion design. We only guarantee
  the forward model is differentiable and exposes a clean interface an inversion can
  call.
- Inelastic processes; full operational PACE processing; the learned-prior machinery.

---

## 2. Architecture

```
                        phase_params θ_p        geometry Ω
                              │                     │
   iops (a, bb_w, bb_p) ──────┼─────────────────────┼─────────┐
                              ▼                     ▼         ▼
                    ┌───────────────────┐   ┌──────────────────────┐
                    │  Rrs_ZTT(θ)       │   │  ΔRrs_emulator(θ)     │
                    │  analytic (JAX)   │ + │  small MLP (Flax)     │  =  Rrs(λ)
                    │  explicit backVSF │   │  residual, ≈smooth    │
                    └───────────────────┘   └──────────────────────┘
                              └──── both differentiable in JAX ─────┘
   θ = (a(λ), bb_w(λ), bb_p(λ), θ_p, Ω)          reference: HydroLight / L23
```

The emulator is trained on `ΔRrs = Rrs_reference − Rrs_ZTT`. Because that residual is
small and smooth (ZTT already captures single-scattering + explicit phase function),
the network is light and its extrapolation is bounded — the point of the hybrid over a
wholly learned model.

---

## 3. Interface and data model

**Public API** (new subpackage `robust/rt/`):

```python
def forward(iops: IOPs, phase_params: PhaseParams, geometry: Geometry,
            wave: Array) -> Array:      # returns Rrs(wave), shape (..., n_wave)
    """Elastic Rrs. Differentiable in JAX; batched over leading axes."""
```

- `IOPs`: a pytree with `a(λ)`, `bb_w(λ)`, `bb_p(λ)` (all m⁻¹; `bb = bb_w + bb_p`).
  Keeping `bb_w` (known pure-water constant) and `bb_p` separate is deliberate — it is
  the water/particle split the synthesis doc shows is essential, and it is free for us.
- `phase_params`: the explicit phase-function descriptor(s) — §4.2.
- `geometry`: `(θ_s, θ_v, Δφ)` (solar zenith, sensor zenith, relative azimuth), plus
  wind speed as an optional scalar.
- Convention: subsurface `rrs = Rrs / (A + B·Rrs)`, **A = 0.52, B = 1.7** (Lee 2002),
  asserted once at the package boundary to match BING.

Everything is JAX arrays; `forward` is `jit`/`vmap`-friendly so a whole L23 batch (3320
scenes × 81 λ) runs in one call, and `jax.grad`/`jax.jacobian` give input sensitivities
for the future inversion.

---

## 4. Components

### 4.1 Reference data

- **In hand (prototype uses this):** L23 elastic (`Hydrolight1{Y:02d}.nc`, X=1 = no
  inelastic) at **three solar-zenith angles** — `Hydrolight100/130/160.nc` = 0°/30°/60°,
  nadir view, fixed Fournier–Forand phase function; 3320 scenes × 81 λ (350–750 nm).
  This already exercises the **solar-zenith geometry** axis. (X=2/4 add Raman/fluor —
  out of scope; kept only for an inelastic-delta sanity check later.)
- **Next (immediately after prototype):** new **HydroLight** runs that explicitly vary
  the **particle phase function** and the **sensor zenith / azimuth** (full BRDF) — the
  axes L23 holds fixed. PB24 (Pitarch & Brando 2025; 5000 IOPs × 1300 geometries) is
  used in parallel as a ready multi-angular cross-comparison.

### 4.2 Phase-function parameterization (DQ4 — my pick, kept simple)

**Primary explicit parameter: the particulate backscattering ratio**

```
B_p = bb_p / b_p        (dimensionless, ~0.005–0.03)
```

realized through a **Fournier–Forand** phase function (one scalar shape index). Rationale
for starting here:
- It is the single scalar to which O25's `η_b`/`γ_b` and ZTT's backward-VSF descriptor
  both reduce at leading order — so it *is* an explicit phase-function-shape input, just
  the simplest one.
- It is **directly available in L23** (`B_p = bbnw / bnw`), so the prototype can compute
  and (later) vary it without new data.
- It is differentiable and one-dimensional — cheap to sweep and to emulate.

**Planned generalization (documented, not built in week 1):** the fuller **ZTT
backward-VSF parameters** (e.g. `β̃(π)` plus a backward-shape term) as a 2–3-parameter
axis, once HydroLight runs sample it. The API's `phase_params` pytree is defined so
adding these later does not change the `forward` signature.

### 4.3 ZTT analytic backbone (JAX)

Implement the ZTT (Twardowski & Tonizzo 2018) forward relation analytically in JAX, with
the backward VSF / `B_p` entering explicitly. This is `Rrs_ZTT(θ)` — pure functions, no
learned parameters, fully differentiable. It is also our **analytical benchmark**: scored
standalone against the reference before the emulator is added.

### 4.4 Residual emulator ΔRrs

- A **small MLP** (Flax) : features → `ΔRrs(λ)`. Candidate input features: `u = bb/(a+bb)`
  (or the `(ω_bw, ω_bp)` split), `B_p`, geometry `(θ_s, θ_v, Δφ)`, and λ (or a per-λ
  output head). Start with a modest width/depth; the target residual is small.
- Trained (Optax) to minimize weighted rRMS of `Rrs_ZTT + ΔRrs` vs the reference, with
  relative weighting (the BING lesson: unweighted fits let red-λ terms run away).
- Regularized to stay small (the hybrid's whole value is a *bounded* correction).

### 4.5 Hybrid assembly

`Rrs = Rrs_ZTT + ΔRrs`, one JAX function, differentiable end-to-end. A flag selects
pure-ZTT, pure-emulator, or hybrid so the three §6-of-synthesis options can be compared
on identical splits.

---

## 5. Software stack & conventions

- **JAX** (+ **Flax** for the MLP, **Optax** for training); run in the `ocean14` env
  (add jax/flax/optax if absent). Data I/O via `xarray` (L23 netCDFs).
- Module layout:
  ```
  robust/rt/
    __init__.py         # exports forward(), types
    types.py            # IOPs, PhaseParams, Geometry pytrees
    ztt.py              # Rrs_ZTT analytic backbone (JAX)
    emulator.py         # Flax MLP ΔRrs, train loop
    hybrid.py           # forward(): ZTT + emulator
    data/l23.py         # L23 loader → IOPs/geometry/Rrs arrays (reuse ocpy loisel23)
    conventions.py      # A,B; wavelength grid; asserts
  reports/py or design/py: figures & validation scripts
  ```
- One config object (dataclass) fixing A/B, wavelength grid, `bb_w` model, phase-function
  parameterization, and seeds — asserted at load so runs are comparable.

---

## 6. Validation protocol (DQ6 — "fast, accurate, differentiable")

No blind target numbers (consistent with the project's unbiased-uncertainty stance);
instead a fixed **protocol** reporting all three axes, with held-out splits:

- **Accurate.** rRMS (rrs-space, relatively weighted) of the model vs the reference,
  reported **per λ, per solar-zenith, and per `B_p` bin**; plus **held-out
  generalization** (seeded split by scene, and — once available — by geometry and by
  phase-function shape the model did not train on). Always shown alongside **Gordon,
  PR05, and O25/L11** on the same splits (the comparison models).
- **Fast.** Throughput (scenes·λ / s, batched) and single-call latency for a full L23
  batch on CPU/GPU; the emulator must not erase the speed advantage over calling the RT
  solver.
- **Differentiable.** A gradient-correctness check: `jax.grad` of `Rrs` w.r.t. `a`, `bb_p`,
  `B_p`, geometry vs central finite differences, agreeing to tolerance across a random
  batch. This is the property the inversion depends on and is treated as a hard gate.

Deliverable: a validation script + a metrics table/figure regenerated on demand.

---

## 7. The 1-week prototype (DQ1)

**Definition of "first high-quality prototype":** a differentiable JAX hybrid
`Rrs = Rrs_ZTT + ΔRrs` trained on **L23 elastic** that (i) **beats standard Gordon** rRMS
across λ and the three solar-zenith angles, (ii) is **competitive with O25** on the same
split, and (iii) **passes the gradient-correctness gate** (§6).

Indicative sequence (order, not rigid days):
1. `robust/rt/` scaffold: types, conventions (A/B, grid), L23 loader (reuse `ocpy`
   `loisel23`), and a batch of `(IOPs, geometry, Rrs)` for X=1, Y∈{0,30,60}.
2. `ztt.py`: analytic ZTT in JAX with `B_p` explicit; validate `Rrs_ZTT` vs L23 and vs
   the Gordon/O25 baselines (reuses the synthesis-doc figure machinery).
3. `emulator.py` + `hybrid.py`: train ΔRrs (Optax) on `L23 − ZTT`; assemble the hybrid.
4. Validation script: the §6 accuracy/speed/gradient protocol; comparison table vs
   Gordon/PR05/O25; a couple of figures.
5. Short write-up of results + honest failure modes → feeds the next round (HydroLight
   phase-function-varied runs, sensor-angle BRDF, ZTT backward-VSF generalization).

---

## 8. Beyond week 1 (forward-model track only)

- Commission **HydroLight** runs varying the **particle phase function** and **sensor
  zenith/azimuth**; retrain/extend the emulator on the richer reference; re-run §6 with
  held-out phase-function shapes.
- Promote `phase_params` from `B_p` to the **ZTT backward-VSF** parameterization (§4.2).
- Add wind-speed / surface effects if the reference shows they matter at target accuracy.
- Freeze the `forward` API as the shared engine for **both** training-data generation and
  the (separately designed) **inversion**.

---

## 9. Open decisions I made on your behalf (flag for correction)

- **DQ4 phase-function parameter** = particulate backscattering ratio `B_p` (FF-based),
  with ZTT backward-VSF as the documented generalization. If you'd rather start directly
  on the ZTT backward-VSF params, say so — it changes §4.2–4.3, not the API.
- **Prototype target** (§7) = "differentiable hybrid on L23 that beats Gordon and matches
  O25, gradient-gated." If "first high-quality prototype" means something narrower (e.g.
  just the ZTT-in-JAX backbone validated) or broader (include new HydroLight runs),
  adjust §7.
- **DQ6 acceptance** = protocol-only (no hard target), three gates (accuracy/speed/grad).
  If you want a specific rRMS or latency target committed, name it.

---

## 10. References

Physics, model lineage, and the L23-derived figures: see
[`context/RT/rt_elastic_model.md`](../context/RT/rt_elastic_model.md) §8. Key models:
Gordon (1988), Park & Ruddick (2005), Lee (2011) / Pitarch (2025, O25), **Twardowski &
Tonizzo (2018, ZTT)**, Mobley (HydroLight), Loisel et al. (2023, L23).
