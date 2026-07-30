# Elastic RT Coding — Prompt 4 (M3: Emulator + hybrid)

## Goals

Implement **Milestone M3**: the **residual emulator** `ΔRrs` and the **hybrid**
`forward()` = `Rrs_ZTT + ΔRrs`. This produces the first trained, end-to-end
differentiable forward model — the core deliverable of the Week-1 prototype.

## Claude

### Skills

Consider `.claude/skills/` (`code-review`, `verify`).

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements*.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M3.
- **Design** — `design/rt_elastic_model.md` §4.4–4.5 (emulator + hybrid assembly) and
  §2 (the residual is small and smooth — the point of the hybrid over a black box).
- **The BING weighting lesson** — relative (∝ rrs) weighting; unweighted fits let the
  red-λ terms run away (`context/RT/rt_elastic_model.md` §4 note).
- **Existing seams** — `robust/rt/ztt.py` (M2), the M1 loader/splits, `types.py`.

## Prompts

1. Read this doc. Execute the 1st task in the "M3" section below.
2. Read this doc. Execute the 2nd task in the "M3" section below.

## M3

### Tasks

1. **Emulator.** `robust/rt/emulator.py`: a **small Flax MLP** `ΔRrs`, inputs e.g.
   `(u` or `(ω_bw, ω_bp)`, `B_p`, geometry, λ`)`; train with **Optax** on
   `Rrs_L23 − Rrs_ZTT`, **relatively weighted**; L2 / size regularization so the residual
   stays small. Train on the M1 train split only.

2. **Hybrid + gate.** `robust/rt/hybrid.py`: `forward()` = `Rrs_ZTT + ΔRrs` with a
   `mode ∈ {ztt, emulator, hybrid}` flag (so the three design-doc options compare on
   identical data). Tests in `robust/tests/test_hybrid.py`:
   - hybrid **beats standard Gordon** rRMS on the **train** split at all three solar
     zeniths;
   - `jax.grad` finite-difference check on the **full `forward`**;
   - **throughput** (scenes·λ/s, batched) recorded — must not collapse vs ZTT alone.

   Update the implementation record; note the branch for JXP.

### Q&A

## Next

→ `rt_elastic_coding_prompt_5.md` (M4: Validation — prototype done).

## Logging

### <Date> (Short summary)

<Detailed description>

## Logs
