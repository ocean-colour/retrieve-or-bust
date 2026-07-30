# Elastic RT Coding — Prompt 3 (M2: ZTT-in-JAX backbone)

## Goals

Implement **Milestone M2**: the **ZTT analytic backbone** in JAX — a differentiable
`Rrs_ZTT(iops, phase_params, geometry, wave)` with the particle phase function
(backward VSF) as an **explicit** input. This is the physically-interpretable half of
the hybrid and our analytical benchmark.

## Claude

### Skills

Consider `.claude/skills/` (`critical-partner` for checking the equation transcription;
`code-review`).

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements*.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M2 (and the Risks note on
  the placeholder backbone).
- **The ZTT paper** — `context/RT/twardowski2018.pdf` (Twardowski & Tonizzo 2018,
  *Applied Sciences*, "Ocean Color Analytical Model Explicitly Dependent on the VSF").
  Transcribe its forward relation; note the backward-VSF / phase-function parameters and
  any reference values/curves you can test against.
- **Synthesis** — `context/RT/rt_elastic_model.md` §2, §3.5 (the phase-function axis;
  `β(π)/bb` = 0.23 water vs 0.12–0.16 particles) and §4 (BING `rrs↔Rrs` convention).
- **Existing seams** — `robust/rt/{types,conventions}.py` and the M1 L23 loader.

## Prompts

1. Read this doc. Execute the 1st task in the "M2" section below.
2. Read this doc. Execute the 2nd task in the "M2" section below.

## M2

### Tasks

1. **Transcribe ZTT into JAX.** Read `twardowski2018.pdf`; implement the ZTT forward
   relation in `robust/rt/ztt.py` as pure JAX functions
   `Rrs_ZTT(iops, phase_params, geometry, wave)`, with the backward-VSF / `B_p` entering
   explicitly. Keep `phase_params` structured so the fuller ZTT backward-VSF parameters
   can replace `B_p` later without changing the signature. Document each transcribed
   equation with its paper reference (eq. number / section).

2. **Validate & gate.** Tests in `robust/tests/test_ztt.py`:
   - **(i)** `Rrs_ZTT` reproduces a **reference case from the paper** (a quoted value or
     digitized curve) to a stated tolerance.
   - **(ii)** **Gradient check** — `jax.grad` of `Rrs_ZTT` vs central finite differences
     agree (tolerance) w.r.t. `a, bb_p, B_p, geometry` (use float64).
   - **(iii)** Report (not gate) the standalone rRMS of `Rrs_ZTT` vs L23 at the three
     solar zeniths, alongside standard Gordon / O25 for reference.

   **De-risk (from the plan).** If any ZTT term is ambiguous or slow to pin down, land a
   Gordon/O25-in-JAX **placeholder backbone** first so M3–M4 proceed end-to-end, and
   flag the gap in the Q&A + implementation record; swap true ZTT in later without
   changing the `forward` signature. Update the implementation record; note the branch.

### Q&A

## Next

→ `rt_elastic_coding_prompt_4.md` (M3: Emulator + hybrid).

## Logging

### <Date> (Short summary)

<Detailed description>

## Logs
