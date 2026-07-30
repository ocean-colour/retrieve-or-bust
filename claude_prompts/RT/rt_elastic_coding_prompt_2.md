# Elastic RT Coding — Prompt 2 (M1: Data & conventions)

## Goals

Implement **Milestone M1**: the data layer and shared conventions — the `Rrs↔rrs`
convention, the IOP/phase/geometry types, and a one-call **L23 loader** that returns JAX
arrays with the `B_p` phase parameter and the held-out splits. This is the foundation
every later milestone consumes.

## Claude

### Skills

Consider using `.claude/skills/` (`code-review`, `verify`) as helpful.

### Working agreements

See `rt_elastic_coding_prompt_1.md` → *Working agreements* (git by JXP; `ocean14`;
CPU-only JAX; reuse `ocpy`/`bing`; BING test conventions; pytest-gated; Fable; log).

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` §M1.
- **Design** — `design/rt_elastic_model.md` §3 (interface/data model) and the A=0.52,
  B=1.7 convention.
- **L23 loader** — `ocpy.hydrolight.loisel23.load_ds(X, Y)` → `Hydrolight{X}{Y:02d}.nc`;
  use **X=1** (elastic) for **Y∈{0,30,60}** (solar zenith). Data at
  `$OS_COLOR_DATA/Loisel2023/` (this laptop: `/Users/xavier/data/Color/Loisel2023/`).
  Variables include `Rrs, a, bb, bbnw, bnw, ...` (3320 scenes × 81 λ, 350–750 nm).
- **Implementation record** — `design/rt_elastic_implementation.md` (update at close).

## Prompts

1. Read this doc. Execute the 1st task in the "M1" section below.
2. Read this doc. Execute the 2nd task in the "M1" section below.
3. Read this doc. Execute the 3rd task in the "M1" section below.

## M1

### Tasks

1. **Conventions.** `robust/rt/conventions.py`: `A_RRS=0.52, B_RRS=1.7`;
   `Rrs_to_rrs`/`rrs_to_Rrs`; the canonical wavelength grid (L23 350–750, 81 bands);
   pure-water `bb_w(λ)`; load-time asserts. Test: `Rrs→rrs→Rrs` round-trips to ~1e-6;
   asserts fire on bad input.

2. **Types.** `robust/rt/types.py`: `IOPs(a, bb_w, bb_p)`, `PhaseParams(B_p, …)`,
   `Geometry(theta_s, theta_v, dphi, wind)` as JAX pytrees with `jaxtyping` shapes.

3. **L23 loader + splits.** `robust/rt/data/l23.py`: load the elastic set via `ocpy`
   for Y∈{0,30,60}; assemble `(IOPs, Geometry, Rrs)` JAX batches; compute
   `B_p = bbnw / bnw`; expose the **seeded splits** (random 20% of scenes; and the
   solar-zenith hold-out: train 0°/30°, test 60°).

   **Gate.** `test_conventions.py` + `test_l23.py`: shapes `(3320, 81)`; `a, bb ≥ 0`;
   `B_p` within ~[0.004, 0.03]; a **golden-value** row cross-checked against the raw
   netCDF. Update the implementation record; note the branch for JXP.

### Q&A

## Next

→ `rt_elastic_coding_prompt_3.md` (M2: ZTT-in-JAX backbone).

## Logging

### <Date> (Short summary)

<Detailed description>

## Logs
