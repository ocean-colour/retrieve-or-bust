# Elastic RT Coding — Prompt 1 (M0: Environment & scaffold)

## Goals

Implement **Milestone M0** of the coding plan
(`design/rt_elastic_model_coding_plan.md`): stand up the environment and the package
scaffold for the differentiable elastic-RT forward model, and create the running
implementation record. Nothing scientific yet — the point is a green, importable base.

## Claude

### Skills

Consider using the skills in `.claude/skills/` (e.g. `critical-partner`, `code-review`,
`verify`) as helpful.

### Working agreements (hold for every M-prompt)

- **Git is handled by JXP** (per `CLAUDE.md`). Work on a branch (suggest
  `rt-elastic-prototype`); each milestone is a reviewable commit/PR. Do **not** run
  state-changing git commands; read-only inspection is fine.
- **Python only**, in the `ocean14` conda env; **CPU-only JAX** for now.
- **Reuse, don't reinvent.** Build on `ocpy` (the `loisel23` L23 loader) and the
  installed `bing` package; follow **BING conventions** — tests in `robust/tests/` as
  `test_*.py` with a `conftest.py` and a `files/` fixtures dir.
- **Every milestone is `pytest`-gated.** Accuracy gates are *relative* (no blind
  targets). Use Fable if you can. Log your work.

## Context

Read before coding:

- **Coding plan** — `design/rt_elastic_model_coding_plan.md` (M0 section + Ground rules
  + Package layout).
- **Design** — `design/rt_elastic_model.md` (the `forward(iops, phase_params, geometry)
  → Rrs` API and the `robust/rt/` module layout).
- **BING conventions** — `Oceanography/python/bing/bing/` (submodule layout) and
  `bing/bing/tests/` (`test_*.py`, `conftest.py`, `files/`).

## Prompts

1. Read this doc. Execute the 1st task in the "M0" section below.
2. Read this doc. Execute the 2nd task in the "M0" section below.
3. Read this doc. Execute the 3rd task in the "M0" section below.

## M0

### Tasks

1. **Create the implementation record.** New file `design/rt_elastic_implementation.md`
   — the running log of what gets built each milestone (mirror PAB's
   `PAB_implementation.md`): a table of milestones with status, and a per-milestone
   section for modules added, tests, and results. Seed it with M0.

2. **Dependencies.** Add `jax`, `flax`, `optax`, `jaxtyping` (CPU) to `requirements.txt`
   and install into `ocean14`. Verify `import jax; jax.numpy.ones(3)` runs on CPU.
   Record versions in the implementation record.

3. **Scaffold the package.** Create `robust/rt/` with stub modules per the coding-plan
   layout (`__init__.py`, `types.py`, `conventions.py`, `data/l23.py`, `ztt.py`,
   `emulator.py`, `hybrid.py`, `validation.py`) and `robust/tests/` with `conftest.py`,
   an empty `files/`, and `test_env.py`.

   **Gate.** `pytest -q` collects and passes; `test_env.py` asserts `import jax` works on
   CPU and `from robust import rt` succeeds. Log the result and update the implementation
   record; note the branch for JXP to commit.

### Q&A

## Next

→ `rt_elastic_coding_prompt_2.md` (M1: Data & conventions).

## Logging

Record work in the Logs section below, format:

### <Date> (Short summary)

<Detailed description of the work and what you learned>

## Logs
