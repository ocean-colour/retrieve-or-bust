# Installation

```{note}
Stub. This page is written at D1 task 4, where every command below is run in
this environment and its real output pasted in. Until then it lists only what
the repository mechanics already state.
```

- Python ≥ 3.12 (the floor declared in `setup.py`).
- The JAX stack (`jax`, `flax`, `optax`, `jaxtyping`) plus the sibling
  packages `ocpy` and `bing` come from the repository's root
  `requirements.txt`.
- The package itself installs with `pip install -e . --no-deps`.
- `robust/rt/files/*.npz` — the trained emulator and correction-head weights —
  ship via `package_data`; without them `forward(..., mode='hybrid')` fails at
  the first call.
