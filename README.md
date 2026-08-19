# retrieve-or-bust

[![CI](https://github.com/ocean-colour/retrieve-or-bust/actions/workflows/ci.yml/badge.svg)](https://github.com/ocean-colour/retrieve-or-bust/actions/workflows/ci.yml)

Our last best effort at IOP Retreivals

## Development

```bash
pip install -r requirements.txt     # full environment, incl. the CPU JAX stack
pip install -e . --no-deps          # the package itself
pytest -q                           # from the repo root
```

The Loisel+2023 reference data lives outside the repo; `ocpy` finds it via
`$OS_COLOR`. Tests that need it **skip** when it is absent, so `pytest -q` stays
meaningful either way (`-ra` prints the skip reasons).

The elastic radiative-transfer forward model lives in `robust/rt/` — see
[`design/rt_elastic_model.md`](design/rt_elastic_model.md) (design),
[`design/rt_elastic_model_coding_plan.md`](design/rt_elastic_model_coding_plan.md)
(milestones), [`design/rt_elastic_implementation.md`](design/rt_elastic_implementation.md)
(what is built), and the notebooks in [`notebooks/RT/`](notebooks/RT/).
