# Installation

Everything on this page was run on 2026-08-30 in the project's development
environment (the `ocean14` conda environment, macOS/arm64), and the outputs are
pasted as they came back. Two commands are exceptions, and both say so where
they appear: the CUDA JAX wheel, which needs an NVIDIA device this machine does
not have, and the full `pip install -r requirements.txt`, which was run as
`--dry-run` here because — as the warning below shows with its own output — the
real thing would have replaced two editable checkouts. Nothing on this page is
an invented transcript.

## Requirements

- **Python ≥ 3.12.** That is the floor declared in `setup.py`
  (`python_requires='>=3.12'`). ReadTheDocs and the CI documentation job both
  build on 3.12; the development environment used for the numbers on this site
  is newer:

  ```console
  $ python --version
  Python 3.14.6
  ```

- **The JAX stack** — `jax`, `jaxtyping` at import time, plus `flax` and
  `optax` for emulator/correction-head *training*. `robust.rt` imports only
  `jax`, `jaxtyping` and `numpy` at module level; `flax`, `optax` and `ocpy`
  are imported inside the functions that need them.
- **`ocpy`**, only if you want to load the Loisel et al. (2023) archive
  yourself. The committed test fixtures do not need it.

## Getting the code

```console
$ git clone https://github.com/ocean-colour/retrieve-or-bust.git
Cloning into 'retrieve-or-bust'...
$ cd retrieve-or-bust
```

## Dependencies

The root `requirements.txt` carries the full working set: the scientific stack,
the CPU JAX block, and the two sibling packages `ocpy` and `bing`.

```console
$ pip install -r requirements.txt
```

:::{warning}
**On a development machine this will replace your editable `bing` and `ocpy`
checkouts.** The last two lines of `requirements.txt` are

```text
git+https://github.com/ocean-colour/ocpy
git+https://github.com/ocean-colour/bing.git
```

and pip resolves them against GitHub, not against whatever you have checked out
locally. Measured here with `--dry-run`, in an environment whose `bing` and
`ocpy` are editable installs pointing at sibling working trees:

```console
$ pip install --dry-run -r requirements.txt
Collecting git+https://github.com/ocean-colour/ocpy (from -r requirements.txt (line 24))
  Cloning https://github.com/ocean-colour/ocpy to /private/var/folders/.../pip-req-build-_omzv_cw
  Resolved https://github.com/ocean-colour/ocpy to commit 3aed28acbeaad1e699ede06f049edd73b3eb41e9
Collecting git+https://github.com/ocean-colour/bing.git (from -r requirements.txt (line 25))
  Cloning https://github.com/ocean-colour/bing.git to /private/var/folders/.../pip-req-build-ys86fnwy
  Resolved https://github.com/ocean-colour/bing.git to commit f242b0ea4102e2f17a0cec5ec912f7fef0eb7c9c
Would install bing-0.0.dev0 ocpy-ocean-0.1.0
```

Note that the GitHub `ocpy` even installs under a *different distribution
name* (`ocpy-ocean`) than the local editable one (`ocpy`), so the two can
coexist in the metadata while only one wins on `import ocpy`.

If you are developing against local `bing`/`ocpy` branches, install the JAX
block on its own instead and leave the checkouts alone:

```console
$ pip install jax flax optax jaxtyping
```

Always run `pip install --dry-run …` first and read the "Would install" /
"Would uninstall" lines before committing to the real install. That is the
house procedure, and it exists because this exact line once cost a working
`bing` branch.
:::

## Installing the package

`robust` itself is installed without dependencies, because the previous step
already resolved them and a second resolution could only pull a different
`jax`:

```console
$ pip install -e . --no-deps
Successfully built retrieve-or-bust
Installing collected packages: retrieve-or-bust
  Attempting uninstall: retrieve-or-bust
    Found existing installation: retrieve-or-bust 0.0.dev0
    Uninstalling retrieve-or-bust-0.0.dev0:
      Successfully uninstalled retrieve-or-bust-0.0.dev0
Successfully installed retrieve-or-bust-0.0.dev0
```

`setup.py` reads the version out of `robust/__init__.py` by regex rather than
importing it, so packaging never needs `jax` present.

### What ships with the package

`setup.py`'s `package_data` block collects three globs — `rt/files/*.npz`,
`rt/data/*.npz` and `tests/files/*.npz` — because `find_packages()` collects
*modules* only, and data files have to be named out loud. The trained weights
are the ones that matter at run time:

```console
$ python -c "
from importlib.resources import files
d = files('robust') / 'rt' / 'files'
for p in sorted(d.iterdir()):
    print(f'  {p.name:<22} {p.stat().st_size:>6} bytes')
"
  emulator_l23.npz         6678 bytes
  fl_corr_l23.npz          4366 bytes
  raman_corr_l23.npz       4330 bytes
```

- `emulator_l23.npz` — the trained residual emulator. Without it the default
  `mode='hybrid'` raises at the first call, with the command that regenerates
  it:

  ```text
  FileNotFoundError: packaged emulator weights not found at .../emulator_l23.npz;
  regenerate with `python design/py/train_emulator.py` (needs $OS_COLOR and the
  L23 netCDFs)
  ```

- `raman_corr_l23.npz`, `fl_corr_l23.npz` — the two learned inelastic
  correction heads. These fail *softer*: if they are missing, `forward` falls
  back to the analytic Raman and fluorescence terms and emits a single
  `MissingCorrectionWarning`, because the analytic backbone is a legitimate
  model in its own right. A warning rather than an error, but not silence.

Alongside them, `rt/data/ed_l23.npz` (1675 bytes) carries the three packaged
L23 downwelling-irradiance spectra, and `tests/files/*.npz` are the committed
50-scene fixtures the test suite and the [Quickstart](quickstart.md) run
against.

## CPU and GPU JAX

`requirements.txt` asks for plain `jax`, which pulls the matching **CPU**
`jaxlib`. That is deliberate: nothing in this repository needs a GPU, and every
number on this site was measured on CPU.

```console
$ python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
0.11.0 cpu [CpuDevice(id=0)]
```

For a CUDA 12 machine the wheel is `jax[cuda12]` instead of `jax` — swap that
one requirement and the rest of the install is unchanged. *This was not run
here*: the development machine has no NVIDIA device, so no transcript is
pasted. Note that on a machine that *does* have one, the CPU wheel prints
`An NVIDIA GPU may be present … Falling back to cpu` once per process; that
line is expected log noise from a CPU-only install, not a failure.

## The L23 data and `$OS_COLOR`

The Loisel et al. (2023) HydroLight archive is not distributed with the
package. Point `$OS_COLOR` at the directory holding the data collections; the
loader reads `$OS_COLOR/Loisel2023/Hydrolight*.nc`.

```console
$ echo $OS_COLOR
/Users/xavier/Projects/Oceanography/data/Color/
```

**Without `$OS_COLOR` the package still works.** `robust.rt` imports, `forward`
runs, and the committed 50-scene fixtures under `robust/tests/files/` feed the
real loaders (they store the loader's *input*, not a snapshot of its output),
so the Quickstart and most of the suite need nothing extra. What you lose is
the full 3320-scene archive: the tests that need it carry the `needs_l23` /
`needs_l23_inelastic` markers and skip themselves.

## Verifying the install

Import and version:

```console
$ python -c "
import robust, robust.rt as rt
print('robust', robust.__version__)
print('forward', rt.forward.__module__ + '.' + rt.forward.__name__)
import jax; print('jax', jax.__version__, jax.devices())
"
robust 0.0.dev0
forward robust.rt.hybrid.forward
jax 0.11.0 [CpuDevice(id=0)]
```

Then the test suite. Both runs below come from a single invocation on
2026-08-30, **with** and then **without** the L23 archive:

```console
$ pytest -q -ra
...
SKIPPED [1] robust/tests/test_inelastic_corr.py:405: trained weights are committed; the fallback path is gone
FAILED robust/tests/test_inelastic_types.py::test_elastic_hash_regression_strict
FAILED robust/tests/test_inelastic_validation.py::test_gate_4_pre_change_pins
2 failed, 480 passed, 1 skipped in 62.84s (0:01:02)

$ env -u OS_COLOR pytest -q
...
2 failed, 445 passed, 36 skipped, 1 warning in 55.86s
```

Thirty-five tests convert from passed to skipped, every one of them with the
reason `L23 elastic Hydrolight data not available ($OS_COLOR)` or its
`(X=2/X=4)` inelastic sibling. That is the whole cost of not having the
archive: the model, the emulator, the correction heads and the pytrees are all
exercised from the committed fixtures either way.

:::{note}
**Treat the counts as a snapshot, not a contract.** They were measured on a
development checkout while a second effort was actively adding tests, and they
climbed by six between two runs an hour apart. What is stable, and what you
should actually check against, is the *shape*: the two failures named below,
one pre-existing skip, and a ~35-test skip delta when `$OS_COLOR` is unset.
:::

:::{note}
**About those two failures — they are expected on a machine other than the one
that anchored the pins, and they are not a broken install.**

Both are the same assertion, a SHA-256 pin on the bytes of the elastic
`Rrs`/`rrs` output over the committed fixture. The elastic pins were anchored
on a different machine from the inelastic ones, and float32 arithmetic is not
bit-reproducible across CPUs and JAX/XLA builds — so on any one machine one
strict set may fail while the other passes. The *closeness* tiers, which are
the guard that actually detects a changed computation, pass everywhere:
measured deviation from the committed reference here is at most **3.0 ULP**
(max relative 3.33e-07 on `Rrs`, 1.64e-07 on `rrs`).

The strict tiers are marked `skipif(CI)`, so this is a development-machine
gate only and GitHub Actions is unaffected. If you see these two and nothing
else, your install is fine. If you see a *closeness* failure, that is a real
finding.
:::

## Building this documentation

The documentation has its own, lighter requirements file — the Sphinx
toolchain plus the real import-time stack, deliberately not the root
`requirements.txt` (nothing here needs `ocpy`'s geospatial extras, and the
`git+` lines above are a build-breaking risk on ReadTheDocs):

```console
$ pip install -r docs/requirements.txt
$ pip install -e . --no-deps
$ python -m sphinx -b html -W --keep-going docs docs/_build/html
```

`-W` turns every warning into an error; that is how CI builds it, so a page
that is not in a toctree, or a `{doc}` reference to a page that does not exist,
fails the build rather than shipping. Autodoc genuinely imports `robust.rt`
here — nothing is mocked — which is why the JAX stack has to be present.
