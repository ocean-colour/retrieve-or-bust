# Data

Everything `robust.rt` claims is measured against **one** dataset: the
Loisel et al. (2023) HydroLight archive, "L23". It is the model's training set,
its truth, and the limit of what the model has been shown to do. This page is
what a user needs in order to load it, and — since the archive is not in the
repository — what is still possible without it.

`robust.rt` reaches L23 through {mod}`robust.rt.data.l23`, a thin wrapper over
`ocpy.hydrolight.loisel23`. The wrapper's job is to turn netCDF files into the
`(IOPs, PhaseParams, Geometry, Rrs)` batches {func}`robust.rt.hybrid.forward`
takes, and to produce the seeded held-out splits every number in the reports was
computed on.

*Sources for this page: the {mod}`robust.rt.data.l23` module docstring and the
docstrings of {class}`~robust.rt.data.l23.L23Batch`,
{class}`~robust.rt.data.l23.L23InelasticBatch`, {class}`~robust.rt.data.l23.Splits`,
{func}`~robust.rt.data.l23.load_batch`, {func}`~robust.rt.data.l23.load_inelastic_batch`,
{func}`~robust.rt.data.l23.make_splits` and {func}`~robust.rt.data.l23.write_fixture`;
`robust/tests/conftest.py` for the skip markers;
[`design/rt_inelastic_model.md`](gh:design/rt_inelastic_model.md) §4.1 for the
scenario levels and truth channels;
[`reports/report_rt_elastic_model.md`](../reports/report_rt_elastic_model.md) §3
and [`reports/report_rt_inelastic_model.md`](../reports/report_rt_inelastic_model.md)
§3 for the splits and the reference-data description. Every number and every
pasted output on this page was measured in this environment when the page was
written — the release-scale figures with `$OS_COLOR` set, the fixture-scale ones
under `env -u OS_COLOR`.*

## Where the data lives, and `$OS_COLOR`

L23 is published on Dryad ([doi:10.6076/D1630T](https://datadryad.org/stash/dataset/doi:10.6076/D1630T))
and is not redistributed here. `ocpy` locates it from one environment variable:

```text
$OS_COLOR/Loisel2023/Hydrolight{X}{Y:02d}.nc
```

with `X` the scenario level and `Y` the solar zenith in degrees. Measured on this
machine:

```console
$ echo $OS_COLOR
/Users/xavier/Projects/Oceanography/data/Color/

$ ls -l $OS_COLOR/Loisel2023/Hydrolight100.nc
-rw-r--r--  1 xavier  staff  18304505  Nov 11  2023  Hydrolight100.nc
```

**18,304,505 bytes each** — 17.5 MiB, which is the "~17 MB" the repository's
own docstrings quote. The model needs nine of them: `Hydrolight1{00,30,60}.nc`
for the elastic set and the six X = 2 / X = 4 files, about 165 MB in total. (The
archive also ships a `*_profile.nc` sibling for each, 727 MB apiece; `robust.rt`
never opens those.)

:::{warning}
**`ocpy` resolves the directory at import time, not at call time.** Setting
`os.environ["OS_COLOR"]` *after* `ocpy.hydrolight.loisel23` has been imported has
no effect — the module has already computed its path. With the variable unset the
import warns and falls back to the working directory:

```text
UserWarning: OS_COLOR not set. Using current directory.
>>> loisel23.l23_path
'./'
```

and a load then fails on a path that makes the cause obvious:

```text
FileNotFoundError: [Errno 2] Unable to synchronously open file
(unable to open file: name = '.../retrieve-or-bust/Hydrolight100.nc', errno = 2)
```

Set the variable in the shell profile, before Python starts.
:::

## The scenario levels: X = 1, X = 2, X = 4

The three files at one zenith are the **same water body, three radiative-transfer
configurations**. That is the whole basis of the inelastic validation: the
difference between two scenarios is exact, per-scene truth for the process that
was switched on.

| X | HydroLight configuration | what it contains |
| --- | --- | --- |
| **1** | no inelastic processes | elastic scattering and absorption only |
| **2** | Raman scattering by water molecules | X = 1 **+ Raman** |
| **4** | Raman **and** chlorophyll-a fluorescence | X = 2 **+ fluorescence** at φ_C = 0.02 |

There is no X = 3 in the archive, and no fluorescence-without-Raman level — which
is why the fluorescence truth is a *difference* rather than a file. The
level numbering is `ocpy`'s and L23's, not this package's;
{data}`~robust.rt.data.l23.ELASTIC_X` is 1 and
{data}`~robust.rt.data.l23.INELASTIC_XS` is `(2, 4)`.
{data}`~robust.rt.data.l23.PHI_C_L23` = 0.02 is the fluorescence quantum yield
HydroLight ran X = 4 at, and is why {class}`robust.rt.types.Inelastic`'s `phi_C`
defaults to that value: the packaged default reproduces the truth channel.

The three zeniths are {data}`~robust.rt.data.l23.ZENITHS` = `(0, 30, 60)`
degrees, nadir view, one fixed Fournier–Forand phase function per scene.

### The truth channels

| channel | expression | what it is truth *for* |
| --- | --- | --- |
| Raman factor | `Rrs_x2 / Rrs_x1` — {attr}`~robust.rt.data.l23.L23InelasticBatch.truth_raman_factor` | the analytic `f_phys` and the corrected `f_R` ({doc}`../model/inelastic`) |
| fluorescence | `Rrs_x4 - Rrs_x2` — {attr}`~robust.rt.data.l23.L23InelasticBatch.truth_fluorescence` | `φ_C·K_fl` and the corrected form ({doc}`../model/fluorescence`) |
| total | `Rrs_x4` | the end-to-end gate ({doc}`validation`) |

Both derived channels have a sign that is a data check rather than a
convention. Measured over the full release in this environment:

```text
truth_raman_factor   1.004739 - 2.514770   (>= 1 everywhere: True)
truth_fluorescence(685 nm)   7.09e-06 - 1.65e-03 sr^-1   (> 0 everywhere: True)
```

Raman only ever adds photons, so a factor below 1 means a data or indexing
error, not a weak signal.

:::{note}
{attr}`~robust.rt.data.l23.L23InelasticBatch.truth_raman_factor`'s own docstring
gives the range as `1.0076-2.51`. That is the **0° file's** range — measured
here as 1.007599–2.514770 at Y = 0 — not the release's: at 60° the minimum falls
to 1.004739. The maximum is the same number. Nothing downstream depends on the
minimum, but the figure is quoted here as measured across all three zeniths.
:::

## The containers

Three dataclasses, none of them a registered pytree. That is deliberate: they are
analysis containers whose `scene` field is host-side integer metadata with no
business being traced or differentiated. The *model inputs* they hold
({class}`~robust.rt.types.IOPs`, {class}`~robust.rt.types.PhaseParams`,
{class}`~robust.rt.types.Geometry`) are pytrees, and are what `forward()` is
handed.

{class}`~robust.rt.data.l23.L23Batch` — the elastic set:
`iops`, `phase_params`, `geometry`, `Rrs`, `wave`, `scene`, with
{attr}`~robust.rt.data.l23.L23Batch.n_sample` /
{attr}`~robust.rt.data.l23.L23Batch.n_wave` /
{attr}`~robust.rt.data.l23.L23Batch.zenith` properties and a
{meth}`~robust.rt.data.l23.L23Batch.validate` boundary check.

{class}`~robust.rt.data.l23.L23InelasticBatch` — the inelastic sibling, same
layout, two differences: `iops` carries `a_ph` (the fluorescence source term)
**and** `a_cdom` (extracted from the release's `ag` field — the source term of
the off-by-default {doc}`../model/cdom_fluorescence` term, which the three
reference channels below do *not* contain), and the reference is three channels
(`Rrs_x1`, `Rrs_x2`, `Rrs_x4`) rather than one. Its
{meth}`~robust.rt.data.l23.L23InelasticBatch.validate` refuses a batch whose
`a_ph` or `a_cdom` is `None` — an inelastic batch without its source terms is a
loader bug, not a configuration.

{class}`~robust.rt.data.l23.Splits` — four boolean masks over the sample axis
plus the held-out scene indices: `scene_train` / `scene_test`,
`zenith_train` / `zenith_test`, `test_scenes`.

**One flat sample axis.** Both loaders stack `n_zenith × n_scene` samples along a
single leading axis, zenith-major, so every leaf shares the batch shape and
`jax.vmap(f, in_axes=0)` needs no thought. `scene` and `geometry.theta_s` label
each sample, which is what makes the per-zenith metrics and both splits
expressible as boolean masks.

## Loading

{func}`~robust.rt.data.l23.load_batch` and
{func}`~robust.rt.data.l23.load_inelastic_batch` are the two entry points; full
signatures are on the {doc}`../api` page under *data.l23*. Both take `zeniths`,
`scenes`, `validate` and a `reader` seam, and `load_batch` additionally takes
`x`. Measured on the full release in this environment:

```text
load_batch()             -> 9960 samples x 81 wavelengths, 350-750 nm
load_inelastic_batch()   -> 9960 samples x 81 wavelengths, three Rrs channels
```

9,960 = {data}`~robust.rt.data.l23.N_SCENES` (3,320) × three zeniths.

Two facts about `load_batch` worth knowing before reading a batch's numbers.
`bb_w` and `bb_p` come from **the file itself** (`bb - bbnw` and `bbnw`), not from
{func}`robust.rt.conventions.bb_w`, so a batch is exactly what L23 says with no
convention drift. The `conventions` table was *derived* from this same
difference, so the loop closes: measured here, the file's `bb - bbnw` at the
scene the table was taken from reproduces
{data}`~robust.rt.conventions.BB_W_L23` to **3.1e-09** relative, and
`test_conventions.py::test_bb_w_matches_l23_netcdf` asserts exactly that at
`rtol=1e-6`. Across all 3,320 scenes the spread widens to **3.4e-06** — float32
storage noise between scenes, not a convention disagreement, and the reason the
scene-independence test tolerates 1e-6 rather than demanding equality. And
`B_p = bbnw / bnw` is **warned about, never clipped**, if it leaves
{data}`~robust.rt.data.l23.B_P_EXPECTED` = (0.004, 0.03): silently squashing it
would hide a change in the reference data. Measured over the release:
`B_p ∈ [0.01026, 0.01800]` across all 806,760 (sample, wavelength) values — a
factor 1.75, against the design's ~7× nominal band, which is the honest limit on
what any phase-function claim here can mean ({doc}`../model/ztt`).

### The splits, and why they are by scene

{func}`~robust.rt.data.l23.make_splits` builds both held-out sets from one seed:

```text
make_splits(load_batch())
  scene_train  7968      scene_test  1992      test_scenes 664
  zenith_train 6640      zenith_test 3320
```

- **The random split holds out water bodies, not samples.** Every zenith of a
  held-out scene is held out with it. Splitting per sample would leak: the same
  IOPs appear three times, and a model could be tested on water it had already
  seen at another sun angle. 664 scenes = 20 %
  ({data}`~robust.rt.data.l23.TEST_FRACTION`) of 3,320.
- **The geometry split holds out one sun angle entirely** — train on 0°/30°, test
  on {data}`~robust.rt.data.l23.HELD_OUT_ZENITH` = 60°. This is the split that
  produces the unseen-zenith cliff the reports lead with, and it is a
  diagnostic, not a gate.
- {data}`~robust.rt.data.l23.SPLIT_SEED` = 23, **fixed forever**: changing it
  invalidates every held-out number in the record.

`make_splits` accepts either container unchanged — it reads only `scene` and
`zenith` — which is what "the inelastic effort reuses the elastic splits
verbatim" means mechanically. Verified here rather than taken on trust: all five
`Splits` fields are equal between `make_splits(elastic_batch)` and
`make_splits(inelastic_batch)`. {func}`~robust.rt.data.l23.select` and
{func}`~robust.rt.data.l23.select_inelastic` apply a mask to a batch.

## The committed fixtures

`robust/tests/files/` holds four `.npz` files, all committed, 689 kB in total:

| file | bytes | what it is |
| --- | --- | --- |
| `l23_small.npz` | 217,744 | the **elastic fixture** — the first 50 scenes × 3 zeniths of the *raw* L23 fields |
| `l23_inelastic_fixture.npz` | 291,450 | the **inelastic sibling** — the same 50 scenes, adding `aph`, `ag` and the X = 2 / X = 4 `Rrs` |
| `elastic_reference_outputs.npz` | 90,040 | pinned elastic `Rrs`/`rrs` for the bit-identity regression |
| `inelastic_default_reference_outputs.npz` | 89,907 | the same for the default inelastic model |

The design decision that makes these worth more than their size: **the fixtures
store the loader's *input*, not its output.** A snapshot of a
{class}`~robust.rt.data.l23.L23Batch` would let a test check only that the
snapshot is unchanged. Storing the raw per-file fields means
{func}`~robust.rt.data.l23.load_batch` itself runs, against real HydroLight
numbers, everywhere the fixture is present — CI included.

{func}`~robust.rt.data.l23.npz_reader` and
{func}`~robust.rt.data.l23.inelastic_npz_reader` are the readers that feed them
to the real loaders. Both **refuse** rather than improvise: `npz_reader` raises if
asked for a scenario or zenith the fixture does not hold, and
`inelastic_npz_reader` cross-checks `a`, `bb` and `Rrs1` against the elastic
fixture so the two files provably describe the same water. Regenerating them
needs the archive: {func}`~robust.rt.data.l23.write_fixture` for the elastic one
(which validates the candidate through the real loader before an atomic
`os.replace`), `design/py/gen_inelastic_fixture.py` for the sibling.

## Without the netCDFs

Concretely, and measured rather than asserted. With `$OS_COLOR` unset, this runs:

```python
from pathlib import Path
from robust.rt.data import l23

FILES = Path("robust/tests/files")

batch = l23.load_batch(reader=l23.npz_reader(FILES / "l23_small.npz"))

inelastic = l23.load_inelastic_batch(
    reader=l23.inelastic_npz_reader(
        FILES / "l23_inelastic_fixture.npz", FILES / "l23_small.npz"
    )
)
splits = l23.make_splits(inelastic)
```

```text
load_batch(elastic fixture):    150 samples x 81 wavelengths, zeniths [0, 30, 60], 50 scenes
load_inelastic_batch(fixtures): 150 samples x 81 wavelengths, 350-750 nm
  a_ph set: True   a_cdom set: True   Rrs_x1/x2/x4 all (150, 81)
make_splits: scene_train 120  scene_test 30  zenith_train 100  zenith_test 50
             test_scenes 10   (identical to the elastic batch's masks: True)
```

So without the archive you can: build both batch types from real HydroLight
numbers; run `forward()` in every mode, elastic and inelastic, since the emulator
and both correction heads are packaged with the distribution
({doc}`../installation`); reproduce the splits; and run **the entire validation
protocol** — every function on the {doc}`validation` page, including the gradient
reports and the speed ratio — at fixture scale. That was exercised in writing
these two pages: all of it ran under `env -u OS_COLOR`.

What you cannot do is reproduce the reports' numbers. Fifty scenes is not 3,320,
and ten held-out scenes is not 664, so fixture-scale metrics are the right
*shape* and the wrong *value*. The release-scale gate lines skip themselves
rather than lying about it — the markers are `needs_l23` and
`needs_l23_inelastic` in `robust/tests/conftest.py`, and each checks that `ocpy`
imports *and* that every required file is on disk:

```console
$ pytest -q -ra
483 passed, 3 skipped in 64.35s

$ env -u OS_COLOR pytest -q -ra
446 passed, 40 skipped, 1 warning in 56.65s
```

Thirty-seven tests convert from passed to skipped — **23** with the reason
`L23 elastic Hydrolight data not available ($OS_COLOR)` and **14** with its
`(X=2/X=4)` sibling, tallied from the `-ra` summary.

Two of the skips in each run — `test_elastic_hash_regression_strict` and
`test_gate_4_pre_change_pins` — are the same in both and are neither data- nor
docs-related: they are the strict bitwise hash tiers, which run only on the
machine whose pins they carry and skip with a reason elsewhere.
{doc}`../installation` explains the mechanism, and {doc}`validation` says what
it means for the acceptance gate.

:::{note}
Treat those counts as a snapshot. They were measured on a checkout under
concurrent development and climbed from 451 to 483 passing over three weeks. The
durable facts are the shape: no failures, the two machine-anchored hash skips,
one pre-existing skip, a ~37-test skip delta.
:::

Training is the one thing that genuinely requires the archive.
`design/py/train_emulator.py`, the correction-head training in
`design/py/`, and `design/py/run_validation.py --inelastic` all need
`$OS_COLOR`; the shipped weights mean no user has to run them
({doc}`../model/corrections`).

## API

| What | Where |
| --- | --- |
| Containers | {class}`~robust.rt.data.l23.L23Batch`, {class}`~robust.rt.data.l23.L23InelasticBatch`, {class}`~robust.rt.data.l23.Splits` |
| Loading | {func}`~robust.rt.data.l23.load_batch`, {func}`~robust.rt.data.l23.load_inelastic_batch` |
| Splitting and subsetting | {func}`~robust.rt.data.l23.make_splits`, {func}`~robust.rt.data.l23.select`, {func}`~robust.rt.data.l23.select_inelastic` |
| The fixture seam | {func}`~robust.rt.data.l23.npz_reader`, {func}`~robust.rt.data.l23.inelastic_npz_reader`, {func}`~robust.rt.data.l23.write_fixture`, {data}`~robust.rt.data.l23.RAW_FIELDS`, {data}`~robust.rt.data.l23.INELASTIC_RAW_FIELDS` |
| Dataset constants | {data}`~robust.rt.data.l23.ELASTIC_X`, {data}`~robust.rt.data.l23.INELASTIC_XS`, {data}`~robust.rt.data.l23.ZENITHS`, {data}`~robust.rt.data.l23.N_SCENES`, {data}`~robust.rt.data.l23.PHI_C_L23`, {data}`~robust.rt.data.l23.B_P_EXPECTED` |
| Split configuration | {data}`~robust.rt.data.l23.SPLIT_SEED`, {data}`~robust.rt.data.l23.TEST_FRACTION`, {data}`~robust.rt.data.l23.HELD_OUT_ZENITH` |

Full signatures are on the {doc}`../api` page under *data.l23*. What is done with
these batches is {doc}`validation`; what the model does with them is
{doc}`../model/overview`.
