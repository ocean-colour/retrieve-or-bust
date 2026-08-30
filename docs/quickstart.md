# Quickstart

Five minutes, one scene, one gradient. By the end of this page you will have
run the elastic forward model on a real Loisel et al. (2023) scene, switched
the two inelastic emission terms on and seen what they change, and
differentiated the result with respect to both an inherent optical property and
the fluorescence quantum yield.

Everything below was executed in the project environment on 2026-08-30, and
every printed line is pasted verbatim from that run. It needs **no `$OS_COLOR`
and no downloaded data** — the scene comes from a fixture committed to the
repository. If your numbers differ in the last digit or two, see the note about
float32 reproducibility on the [Installation](installation.md) page.

## 1. Check the environment

```python
import jax
import jax.numpy as jnp

import robust
import robust.rt as rt

print("robust", robust.__version__)
print("jax   ", jax.__version__)
print("devices", jax.devices())
```

```text
robust 0.0.dev0
jax    0.11.0
devices [CpuDevice(id=0)]
```

`robust.rt` re-exports everything this page uses: the function `forward`, and
the four argument types `IOPs`, `PhaseParams`, `Geometry` and `Inelastic`.

## 2. Load an L23 scene

`robust.rt.data.l23` reads the L23 HydroLight archive, but it also takes a
*reader*, and the repository ships a 50-scene fixture that feeds the real
loader with the real file's contents. That is what the test suite uses, and it
is what makes this page runnable anywhere.

```python
from pathlib import Path

from robust.rt.data import l23

FILES = Path(robust.__file__).parent / "tests" / "files"
batch = l23.load_inelastic_batch(
    reader=l23.inelastic_npz_reader(
        FILES / "l23_inelastic_fixture.npz", FILES / "l23_small.npz"
    )
)
print(f"samples {batch.n_sample}   wavelengths {batch.n_wave}")
print(f"wave    {batch.wave[0]:.0f} ... {batch.wave[-1]:.0f} nm")
print(f"zeniths {sorted(set(batch.zenith.tolist()))}")
```

```text
samples 150   wavelengths 81
wave    350 ... 750 nm
zeniths [0.0, 30.0, 60.0]
```

150 samples is 50 IOP scenes at each of three solar zeniths, on the canonical
81-point 350–750 nm grid. The *inelastic* loader is used rather than
`load_batch` because it carries `a_ph`, the phytoplankton absorption that the
fluorescence source term needs. For elastic-only work, `l23.load_batch(...)`
with the `l23_small.npz` fixture is the lighter option.

## 3. Build the three model arguments

`forward` takes pytrees, not a batch object. Pulling one sample out of the
batch gives exactly the three arguments a caller with their own IOPs would
construct by hand:

```python
i = 0  # first scene, first solar zenith

iops = rt.IOPs(
    a=batch.iops.a[i],
    bb_w=batch.iops.bb_w[i],
    bb_p=batch.iops.bb_p[i],
    a_ph=batch.iops.a_ph[i],
)
phase_params = rt.PhaseParams(B_p=batch.phase_params.B_p[i])
geometry = rt.Geometry.nadir(batch.geometry.theta_s[i])
wave = batch.wave

iops.validate(wave=wave)
phase_params.validate()
geometry.validate()

print(f"theta_s   {float(geometry.theta_s):.0f} deg (view: nadir)")
print(f"a(440)    {float(jnp.interp(440.0, wave, iops.a)):.4f} m^-1")
print(f"a_ph(440) {float(jnp.interp(440.0, wave, iops.a_ph)):.4f} m^-1")
print(f"bb_p(440) {float(jnp.interp(440.0, wave, iops.bb_p)):.5f} m^-1")
print(f"B_p(440)  {float(jnp.interp(440.0, wave, phase_params.B_p)):.4f}")
```

```text
theta_s   0 deg (view: nadir)
a(440)    0.0167 m^-1
a_ph(440) 0.0037 m^-1
bb_p(440) 0.00072 m^-1
B_p(440)  0.0132
```

Three things worth noticing about the arguments:

- **`IOPs` splits backscattering into `bb_w` and `bb_p`.** Water
  backscattering is a known constant of the medium; particulate
  backscattering is what a retrieval is actually after. Keeping them apart in
  the type means the split is never a caller's guess.
- **`a_ph` is optional.** Leave it out and you get the elastic model plus
  Raman; fluorescence needs it, because `b_F = phi_C · a_ph` *is* the source
  term. That is a physical requirement, not an API preference.
- **`validate()` is a boundary check.** Call it on the way in, not inside
  `jit`.

The scene drawn here is clear water — `a(440) = 0.017 m⁻¹` — which is why the
inelastic contributions below are as large as they are relative to the elastic
signal.

## 4. Run the elastic forward model

```python
Rrs = rt.forward(iops, phase_params, geometry, wave)

print(f"Rrs shape {Rrs.shape}, dtype {Rrs.dtype}")
for lam in (440.0, 550.0, 685.0):
    print(f"  Rrs({lam:.0f}) = {float(jnp.interp(lam, wave, Rrs)):.6e} sr^-1")
```

```text
Rrs shape (81,), dtype float32
  Rrs(440) = 8.527968e-03 sr^-1
  Rrs(550) = 1.207069e-03 sr^-1
  Rrs(685) = 7.285256e-05 sr^-1
```

That is the default `mode='hybrid'`: the Twardowski & Tonizzo analytic backbone
plus the trained residual emulator, above-water remote-sensing reflectance in
sr⁻¹. There are no leading batch axes here because one sample was selected; the
same call over `batch.iops` directly would return `(150, 81)`.

## 5. Switch the inelastic terms on

The fifth argument of `forward` is keyword-only. `inelastic=None`, the default,
is the elastic model and is bit-identical to it by construction — the `None`
branch takes the pre-existing code route rather than multiplying by one.
Passing an `Inelastic()` instance turns on Raman scattering by water and
chlorophyll-a fluorescence together:

```python
Rrs_inel = rt.forward(
    iops, phase_params, geometry, wave, inelastic=rt.Inelastic()
)

print(f"identical to elastic? {bool(jnp.all(Rrs_inel == Rrs))}")
for lam in (440.0, 550.0, 685.0):
    e = float(jnp.interp(lam, wave, Rrs))
    t = float(jnp.interp(lam, wave, Rrs_inel))
    print(f"  {lam:.0f} nm: {e:.6e} -> {t:.6e}   ({100 * (t / e - 1):+.2f} %)")
```

```text
identical to elastic? False
  440 nm: 8.527968e-03 -> 9.241643e-03   (+8.37 %)
  550 nm: 1.207069e-03 -> 1.505374e-03   (+24.71 %)
  685 nm: 7.285256e-05 -> 1.225158e-04   (+68.17 %)
```

## 6. Where the difference lives

Raman scattering redistributes blue photons into a broad gain across the red;
fluorescence adds a narrow emission peak at 685 nm. Both show up where the
elastic signal is weakest, which is why they matter far more than their
absolute magnitude suggests.

```python
band = (wave >= 550.0) & (wave <= 700.0)
rel = (Rrs_inel - Rrs) / Rrs

print(f"550-700 nm ({int(band.sum())} bands)")
print(f"  median increment {100 * float(jnp.median(rel[band])):+.2f} %")
print(f"  max    increment {100 * float(jnp.max(rel[band])):+.2f} %"
      f" at {float(wave[band][jnp.argmax(rel[band])]):.0f} nm")

# Raman alone vs fluorescence alone, at the 685 nm emission peak.
Rrs_raman = rt.forward(iops, phase_params, geometry, wave,
                       inelastic=rt.Inelastic(fluorescence=False))
Rrs_fl = rt.forward(iops, phase_params, geometry, wave,
                    inelastic=rt.Inelastic(raman=False))
at685 = lambda y: float(jnp.interp(685.0, wave, y))
print(f"685 nm: elastic {at685(Rrs):.6e}")
print(f"        + Raman only      {at685(Rrs_raman):.6e}"
      f"  ({100 * (at685(Rrs_raman) / at685(Rrs) - 1):+.2f} %)")
print(f"        + fluorescence only {at685(Rrs_fl):.6e}"
      f"  ({100 * (at685(Rrs_fl) / at685(Rrs) - 1):+.2f} %)")
print(f"        + both              {at685(Rrs_inel):.6e}"
      f"  ({100 * (at685(Rrs_inel) / at685(Rrs) - 1):+.2f} %)")
```

```text
550-700 nm (31 bands)
  median increment +27.92 %
  max    increment +68.17 % at 685 nm
685 nm: elastic 7.285256e-05
        + Raman only      9.287027e-05  (+27.48 %)
        + fluorescence only 1.024981e-04  (+40.69 %)
        + both              1.225158e-04  (+68.17 %)
```

The band maximum falls exactly on 685 nm, the fluorescence emission peak, and
the split confirms why: Raman contributes +27.48 % there and fluorescence
+40.69 %, which sum to the +68.17 % of the two together. That additivity is not
a coincidence — the composition law multiplies the elastic spectrum by the
Raman factor and *adds* the fluorescence term, so the two increments are
independent of each other by construction.

`raman` and `fluorescence` are *static* fields: they select code paths, so
`jit` specializes on them, one compilation per configuration.

:::{note}
These are single-scene numbers from a clear-water scene, shown to make the
mechanics concrete. They are **not** the model's accuracy. The measured
per-process errors over the L23 ensemble, and the conditions under which they
degrade, live in the two reports in `reports/` and are summarised on the
scope-and-limitations page.
:::

## 7. Take a gradient

This is the point of writing the model in JAX. `forward` is differentiable with
respect to every leaf of every argument, so a gradient with respect to
absorption and a gradient with respect to the fluorescence quantum yield are
the same one-line operation.

```python
def Rrs685(iops, inelastic):
    """Rrs at the 685 nm fluorescence peak — a scalar, so ``grad`` applies."""
    out = rt.forward(iops, phase_params, geometry, wave, inelastic=inelastic)
    return jnp.interp(685.0, wave, out)


inel = rt.Inelastic()

d_iops = jax.grad(Rrs685, argnums=0)(iops, inel)
d_inel = jax.grad(Rrs685, argnums=1)(iops, inel)

print(f"dRrs(685)/da     : shape {d_iops.a.shape}, "
      f"at 685 nm {float(jnp.interp(685.0, wave, d_iops.a)):+.4e}")
print(f"                   at 440 nm {float(jnp.interp(440.0, wave, d_iops.a)):+.4e}")
print(f"dRrs(685)/da_ph  : at 685 nm "
      f"{float(jnp.interp(685.0, wave, d_iops.a_ph)):+.4e}")
print(f"dRrs(685)/dphi_C : {float(d_inel.phi_C):+.6e} sr^-1")
```

```text
dRrs(685)/da     : shape (81,), at 685 nm -2.3508e-04
                   at 440 nm -1.1346e-06
dRrs(685)/da_ph  : at 685 nm +2.3789e-04
dRrs(685)/dphi_C : +1.482279e-03 sr^-1
```

Read the signs, because they are the physics:

- `∂Rrs(685)/∂a` is **negative at 685 nm** — more absorption there, less
  reflectance — and, with the inelastic terms on, also nonzero at **440 nm**,
  four orders of magnitude smaller. Absorption at 440 nm cannot affect an
  *elastic* reflectance at 685 nm at all, so that number is entirely an
  inelastic effect.
- `∂Rrs(685)/∂a_ph` is **positive**, and comparable in magnitude to the `a`
  derivative. Phytoplankton absorption enters twice — once inside the total
  `a`, where it darkens, and once as the fluorescence source `b_F = phi_C ·
  a_ph`, where it brightens.
- The gradient comes back *labelled*: `grad` of a scalar function of an `IOPs`
  returns an `IOPs`, and of an `Inelastic` returns an `Inelastic`. There is no
  index bookkeeping to get wrong.

Turning the two processes on one at a time says which one owns the 440 nm
sensitivity. Same gradient, four configurations:

| `inelastic=` | `∂Rrs(685)/∂a` at 440 nm | at 685 nm | `∂Rrs(685)/∂a_ph` at 685 nm |
| --- | --- | --- | --- |
| `None` (elastic) | `+0.0000e+00` | `-1.4788e-04` | `+0.0000e+00` |
| `Inelastic(fluorescence=False)` | `+0.0000e+00` | `-1.8272e-04` | `+0.0000e+00` |
| `Inelastic(raman=False)` | `-1.1346e-06` | `-2.0024e-04` | `+2.3789e-04` |
| `Inelastic()` | `-1.1346e-06` | `-2.3508e-04` | `+2.3789e-04` |

It is **fluorescence**, not Raman. Fluorescence excitation is broadband, so
absorption anywhere in the blue-green changes how much light is available to
excite the 685 nm emission. Raman excitation is not broadband: a 685 nm
emission comes from one place, the 3400 cm⁻¹ Stokes shift below it, and sure
enough the Raman-only gradient with respect to `a` is exactly zero everywhere
except three wavelengths on the 81-point grid —

```text
raman-only d/da nonzero at wavelengths: [555. 560. 685.]
values: [-3.0453191e-05 -4.1577973e-06 -1.8272053e-04]
```

— the 685 nm emission band itself, and the two grid points bracketing
1 / (1/685 nm + 3400 cm⁻¹) = **555.6 nm**, its excitation wavelength. The
gradient found the physics without being told about it.

Because the fluorescence term is linear in `phi_C` by construction
(`Rrs_fl = phi_C · K_fl`), that last derivative *is* the fluorescence kernel —
which is checkable in one line:

```python
print(f"phi_C * dRrs/dphi_C     = {float(d_inel.phi_C) * float(inel.phi_C):.6e}")
print(f"Rrs_fl(685) - Rrs(685)  = {at685(Rrs_fl) - at685(Rrs):.6e}")
```

```text
phi_C * dRrs/dphi_C     = 2.964558e-05
Rrs_fl(685) - Rrs(685)  = 2.964558e-05
```

Identical to every printed digit, as linearity requires.

## Where to go next

- [The model in one page](model/overview.md) — the composition law, what each
  term is, and which module owns it.
- [API reference](api.rst) — `forward`'s full signature, the three `mode`
  values, and the four pytrees.
- An **executed notebook version** of this page, with the spectra plotted,
  arrives as `docs/quickstart_nb.ipynb`. It is written at the D2 milestone and
  is not part of the site yet; until then this page is the runnable
  introduction.
