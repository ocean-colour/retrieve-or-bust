"""
Pitarch & Brando (2024) multi-angular reference batches -- "PB24".

The dataset M5 is built on, and the first one this package has held that varies
the two axes the Week-1 prototype could not speak to: the **particle phase
function** and the **full BRDF**.

**What it is.** 5000 HydroLight realisations at ``$OS_COLOR/SD/v5``, each a
separate netCDF in two spectral resolutions -- ``SD_OLCI_no_R_NNNN.nc`` (12 OLCI
bands, 400-753 nm) and ``SD_hyp_no_R_NNNN.nc`` (451 bands, 350-800 nm at 1 nm) --
on a shared ``theta_s`` (10) x ``theta`` (10) x ``phi`` (13) grid, i.e. **1300
geometries per realisation**. Each file carries IOP *components*, both ``rrs``
and ``Rrs``, and ``Q``, ``mu_d``, ``mu_u``, ``mu_tot`` and seven K's. A
``.mat`` sidecar labels every realisation with one of 12 optical water classes.

**Measured properties** (M5 task 0; two independent samplings, one adversarial):

- The particle backscatter ratio **varies per realisation**: ``bbph/bph`` is flat
  in wavelength within a file (max/min <= 1.0008) but spans **0.0010-0.0358
  (~30x)** across files; ``bbNAP/bNAP`` spans 0.0100-0.0200. Bulk ``B_p`` spans
  **6.2x**, against L23's 1.7x. This is what makes a held-out-``B_p`` split
  constructible, and it is the reason M5 exists.
- Bulk ``B_p`` is **not** flat in wavelength (median 3% within a file, up to 17%)
  even though each component ratio is, because the phytoplankton/NAP mix shifts
  across the spectrum. Carried as a spectrum, like L23's.
- ``C``, ``N``, ``Y`` are **labels, not generators**: ``S_g``, ``S_NAP``,
  ``aNAP*``, ``aph*`` and ``bph/C`` all vary independently, and normalised
  ``aph`` shapes are drawn from a finite library and reused across files. So a
  split on chlorophyll is not a split on the IOPs.
- ``rrs`` is **exactly 0** at a handful of grazing geometries -- in the OLCI band
  set, only at 753 nm with ``theta_s = 80``, ``theta = 87.5`` (float32 underflow;
  ``Rrs`` is non-zero there). Since the metric divides by truth, see
  ``drop_zero_rrs``.

**The angle window.** JXP's Q14 answer sets the sanctioned envelope at
**0-70 degrees in both zeniths**, holding PB24's 80/87.75 shell out as a
deliberate extrapolation test. :func:`load_batch` therefore defaults to
``angles="window"``; ``angles="shell"`` is its complement and ``"all"`` is
everything. The zero-``rrs`` samples live entirely in the shell.

**Layout: one flat sample axis**, as in :mod:`robust.rt.data.l23` -- a sample is
one ``(realisation, theta_s, theta_v, dphi)`` combination carrying an ``n_wave``
spectrum, so every leaf shares the batch shape and ``jax.vmap(f, in_axes=0)``
works. Note what that costs: the IOPs depend only on the realisation and the
``mu``/K fields only on ``(realisation, theta_s)``, so flattening duplicates them
832x and 104x respectively. That is the price of a uniform batch, and it is why
**subsampling is an explicit argument** rather than a default: the full window is
5000 x 832 = 4.16 M samples.

**Nothing here is a hidden sample.** Every load returns a :class:`LoadReport`
saying how many realisations and geometries were kept out of how many, and how
many samples the zero-``rrs`` filter removed. A loader that quietly drops data is
indistinguishable from one that has none.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .. import conventions
from ..types import Geometry, IOPs, PhaseParams

__all__ = [  # noqa: RUF022  - grouped by role
    # Dataset constants
    "SUBDIR",
    "N_REALISATION",
    "RESOLUTIONS",
    "CLASSES_FILE",
    "N_CLASS",
    "ANGLE_WINDOW_MAX",
    "ANGLE_MODES",
    "B_P_EXPECTED",
    # Split configuration
    "SPLIT_SEED",
    "TEST_FRACTION",
    "BP_BAND_QUANTILES",
    "SPLIT_KINDS",
    "DEFAULT_SPLIT_KINDS",
    "CONFOUND_SEEDS",
    # Containers
    "PB24Batch",
    "LoadReport",
    "Splits",
    "SplitReport",
    # Functions
    "load_batch",
    "make_splits",
    "confound_reference",
    "select",
    "data_dir",
    "file_path",
    "read_classes",
    # Cached-fixture seam
    "RAW_FIELDS",
    "EXTRA_FIELDS",
    "DEFAULT_EXTRAS",
    "write_fixture",
    "npz_reader",
]

#: Location of the release under ``$OS_COLOR``.
SUBDIR = ("SD", "v5")

#: Realisations in the release, indexed 1..5000 in the file names.
N_REALISATION = 5000

#: Spectral resolutions, mapped to their file-name prefix and wavelength grid.
#: The OLCI set is what M5 trains on (JXP's Q12 answer); the hyperspectral set
#: opens through the same reader when the wavelength-interpolation question comes
#: up, which is why this is a table rather than a hard-coded prefix.
RESOLUTIONS = {
    "olci": ("SD_OLCI_no_R_", "olci"),
    "hyp": ("SD_hyp_no_R_", None),
}

#: The optical-water-class sidecar, and how many classes it defines.
CLASSES_FILE = "classes_Rrs_OLCI_v5_20240214.mat"
N_CLASS = 12

#: Q14: train and sanction 0-70 degrees in both zeniths; hold the rest out.
ANGLE_WINDOW_MAX = 70.0

#: Angle-selection modes for :func:`load_batch`.
ANGLE_MODES = ("window", "shell", "all")

#: The design's nominal range for ``B_p`` (§4.2). Advisory: the loader *warns*
#: outside it, never clips -- same contract as :mod:`robust.rt.data.l23`.
B_P_EXPECTED = (0.004, 0.03)

#: Seed for every random split. Arbitrary but fixed forever -- changing it
#: invalidates every held-out number computed from it.
SPLIT_SEED = 23

#: Fraction of *realisations* held out by the random split, matching L23's.
TEST_FRACTION = 0.2

#: Quantile band of per-realisation mean ``B_p`` held out by the ``bp_band``
#: split. An **interior** band by design: holding out the top or the bottom would
#: test extrapolation in ``B_p``, and extrapolation is already the geometry
#: split's job. What M5 needs to know first is whether the model interpolates
#: across phase functions it has not seen -- the honest test of the design §4.2
#: parameterization. The width matches :data:`TEST_FRACTION` so the three splits
#: hold out comparable amounts.
BP_BAND_QUANTILES = (0.4, 0.6)

#: The splits M5 defines.
SPLIT_KINDS = ("realisation", "bp_band", "geometry")

#: Seeds :func:`confound_reference` averages over by default.
CONFOUND_SEEDS = tuple(range(12))

#: What :func:`make_splits` builds unless told otherwise -- the two a default
#: (window) load can support. ``"geometry"`` needs ``angles="all"``, so asking
#: for it is deliberate rather than accidental.
DEFAULT_SPLIT_KINDS = ("realisation", "bp_band")

#: Raw per-realisation fields the loader consumes. A ``reader`` must supply all
#: of them. Deliberately complete rather than minimal: the fixture and any cache
#: are gated **bit-identically**, so a field added later invalidates every cache
#: in existence. Cheaper to store the lot once.
RAW_FIELDS = (
    # coordinates
    "wave",
    "theta_s",
    "theta",
    "phi",
    # generating labels (not generators -- see the module docstring)
    "C",
    "N",
    "Y",
    # IOP components, (wave,)
    "aw",
    "aph",
    "ag",
    "aNAP",
    "bw",
    "bph",
    "bNAP",
    "bbw",
    "bbph",
    "bbNAP",
    # reflectance on the full 4-D grid, (wave, phi, theta, theta_s)
    "rrs",
    "Rrs",
    "Q",
    # AOPs on (wave, theta_s)
    "mu_d",
    "mu_u",
    "mu_tot",
    "Kd",
    "Ku",
    "Ko",
    "Kod",
    "Kou",
    "Knet",
    "KLu",
    "R",
)

#: Fields that are per ``(wave, theta_s)`` and become per-sample spectra by
#: indexing each sample's solar zenith.
EXTRA_FIELDS = (
    "mu_d",
    "mu_u",
    "mu_tot",
    "Kd",
    "Ku",
    "Ko",
    "Kod",
    "Kou",
    "Knet",
    "KLu",
    "R",
)

#: What :func:`load_batch` materialises by default. ``Q`` is per-geometry and is
#: always carried; these are the ones task 13 (ZTT's internals against
#: HydroLight) consumes. The K's are available but cost ~200 MB each at the full
#: window, so they are opt-in.
DEFAULT_EXTRAS = ("mu_d", "mu_u", "mu_tot")


@dataclass(frozen=True)
class LoadReport:
    """What a load kept, and what it dropped.

    Returned with every batch so a subsample is visible at the call site instead
    of implied. M4's review settled the rule this implements: a cap that is not
    reported reads as "we covered everything".

    Attributes
    ----------
    n_realisation_available, n_realisation : int
        Realisations in the source and in the batch.
    n_geometry_available, n_geometry : int
        Geometries per realisation before and after the angle mode and stride.
    angles : str
        Which angle mode was applied (:data:`ANGLE_MODES`).
    geometry_stride : int or tuple of int
        Stride applied within the selected geometries; 1 means none. A tuple is
        per angle axis and preserves the grid's product structure.
    n_dropped_angle : int
        Geometries removed by the angle mode alone.
    n_dropped_stride : int
        Geometries removed by the stride alone.
    n_dropped_zero_rrs : int
        Samples removed because their spectrum contained a zero ``rrs``.
    n_zero_bands : int
        Individual ``(sample, band)`` values that were zero -- always at least
        ``n_dropped_zero_rrs``, and the gap is what dropping whole spectra costs.
    coverage : dict of str to tuple
        ``{axis: (kept, available)}`` distinct angle values per axis, after the
        stride. Present because a stride over a flattened grid can **alias**:
        the azimuth axis has 13 values, so ``geometry_stride=13`` keeps a single
        azimuth and silently deletes the BRDF dimension the milestone is about.
    """

    n_realisation_available: int
    n_realisation: int
    n_geometry_available: int
    n_geometry: int
    angles: str
    geometry_stride: int
    n_dropped_angle: int
    n_dropped_stride: int
    n_dropped_zero_rrs: int
    n_zero_bands: int
    coverage: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def aliased_axes(self) -> tuple[str, ...]:
        """Angle axes the stride lost values from -- empty is the healthy case."""
        return tuple(
            axis for axis, (kept, avail) in self.coverage.items() if kept < avail
        )

    def summary(self) -> str:
        """One line, for a log or a notebook cell."""
        aliased = self.aliased_axes
        tail = f"; ALIASED on {list(aliased)}" if aliased else ""
        return (
            f"PB24: {self.n_realisation}/{self.n_realisation_available} realisations x "
            f"{self.n_geometry}/{self.n_geometry_available} geometries "
            f"(angles={self.angles}, stride={self.geometry_stride}); dropped "
            f"{self.n_dropped_angle} by angle, {self.n_dropped_stride} by stride, "
            f"{self.n_dropped_zero_rrs} samples for zero rrs "
            f"({self.n_zero_bands} zero bands){tail}"
        )


@dataclass(frozen=True)
class PB24Batch:
    """A stacked PB24 batch: model inputs, both reflectances, and labels.

    Like :class:`robust.rt.data.l23.L23Batch` this is **not** a registered pytree
    -- it is an analysis container, and ``realisation``/``water_class`` are
    host-side integer metadata. The model inputs it holds are pytrees.

    The field list deliberately exceeds L23's. ``L23Batch`` carries ``Rrs`` alone
    and derives ``rrs`` through the nadir Lee-2002 map, which PB24 itself shows is
    wrong off-nadir by a median 45.7% at ``theta_v = 60`` -- so ``rrs`` is carried
    **as tabulated**, and scoring never passes through that map.

    Attributes
    ----------
    iops : IOPs
        ``(n_sample, n_wave)`` per field. ``bb_w`` is the file's own ``bbw``, not
        :func:`robust.rt.conventions.bb_w` -- nothing guarantees another campaign
        shares L23's water column.
    phase_params : PhaseParams
        ``B_p = (bbph + bbNAP) / (bph + bNAP)`` as a spectrum.
    geometry : Geometry
        Per-sample ``theta_s``, ``theta_v``, ``dphi`` in degrees.
    rrs, Rrs : Array
        Subsurface and above-water reflectance as tabulated, sr^-1.
    wave : Array
        Wavelengths (nm).
    realisation : numpy.ndarray
        1-based realisation index per sample, matching the file names.
    labels : dict of str to numpy.ndarray
        ``C`` (chlorophyll), ``N`` (NAP) and ``Y`` (a_cdom(440)) per sample, as
        NumPy. Called *labels* rather than parameters on purpose: they do not
        determine the IOPs (see the module docstring), and their job here is to
        let a split measure the confound it induces. Host-side, like
        ``realisation`` -- never traced.
    water_class : numpy.ndarray or None
        Optical water class in ``[1, N_CLASS]`` per sample, if the sidecar was
        read.
    aops : dict of str to Array
        Extra per-sample spectra -- ``Q`` always, plus whichever of
        :data:`EXTRA_FIELDS` was requested.
    report : LoadReport or None
        What this load kept and dropped.
    """

    iops: IOPs
    phase_params: PhaseParams
    geometry: Geometry
    rrs: Float[Array, "sample wave"]
    Rrs: Float[Array, "sample wave"]
    wave: Float[Array, " wave"]
    realisation: np.ndarray
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    water_class: np.ndarray | None = None
    aops: dict[str, Float[Array, "sample wave"]] = field(default_factory=dict)
    report: LoadReport | None = None

    @property
    def n_sample(self) -> int:
        """Number of samples (realisations x kept geometries)."""
        return int(self.rrs.shape[0])

    @property
    def n_wave(self) -> int:
        """Number of wavelength bands."""
        return int(self.rrs.shape[-1])

    @property
    def theta_s(self) -> np.ndarray:
        """Per-sample solar zenith, degrees, as NumPy -- handy for grouping."""
        return np.asarray(self.geometry.theta_s)

    @property
    def theta_v(self) -> np.ndarray:
        """Per-sample sensor zenith, degrees, as NumPy."""
        return np.asarray(self.geometry.theta_v)

    @property
    def dphi(self) -> np.ndarray:
        """Per-sample relative azimuth, degrees, as NumPy."""
        return np.asarray(self.geometry.dphi)

    def validate(self, *, grid: str = "olci") -> None:
        """Raise ``ValueError`` unless the batch is self-consistent and physical.

        Boundary check; not for use under ``jit``.

        Parameters
        ----------
        grid : str, optional
            Wavelength grid the batch claims to be on.

        Raises
        ------
        ValueError
            On a shape mismatch, a bad wavelength grid, a non-physical IOP, an
            out-of-range angle, or a negative reflectance.
        """
        conventions.check_wave(self.wave, name="PB24Batch.wave", grid=grid)
        self.iops.validate(wave=self.wave, grid=grid)
        self.phase_params.validate()
        self.geometry.validate()
        conventions.check_rrs(self.rrs, name="PB24Batch.rrs", subsurface=True)
        conventions.check_rrs(self.Rrs, name="PB24Batch.Rrs", subsurface=False)

        if self.rrs.shape != self.iops.a.shape:
            raise ValueError(
                f"PB24Batch: rrs {self.rrs.shape} does not match "
                f"IOPs.a {self.iops.a.shape}"
            )
        if self.Rrs.shape != self.rrs.shape:
            raise ValueError(
                f"PB24Batch: Rrs {self.Rrs.shape} does not match rrs {self.rrs.shape}"
            )
        if self.realisation.shape != (self.n_sample,):
            raise ValueError(
                f"PB24Batch: realisation {self.realisation.shape} does not label "
                f"{self.n_sample} samples"
            )
        for name, values in self.labels.items():
            if values.shape != (self.n_sample,):
                raise ValueError(
                    f"PB24Batch: labels[{name!r}] {values.shape} does not label "
                    f"{self.n_sample} samples"
                )
        if self.water_class is not None and self.water_class.shape != (self.n_sample,):
            raise ValueError(
                f"PB24Batch: water_class {self.water_class.shape} does not label "
                f"{self.n_sample} samples"
            )
        for name, values in self.aops.items():
            if values.shape != self.rrs.shape:
                raise ValueError(
                    f"PB24Batch: aops[{name!r}] {values.shape} does not match "
                    f"rrs {self.rrs.shape}"
                )


def data_dir() -> Path:
    """The release directory, from ``$OS_COLOR``.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    RuntimeError
        If ``$OS_COLOR`` is unset, or the directory is missing -- with the path
        that was tried, since a wrong mount is the usual cause.
    """
    root = os.environ.get("OS_COLOR")
    if not root:
        raise RuntimeError(
            "pb24: $OS_COLOR is not set, so the PB24 release cannot be located; "
            "use a fixture via npz_reader() instead"
        )
    path = Path(root).joinpath(*SUBDIR)
    if not path.is_dir():
        raise RuntimeError(f"pb24: {path} is not a directory (from $OS_COLOR)")
    return path


def file_path(index: int, resolution: str = "olci") -> Path:
    """Path of one realisation's netCDF.

    Parameters
    ----------
    index : int
        1-based realisation index, as in the file names.
    resolution : str, optional
        A key of :data:`RESOLUTIONS`.

    Returns
    -------
    pathlib.Path
    """
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f"pb24: unknown resolution {resolution!r}; known: {sorted(RESOLUTIONS)}"
        )
    if not 1 <= int(index) <= N_REALISATION:
        raise ValueError(
            f"pb24: realisation index {index} outside [1, {N_REALISATION}]"
        )
    prefix = RESOLUTIONS[resolution][0]
    return data_dir() / f"{prefix}{int(index):04d}.nc"


def read_classes() -> np.ndarray:
    """Optical water class per realisation, from the ``.mat`` sidecar.

    Returns
    -------
    numpy.ndarray
        Shape ``(N_REALISATION,)``, values in ``[1, N_CLASS]``. Note the classes
        are very unbalanced (84 to 1042 of 5000), which matters for any
        stratified split.
    """
    from scipy.io import loadmat

    raw = loadmat(data_dir() / CLASSES_FILE)["i_classes"]
    return np.asarray(raw, dtype=int).reshape(-1)


def _read_file(index: int, resolution: str = "olci") -> dict[str, np.ndarray]:
    """Read one PB24 realisation into plain NumPy -- the default ``reader``.

    Parameters
    ----------
    index : int
        1-based realisation index.
    resolution : str, optional
        A key of :data:`RESOLUTIONS`.

    Returns
    -------
    dict
        The :data:`RAW_FIELDS`. ``rrs``/``Rrs``/``Q`` are transposed to
        ``(theta_s, theta, phi, wave)`` here, so the flattening order is fixed in
        one place rather than at every call site.
    """
    import xarray as xr

    with xr.open_dataset(file_path(index, resolution)) as ds:
        out: dict[str, np.ndarray] = {
            "wave": np.asarray(ds["lambda"].values, dtype=float),
            "theta_s": np.asarray(ds["theta_s"].values, dtype=float),
            "theta": np.asarray(ds["theta"].values, dtype=float),
            "phi": np.asarray(ds["phi"].values, dtype=float),
        }
        for name in ("C", "N", "Y"):
            out[name] = np.asarray(ds[name].values, dtype=float).reshape(())
        for name in ("rrs", "Rrs", "Q"):
            out[name] = np.asarray(
                ds[name].transpose("theta_s", "theta", "phi", "lambda").values,
                dtype=float,
            )
        for name in RAW_FIELDS:
            if name in out:
                continue
            values = ds[name]
            if set(values.dims) == {"lambda", "theta_s"}:
                out[name] = np.asarray(
                    values.transpose("theta_s", "lambda").values, dtype=float
                )
            else:
                out[name] = np.asarray(values.values, dtype=float)
    return out


def _geometry_table(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Flatten the angle grid to one geometry axis, C-order ``(theta_s, theta, phi)``.

    Returns
    -------
    tuple of ndarray
        ``(theta_s, theta_v, dphi, theta_s_index)``, each of length
        ``n_theta_s * n_theta * n_phi``. The index is kept because the ``mu``/K
        fields are tabulated per solar zenith and must be gathered by it.
    """
    ts, tv, ph = raw["theta_s"], raw["theta"], raw["phi"]
    i_ts, i_tv, i_ph = np.meshgrid(
        np.arange(ts.size), np.arange(tv.size), np.arange(ph.size), indexing="ij"
    )
    return (
        ts[i_ts].reshape(-1),
        tv[i_tv].reshape(-1),
        ph[i_ph].reshape(-1),
        i_ts.reshape(-1),
    )


def _apply_stride(selected: np.ndarray, stride, angles) -> np.ndarray:
    """Thin the selected geometries, either flat or **per axis**.

    An ``int`` strides the flattened list, which is simple and has two failure
    modes: it aliases against an axis length (the 13 azimuths, record §7.7) and --
    worse -- it does not preserve the **product structure**. A strided flat list
    populates only a fraction of the ``(theta_s, theta_v, dphi)`` cells, so
    anything that fits a *gridded* table on the same batch, like
    :func:`robust.rt.baselines.fit_o25_table`, finds most cells empty and refuses.

    A 3-tuple ``(s_theta_s, s_theta_v, s_dphi)`` strides each angle axis
    independently and keeps the full product of what survives. That is the right
    shape of subsample whenever the batch has to serve a gridded fit as well as a
    network, which on this milestone is always.

    Parameters
    ----------
    selected : numpy.ndarray
        Indices into the flattened geometry list, after the angle mode.
    stride : int or tuple of int
        Flat stride, or one per angle axis.
    angles : tuple of numpy.ndarray
        ``(theta_s, theta_v, dphi)`` over the *whole* flattened grid.

    Returns
    -------
    numpy.ndarray
        The kept indices, ascending.
    """
    if isinstance(stride, (int, np.integer)):
        return selected[:: int(stride)]

    if len(stride) != 3:
        raise ValueError(
            f"load_batch: a per-axis geometry_stride needs three values "
            f"(theta_s, theta_v, dphi); got {stride!r}"
        )
    keep = np.ones(selected.size, dtype=bool)
    for step, values in zip(stride, angles, strict=True):
        step = int(step)
        if step < 1:
            raise ValueError(
                f"load_batch: geometry_stride entries must be >= 1; {stride!r}"
            )
        present = np.unique(values[selected])
        wanted = set(present[::step].tolist())
        keep &= np.isin(values[selected], list(wanted))
    return selected[keep]


def _angle_mask(theta_s: np.ndarray, theta_v: np.ndarray, angles: str) -> np.ndarray:
    """Boolean mask over flattened geometries for an angle mode."""
    if angles not in ANGLE_MODES:
        raise ValueError(f"pb24: angles must be one of {ANGLE_MODES}; got {angles!r}")
    inside = (theta_s <= ANGLE_WINDOW_MAX) & (theta_v <= ANGLE_WINDOW_MAX)
    if angles == "window":
        return inside
    if angles == "shell":
        return ~inside
    return np.ones_like(inside, dtype=bool)


def write_fixture(
    path,
    *,
    realisations=(1, 2, 3),
    resolution: str = "olci",
) -> None:
    """Snapshot whole realisations of the raw PB24 fields to an ``.npz``.

    Stores the **raw per-file fields**, not an assembled :class:`PB24Batch`, for
    the reason :func:`robust.rt.data.l23.write_fixture` gives: a snapshot of the
    loader's output would only ever prove the snapshot unchanged, while storing
    its *input* means :func:`load_batch` itself runs against real numbers
    everywhere the fixture is available, CI included.

    The **whole** 1300-geometry grid is kept for each realisation, because the
    angle window and its shell are exactly what the tests need to exercise -- a
    fixture holding only in-window geometries could not test the filter that
    removes the rest.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.npz``.
    realisations : sequence of int, optional
        1-based realisation indices to include. Three realisations of the OLCI
        set is ~600 kB; the full 5000 is ~1 GB and belongs outside the repo.
    resolution : str, optional
        A key of :data:`RESOLUTIONS`.

    Notes
    -----
    Values are stored as float32, the dtype the netCDF itself uses, so the
    fixture is bit-faithful to the source. Written via a temporary file, loaded
    back through :func:`npz_reader` and :func:`load_batch` -- the real consumers
    -- and only then moved into place with an atomic :func:`os.replace`.
    """
    realisations = tuple(int(i) for i in realisations)
    if not realisations:
        raise ValueError("write_fixture: `realisations` must name at least one file")

    arrays: dict[str, np.ndarray] = {
        "realisations": np.asarray(realisations),
        "resolution": np.asarray(resolution),
    }
    for index in realisations:
        raw = _read_file(index, resolution)
        for name in ("wave", "theta_s", "theta", "phi"):
            arrays[name] = raw[name].astype(np.float32)
        for name in RAW_FIELDS:
            if name in ("wave", "theta_s", "theta", "phi"):
                continue
            arrays[f"{name}_{index}"] = np.asarray(raw[name], dtype=np.float32)

    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        # drop_zero_rrs=False deliberately: this verifies the *snapshot*, so it
        # must round-trip every sample the files hold. With the filter on, a
        # realisation carrying a grazing-geometry zero (993 does) legitimately
        # loads fewer samples than it stores, and the count check below would
        # reject a perfectly good fixture. Asking for the filter here would test
        # the filter, which the suite does separately.
        batch = load_batch(
            realisations=realisations,
            angles="all",
            drop_zero_rrs=False,
            reader=npz_reader(tmp),
        )
        batch.validate()
        expected = len(realisations) * batch.report.n_geometry_available
        if batch.n_sample != expected:
            raise ValueError(
                f"write_fixture: snapshot loads {batch.n_sample} samples, expected "
                f"{expected}; {path} left untouched"
            )
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def npz_reader(path):
    """A :func:`load_batch` ``reader`` backed by a :func:`write_fixture` file.

    Parameters
    ----------
    path : str or pathlib.Path
        An ``.npz`` written by :func:`write_fixture`.

    Returns
    -------
    callable
        ``(index, resolution) -> dict`` of :data:`RAW_FIELDS`.

    Raises
    ------
    ValueError
        If asked for a realisation or resolution the fixture does not hold.
        Serving the wrong realisation silently would be far worse than failing.
    """
    data = np.load(path)
    available = tuple(int(i) for i in data["realisations"])
    stored_resolution = str(data["resolution"])

    def read(index: int, resolution: str = "olci") -> dict[str, np.ndarray]:
        if resolution != stored_resolution:
            raise ValueError(
                f"npz_reader: fixture holds resolution {stored_resolution!r}, "
                f"was asked for {resolution!r}"
            )
        if int(index) not in available:
            raise ValueError(
                f"npz_reader: fixture holds realisations {available}, was asked "
                f"for {index}"
            )
        out = {
            name: np.asarray(data[name], dtype=float)
            for name in ("wave", "theta_s", "theta", "phi")
        }
        for name in RAW_FIELDS:
            if name in out:
                continue
            out[name] = np.asarray(data[f"{name}_{int(index)}"], dtype=float)
        return out

    return read


def load_batch(
    *,
    realisations=None,
    angles: str = "window",
    geometry_stride: int = 1,
    extras: tuple[str, ...] = DEFAULT_EXTRAS,
    drop_zero_rrs: bool = True,
    water_classes=None,
    validate: bool = True,
    resolution: str = "olci",
    reader=None,
) -> PB24Batch:
    """Load PB24 as one stacked batch of JAX arrays.

    Parameters
    ----------
    realisations : int or sequence of int, optional
        An ``int`` takes the first *n* realisations; a sequence names 1-based
        indices; ``None`` takes all :data:`N_REALISATION`. **There is no default
        subsample** -- ``None`` really means all 5000, which is ~1 GB and a few
        minutes of netCDF reads, so pass a cache via ``reader`` for repeat work.
    angles : {"window", "shell", "all"}, optional
        Which geometries to keep. ``"window"`` (default) is Q14's sanctioned
        envelope, ``theta_s`` and ``theta_v`` both <= :data:`ANGLE_WINDOW_MAX`;
        ``"shell"`` is its complement, the deliberate extrapolation set; ``"all"``
        is the whole 1300.
    geometry_stride : int or tuple of int, optional
        Thin the selected geometries. An ``int`` strides the flattened list; a
        3-tuple ``(s_theta_s, s_theta_v, s_dphi)`` strides each angle axis and
        **keeps the full product**, which is what a gridded fit on the same batch
        needs (see :func:`_apply_stride`). Either way this is the subsampling knob
        Q12 sanctioned and it is **explicit** by design: what it drops is reported
        in :class:`LoadReport`, never assumed.
    extras : tuple of str, optional
        Which of :data:`EXTRA_FIELDS` to materialise as per-sample spectra.
        ``Q`` is always carried. Each extra costs one more ``(n_sample, n_wave)``
        array.
    drop_zero_rrs : bool, optional
        Drop samples whose spectrum contains a zero ``rrs`` (default True).
        See the Notes -- this is a lossy choice, and it is reported.
    water_classes : array_like or "auto", optional
        Per-realisation classes, 1-based, as :func:`read_classes` returns.
        ``None`` (the default) attaches none; ``"auto"`` reads the ``.mat``
        sidecar and warns if it cannot. The default is *not* "auto" on purpose --
        a loader that silently behaves differently depending on whether a mount
        is present makes a fixture-backed test non-deterministic, and that is
        exactly the class of surprise this module is trying not to have.
    validate : bool, optional
        Run :meth:`PB24Batch.validate` before returning (default True).
    resolution : str, optional
        A key of :data:`RESOLUTIONS`.
    reader : callable, optional
        ``(index, resolution) -> dict`` of :data:`RAW_FIELDS`. Defaults to
        reading the netCDFs; :func:`npz_reader` feeds the *real* loader real
        numbers where the dataset is absent, CI included.

    Returns
    -------
    PB24Batch
        Samples ordered realisation-major, then geometry in C-order
        ``(theta_s, theta, phi)``.

    Warns
    -----
    UserWarning
        If ``B_p`` falls outside :data:`B_P_EXPECTED`. Reported, never clipped.

    Notes
    -----
    **On ``drop_zero_rrs``.** The zeros are per *band*, not per spectrum -- in the
    OLCI set they occur only at 753 nm with ``theta_s = 80``, ``theta = 87.5`` --
    so dropping the whole spectrum discards eleven good bands to remove one bad
    one. It is nonetheless the right call here: :func:`robust.rt.validation.rrms`
    divides by truth over rectangular ``(sample, wave)`` arrays and has no mask,
    so an unfiltered zero is an ``inf`` in the first score computed. When M5's
    task 6 gives the metric a mask, per-band exclusion becomes available and this
    default should be revisited. The report records both counts, so the cost is
    visible: ``n_dropped_zero_rrs`` spectra removed to exclude ``n_zero_bands``
    values.

    **On ``bb_w``.** Taken from the file's own ``bbw``, never from
    :func:`robust.rt.conventions.bb_w` -- that table is L23's water column, and
    PB24 tabulates its own. It also means the 753 nm band needs no extrapolation.
    """
    read = _read_file if reader is None else reader

    if realisations is None:
        wanted = tuple(range(1, N_REALISATION + 1))
    elif isinstance(realisations, (int, np.integer)):
        wanted = tuple(range(1, int(realisations) + 1))
    else:
        wanted = tuple(int(i) for i in realisations)
    if not wanted:
        raise ValueError("load_batch: `realisations` selects nothing")
    if isinstance(geometry_stride, (int, np.integer)) and geometry_stride < 1:
        raise ValueError(
            f"load_batch: geometry_stride must be >= 1; got {geometry_stride}"
        )
    unknown = set(extras) - set(EXTRA_FIELDS)
    if unknown:
        raise ValueError(
            f"load_batch: unknown extras {sorted(unknown)}; known: {list(EXTRA_FIELDS)}"
        )

    if isinstance(water_classes, str):
        if water_classes != "auto":
            raise ValueError(
                f"load_batch: water_classes must be an array, None or 'auto'; "
                f"got {water_classes!r}"
            )
        try:
            water_classes = read_classes()
        except Exception as exc:  # noqa: BLE001 - a missing sidecar is not fatal
            warnings.warn(
                f"pb24: water_classes='auto' could not read the sidecar ({exc}); "
                "the batch will carry no classes.",
                UserWarning,
                stacklevel=2,
            )
            water_classes = None
    elif water_classes is not None:
        water_classes = np.asarray(water_classes, dtype=int).reshape(-1)

    reference: dict[str, np.ndarray] | None = None
    keep: np.ndarray | None = None
    theta_s = theta_v = dphi = zenith_index = None
    n_geometry_available = 0
    n_dropped_angle = n_dropped_stride = 0

    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("a", "bb_w", "bb_p", "B_p", "rrs", "Rrs", "Q")
    }
    parts.update({name: [] for name in extras})
    realisation_parts, class_parts = [], []
    label_parts: dict[str, list[np.ndarray]] = {"C": [], "N": [], "Y": []}

    for index in wanted:
        raw = read(index, resolution)

        if reference is None:
            reference = raw
            grid_name = RESOLUTIONS[resolution][1]
            conventions.check_wave(
                raw["wave"], name=f"PB24 lambda (realisation {index})", grid=grid_name
            )
            theta_s, theta_v, dphi, zenith_index = _geometry_table(raw)
            n_geometry_available = theta_s.size
            mask = _angle_mask(theta_s, theta_v, angles)
            n_dropped_angle = int((~mask).sum())
            selected = np.flatnonzero(mask)
            keep = _apply_stride(selected, geometry_stride, (theta_s, theta_v, dphi))
            n_dropped_stride = int(selected.size - keep.size)
            if keep.size == 0:
                raise ValueError(
                    f"load_batch: angles={angles!r} with stride {geometry_stride} "
                    "selects no geometries"
                )
            coverage = {
                axis: (
                    int(np.unique(values[keep]).size),
                    int(np.unique(values[selected]).size),
                )
                for axis, values in (
                    ("theta_s", theta_s),
                    ("theta_v", theta_v),
                    ("dphi", dphi),
                )
            }
            # Only a *flat* stride can lose an angle by accident. A per-axis
            # stride drops whole values because that is what it was asked to do,
            # so warning about it would train the reader to ignore the warning.
            if isinstance(geometry_stride, (int, np.integer)):
                _warn_if_stride_aliases(coverage, geometry_stride)
            theta_s, theta_v, dphi = theta_s[keep], theta_v[keep], dphi[keep]
            zenith_index = zenith_index[keep]
        else:
            # The grid is identical across the release -- verified across 50
            # files at M5 task 0 -- and this is where that stops being an
            # assumption. Concatenating samples from two different angle grids
            # would produce a batch whose geometry labels are silently wrong.
            for name in ("wave", "theta_s", "theta", "phi"):
                if not np.array_equal(raw[name], reference[name]):
                    raise ValueError(
                        f"load_batch: realisation {index} has a different "
                        f"{name!r} grid than realisation {wanted[0]}"
                    )

        n_geometry = keep.size
        a = raw["aw"] + raw["aph"] + raw["ag"] + raw["aNAP"]
        bb_p = raw["bbph"] + raw["bbNAP"]
        b_p = raw["bph"] + raw["bNAP"]

        parts["a"].append(np.broadcast_to(a, (n_geometry, a.size)))
        parts["bb_w"].append(np.broadcast_to(raw["bbw"], (n_geometry, a.size)))
        parts["bb_p"].append(np.broadcast_to(bb_p, (n_geometry, a.size)))
        parts["B_p"].append(np.broadcast_to(bb_p / b_p, (n_geometry, a.size)))
        for name in ("rrs", "Rrs", "Q"):
            parts[name].append(raw[name].reshape(-1, a.size)[keep])
        for name in extras:
            parts[name].append(raw[name][zenith_index])

        realisation_parts.append(np.full(n_geometry, index, dtype=int))
        for name in label_parts:
            label_parts[name].append(np.full(n_geometry, float(raw[name])))
        if water_classes is not None:
            class_parts.append(np.full(n_geometry, int(water_classes[index - 1])))

    stacked = {name: np.concatenate(values) for name, values in parts.items()}
    _warn_if_B_p_unexpected(stacked["B_p"])

    realisation = np.concatenate(realisation_parts)
    labels = {name: np.concatenate(values) for name, values in label_parts.items()}
    water_class = np.concatenate(class_parts) if class_parts else None
    n_repeat = len(wanted)
    geometry = {
        "theta_s": np.tile(theta_s, n_repeat),
        "theta_v": np.tile(theta_v, n_repeat),
        "dphi": np.tile(dphi, n_repeat),
    }

    zero_bands = int((stacked["rrs"] <= 0.0).sum())
    n_dropped_zero = 0
    if drop_zero_rrs and zero_bands:
        good = ~np.any(stacked["rrs"] <= 0.0, axis=-1)
        n_dropped_zero = int((~good).sum())
        stacked = {name: values[good] for name, values in stacked.items()}
        geometry = {name: values[good] for name, values in geometry.items()}
        realisation = realisation[good]
        labels = {name: values[good] for name, values in labels.items()}
        if water_class is not None:
            water_class = water_class[good]

    report = LoadReport(
        n_realisation_available=N_REALISATION,
        n_realisation=len(wanted),
        n_geometry_available=n_geometry_available,
        n_geometry=int(keep.size),
        angles=angles,
        geometry_stride=geometry_stride,
        n_dropped_angle=n_dropped_angle,
        n_dropped_stride=n_dropped_stride,
        n_dropped_zero_rrs=n_dropped_zero,
        n_zero_bands=zero_bands,
        coverage=coverage,
    )

    batch = PB24Batch(
        iops=IOPs(
            a=jnp.asarray(stacked["a"]),
            bb_w=jnp.asarray(stacked["bb_w"]),
            bb_p=jnp.asarray(stacked["bb_p"]),
        ),
        phase_params=PhaseParams(B_p=jnp.asarray(stacked["B_p"])),
        geometry=Geometry(
            theta_s=jnp.asarray(geometry["theta_s"]),
            theta_v=jnp.asarray(geometry["theta_v"]),
            dphi=jnp.asarray(geometry["dphi"]),
        ),
        rrs=jnp.asarray(stacked["rrs"]),
        Rrs=jnp.asarray(stacked["Rrs"]),
        wave=jnp.asarray(reference["wave"]),
        realisation=realisation,
        labels=labels,
        water_class=water_class,
        aops={"Q": jnp.asarray(stacked["Q"])}
        | {name: jnp.asarray(stacked[name]) for name in extras},
        report=report,
    )

    if validate:
        batch.validate(grid=RESOLUTIONS[resolution][1])
    return batch


def _warn_if_stride_aliases(coverage, stride: int) -> None:
    """Warn if the stride deleted whole angle values rather than thinning them.

    Flattening ``(theta_s, theta, phi)`` in C order puts azimuth innermost, so a
    stride sharing a factor with an axis length keeps a *periodic* subset rather
    than a representative one -- ``geometry_stride=13`` on the 13-value azimuth
    axis yields one azimuth and no BRDF at all. Found by running the loader, not
    by reading it, which is why the report carries the coverage.
    """
    lost = {
        axis: (kept, avail) for axis, (kept, avail) in coverage.items() if kept < avail
    }
    if lost:
        detail = ", ".join(
            f"{axis}: {kept} of {avail} values" for axis, (kept, avail) in lost.items()
        )
        warnings.warn(
            f"geometry_stride={stride} loses whole angle values -- {detail}. "
            "Either the stride shares a factor with an axis length (the azimuth "
            "axis has 13 values, so stride 13 keeps one azimuth and no BRDF at "
            "all) or it is simply too coarse to reach every value. A stride "
            "coprime with 13 and 10, and well below the geometry count, avoids "
            "both.",
            UserWarning,
            stacklevel=3,
        )


def _warn_if_B_p_unexpected(B_p: np.ndarray) -> None:
    """Warn if ``B_p`` leaves :data:`B_P_EXPECTED`; never modify it."""
    lo, hi = B_P_EXPECTED
    outside = (B_p < lo) | (B_p > hi)
    if outside.any():
        warnings.warn(
            f"B_p leaves the expected range {B_P_EXPECTED}: "
            f"{int(outside.sum())} of {B_p.size} values, observed range "
            f"[{B_p.min():.5g}, {B_p.max():.5g}]. Reported, not clipped -- "
            "PB24 spans a wider band than L23, so this is informative, not wrong.",
            UserWarning,
            stacklevel=3,
        )


def select(batch: PB24Batch, mask: np.ndarray) -> PB24Batch:
    """Subset a batch along its sample axis.

    Parameters
    ----------
    batch : PB24Batch
        The batch to subset.
    mask : numpy.ndarray
        Boolean mask of length ``batch.n_sample``.

    Returns
    -------
    PB24Batch
        A new batch holding only the selected samples; ``wave`` and ``report``
        are carried through unchanged, the latter because it describes the
        *load*, not the selection.

    Raises
    ------
    ValueError
        If ``mask`` is not a boolean array of the right length.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (batch.n_sample,):
        raise ValueError(
            f"select: expected a boolean mask of shape ({batch.n_sample},); got "
            f"dtype {mask.dtype}, shape {mask.shape}"
        )

    keep = jnp.asarray(np.flatnonzero(mask))
    return PB24Batch(
        iops=IOPs(
            a=batch.iops.a[keep],
            bb_w=batch.iops.bb_w[keep],
            bb_p=batch.iops.bb_p[keep],
        ),
        phase_params=PhaseParams(B_p=batch.phase_params.B_p[keep]),
        geometry=Geometry(
            theta_s=batch.geometry.theta_s[keep],
            theta_v=batch.geometry.theta_v[keep],
            dphi=batch.geometry.dphi[keep],
        ),
        rrs=batch.rrs[keep],
        Rrs=batch.Rrs[keep],
        wave=batch.wave,
        realisation=batch.realisation[mask],
        labels={name: values[mask] for name, values in batch.labels.items()},
        water_class=None if batch.water_class is None else batch.water_class[mask],
        aops={name: values[keep] for name, values in batch.aops.items()},
        report=batch.report,
    )


@dataclass(frozen=True)
class SplitReport:
    """What one split holds out, and what else it holds out by accident.

    The second half is the point. A split on ``B_p`` is not a split on ``B_p``
    alone: the particle backscatter ratio correlates with chlorophyll across
    PB24's realisations -- **corr(log B_p, log C) = -0.49** over 600 realisations
    -- so holding out a band of ``B_p`` also shifts the water type. That does not
    make the split wrong; it is still the only way to test phase-function
    generalisation with this data. But a result quoted without it invites the
    reader to attribute the whole error to the phase function.

    So the confound is **measured on the split that was actually built** and
    carried with it, rather than described once in a design document and
    forgotten by the time the number is quoted.

    **Read the ratios against :func:`confound_reference`, not against 1.0.** A
    random hold-out of the same size already moves these labels a long way,
    because PB24's label distributions are heavy-tailed: over 12 seeds at 600
    realisations the random split's median-chlorophyll ratio ranges over
    **[0.53, 1.90]**. So "test/train = 1.5" is unremarkable on its own, and only
    a shift outside that band is the split's own doing. ``B_p_mean`` is the
    exception: the random split holds it to [0.98, 1.08], so it is the one entry
    where a departure is immediately meaningful.

    Note also what these ratios *cannot* show. ``bp_band`` holds out an interior
    band, so its train side is both tails and the two medians nearly coincide --
    ``B_p_mean`` reads ~1.0 for a split whose entire purpose is to separate
    ``B_p``. The separation lives in :attr:`detail` (the band edges and the train
    tails), not in a ratio of medians.

    Attributes
    ----------
    kind : str
        One of :data:`SPLIT_KINDS`.
    n_train, n_test : int
        Samples each side.
    n_train_realisation, n_test_realisation : int
        Distinct realisations each side. For ``"geometry"`` these are equal and
        cover the whole batch: that split divides angles, not water bodies.
    detail : dict
        Kind-specific numbers -- band edges, angle ranges, held-out counts.
    confound : dict
        ``test/train`` ratio of the median of each label and of two IOP
        summaries. Compare against :func:`confound_reference`, which measures
        what a random hold-out of the same size does -- not against 1.0.
    """

    kind: str
    n_train: int
    n_test: int
    n_train_realisation: int
    n_test_realisation: int
    detail: dict[str, float] = field(default_factory=dict)
    confound: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One line, for a log, a table, or a notebook cell."""
        worst = ""
        if self.confound:
            name, value = max(
                self.confound.items(), key=lambda kv: abs(np.log(max(kv[1], 1e-12)))
            )
            worst = f"; largest side-effect {name} x{value:.2f}"
        return (
            f"{self.kind}: {self.n_train} train / {self.n_test} test samples "
            f"({self.n_train_realisation} / {self.n_test_realisation} "
            f"realisations){worst}"
        )


@dataclass(frozen=True)
class Splits:
    """Boolean masks over a batch's sample axis, one pair per split kind.

    Attributes
    ----------
    masks : dict of str to tuple
        ``{kind: (train, test)}``, each a boolean array of length
        ``batch.n_sample``.
    reports : dict of str to SplitReport
        One per kind.
    seed : int
        The seed the random draws used.
    """

    masks: dict[str, tuple[np.ndarray, np.ndarray]]
    reports: dict[str, SplitReport]
    seed: int

    def train(self, kind: str) -> np.ndarray:
        """Training mask for ``kind``."""
        return self.masks[kind][0]

    def test(self, kind: str) -> np.ndarray:
        """Held-out mask for ``kind``."""
        return self.masks[kind][1]

    def summary(self) -> str:
        """One line per split."""
        return "\n".join(self.reports[kind].summary() for kind in self.masks)


def _per_realisation_B_p(batch: PB24Batch) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``B_p`` per realisation, and the realisation indices it belongs to.

    ``B_p`` is duplicated across a realisation's geometries (it is an IOP), so the
    mean over samples and wavelengths is exact rather than an approximation --
    but it is taken per realisation, not per sample, because the split must move
    whole water bodies.
    """
    B_p = np.asarray(batch.phase_params.B_p).mean(axis=-1)
    realisations = np.unique(batch.realisation)
    means = np.array([B_p[batch.realisation == index].mean() for index in realisations])
    return realisations, means


def _confound(
    batch: PB24Batch, train: np.ndarray, test: np.ndarray
) -> dict[str, float]:
    """``test/train`` ratio of median labels and IOP summaries across a split.

    Everything a split moves *other* than the thing it meant to move. Medians
    rather than means because PB24's label distributions are heavy-tailed (``C``
    reaches 938 mg m^-3), where a mean would report one realisation's opinion.
    """
    out: dict[str, float] = {}
    columns = dict(batch.labels)
    a = np.asarray(batch.iops.a).mean(axis=-1)
    bb_p = np.asarray(batch.iops.bb_p).mean(axis=-1)
    B_p = np.asarray(batch.phase_params.B_p).mean(axis=-1)
    columns.update({"a_mean": a, "bb_p_mean": bb_p, "B_p_mean": B_p})

    for name, values in columns.items():
        lo = float(np.median(values[train]))
        hi = float(np.median(values[test]))
        out[name] = hi / lo if lo > 0.0 else float("nan")
    return out


def make_splits(
    batch: PB24Batch,
    *,
    kinds: tuple[str, ...] = DEFAULT_SPLIT_KINDS,
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    bp_quantiles: tuple[float, float] = BP_BAND_QUANTILES,
) -> Splits:
    """Build M5's held-out splits over a PB24 batch.

    Three kinds, each answering a different question:

    ``"realisation"``
        Random ``test_fraction`` of realisations, held out **whole**. Every
        geometry of a held-out water body goes with it -- splitting per sample
        would leak, since the same IOPs appear at 832 geometries. M4's scene
        split, transplanted.
    ``"bp_band"``
        An interior band of per-realisation mean ``B_p``
        (:data:`BP_BAND_QUANTILES`), held out whole. **The reason M5 exists**: it
        converts "phase-function generalisation untested" into a number. Read its
        :attr:`SplitReport.confound` before quoting the result.
    ``"geometry"``
        Train on Q14's window, test on the 80/87.75 shell. The direct successor
        to M4's unseen-60 split -- the half the prototype lost -- so a bad number
        here is information, not a failure.

    Parameters
    ----------
    batch : PB24Batch
        The batch to split.
    kinds : tuple of str, optional
        Which splits to build; defaults to :data:`DEFAULT_SPLIT_KINDS`.
        ``"geometry"`` requires a batch loaded with ``angles="all"``.
    seed : int, optional
        Seed for the random draw; defaults to :data:`SPLIT_SEED`.
    test_fraction : float, optional
        Fraction of realisations held out by ``"realisation"``.
    bp_quantiles : tuple of float, optional
        ``(lo, hi)`` quantiles of per-realisation mean ``B_p`` held out by
        ``"bp_band"``.

    Returns
    -------
    Splits

    Raises
    ------
    ValueError
        On an unknown kind, an out-of-range fraction or quantile pair, or **any
        split whose test or train side would be empty**. An empty side is the
        failure mode that matters here: it does not crash, it silently scores
        nothing and reports a perfect number, which is exactly how the geometry
        split would behave on a default (window-only) load.
    """
    unknown = tuple(k for k in kinds if k not in SPLIT_KINDS)
    if unknown:
        raise ValueError(
            f"make_splits: unknown split kind(s) {list(unknown)}; "
            f"known: {list(SPLIT_KINDS)}"
        )
    if not kinds:
        raise ValueError("make_splits: `kinds` selects nothing")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"make_splits: test_fraction must lie in (0, 1); got {test_fraction}"
        )
    lo_q, hi_q = bp_quantiles
    if not 0.0 <= lo_q < hi_q <= 1.0:
        raise ValueError(
            f"make_splits: bp_quantiles must satisfy 0 <= lo < hi <= 1; "
            f"got {bp_quantiles}"
        )

    realisations, bp_mean = _per_realisation_B_p(batch)
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reports: dict[str, SplitReport] = {}

    for kind in kinds:
        if kind == "realisation":
            n_test = max(1, int(round(test_fraction * realisations.size)))
            drawn = np.random.default_rng(seed).permutation(realisations)
            held = np.sort(drawn[:n_test])
            test = np.isin(batch.realisation, held)
            detail = {
                "n_realisation_held": float(held.size),
                "test_fraction_requested": float(test_fraction),
                "test_fraction_realised": float(held.size / realisations.size),
            }
        elif kind == "bp_band":
            lo, hi = np.quantile(bp_mean, [lo_q, hi_q])
            held = realisations[(bp_mean >= lo) & (bp_mean <= hi)]
            test = np.isin(batch.realisation, held)
            trained_on = bp_mean[~np.isin(realisations, held)]
            detail = {
                "B_p_band_lo": float(lo),
                "B_p_band_hi": float(hi),
                # The train side is *both tails*, so quoting only its range would
                # imply it straddles the band. These say so explicitly.
                "B_p_train_lo": float(trained_on.min()),
                "B_p_train_hi": float(trained_on.max()),
                "n_train_inside_band": float(
                    int(((trained_on >= lo) & (trained_on <= hi)).sum())
                ),
                "n_realisation_held": float(held.size),
                "B_p_span_all": float(bp_mean.max() / bp_mean.min()),
            }
        else:  # "geometry"
            # A sample is "beyond the window" if *either* zenith is, so the
            # quantity that decides the split is the larger of the two. Report
            # that, not the two axes separately: the max of theta_s alone would
            # read 70 in the test set and look like no extrapolation at all.
            worst = np.maximum(batch.theta_s, batch.theta_v)
            test = worst > ANGLE_WINDOW_MAX
            held = realisations
            detail = {
                "worst_angle_max_train": float(worst[~test].max())
                if (~test).any()
                else float("nan"),
                "worst_angle_min_test": float(worst[test].min())
                if test.any()
                else float("nan"),
                "worst_angle_max_test": float(worst[test].max())
                if test.any()
                else float("nan"),
            }

        train = ~test
        if not test.any() or not train.any():
            side = "test" if not test.any() else "train"
            extra = (
                " -- the batch was loaded with angles='window', which holds no "
                "shell samples; load with angles='all' for this split"
                if kind == "geometry"
                else ""
            )
            raise ValueError(
                f"make_splits: the {kind!r} split leaves its {side} side empty"
                f"{extra}. An empty side scores nothing and reports it as success."
            )

        masks[kind] = (train, test)
        reports[kind] = SplitReport(
            kind=kind,
            n_train=int(train.sum()),
            n_test=int(test.sum()),
            n_train_realisation=int(np.unique(batch.realisation[train]).size),
            n_test_realisation=int(np.unique(batch.realisation[test]).size),
            detail=detail,
            confound=_confound(batch, train, test),
        )

    return Splits(masks=masks, reports=reports, seed=seed)


def confound_reference(
    batch: PB24Batch,
    *,
    test_fraction: float = TEST_FRACTION,
    seeds=CONFOUND_SEEDS,
) -> dict[str, tuple[float, float, float]]:
    """What a *random* hold-out of the same size does to the same labels.

    The yardstick :attr:`SplitReport.confound` needs. On its own a ratio like
    ``C = 1.5`` is uninterpretable: PB24's label distributions are heavy-tailed,
    so a purely random draw of 20% of realisations moves the median chlorophyll
    over roughly ``[0.53, 1.90]`` (12 seeds, 600 realisations). Only a shift
    outside that band is attributable to the split's own criterion.

    Parameters
    ----------
    batch : PB24Batch
        The batch the comparison is about.
    test_fraction : float, optional
        Hold-out size, matching the split being judged.
    seeds : iterable of int, optional
        Random seeds to average over. More seeds, tighter band.

    Returns
    -------
    dict
        ``{label: (median, min, max)}`` of the ``test/train`` ratio across seeds.

    Notes
    -----
    ``B_p_mean`` is the useful entry: a random draw holds it near 1 (measured
    ``[0.98, 1.08]``), so a departure there means something, while the other
    labels need this reference to be read at all.
    """
    collected: dict[str, list[float]] = {}
    for seed in seeds:
        report = make_splits(
            batch, kinds=("realisation",), seed=int(seed), test_fraction=test_fraction
        ).reports["realisation"]
        for name, value in report.confound.items():
            collected.setdefault(name, []).append(value)
    return {
        name: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
        for name, v in collected.items()
    }
