"""
PB24 consolidation caches: the 5,000 hyperspectral netCDFs, read once, as
memory-mappable float32 ``.npy`` files (M8 task 2).

**What it is.** Every later M8 task needs PB24's IOPs, AOPs and reflectances
for thousands of realisations at once. Reading them per file costs ~0.06 s a
file through :mod:`ocpy.hydrolight.pb24`, every time. This module opens each
``SD_hyp_no_R_NNNN.nc`` **once**, through that reader (reuse, not
reimplementation), and writes one ``.npy`` per field plus a JSON sidecar.
Downstream code slices memmaps and never opens a netCDF again. The OLCI set is
small enough to read per file and is **not** cached.

**Where it lives.** ``$OS_COLOR/SD/derived/pb24_v{CACHE_VERSION}/`` -- next
to the data, not in git (~36 GB). :func:`default_cache_dir` resolves it from
``ocpy.hydrolight.pb24.pb24_path``, so it follows ``$OS_COLOR``; every
function takes ``cache_dir=`` to override it. Build it with
``design/py/build_pb24_cache.py``.

**Layout.** One realisation per leading row, rows in ascending realisation id
(:attr:`PB24Cache.realisations`); λ is always the last, contiguous axis:

- ``C``, ``N``, ``Y``: ``(n_real,)``, from ``(scalar,)``;
- the ten IOP components: ``(n_real, 451 λ)``, from ``(lambda,)``;
- the eleven AOPs and ``Esdir_Es_ratio``: ``(n_real, 10 θs, 451 λ)``, from
  ``(lambda, theta_s)``;
- ``rrs``, ``Rrs``, ``Q``: ``(n_real, 10 θs, 10 θ, 13 φ, nb λ)``, from
  ``(lambda, phi, theta, theta_s)``.

The length-1 ``scalar`` dim is dropped and the ``(λ, θs)`` and 4-D fields are
transposed so that realisation, then sun, then view angles lead and λ is
innermost: a spectrum at one geometry is one contiguous run, and one
realisation's 4-D block (2.3 MB) is one contiguous slab. ``nb`` is the number
of bands kept in the 4-D fields: 451 in the real cache; a miniature (the test
fixture) may keep a subset, recorded in the sidecar as ``four_d_bands``
(indices into the full λ grid). The table fields always keep all 451 bands.

The sidecar (``sidecar.json``) holds the realisation ids, the grids, every
field's shape, dims and dtype, the source file names and byte sizes, the
resolution, :data:`CACHE_VERSION` and the reader's ``READER_VERSION``. It has
**no timestamp** and its keys are sorted, so a rebuild is byte-identical, the
sidecar included. :func:`open_cache` refuses a cache whose versions differ
from the running code's.

**Why bit-identity is a property, not a tolerance.** The files store float32;
the reader returns float32; the cache stores those same float32 words,
transposed (a permutation, which moves values without touching them) and
nothing else -- no float64 round trip, no arithmetic. So every cached value
*is* the file's value, and :func:`verify` checks it with exact equality of
the raw 32-bit words, not with ``allclose``.

**Atomicity.** :func:`build` writes every array to a ``*.partial`` name,
deletes any old sidecar, renames the arrays into place and writes the sidecar
last. A crashed build therefore leaves no sidecar, and :func:`open_cache`
refuses a directory without one.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Version of the cache layout. Bump it when what :func:`build` writes changes;
#: the default directory carries it (``pb24_v{N}``) and :func:`open_cache`
#: refuses a cache written under another version.
CACHE_VERSION = 1

#: Name of the JSON sidecar inside a cache directory.
SIDECAR = "sidecar.json"

#: Suffix of the not-yet-committed arrays during a build.
PARTIAL_SUFFIX = ".partial"

#: Constituent labels, one number per realisation (dim ``scalar`` in the files).
SCALAR_FIELDS = ("C", "N", "Y")

#: IOP components on λ.
IOP_FIELDS = ("aw", "aph", "ag", "aNAP", "bw", "bph", "bNAP", "bbw", "bbph", "bbNAP")

#: Surface AOPs on (λ, θs), and the above-surface direct-to-total irradiance
#: ratio (1 - it is the diffuse fraction).
AOP_FIELDS = (
    "Kd",
    "Ku",
    "Ko",
    "Kod",
    "Kou",
    "Knet",
    "KLu",
    "mu_d",
    "mu_u",
    "mu_tot",
    "R",
    "Esdir_Es_ratio",
)

#: The per-realisation table (≈ 1.2 GB for 5000 realisations).
TABLE_FIELDS = SCALAR_FIELDS + IOP_FIELDS + AOP_FIELDS

#: The full 4-D reflectances (11.7 GB each for 5000 realisations).
FOUR_D_FIELDS = ("rrs", "Rrs", "Q")

#: Everything cached.
FIELDS = TABLE_FIELDS + FOUR_D_FIELDS

#: Each field's dims in the netCDF files (checked on every file read).
SOURCE_DIMS = {
    **{name: ("scalar",) for name in SCALAR_FIELDS},
    **{name: ("lambda",) for name in IOP_FIELDS},
    **{name: ("lambda", "theta_s") for name in AOP_FIELDS},
    **{name: ("lambda", "phi", "theta", "theta_s") for name in FOUR_D_FIELDS},
}

#: Each field's dims in the cache, after the leading ``realisation`` axis.
CACHE_DIMS = {
    **{name: () for name in SCALAR_FIELDS},
    **{name: ("lambda",) for name in IOP_FIELDS},
    **{name: ("theta_s", "lambda") for name in AOP_FIELDS},
    **{name: ("theta_s", "theta", "phi", "lambda") for name in FOUR_D_FIELDS},
}

#: The four angle/band axes of a 4-D field, in cache order.
FOUR_D_AXES = ("theta_s", "theta", "phi", "lambda")

#: Default seed for :func:`verify`'s 4-D sample.
VERIFY_SEED = 24

#: The realisations the M8 task-2 gate checks in full.
GATE_FULL_REALISATIONS = (1, 2500, 5000)


def default_cache_dir() -> Path:
    """Where the full cache lives: ``$OS_COLOR/SD/derived/pb24_v{N}/``.

    Resolved through ``ocpy.hydrolight.pb24.pb24_path`` (``$OS_COLOR/SD/v5``),
    so it follows ``$OS_COLOR``.

    Returns
    -------
    pathlib.Path
    """
    from ocpy.hydrolight import pb24

    return Path(pb24.pb24_path).parent / "derived" / f"pb24_v{CACHE_VERSION}"


def _resolve(cache_dir) -> Path:
    return default_cache_dir() if cache_dir is None else Path(cache_dir)


def _reader_version() -> int:
    from ocpy.hydrolight import pb24

    return int(pb24.READER_VERSION)


def _band_indices(bands, n_lambda: int) -> np.ndarray:
    """Validate a 4-D band subset (indices into the full λ grid)."""
    if bands is None:
        return np.arange(n_lambda, dtype=int)
    idx = np.asarray(bands, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("bands must not be empty")
    if np.any(np.diff(idx) <= 0):
        raise ValueError("bands must be strictly increasing")
    if idx[0] < 0 or idx[-1] >= n_lambda:
        raise ValueError(f"bands must lie in [0, {n_lambda})")
    return idx


def _extract(ds, field_names, bands: np.ndarray) -> dict:
    """One realisation's fields from a dataset, in cache layout, float32 as read.

    Checks each field's dims and dtype against :data:`SOURCE_DIMS`; no value
    is touched (transposes and band selection only).
    """
    out = {}
    for name in field_names:
        var = ds[name]
        if tuple(var.dims) != SOURCE_DIMS[name]:
            raise ValueError(
                f"PB24 field {name!r} has dims {var.dims}, expected "
                f"{SOURCE_DIMS[name]}; the cache layout no longer matches the files"
            )
        if var.dtype != np.float32:
            raise ValueError(f"PB24 field {name!r} is {var.dtype}, expected float32")
        values = var.values
        if name in SCALAR_FIELDS:
            arr = values.reshape(())
        elif name in IOP_FIELDS:
            arr = values
        elif name in AOP_FIELDS:
            arr = values.T  # (lambda, theta_s) -> (theta_s, lambda)
        else:  # (lambda, phi, theta, theta_s) -> (theta_s, theta, phi, lambda)
            arr = values.transpose(3, 2, 1, 0)
            if bands.size != values.shape[0]:
                arr = arr[..., bands]
        out[name] = np.array(arr, order="C", copy=True)  # keeps 0-d as 0-d
    return out


def _load_one(task):
    """Worker: read one realisation through the ocpy reader, in cache layout."""
    index, resolution, field_names, bands = task
    from ocpy.hydrolight import pb24

    ds = pb24.load_ds(int(index), resolution=resolution)
    return int(index), _extract(ds, field_names, bands)


def _map_ordered(func, tasks, workers: int):
    """``map`` in task order, optionally across ``workers`` processes.

    The order of results never depends on ``workers``, which is what keeps a
    build byte-identical regardless of parallelism. The pool forks, so it
    works from a script without a ``__main__`` guard or from stdin (no netCDF
    file is open in the parent when it starts: the reader reads eagerly and
    closes). If JAX is already imported, forking a multithreaded process can
    deadlock, so the pool is spawned instead (the caller then needs an
    importable ``__main__``, as pytest and the builder have).
    """
    if workers <= 1:
        for task in tasks:
            yield func(task)
        return
    ctx = multiprocessing.get_context("spawn" if "jax" in sys.modules else "fork")
    with ctx.Pool(workers) as pool:
        yield from pool.imap(func, tasks, chunksize=4)


def _sizes(cache_dir: Path, names) -> dict:
    return {name: (cache_dir / name).stat().st_size for name in names}


class _NpyStream:
    """Write a C-order float32 ``.npy`` one leading row at a time.

    The header is the one ``np.save`` / ``np.lib.format.open_memmap`` write
    for that shape and dtype; the rows are appended in order with plain
    sequential writes, so an 11.7 GB array is never in RAM. (Writing through
    ``open_memmap`` was tried first: on the ZFS data volume, ``msync`` of the
    35 GB of dirty mapped pages stalled for tens of minutes. Sequential
    writes produce the same bytes.)
    """

    def __init__(self, path: Path, shape: tuple):
        self.path = Path(path)
        self.shape = tuple(int(n) for n in shape)
        self.row_shape = self.shape[1:]
        self.rows = 0
        self.f = open(self.path, "wb")
        header = {
            "descr": np.lib.format.dtype_to_descr(np.dtype("<f4")),
            "fortran_order": False,
            "shape": self.shape,
        }
        np.lib.format.write_array_header_1_0(self.f, header)
        self.data_offset = self.f.tell()

    def write_row(self, values: np.ndarray) -> None:
        """Append one row (shape ``shape[1:]``, float32) verbatim."""
        if values.shape != self.row_shape or values.dtype != np.float32:
            raise ValueError(
                f"{self.path.name}: row {values.shape} {values.dtype}, expected "
                f"{self.row_shape} float32"
            )
        if self.rows >= self.shape[0]:
            raise ValueError(f"{self.path.name}: more than {self.shape[0]} rows")
        self.f.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
        self.rows += 1

    def close(self) -> None:
        """Check every row was written, flush and fsync."""
        if self.rows != self.shape[0]:
            raise ValueError(f"{self.path.name}: {self.rows} of {self.shape[0]} rows")
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        expect = self.data_offset + 4 * int(np.prod(self.shape))
        if self.path.stat().st_size != expect:
            raise OSError(f"{self.path}: size {self.path.stat().st_size} != {expect}")

    def abort(self) -> None:
        """Close without checks (the caller deletes the file)."""
        if not self.f.closed:
            self.f.close()


@dataclass
class BuildSummary:
    """What :func:`build` wrote.

    Attributes
    ----------
    cache_dir : pathlib.Path
    n_realisations : int
    sizes : dict
        File name -> bytes, the sidecar included.
    wall_s : float
        Wall time of the build, seconds.
    workers : int
    """

    cache_dir: Path
    n_realisations: int
    sizes: dict
    wall_s: float
    workers: int

    @property
    def total_bytes(self) -> int:
        """Bytes on disk, all files."""
        return int(sum(self.sizes.values()))

    def table(self) -> str:
        """A markdown table of file sizes, then the total and the wall time."""
        lines = ["| file | bytes | GB |", "|---|---:|---:|"]
        for name, size in sorted(self.sizes.items()):
            lines.append(f"| `{name}` | {size:,} | {size / 1e9:.4f} |")
        lines.append(
            f"| **total** | {self.total_bytes:,} | {self.total_bytes / 1e9:.3f} |"
        )
        lines.append("")
        lines.append(
            f"{self.n_realisations} realisations, {self.workers} worker(s), "
            f"wall time {self.wall_s:.1f} s."
        )
        return "\n".join(lines)


def build(
    cache_dir=None,
    realisations=None,
    resolution: str = "hyp",
    workers: int = 1,
    progress: bool = False,
    bands=None,
) -> BuildSummary:
    """Read each PB24 file once and write the cache.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Destination (created). Defaults to :func:`default_cache_dir`. Anything
        already there under the same names is replaced, the sidecar last.
    realisations : sequence of int, optional
        Realisation ids to cache; stored in ascending order. Defaults to every
        id ``ocpy.hydrolight.pb24.available(resolution)`` finds.
    resolution : str, optional
        ``"hyp"`` (default). ``"olci"`` works but is not what the task caches.
    workers : int, optional
        Reader processes. The output is byte-identical for any value.
    progress : bool, optional
        Print a progress line to stderr every 250 realisations.
    bands : sequence of int, optional
        Indices into the full λ grid to keep in the **4-D** fields (strictly
        increasing). Defaults to all bands. Recorded in the sidecar as
        ``four_d_bands``; the table fields always keep every band. Used for
        the committed test fixture.

    Returns
    -------
    BuildSummary
    """
    from ocpy.hydrolight import pb24

    t0 = time.perf_counter()
    cache_dir = _resolve(cache_dir)
    if realisations is None:
        realisations = pb24.available(resolution)
    ids = np.asarray(sorted({int(r) for r in realisations}), dtype=int)
    if ids.size == 0:
        raise FileNotFoundError(
            f"No PB24 {resolution!r} realisations to cache in {pb24.pb24_path}"
        )
    grids = pb24.grids(resolution)
    n_lambda = grids["lambda"].size
    band_idx = _band_indices(bands, n_lambda)

    angle_len = {name: grids[name].size for name in ("theta_s", "theta", "phi")}
    angle_len["lambda"] = n_lambda
    shapes = {}
    for name in FIELDS:
        dims = CACHE_DIMS[name]
        shape = [angle_len[d] for d in dims]
        if name in FOUR_D_FIELDS:
            shape[-1] = band_idx.size
        shapes[name] = (int(ids.size), *shape)

    sources = []
    for r in ids:
        path = Path(pb24.file_path(int(r), resolution))
        sources.append({"name": path.name, "bytes": int(path.stat().st_size)})

    cache_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    partial = {name: cache_dir / f"{name}.npy{PARTIAL_SUFFIX}" for name in FIELDS}
    try:
        writers = {name: _NpyStream(partial[name], shapes[name]) for name in FIELDS}
        tasks = [(int(r), resolution, FIELDS, band_idx) for r in ids]
        for row, (index, values) in enumerate(_map_ordered(_load_one, tasks, workers)):
            if index != ids[row]:  # cannot happen with an ordered map; guard anyway
                raise RuntimeError(f"row {row}: got realisation {index}")
            for name in FIELDS:
                writers[name].write_row(values[name])
            if progress and ((row + 1) % 250 == 0 or row + 1 == ids.size):
                print(
                    f"  {row + 1}/{ids.size} realisations "
                    f"({time.perf_counter() - t0:.0f} s)",
                    file=sys.stderr,
                    flush=True,
                )
        for writer in writers.values():
            writer.close()
        writers = {}

        sidecar = {
            "cache_version": CACHE_VERSION,
            "reader": "ocpy.hydrolight.pb24",
            "reader_version": _reader_version(),
            "resolution": resolution,
            "realisations": [int(r) for r in ids],
            "grids": {
                name: [float(v) for v in grids[name]] for name in pb24.GRID_NAMES
            },
            "four_d_bands": [int(b) for b in band_idx],
            "fields": {
                name: {
                    "file": f"{name}.npy",
                    "dims": ["realisation", *CACHE_DIMS[name]],
                    "source_dims": list(SOURCE_DIMS[name]),
                    "shape": list(shapes[name]),
                    "dtype": "float32",
                }
                for name in FIELDS
            },
            "sources": sources,
        }
        text = json.dumps(sidecar, sort_keys=True, indent=1) + "\n"

        # Commit: no sidecar while arrays are being swapped, sidecar last.
        (cache_dir / SIDECAR).unlink(missing_ok=True)
        for name in FIELDS:
            os.replace(partial[name], cache_dir / f"{name}.npy")
        tmp = cache_dir / f"{SIDECAR}{PARTIAL_SUFFIX}"
        tmp.write_text(text)
        os.replace(tmp, cache_dir / SIDECAR)
    finally:
        for writer in writers.values():
            writer.abort()
        for path in partial.values():
            path.unlink(missing_ok=True)

    names = [f"{name}.npy" for name in FIELDS] + [SIDECAR]
    return BuildSummary(
        cache_dir=cache_dir,
        n_realisations=int(ids.size),
        sizes=_sizes(cache_dir, names),
        wall_s=time.perf_counter() - t0,
        workers=int(workers),
    )


def _as_index(idx: np.ndarray):
    """An index array as a ``slice`` when it is an increasing progression.

    Slices give numpy views of a memmap; arrays give copies.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.ndim == 0:
        return int(idx)
    if idx.size == 0:
        return idx
    if idx.size == 1:
        return slice(int(idx[0]), int(idx[0]) + 1)
    step = np.diff(idx)
    if step[0] > 0 and np.all(step == step[0]):
        return slice(int(idx[0]), int(idx[-1]) + 1, int(step[0]))
    return idx


def _apply(arr, selectors):
    """Index ``arr`` axis by axis: basic selectors first (views), then arrays.

    ``selectors`` is one entry per leading axis: ``slice``, ``int`` or an
    int array. Arrays are applied one axis at a time with :func:`numpy.take`,
    so several of them select an outer product (not numpy's broadcast).
    """
    basic, fancy = [], []
    for axis, sel in enumerate(selectors):
        if isinstance(sel, np.ndarray):
            basic.append(slice(None))
            fancy.append((axis, sel))
        else:
            basic.append(sel)
    out = arr[tuple(basic)]
    for axis, sel in fancy:
        dropped = sum(isinstance(s, int) for s in selectors[:axis])
        out = np.take(out, sel, axis=axis - dropped)
    return out


@dataclass
class PB24Cache:
    """An opened cache: the sidecar plus lazily memory-mapped arrays.

    Build it with :func:`open_cache`. Arrays are opened with
    ``np.load(..., mmap_mode="r")`` on first use and are read-only.

    Attributes
    ----------
    cache_dir : pathlib.Path
    sidecar : dict
    """

    cache_dir: Path
    sidecar: dict
    _arrays: dict = field(default_factory=dict, repr=False)

    # -- metadata -----------------------------------------------------------
    @property
    def realisations(self) -> np.ndarray:
        """Realisation ids, one per leading row, ascending."""
        return np.asarray(self.sidecar["realisations"], dtype=int)

    @property
    def grids(self) -> dict:
        """``lambda``, ``theta_s``, ``theta``, ``phi`` as float32 arrays."""
        return {
            name: np.asarray(values, dtype=np.float32)
            for name, values in self.sidecar["grids"].items()
        }

    @property
    def four_d_bands(self) -> np.ndarray:
        """Indices into the full λ grid kept by the 4-D fields."""
        return np.asarray(self.sidecar["four_d_bands"], dtype=int)

    @property
    def four_d_wavelengths(self) -> np.ndarray:
        """Wavelengths (nm) of the 4-D fields' last axis."""
        return self.grids["lambda"][self.four_d_bands]

    @property
    def fields(self) -> tuple:
        """The cached field names."""
        return tuple(self.sidecar["fields"])

    # -- index helpers --------------------------------------------------------
    def realisation_index(self, ids) -> np.ndarray | int:
        """Row positions of realisation ids.

        Parameters
        ----------
        ids : int or sequence of int

        Returns
        -------
        int or numpy.ndarray
            A scalar for a scalar id, else an int array of the same length.

        Raises
        ------
        KeyError
            If any id is not in the cache.
        """
        real = self.realisations
        scalar = np.ndim(ids) == 0
        want = np.atleast_1d(np.asarray(ids, dtype=int))
        pos = np.searchsorted(real, want)
        pos = np.clip(pos, 0, real.size - 1)
        missing = want[real[pos] != want]
        if missing.size:
            raise KeyError(
                f"realisations not in the cache: {missing[:10].tolist()}"
                + (" ..." if missing.size > 10 else "")
            )
        return int(pos[0]) if scalar else pos

    def angle_index(self, axis: str, values) -> np.ndarray | int:
        """Positions of exact angle values on ``theta_s``, ``theta`` or ``phi``.

        Raises
        ------
        KeyError
            If a value is not a grid node (no nearest-node rounding).
        """
        if axis not in ("theta_s", "theta", "phi"):
            raise ValueError(f"axis must be theta_s, theta or phi, got {axis!r}")
        return self._exact(self.grids[axis], values, axis)

    def band_index(self, wavelengths) -> np.ndarray | int:
        """Positions of exact wavelengths (nm) on the full λ grid."""
        return self._exact(self.grids["lambda"], wavelengths, "lambda")

    @staticmethod
    def _exact(grid, values, label):
        scalar = np.ndim(values) == 0
        want = np.atleast_1d(np.asarray(values, dtype=float))
        g = grid.astype(float)
        hits = [np.flatnonzero(g == v) for v in want]
        pos = np.array([h[0] if h.size else -1 for h in hits], dtype=int)
        if np.any(pos < 0):
            raise KeyError(f"{label}: {want[pos < 0].tolist()} not on the grid")
        return int(pos[0]) if scalar else pos

    # -- arrays ---------------------------------------------------------------
    def array(self, name: str) -> np.memmap:
        """The whole memory-mapped array of one field (read-only).

        Raises
        ------
        KeyError
            For a name not in the cache.
        ValueError
            If the file's shape or dtype disagrees with the sidecar.
        """
        if name not in self.sidecar["fields"]:
            raise KeyError(f"{name!r} is not cached; fields are {self.fields}")
        if name not in self._arrays:
            meta = self.sidecar["fields"][name]
            arr = np.load(self.cache_dir / meta["file"], mmap_mode="r")
            if list(arr.shape) != meta["shape"] or arr.dtype != np.dtype(meta["dtype"]):
                raise ValueError(
                    f"{self.cache_dir / meta['file']}: shape {arr.shape} dtype "
                    f"{arr.dtype}, sidecar says {meta['shape']} {meta['dtype']}"
                )
            self._arrays[name] = arr
        return self._arrays[name]

    def _rows(self, realisations):
        if realisations is None:
            return slice(None)
        return _as_index(self.realisation_index(realisations))

    def table(self, name: str, realisations=None) -> np.ndarray:
        """A table field, for all or some realisations.

        Parameters
        ----------
        name : str
            One of :data:`TABLE_FIELDS`.
        realisations : int or sequence of int, optional
            Realisation **ids** (not rows). Default: all.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, *CACHE_DIMS[name])`` (no leading axis for a scalar
            id). A view of the memmap where the ids are an evenly spaced run
            of rows, else a copy.
        """
        if name not in TABLE_FIELDS:
            raise KeyError(f"{name!r} is not a table field")
        arr = self.array(name)
        return _apply(arr, [self._rows(realisations)])

    def four_d(
        self,
        name: str,
        realisations=None,
        theta_s=None,
        theta=None,
        phi=None,
        bands=None,
    ) -> np.ndarray:
        """Slice a 4-D field by realisation ids and grid **indices**.

        Parameters
        ----------
        name : str
            ``"rrs"``, ``"Rrs"`` or ``"Q"``.
        realisations : int or sequence of int, optional
            Realisation ids. Default: all.
        theta_s, theta, phi : int, slice or sequence of int, optional
            Indices on the sun-zenith, view-zenith and azimuth grids. Default:
            all. Use :meth:`four_d_at` to select by value.
        bands : int, slice or sequence of int, optional
            Indices into the **full** λ grid (as :meth:`band_index` returns);
            each must be among :attr:`four_d_bands`. Default: every stored
            band.

        Returns
        -------
        numpy.ndarray
            Axes ``(realisation, theta_s, theta, phi, lambda)``, with an axis
            dropped for each scalar selector. A view where every selector is
            a scalar, a slice or an evenly spaced run; otherwise a copy.
        """
        if name not in FOUR_D_FIELDS:
            raise KeyError(f"{name!r} is not a 4-D field")
        arr = self.array(name)
        selectors = [self._rows(realisations)]
        for sel in (theta_s, theta, phi):
            selectors.append(_angle_selector(sel))
        selectors.append(self._band_selector(bands))
        return _apply(arr, selectors)

    def four_d_at(
        self,
        name: str,
        realisations=None,
        theta_s=None,
        theta=None,
        phi=None,
        wavelengths=None,
    ) -> np.ndarray:
        """:meth:`four_d` with angles (deg) and wavelengths (nm) given as values.

        Values must be grid nodes exactly (no interpolation, no rounding).
        """
        return self.four_d(
            name,
            realisations=realisations,
            theta_s=None if theta_s is None else self.angle_index("theta_s", theta_s),
            theta=None if theta is None else self.angle_index("theta", theta),
            phi=None if phi is None else self.angle_index("phi", phi),
            bands=None if wavelengths is None else self.band_index(wavelengths),
        )

    def _band_selector(self, bands):
        """Map full-grid band indices to positions on the stored band axis."""
        if bands is None:
            return slice(None)
        stored = self.four_d_bands
        if isinstance(bands, slice):
            bands = np.arange(self.grids["lambda"].size)[bands]
        scalar = np.ndim(bands) == 0
        want = np.atleast_1d(np.asarray(bands, dtype=int))
        pos = np.searchsorted(stored, want)
        pos = np.clip(pos, 0, stored.size - 1)
        bad = want[stored[pos] != want]
        if bad.size:
            raise KeyError(
                f"bands {bad[:10].tolist()} are not stored in this cache's 4-D "
                f"fields (four_d_bands has {stored.size} bands)"
            )
        return int(pos[0]) if scalar else _as_index(pos)


def _angle_selector(sel):
    if sel is None:
        return slice(None)
    if isinstance(sel, slice):
        return sel
    if np.ndim(sel) == 0:
        return int(sel)
    return _as_index(np.asarray(sel, dtype=int))


def read_sidecar(cache_dir=None) -> dict:
    """The sidecar of a cache directory, as a dict.

    Raises
    ------
    FileNotFoundError
        If the directory has no sidecar (absent, or a build that never
        finished).
    """
    path = _resolve(cache_dir) / SIDECAR
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: no PB24 cache here (build it with "
            "design/py/build_pb24_cache.py)"
        )
    return json.loads(path.read_text())


def open_cache(cache_dir=None) -> PB24Cache:
    """Open a cache for reading.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Defaults to :func:`default_cache_dir`.

    Returns
    -------
    PB24Cache

    Raises
    ------
    FileNotFoundError
        No sidecar in the directory.
    ValueError
        The sidecar's ``cache_version`` is not :data:`CACHE_VERSION`, or its
        ``reader_version`` is not the installed reader's ``READER_VERSION`` --
        a stale cache is refused, not reused.
    """
    cache_dir = _resolve(cache_dir)
    sidecar = read_sidecar(cache_dir)
    if sidecar.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"{cache_dir}: cache version {sidecar.get('cache_version')}, expected "
            f"{CACHE_VERSION}; rebuild with design/py/build_pb24_cache.py"
        )
    if sidecar.get("reader_version") != _reader_version():
        raise ValueError(
            f"{cache_dir}: built with ocpy.hydrolight.pb24 READER_VERSION "
            f"{sidecar.get('reader_version')}, installed reader is "
            f"{_reader_version()}; rebuild with design/py/build_pb24_cache.py"
        )
    return PB24Cache(cache_dir=cache_dir, sidecar=sidecar)


@dataclass
class VerifyReport:
    """What :func:`verify` compared.

    Attributes
    ----------
    n_realisations : int
        Realisations re-read from the netCDFs.
    full : tuple
        Realisations whose 4-D fields were compared in full.
    fields : tuple
        Fields compared.
    n_table_values : int
        Table values compared (every one, for every realisation).
    n_four_d_values : int
        4-D values compared (in full for ``full``, a sample elsewhere).
    n_four_d_total : int
        4-D values in the cache for the realisations checked.
    mismatches : list
        ``(realisation, field, n_differing)`` for every failure.
    fraction, seed
        The sampling parameters.
    wall_s : float
    """

    n_realisations: int
    full: tuple
    fields: tuple
    n_table_values: int
    n_four_d_values: int
    n_four_d_total: int
    mismatches: list
    fraction: float
    seed: int
    wall_s: float

    @property
    def ok(self) -> bool:
        """True when every compared value is bit-identical."""
        return not self.mismatches

    def summary(self) -> str:
        """One paragraph for the builder's report."""
        share = self.n_four_d_values / self.n_four_d_total if self.n_four_d_total else 0
        verdict = "all bit-identical" if self.ok else f"MISMATCHES: {self.mismatches}"
        return (
            f"verify: {self.n_realisations} realisations re-read through the ocpy "
            f"reader; {self.n_table_values:,} table values compared (all of them); "
            f"{self.n_four_d_values:,} of {self.n_four_d_total:,} 4-D values "
            f"compared ({share:.2%}: realisations {list(self.full)} in full, a "
            f"random {self.fraction:.0%} of the rest, seed {self.seed}); "
            f"{verdict}"
            f" ({self.wall_s:.1f} s)."
        )


def _same_bits(a: np.ndarray, b: np.ndarray) -> int:
    """Number of float32 values whose 32-bit words differ (0 means identical).

    Stricter than ``np.array_equal(a, b, equal_nan=True)``: it also separates
    -0 from +0 and distinguishes NaN payloads.
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


def _verify_one(task):
    """Worker: compare one realisation's cache rows with a fresh file read."""
    cache_dir, index, fields, fraction, seed, full = task
    from ocpy.hydrolight import pb24

    cache = open_cache(cache_dir)
    row = cache.realisation_index(index)
    ds = pb24.load_ds(index, resolution=cache.sidecar["resolution"])
    truth = _extract(ds, fields, cache.four_d_bands)
    n_table = n4 = n4_total = 0
    bad = []
    for name in fields:
        cached = cache.array(name)[row]
        if name in TABLE_FIELDS:
            n_table += cached.size
            nd = _same_bits(cached, truth[name])
            # Belt and braces: the spec's own test, too.
            if not np.array_equal(cached, truth[name], equal_nan=True):
                nd = max(nd, 1)
        else:
            n4_total += cached.size
            if index in full or fraction >= 1:
                n4 += cached.size
                nd = _same_bits(cached, truth[name])
                if not np.array_equal(cached, truth[name], equal_nan=True):
                    nd = max(nd, 1)
            else:
                rng = np.random.default_rng([seed, index, FOUR_D_FIELDS.index(name)])
                k = max(1, int(round(fraction * cached.size)))
                pick = np.sort(rng.choice(cached.size, size=k, replace=False))
                a = np.asarray(cached).reshape(-1)[pick]
                b = truth[name].reshape(-1)[pick]
                n4 += k
                nd = _same_bits(a, b)
                if not np.array_equal(a, b, equal_nan=True):
                    nd = max(nd, 1)
        if nd:
            bad.append((index, name, nd))
    return n_table, n4, n4_total, bad


def verify(
    cache_dir=None,
    realisations=None,
    fields=None,
    fraction: float = 0.02,
    seed: int = VERIFY_SEED,
    full=(),
    workers: int = 1,
    raise_on_mismatch: bool = True,
) -> VerifyReport:
    """Re-read realisations through the ocpy reader and check bit-identity.

    For each realisation: every table field in ``fields`` is compared in
    full; each 4-D field is compared in full if the realisation is in
    ``full`` (or ``fraction >= 1``), else on a random ``fraction`` of its
    values drawn with ``numpy.random.default_rng([seed, id, field])`` -- so
    the sample does not depend on ``workers`` or on which other
    realisations are checked. Equality is of the raw 32-bit words, which
    implies ``np.array_equal(..., equal_nan=True)`` (also checked).

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
    realisations : sequence of int, optional
        Ids to check. Default: every realisation in the cache.
    fields : sequence of str, optional
        Default: every cached field.
    fraction : float, optional
        Share of each 4-D array sampled outside ``full``. Default 0.02.
    seed : int, optional
    full : sequence of int, optional
        Ids whose 4-D fields are compared in full.
    workers : int, optional
    raise_on_mismatch : bool, optional
        Raise ``AssertionError`` on any difference (default) rather than only
        report it.

    Returns
    -------
    VerifyReport
    """
    t0 = time.perf_counter()
    cache = open_cache(cache_dir)
    fields = tuple(cache.fields if fields is None else fields)
    for name in fields:
        if name not in cache.sidecar["fields"]:
            raise KeyError(f"{name!r} is not cached")
    ids = cache.realisations if realisations is None else realisations
    ids = [int(r) for r in np.atleast_1d(ids)]
    cache.realisation_index(ids)  # raise early on unknown ids
    from ocpy.hydrolight import pb24

    ref = pb24.grids(cache.sidecar["resolution"])
    for name, values in cache.grids.items():
        if _same_bits(values, ref[name]):
            raise AssertionError(f"cached grid {name!r} differs from the reader's")
    full = tuple(sorted({int(r) for r in full}))
    tasks = [
        (str(cache.cache_dir), r, fields, float(fraction), int(seed), full) for r in ids
    ]
    n_table = n4 = n4_total = 0
    bad = []
    for t, f, tot, b in _map_ordered(_verify_one, tasks, workers):
        n_table += t
        n4 += f
        n4_total += tot
        bad.extend(b)
    report = VerifyReport(
        n_realisations=len(ids),
        full=tuple(r for r in full if r in set(ids)),
        fields=fields,
        n_table_values=n_table,
        n_four_d_values=n4,
        n_four_d_total=n4_total,
        mismatches=bad,
        fraction=float(fraction),
        seed=int(seed),
        wall_s=time.perf_counter() - t0,
    )
    if raise_on_mismatch and not report.ok:
        raise AssertionError(report.summary())
    return report


# -- packing a miniature into one file --------------------------------------------

#: The fixed timestamp written into every zip member, so packing is deterministic.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def pack(cache_dir, zip_path) -> Path:
    """Pack a (small) cache directory into one deterministic zip file.

    Used for the committed test miniature: one file in the tree instead of
    one ``.npy`` per field. Members are written in sorted name order with a
    fixed timestamp, so packing the same directory twice gives identical
    bytes. :func:`unpack` restores a directory :func:`open_cache` accepts.

    Parameters
    ----------
    cache_dir : path-like
        A directory written by :func:`build` (sidecar included).
    zip_path : path-like
        Destination ``.zip``; overwritten if present.

    Returns
    -------
    pathlib.Path
        ``zip_path``.
    """
    import zipfile

    src = _resolve(cache_dir)
    if not (src / SIDECAR).is_file():
        raise FileNotFoundError(f"no {SIDECAR} in {src}: not a cache")
    dst = Path(zip_path)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in src.iterdir() if p.is_file()):
            info = zipfile.ZipInfo(path.name, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return dst


def unpack(zip_path, dest) -> Path:
    """Extract a zip written by :func:`pack` into ``dest`` and return it.

    Parameters
    ----------
    zip_path : path-like
    dest : path-like
        Directory to create (or fill).

    Returns
    -------
    pathlib.Path
        ``dest``, ready for :func:`open_cache`.
    """
    import zipfile

    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if any("/" in n or n.startswith("..") for n in names):
            raise ValueError(f"{zip_path}: unexpected member paths {names}")
        zf.extractall(out)
    return out
