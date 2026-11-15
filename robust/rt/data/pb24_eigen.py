"""
The eigen µ∞ of every PB24 (realisation, λ) — a derived product (M8 task 6).

µ∞ is an IOP: the asymptotic average cosine of the azimuthally averaged RTE,
computed by :func:`robust.rt.asymptotic.mu_infinity_eigen` from PB24's own
inputs -- absorption, pure-water scattering, and the two Fournier–Forand
particle components with each realisation's backscatter ratios (B_ph, B_NAP;
Pitarch & Brando 2025 §1.1). PB24 has no depth profiles, so this is the only
physical µ∞ it can be scored against (``rt_fit_pb24.md`` Q15).

The product is built by ``design/py/run_pb24_baseline.py --eigen-only`` and
lives next to the consolidation cache, **not inside it** (the cache directory
is verified file by file against its sidecar):
``$OS_COLOR/SD/derived/pb24_mu_inf_eigen_v{N}.npz``. It is written with fixed
zip timestamps and a JSON provenance member without a timestamp, so a rebuild
is byte-identical. Nothing here is on the model path.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "EIGEN_VERSION",
    "EIGEN_FILE",
    "MuInfEigen",
    "default_eigen_path",
    "save_mu_inf_eigen",
    "load_mu_inf_eigen",
]

#: Version of the product's layout and recipe; a reader refuses any other.
EIGEN_VERSION = 1

#: File name, beside the ``pb24_v1`` cache directory.
EIGEN_FILE = f"pb24_mu_inf_eigen_v{EIGEN_VERSION}.npz"

_ARRAYS = ("realisations", "wave", "mu_inf", "B_ph", "B_nap")


@dataclass(frozen=True)
class MuInfEigen:
    """The eigen µ∞ product.

    Attributes
    ----------
    realisations : numpy.ndarray
        ``(n,)`` 1-based ids, ascending.
    wave : numpy.ndarray
        ``(n_wave,)`` nm.
    mu_inf : numpy.ndarray
        ``(n, n_wave)`` float64.
    B_ph, B_nap : numpy.ndarray
        ``(n,)`` the FF backscatter ratios each realisation was solved with.
    provenance : dict
        Recipe (grid, streams, truncation, ...), source and measured
        interpolation error.
    """

    realisations: np.ndarray
    wave: np.ndarray
    mu_inf: np.ndarray
    B_ph: np.ndarray
    B_nap: np.ndarray
    provenance: dict

    def rows(self, ids) -> np.ndarray:
        """Row indices of realisation ``ids`` (raises if any is absent)."""
        ids = np.atleast_1d(np.asarray(ids, dtype=int))
        pos = np.searchsorted(self.realisations, ids)
        pos = np.clip(pos, 0, self.realisations.size - 1)
        bad = ids[self.realisations[pos] != ids]
        if bad.size:
            raise KeyError(f"realisations {bad[:10].tolist()} are not in the product")
        return pos


def default_eigen_path() -> Path:
    """``$OS_COLOR/SD/derived/`` + :data:`EIGEN_FILE`.

    Returns
    -------
    pathlib.Path
    """
    from . import pb24_cache

    return pb24_cache.default_cache_dir().parent / EIGEN_FILE


def save_mu_inf_eigen(
    path, *, realisations, wave, mu_inf, B_ph, B_nap, provenance: dict
) -> Path:
    """Write the product atomically and deterministically.

    Parameters
    ----------
    path : str or pathlib.Path
    realisations, wave, mu_inf, B_ph, B_nap : array_like
        See :class:`MuInfEigen`.
    provenance : dict
        JSON-serialisable; must not carry a timestamp (a rebuild must be
        byte-identical). ``version`` is added.

    Returns
    -------
    pathlib.Path
    """
    path = Path(path)
    arrays = {
        "realisations": np.asarray(realisations, dtype=np.int64),
        "wave": np.asarray(wave, dtype=np.float64),
        "mu_inf": np.asarray(mu_inf, dtype=np.float64),
        "B_ph": np.asarray(B_ph, dtype=np.float64),
        "B_nap": np.asarray(B_nap, dtype=np.float64),
    }
    n = arrays["realisations"].size
    if arrays["mu_inf"].shape != (n, arrays["wave"].size):
        raise ValueError(
            f"save_mu_inf_eigen: mu_inf {arrays['mu_inf'].shape} should be "
            f"({n}, {arrays['wave'].size})"
        )
    meta = dict(provenance) | {"version": EIGEN_VERSION}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in _ARRAYS:
            member = io.BytesIO()
            np.save(member, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, member.getvalue())
        info = zipfile.ZipInfo("provenance.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, json.dumps(meta, indent=1, sort_keys=True) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)
    return path


def load_mu_inf_eigen(path=None) -> MuInfEigen:
    """Read the product (default :func:`default_eigen_path`).

    Raises
    ------
    FileNotFoundError
        If it is absent (build it with ``run_pb24_baseline.py --eigen-only``).
    ValueError
        On a version mismatch.
    """
    path = Path(default_eigen_path() if path is None else path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: no eigen µ∞ product (design/py/run_pb24_baseline.py --eigen-only)"
        )
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("provenance.json"))
        if meta.get("version") != EIGEN_VERSION:
            raise ValueError(
                f"{path}: eigen µ∞ version {meta.get('version')}, expected "
                f"{EIGEN_VERSION}"
            )
        arrays = {
            name: np.load(io.BytesIO(zf.read(f"{name}.npy")), allow_pickle=False)
            for name in _ARRAYS
        }
    return MuInfEigen(**arrays, provenance=meta)
