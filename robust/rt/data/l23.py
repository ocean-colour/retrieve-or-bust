"""
Loisel+2023 elastic reference batches.

Wraps ``ocpy.hydrolight.loisel23`` (reuse, not reimplementation) to turn the L23
netCDFs into the ``(IOPs, PhaseParams, Geometry, Rrs)`` batches the model trains
and scores against, plus the seeded held-out splits the M4 gate needs.

The elastic set is ``Hydrolight1{Y:02d}.nc`` -- X=1 means no inelastic processes --
at three solar-zenith angles, ``Y in {0, 30, 60}``: 3320 scenes x 81 wavelengths
(350-750 nm), nadir view, fixed Fournier-Forand phase function. The files live
outside the repo (~17 MB each); ``ocpy`` resolves the directory from ``$OS_COLOR``.

**Layout: one flat sample axis.** :func:`load_batch` returns ``n_zenith *
n_scene`` samples stacked along a single leading axis, so every leaf shares the
batch shape and ``jax.vmap(f, in_axes=0)`` just works. :attr:`L23Batch.scene`
and ``geometry.theta_s`` label each sample, which is what makes the per-zenith
metrics and the splits expressible as boolean masks.

**Measured properties of the current release** (verified across all three files;
tests pin each one):

- The IOP fields (``a``, ``bb``, ``bbnw``, ``bnw``, ...) are **bit-identical**
  across the three zenith files -- the same 3320 water bodies illuminated three
  ways. Only ``Rrs`` differs. This loader still reads each file's own IOPs and
  concatenates them, rather than tiling one copy: the saving would be ~13 MB and
  the cost would be silent breakage if a future release ever varied them.
- ``Rrs`` falls with solar zenith: median ``Rrs(60 deg) / Rrs(0 deg) = 0.949``,
  ``Rrs(30 deg) / Rrs(0 deg) = 0.990``. A ~5% effect at 60 deg, and the only
  geometry signal in hand -- notably, standard Gordon has no zenith dependence
  at all, which is what the M3/M4 comparison exploits.
- ``B_p = bbnw / bnw`` lies in **[0.01026, 0.01800]** over all 268,920
  (scene, wavelength) values at every zenith -- comfortably inside the design's
  nominal ~[0.004, 0.03], with no failure in the UV or the far red. ``bnw`` never
  approaches zero (minimum 6.1e-3), so the ratio is safe.
- ``B_p`` is **not** wavelength-independent within a scene (e.g. 0.0134 at 350 nm
  to 0.0125 at 750 nm), so it is carried as a spectrum, not a scalar.

**An honest limitation.** L23 spans only a factor ~1.75 in ``B_p`` (0.0103 to
0.0180) where the design's nominal band spans a factor ~7. So the prototype trains
on a narrow slice of phase-function space, and "explicit phase-function
dependence" is exercised only weakly until M5's HydroLight runs vary it properly.
Recorded here rather than discovered at M5.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from .. import conventions
from ..types import Geometry, IOPs, PhaseParams

__all__ = [  # noqa: RUF022  - grouped by role
    # Dataset constants
    "ELASTIC_X",
    "ZENITHS",
    "N_SCENES",
    "B_P_EXPECTED",
    # Split configuration
    "SPLIT_SEED",
    "TEST_FRACTION",
    "HELD_OUT_ZENITH",
    # Containers
    "L23Batch",
    "Splits",
    # Functions
    "load_batch",
    "make_splits",
    "select",
    # Cached-fixture seam
    "RAW_FIELDS",
    "write_fixture",
    "npz_reader",
    # Inelastic scenarios (inelastic coding plan, M1)
    "INELASTIC_XS",
    "PHI_C_L23",
    "INELASTIC_RAW_FIELDS",
    "L23InelasticBatch",
    "load_inelastic_batch",
    "select_inelastic",
    "inelastic_npz_reader",
]

#: L23 scenario index for the **elastic** set (no Raman, no fluorescence).
ELASTIC_X = 1

#: Solar zenith angles available in the elastic set, degrees.
ZENITHS = (0, 30, 60)

#: IOP scenarios per file.
N_SCENES = 3320

#: The design's nominal range for the particulate backscattering ratio (§4.2).
#: Advisory, not definitional: :func:`load_batch` *warns* outside it rather than
#: raising or clipping, since a synthetic sweep may legitimately leave it.
B_P_EXPECTED = (0.004, 0.03)

#: Seed for the scene split. Arbitrary but fixed forever -- changing it
#: invalidates every held-out number in the record.
SPLIT_SEED = 23

#: Fraction of *scenes* held out by the random split (coding plan CQ6).
TEST_FRACTION = 0.2

#: Solar zenith the geometry split holds out entirely (train 0/30, test 60).
HELD_OUT_ZENITH = 60


@dataclass(frozen=True)
class L23Batch:
    """A stacked L23 batch: model inputs, the reference ``Rrs``, and labels.

    Deliberately **not** a registered pytree. It is an analysis container, not an
    argument of :func:`robust.rt.forward` -- and :attr:`scene` is host-side
    integer metadata that has no business being traced or differentiated. The
    model inputs it holds (:attr:`iops`, :attr:`phase_params`, :attr:`geometry`)
    are pytrees.

    Attributes
    ----------
    iops : IOPs
        Absorption and the water/particle backscattering split, shape
        ``(n_sample, n_wave)`` per field.
    phase_params : PhaseParams
        ``B_p`` as a spectrum, shape ``(n_sample, n_wave)``.
    geometry : Geometry
        Per-sample solar zenith; nadir view, as L23 fixes it.
    Rrs : Array
        The HydroLight reference, sr^-1, shape ``(n_sample, n_wave)``.
    wave : Array
        Wavelengths (nm), shape ``(n_wave,)``.
    scene : numpy.ndarray
        IOP-scenario index in ``[0, N_SCENES)`` for each sample. NumPy, not JAX:
        it is a label used for splitting on the host.
    """

    iops: IOPs
    phase_params: PhaseParams
    geometry: Geometry
    Rrs: Float[Array, "sample wave"]
    wave: Float[Array, " wave"]
    scene: np.ndarray

    @property
    def n_sample(self) -> int:
        """Number of samples (scenes x zeniths)."""
        return int(self.Rrs.shape[0])

    @property
    def n_wave(self) -> int:
        """Number of wavelength bands."""
        return int(self.Rrs.shape[-1])

    @property
    def zenith(self) -> np.ndarray:
        """Per-sample solar zenith in degrees, as NumPy -- handy for grouping."""
        return np.asarray(self.geometry.theta_s)

    def validate(self) -> None:
        """Raise ``ValueError`` unless the batch is self-consistent and physical.

        Boundary check; not for use under ``jit``.

        Raises
        ------
        ValueError
            On a shape mismatch, a bad wavelength grid, a non-physical IOP,
            ``B_p`` outside ``(0, 1]``, an out-of-range angle, or a non-positive
            ``Rrs``.
        """
        conventions.check_wave(self.wave)
        self.iops.validate(wave=self.wave)
        self.phase_params.validate()
        self.geometry.validate()
        # Rrs is above-water here, so there is no rrs_to_Rrs pole to respect.
        conventions.check_rrs(self.Rrs, name="L23Batch.Rrs", subsurface=False)

        if self.Rrs.shape != self.iops.a.shape:
            raise ValueError(
                f"L23Batch: Rrs {self.Rrs.shape} does not match "
                f"IOPs.a {self.iops.a.shape}"
            )
        if self.scene.shape != (self.n_sample,):
            raise ValueError(
                f"L23Batch: scene {self.scene.shape} does not label "
                f"{self.n_sample} samples"
            )


@dataclass(frozen=True)
class Splits:
    """Boolean masks over a batch's sample axis for the two held-out tests.

    The M4 acceptance gate requires the hybrid to beat standard Gordon on **both**
    of these (coding plan CQ6), which is why they are produced together and from
    one seed.

    Attributes
    ----------
    scene_train, scene_test : numpy.ndarray
        The random split, held out **by scene**: every zenith of a held-out scene
        is held out with it. Splitting per *sample* instead would leak, since the
        same water body appears three times, and a model could be tested on IOPs
        it had already seen at another sun angle.
    zenith_train, zenith_test : numpy.ndarray
        The geometry split: train on 0 deg/30 deg, test on the unseen
        :data:`HELD_OUT_ZENITH`.
    test_scenes : numpy.ndarray
        The held-out scene indices, so the split can be reproduced or reported.
    """

    scene_train: np.ndarray
    scene_test: np.ndarray
    zenith_train: np.ndarray
    zenith_test: np.ndarray
    test_scenes: np.ndarray


#: The raw per-file fields the loader consumes. A ``reader`` must supply these.
RAW_FIELDS = ("wave", "Rrs", "a", "bb", "bbnw", "bnw")


def _read_file(x: int, zenith: int) -> dict[str, np.ndarray]:
    """Read one L23 file into plain NumPy arrays -- the default ``reader``.

    Parameters
    ----------
    x : int
        L23 scenario index (1 = elastic).
    zenith : int
        Solar zenith angle in degrees (0, 30, or 60).

    Returns
    -------
    dict
        The :data:`RAW_FIELDS`.
    """
    from ocpy.hydrolight import loisel23

    ds = loisel23.load_ds(x, zenith)
    return {
        "wave": np.asarray(ds.Lambda.data, dtype=float),
        "Rrs": np.asarray(ds.Rrs.data, dtype=float),
        "a": np.asarray(ds.a.data, dtype=float),
        "bb": np.asarray(ds.bb.data, dtype=float),
        "bbnw": np.asarray(ds.bbnw.data, dtype=float),
        "bnw": np.asarray(ds.bnw.data, dtype=float),
    }


def write_fixture(
    path,
    *,
    zeniths: tuple[int, ...] = ZENITHS,
    x: int = ELASTIC_X,
    n_scene: int = 50,
) -> None:
    """Snapshot the first ``n_scene`` scenes of the raw L23 fields to an ``.npz``.

    Deliberately stores the **raw per-file fields**, not an assembled
    :class:`L23Batch`. A snapshot of the loader's *output* would let tests check
    only that the snapshot is unchanged; storing its *input* means
    :func:`load_batch` itself runs -- against real numbers -- everywhere the
    fixture is available, including CI with no ``$OS_COLOR`` mount.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination ``.npz``. Compressed; ~50 scenes x 3 zeniths is ~200 kB.
    zeniths : tuple of int, optional
        Zeniths to include.
    x : int, optional
        L23 scenario index.
    n_scene : int, optional
        Number of leading scenes to keep.

    Notes
    -----
    Requires the L23 dataset. Values are stored as float32, the dtype the netCDF
    itself uses, so the fixture is bit-faithful to the source.

    **Written via a temporary file and verified before it replaces anything.** The
    destination is normally ``robust/tests/files/l23_small.npz``, which is committed
    and which the entire suite depends on when ``$OS_COLOR`` is absent, so an
    interrupted write or a snapshot that does not load would break CI with no
    obvious cause. The candidate is loaded back through :func:`npz_reader` and
    :func:`load_batch` -- the real consumers -- and only then moved into place with
    an atomic :func:`os.replace`. (The same defect class was reported against
    ``design/py/train_emulator.py`` in PR #11: validate first, overwrite second.)
    """
    arrays: dict[str, np.ndarray] = {
        "zeniths": np.asarray(zeniths),
        "x": np.asarray(x),
        "n_scene": np.asarray(n_scene),
    }
    for zenith in zeniths:
        raw = _read_file(x, zenith)
        arrays["wave"] = raw["wave"].astype(np.float32)
        for field in RAW_FIELDS:
            if field == "wave":
                continue
            arrays[f"{field}_{zenith}"] = raw[field][:n_scene].astype(np.float32)

    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        # Prove the snapshot is usable by the code that will use it, not merely that
        # savez returned. A fixture that loads to the wrong thing is worse than none.
        batch = load_batch(zeniths=zeniths, x=x, reader=npz_reader(tmp))
        batch.validate()
        if batch.n_sample != len(zeniths) * n_scene:
            raise ValueError(
                f"write_fixture: snapshot loads {batch.n_sample} samples, expected "
                f"{len(zeniths) * n_scene}; {path} left untouched"
            )
        # mkstemp creates 0600; a committed artefact must be readable by everyone
        # who can read the repo, so restore the permissions a plain open() would
        # have given it under the process umask.
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
        ``(x, zenith) -> dict`` of :data:`RAW_FIELDS`, for ``load_batch(...,
        reader=...)``.

    Raises
    ------
    ValueError
        If asked for a scenario or zenith the fixture does not hold -- silently
        serving the wrong angle would be far worse than failing.
    """
    data = np.load(path)
    available = tuple(int(z) for z in data["zeniths"])
    stored_x = int(data["x"])

    def read(x: int, zenith: int) -> dict[str, np.ndarray]:
        if x != stored_x:
            raise ValueError(
                f"npz_reader: fixture holds X={stored_x}, was asked for X={x}"
            )
        if int(zenith) not in available:
            raise ValueError(
                f"npz_reader: fixture holds zeniths {available}, was asked for {zenith}"
            )
        out = {"wave": np.asarray(data["wave"], dtype=float)}
        for field in RAW_FIELDS:
            if field == "wave":
                continue
            out[field] = np.asarray(data[f"{field}_{int(zenith)}"], dtype=float)
        return out

    return read


def load_batch(
    zeniths: tuple[int, ...] = ZENITHS,
    *,
    x: int = ELASTIC_X,
    scenes: np.ndarray | slice | None = None,
    validate: bool = True,
    reader=None,
) -> L23Batch:
    """Load the L23 elastic set as one stacked batch of JAX arrays.

    Parameters
    ----------
    zeniths : tuple of int, optional
        Solar zenith angles to include, degrees. Defaults to all of
        :data:`ZENITHS`.
    x : int, optional
        L23 scenario index; defaults to :data:`ELASTIC_X` (elastic). Passing
        X=2/4 would bring in Raman/fluorescence, which this model does not
        represent -- available only for a deliberate inelastic-delta check.
    scenes : array_like or slice, optional
        Subset of IOP scenarios to keep, applied identically at every zenith.
        Useful for fast tests; ``None`` keeps all :data:`N_SCENES`.
    validate : bool, optional
        Run :meth:`L23Batch.validate` before returning (default True).
    reader : callable, optional
        ``(x, zenith) -> dict`` of :data:`RAW_FIELDS`. Defaults to reading the
        netCDFs. The seam exists so a cached fixture (:func:`npz_reader`) can feed
        the *real* loader real numbers where the dataset is absent -- CI included.

    Returns
    -------
    L23Batch
        Samples ordered zenith-major: all scenes at ``zeniths[0]``, then all at
        ``zeniths[1]``, and so on.

    Warns
    -----
    UserWarning
        If ``B_p`` falls outside :data:`B_P_EXPECTED`. Reported, never clipped --
        silently squashing it would hide a change in the reference data.

    Notes
    -----
    ``bb_w`` and ``bb_p`` come from **the file itself** (``bb - bbnw`` and
    ``bbnw``), not from :func:`robust.rt.conventions.bb_w`, so a batch is exactly
    what L23 says with no convention drift. The two agree to ~1e-7 relative (the
    table in ``conventions`` was derived from this same difference); a test
    asserts that, closing the loop between the modules.
    """
    if not zeniths:
        raise ValueError("load_batch: `zeniths` must name at least one angle")

    read = _read_file if reader is None else reader
    wave_ref: np.ndarray | None = None
    a_parts, bb_w_parts, bb_p_parts, bp_parts, rrs_parts = [], [], [], [], []
    theta_parts, scene_parts = [], []

    for zenith in zeniths:
        raw = read(x, zenith)

        if wave_ref is None:
            wave_ref = raw["wave"]
            conventions.check_wave(wave_ref, name=f"L23 Lambda (Y={zenith})")
        elif not np.array_equal(raw["wave"], wave_ref):
            raise ValueError(
                f"load_batch: Y={zenith} has a different wavelength grid than "
                f"Y={zeniths[0]}"
            )

        index = slice(None) if scenes is None else scenes
        a = raw["a"][index]
        bb = raw["bb"][index]
        bbnw = raw["bbnw"][index]
        bnw = raw["bnw"][index]

        a_parts.append(a)
        # Straight from the file: water is what total minus non-water leaves.
        bb_w_parts.append(bb - bbnw)
        bb_p_parts.append(bbnw)
        bp_parts.append(bbnw / bnw)
        rrs_parts.append(raw["Rrs"][index])
        theta_parts.append(np.full(a.shape[0], float(zenith)))
        scene_parts.append(np.arange(raw["a"].shape[0])[index])

    B_p = np.concatenate(bp_parts)
    _warn_if_B_p_unexpected(B_p)

    batch = L23Batch(
        iops=IOPs(
            a=jnp.asarray(np.concatenate(a_parts)),
            bb_w=jnp.asarray(np.concatenate(bb_w_parts)),
            bb_p=jnp.asarray(np.concatenate(bb_p_parts)),
        ),
        phase_params=PhaseParams(B_p=jnp.asarray(B_p)),
        geometry=Geometry.nadir(jnp.asarray(np.concatenate(theta_parts))),
        Rrs=jnp.asarray(np.concatenate(rrs_parts)),
        wave=jnp.asarray(wave_ref),
        scene=np.concatenate(scene_parts),
    )

    if validate:
        batch.validate()
    return batch


def _warn_if_B_p_unexpected(B_p: np.ndarray) -> None:
    """Warn if ``B_p`` leaves :data:`B_P_EXPECTED`; never modify it."""
    lo, hi = B_P_EXPECTED
    outside = (B_p < lo) | (B_p > hi)
    if outside.any():
        warnings.warn(
            f"B_p leaves the expected range {B_P_EXPECTED}: "
            f"{int(outside.sum())} of {B_p.size} values, observed range "
            f"[{B_p.min():.5g}, {B_p.max():.5g}]. Reported, not clipped -- "
            "check whether the reference data changed.",
            UserWarning,
            stacklevel=3,
        )


def make_splits(
    batch: L23Batch,
    *,
    seed: int = SPLIT_SEED,
    test_fraction: float = TEST_FRACTION,
    held_out_zenith: int = HELD_OUT_ZENITH,
) -> Splits:
    """Build the two seeded held-out splits of the coding plan (CQ6).

    Parameters
    ----------
    batch : L23Batch
        The batch to split.
    seed : int, optional
        Seed for the scene draw; defaults to :data:`SPLIT_SEED`. Fixed so the
        held-out numbers in the record stay reproducible.
    test_fraction : float, optional
        Fraction of *scenes* held out, default :data:`TEST_FRACTION`.
    held_out_zenith : int, optional
        Solar zenith excluded from the geometry-split training set.

    Returns
    -------
    Splits
        Four boolean masks over the sample axis, plus the held-out scene indices.

    Raises
    ------
    ValueError
        If ``test_fraction`` is not in (0, 1), or if ``held_out_zenith`` is absent
        from the batch (which would leave the geometry test set empty -- a silently
        vacuous gate).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"make_splits: test_fraction must lie in (0, 1); got {test_fraction}"
        )

    zenith = batch.zenith
    if not np.any(zenith == held_out_zenith):
        raise ValueError(
            f"make_splits: the batch has no samples at {held_out_zenith} deg "
            f"(present: {sorted(set(zenith.tolist()))}), so the geometry "
            "hold-out would be empty"
        )

    unique_scenes = np.unique(batch.scene)
    n_test = max(1, int(round(test_fraction * unique_scenes.size)))
    drawn = np.random.default_rng(seed).permutation(unique_scenes)
    test_scenes = np.sort(drawn[:n_test])

    # Held out BY SCENE, so all zeniths of a scene move together. Splitting per
    # sample would leak: the same water body appears once per zenith.
    scene_test = np.isin(batch.scene, test_scenes)

    return Splits(
        scene_train=~scene_test,
        scene_test=scene_test,
        zenith_train=zenith != held_out_zenith,
        zenith_test=zenith == held_out_zenith,
        test_scenes=test_scenes,
    )


def select(batch: L23Batch, mask: np.ndarray) -> L23Batch:
    """Subset a batch along its sample axis.

    Parameters
    ----------
    batch : L23Batch
        The batch to subset.
    mask : numpy.ndarray
        Boolean mask of length ``batch.n_sample``, e.g. one of the
        :class:`Splits` fields.

    Returns
    -------
    L23Batch
        A new batch holding only the selected samples. ``wave`` is unchanged.

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
    return L23Batch(
        iops=_take_tree(batch.iops, keep),
        phase_params=_take_tree(batch.phase_params, keep),
        geometry=_take_geometry(batch.geometry, keep),
        Rrs=batch.Rrs[keep],
        wave=batch.wave,
        scene=batch.scene[mask],
    )


def _take_tree(tree, keep):
    """Subset a per-sample pytree leaf-wise.

    ``tree_map`` rather than a field-by-field rebuild: a hand-enumerated
    rebuild silently dropped ``IOPs.a_ph`` (PR #14 review, record §2.7
    finding 2 — the defect class ``gradient_report``'s docstring records).
    Every present leaf is subset, including fields added later, with no
    per-field code here.
    """
    return jax.tree_util.tree_map(lambda leaf: leaf[keep], tree)


def _take_geometry(geometry: Geometry, keep) -> Geometry:
    """Subset a Geometry's per-sample fields, carrying the Ed override whole.

    Deliberately NOT a blanket ``tree_map``: the ``Ed`` override is one sky
    for the whole batch — a pair of 1-D *spectral* arrays, not per-sample
    leaves — so indexing it by ``keep`` would corrupt it.
    """
    return Geometry(
        theta_s=geometry.theta_s[keep],
        theta_v=geometry.theta_v[keep],
        dphi=geometry.dphi[keep],
        wind=None if geometry.wind is None else geometry.wind[keep],
        Ed=geometry.Ed,
    )


# ------------------------------------------------------ inelastic scenarios --
#: The two inelastic L23 scenarios: X=2 adds Raman scattering to the elastic
#: X=1, X=4 adds Raman *and* chlorophyll-a fluorescence.
INELASTIC_XS = (2, 4)

#: The chlorophyll fluorescence quantum yield HydroLight used for the X=4
#: runs. The truth channel ``Rrs_X4 - Rrs_X2`` is therefore fluorescence at
#: exactly this yield -- the reason ``Inelastic.phi_C`` defaults to it.
PHI_C_L23 = 0.02

#: The per-zenith fields an inelastic ``reader`` must supply. One read spans
#: all three scenarios (the IOPs and ``aph`` are bit-identical across X --
#: verified 2026-08-24 and asserted by the default reader), so unlike the
#: elastic :data:`RAW_FIELDS` contract the reader takes only ``zenith``.
INELASTIC_RAW_FIELDS = (
    "wave",
    "a",
    "bb",
    "bbnw",
    "bnw",
    "aph",
    "ag",
    "Rrs1",
    "Rrs2",
    "Rrs4",
)


@dataclass(frozen=True)
class L23InelasticBatch:
    """A stacked L23 inelastic batch: inputs, per-scenario ``Rrs``, labels.

    The inelastic sibling of :class:`L23Batch` (same layout decisions: one
    flat zenith-major sample axis, host-side ``scene`` labels, deliberately
    not a pytree). Differences: :attr:`iops` carries ``a_ph`` (the
    fluorescence source term), and the reference is three channels rather
    than one -- the paired scenarios whose differences are exact truth for
    the inelastic terms.

    :func:`make_splits` accepts this container unchanged (it reads only
    ``scene`` and ``zenith``), which is what "reuse the elastic splits
    verbatim" means mechanically: identical scene labels + identical seed =
    identical masks, and a test proves it rather than trusting this
    docstring.

    Attributes
    ----------
    iops : IOPs
        With ``a_ph`` and ``a_cdom`` set; identical across the X scenarios
        (asserted at read time).
    phase_params : PhaseParams
        ``B_p`` spectrum, as elastic.
    geometry : Geometry
        Per-sample solar zenith; nadir view.
    Rrs_x1, Rrs_x2, Rrs_x4 : Array
        The elastic, +Raman, and +Raman+fluorescence references (sr^-1),
        shape ``(n_sample, n_wave)`` each.
    wave : Array
        Wavelengths (nm).
    scene : numpy.ndarray
        IOP-scenario labels, as elastic.
    """

    iops: IOPs
    phase_params: PhaseParams
    geometry: Geometry
    Rrs_x1: Float[Array, "sample wave"]
    Rrs_x2: Float[Array, "sample wave"]
    Rrs_x4: Float[Array, "sample wave"]
    wave: Float[Array, " wave"]
    scene: np.ndarray

    @property
    def n_sample(self) -> int:
        """Number of samples (scenes x zeniths)."""
        return int(self.Rrs_x1.shape[0])

    @property
    def n_wave(self) -> int:
        """Number of wavelength bands."""
        return int(self.Rrs_x1.shape[-1])

    @property
    def zenith(self) -> np.ndarray:
        """Per-sample solar zenith in degrees, as NumPy."""
        return np.asarray(self.geometry.theta_s)

    @property
    def truth_raman_factor(self) -> Float[Array, "sample wave"]:
        """The Raman-correction truth channel ``Rrs_X2 / Rrs_X1`` (design §4.1).

        What the analytic ``f_phys`` (M2) and the corrected ``f_R`` (M3) are
        scored against. Measured range in L23: 1.0076-2.51 -- Raman only ever
        adds photons, so a value below 1 signals a data or indexing error.
        """
        return self.Rrs_x2 / self.Rrs_x1

    @property
    def truth_fluorescence(self) -> Float[Array, "sample wave"]:
        """The additive fluorescence truth ``Rrs_X4 - Rrs_X2`` at phi_C = 0.02.

        Strictly positive at the 685 nm peak in every L23 sample.
        """
        return self.Rrs_x4 - self.Rrs_x2

    def validate(self) -> None:
        """Raise ``ValueError`` unless the batch is self-consistent and physical.

        Boundary check; not for use under ``jit``. Includes ``a_ph`` presence:
        an inelastic batch without the fluorescence source term is a loader
        bug, not a configuration.
        """
        conventions.check_wave(self.wave)
        if self.iops.a_ph is None:
            raise ValueError(
                "L23InelasticBatch: iops.a_ph is None -- the inelastic loader "
                "must supply the fluorescence source term"
            )
        if self.iops.a_cdom is None:
            raise ValueError(
                "L23InelasticBatch: iops.a_cdom is None -- the inelastic loader "
                "must supply the CDOM-fluorescence source term"
            )
        self.iops.validate(wave=self.wave)
        self.phase_params.validate()
        self.geometry.validate()
        for name, channel in (
            ("Rrs_x1", self.Rrs_x1),
            ("Rrs_x2", self.Rrs_x2),
            ("Rrs_x4", self.Rrs_x4),
        ):
            conventions.check_rrs(channel, name=name, subsurface=False)
            if channel.shape != self.iops.a.shape:
                raise ValueError(
                    f"L23InelasticBatch: {name} {channel.shape} does not match "
                    f"IOPs.a {self.iops.a.shape}"
                )
        if self.scene.shape != (self.n_sample,):
            raise ValueError(
                f"L23InelasticBatch: scene {self.scene.shape} does not label "
                f"{self.n_sample} samples"
            )


def _read_inelastic_file(zenith: int) -> dict[str, np.ndarray]:
    """Read one zenith of all three scenarios -- the default inelastic reader.

    The IOPs and ``aph`` are **bit-identical** across X=1/2/4 (the same water,
    three radiative-transfer configurations); read once from X=1 and asserted
    against the others, so a future release that varies them stops the loader
    instead of silently mixing scenarios.
    """
    from ocpy.hydrolight import loisel23

    datasets = {x: loisel23.load_ds(x, zenith) for x in (1, *INELASTIC_XS)}
    ds1 = datasets[1]
    out = {
        "wave": np.asarray(ds1.Lambda.data, dtype=float),
        "a": np.asarray(ds1.a.data, dtype=float),
        "bb": np.asarray(ds1.bb.data, dtype=float),
        "bbnw": np.asarray(ds1.bbnw.data, dtype=float),
        "bnw": np.asarray(ds1.bnw.data, dtype=float),
        "aph": np.asarray(ds1.aph.data, dtype=float),
        "ag": np.asarray(ds1.ag.data, dtype=float),
    }
    for x, ds in datasets.items():
        for key in ("a", "bb", "aph", "ag"):
            if not np.array_equal(np.asarray(ds[key].data, dtype=float), out[key]):
                raise ValueError(
                    f"L23 inelastic read: {key} differs between X=1 and X={x} "
                    f"at Y={zenith} -- scenarios no longer share their inputs"
                )
        out[f"Rrs{x}"] = np.asarray(ds.Rrs.data, dtype=float)
    return out


def inelastic_npz_reader(path, elastic_path):
    """An inelastic ``reader`` backed by the sibling + elastic fixtures.

    The sibling fixture (CQ4) deliberately stores only what the elastic
    fixture lacks -- ``aph``, ``ag``, and the X=2/X=4 ``Rrs`` -- plus ``a``/``bb``/
    ``Rrs1`` copies used here to *prove* the two files describe the same 50
    scenes (a mismatch raises rather than silently pairing different water).
    ``bbnw``/``bnw`` come from the elastic fixture.

    Parameters
    ----------
    path : str or pathlib.Path
        The sibling fixture (``l23_inelastic_fixture.npz``).
    elastic_path : str or pathlib.Path
        The elastic fixture (``l23_small.npz``), same scenes and zeniths.

    Returns
    -------
    callable
        ``(zenith) -> dict`` of :data:`INELASTIC_RAW_FIELDS`.
    """
    sibling = np.load(path)
    elastic_read = npz_reader(elastic_path)
    available = tuple(int(z) for z in sibling["zeniths"])

    def read(zenith: int) -> dict[str, np.ndarray]:
        if int(zenith) not in available:
            raise ValueError(
                f"inelastic_npz_reader: fixture holds zeniths {available}, "
                f"was asked for {zenith}"
            )
        elastic = elastic_read(ELASTIC_X, zenith)
        out = {
            "wave": np.asarray(sibling["wave"], dtype=float),
            "bbnw": elastic["bbnw"],
            "bnw": elastic["bnw"],
        }
        for field, elastic_key in (("a", "a"), ("bb", "bb"), ("Rrs1", "Rrs")):
            ours = np.asarray(sibling[f"{field}_{int(zenith)}"], dtype=float)
            if not np.array_equal(ours, elastic[elastic_key]):
                raise ValueError(
                    f"inelastic_npz_reader: {field} at Y={zenith} disagrees "
                    "with the elastic fixture -- the two fixtures do not "
                    "describe the same scenes"
                )
            out[field] = ours
        for field in ("aph", "ag", "Rrs2", "Rrs4"):
            out[field] = np.asarray(sibling[f"{field}_{int(zenith)}"], dtype=float)
        return out

    return read


def load_inelastic_batch(
    zeniths: tuple[int, ...] = ZENITHS,
    *,
    scenes: np.ndarray | slice | None = None,
    validate: bool = True,
    reader=None,
) -> L23InelasticBatch:
    """Load the paired X=1/2/4 scenarios as one stacked inelastic batch.

    The inelastic sibling of :func:`load_batch`, sharing its layout exactly
    (zenith-major flat sample axis, same scene labels) so the **elastic
    splits apply verbatim**: ``make_splits`` on this batch equals
    ``make_splits`` on the elastic batch, seed for seed.

    Parameters
    ----------
    zeniths, scenes, validate
        As :func:`load_batch`.
    reader : callable, optional
        ``(zenith) -> dict`` of :data:`INELASTIC_RAW_FIELDS`. Defaults to
        reading the six netCDFs; :func:`inelastic_npz_reader` feeds the same
        loader from the committed fixtures -- CI included.

    Returns
    -------
    L23InelasticBatch
    """
    if not zeniths:
        raise ValueError("load_inelastic_batch: `zeniths` must name at least one angle")

    read = _read_inelastic_file if reader is None else reader
    wave_ref: np.ndarray | None = None
    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "a",
            "bb_w",
            "bb_p",
            "a_ph",
            "a_cdom",
            "B_p",
            "Rrs1",
            "Rrs2",
            "Rrs4",
        )
    }
    theta_parts, scene_parts = [], []

    for zenith in zeniths:
        raw = read(zenith)

        if wave_ref is None:
            wave_ref = raw["wave"]
            conventions.check_wave(wave_ref, name=f"L23 Lambda (Y={zenith})")
        elif not np.array_equal(raw["wave"], wave_ref):
            raise ValueError(
                f"load_inelastic_batch: Y={zenith} has a different wavelength "
                f"grid than Y={zeniths[0]}"
            )

        index = slice(None) if scenes is None else scenes
        a = raw["a"][index]
        bbnw = raw["bbnw"][index]

        parts["a"].append(a)
        parts["bb_w"].append(raw["bb"][index] - bbnw)
        parts["bb_p"].append(bbnw)
        parts["a_ph"].append(raw["aph"][index])
        parts["a_cdom"].append(raw["ag"][index])
        parts["B_p"].append(bbnw / raw["bnw"][index])
        for x in (1, *INELASTIC_XS):
            parts[f"Rrs{x}"].append(raw[f"Rrs{x}"][index])
        theta_parts.append(np.full(a.shape[0], float(zenith)))
        scene_parts.append(np.arange(raw["a"].shape[0])[index])

    B_p = np.concatenate(parts["B_p"])
    _warn_if_B_p_unexpected(B_p)

    batch = L23InelasticBatch(
        iops=IOPs(
            a=jnp.asarray(np.concatenate(parts["a"])),
            bb_w=jnp.asarray(np.concatenate(parts["bb_w"])),
            bb_p=jnp.asarray(np.concatenate(parts["bb_p"])),
            a_ph=jnp.asarray(np.concatenate(parts["a_ph"])),
            a_cdom=jnp.asarray(np.concatenate(parts["a_cdom"])),
        ),
        phase_params=PhaseParams(B_p=jnp.asarray(B_p)),
        geometry=Geometry.nadir(jnp.asarray(np.concatenate(theta_parts))),
        Rrs_x1=jnp.asarray(np.concatenate(parts["Rrs1"])),
        Rrs_x2=jnp.asarray(np.concatenate(parts["Rrs2"])),
        Rrs_x4=jnp.asarray(np.concatenate(parts["Rrs4"])),
        wave=jnp.asarray(wave_ref),
        scene=np.concatenate(scene_parts),
    )

    if validate:
        batch.validate()
    return batch


def select_inelastic(batch: L23InelasticBatch, mask: np.ndarray) -> L23InelasticBatch:
    """Subset an inelastic batch along its sample axis.

    The inelastic sibling of :func:`select`, sharing its helpers -- so the
    PR #14 lesson (subset leaf-wise; carry the Ed override whole) is applied
    once, not maintained twice.

    Parameters
    ----------
    batch : L23InelasticBatch
        The batch to subset.
    mask : numpy.ndarray
        Boolean mask of length ``batch.n_sample``.

    Returns
    -------
    L23InelasticBatch

    Raises
    ------
    ValueError
        If ``mask`` is not a boolean array of the right length.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (batch.n_sample,):
        raise ValueError(
            f"select_inelastic: expected a boolean mask of shape "
            f"({batch.n_sample},); got dtype {mask.dtype}, shape {mask.shape}"
        )

    keep = jnp.asarray(np.flatnonzero(mask))
    return L23InelasticBatch(
        iops=_take_tree(batch.iops, keep),
        phase_params=_take_tree(batch.phase_params, keep),
        geometry=_take_geometry(batch.geometry, keep),
        Rrs_x1=batch.Rrs_x1[keep],
        Rrs_x2=batch.Rrs_x2[keep],
        Rrs_x4=batch.Rrs_x4[keep],
        wave=batch.wave,
        scene=batch.scene[mask],
    )
