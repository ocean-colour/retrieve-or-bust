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
    "OPTIONAL_RAW_FIELDS",
    "L23_FF_B",
    "INELASTIC_XS",
    "PHI_C_L23",
    "INELASTIC_RAW_FIELDS",
    "L23InelasticBatch",
    "load_inelastic_batch",
    "select_inelastic",
    "inelastic_npz_reader",
    # Depth profiles (M7 task 1)
    "PROFILE_FIELDS",
    "PROFILE_E_MIN",
    "ASYMPTOTIC_SPREAD",
    "PROFILE_MIN_DEPTH",
    "FLAG_UNUSABLE",
    "FLAG_NOT_ASYMPTOTIC",
    "FLAG_ASYMPTOTIC",
    "PROFILE_FLAG_NAMES",
    "PROFILE_PRODUCT_VERSION",
    "PROFILE_FIXTURE_DEPTHS",
    "PROFILE_FIXTURE_SCENES",
    "load_mu_profiles",
    "L23ProfileProduct",
    "build_profile_product",
    "default_profile_product_path",
    "save_profile_product",
    "load_profile_product",
    "write_profile_fixture",
    "profile_npz_reader",
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
    argument of :func:`~robust.rt.hybrid.forward` -- and :attr:`scene` is host-side
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

#: Fields a ``reader`` *may* supply (M7 task 2). With them, :func:`load_batch`
#: fills ``PhaseParams.ff_B`` / ``ff_weight`` -- L23's own two-component FF phase
#: function. They are optional, not part of :data:`RAW_FIELDS`, because the
#: committed ``l23_small.npz`` predates them and its bytes are hash-pinned; a
#: fixture-backed batch simply carries no FF information, which ZTT treats as
#: "use Sullivan".
OPTIONAL_RAW_FIELDS = ("bph", "bd")

#: Backscatter ratios of L23's two particle components, phytoplankton then
#: non-algal particles (Loisel et al. 2023, §2 and Eqs. 8, 11) -- the order of
#: ``PhaseParams.ff_B``'s component axis.
L23_FF_B = (0.010, 0.018)


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
        "bph": np.asarray(ds.bph.data, dtype=float),
        "bd": np.asarray(ds.bd.data, dtype=float),
    }


def write_fixture(
    path,
    *,
    zeniths: tuple[int, ...] = ZENITHS,
    x: int = ELASTIC_X,
    n_scene: int = 50,
    optional: bool = False,
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
    optional : bool, optional
        Also store :data:`OPTIONAL_RAW_FIELDS` (``bph``, ``bd``), so batches
        from the fixture carry L23's FF phase-function fields (M7). Off by
        default: ``l23_small.npz`` predates them and its bytes are hash-pinned.

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
        fields = RAW_FIELDS + (OPTIONAL_RAW_FIELDS if optional else ())
        for field in fields:
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
        """Serve one ``(x, zenith)`` slice of the fixture as a
        :data:`RAW_FIELDS` dict, raising if the fixture does not hold it."""
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
        # Optional M7 fields, served only by a fixture written with them.
        for field in OPTIONAL_RAW_FIELDS:
            key = f"{field}_{int(zenith)}"
            if key in data.files:
                out[field] = np.asarray(data[key], dtype=float)
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
    what L23 says with no convention drift. The two agree closely, since the
    table in ``conventions`` was derived from this same difference: measured,
    ``bb - bbnw`` reproduces :data:`robust.rt.conventions.BB_W_L23` to 3.1e-09
    relative at the scene the table was taken from, and to 3.4e-06 across all
    3320 scenes (float32 storage noise). ``test_bb_w_matches_l23_netcdf``
    asserts ``rtol=1e-6``, closing the loop between the modules.
    """
    if not zeniths:
        raise ValueError("load_batch: `zeniths` must name at least one angle")

    read = _read_file if reader is None else reader
    wave_ref: np.ndarray | None = None
    a_parts, bb_w_parts, bb_p_parts, bp_parts, rrs_parts = [], [], [], [], []
    theta_parts, scene_parts, ffw_parts = [], [], []

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
        if all(field in raw for field in OPTIONAL_RAW_FIELDS):
            # Scattering fractions, component axis last: (sample, wave, 2).
            parts = np.stack([raw["bph"][index], raw["bd"][index]], axis=-1)
            ffw_parts.append(parts / parts.sum(axis=-1, keepdims=True))
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
        phase_params=_l23_phase_params(B_p, ffw_parts, len(zeniths)),
        geometry=Geometry.nadir(jnp.asarray(np.concatenate(theta_parts))),
        Rrs=jnp.asarray(np.concatenate(rrs_parts)),
        wave=jnp.asarray(wave_ref),
        scene=np.concatenate(scene_parts),
    )

    if validate:
        batch.validate()
    return batch


def _l23_phase_params(B_p, ffw_parts, n_zenith) -> PhaseParams:
    """``PhaseParams`` for an L23 batch: ``B_p``, plus the FF mixture when every
    zenith's reader supplied the component scattering coefficients."""
    if len(ffw_parts) != n_zenith:
        return PhaseParams(B_p=jnp.asarray(B_p))
    ff_weight = np.concatenate(ffw_parts)
    ff_B = np.broadcast_to(np.asarray(L23_FF_B), ff_weight.shape)
    return PhaseParams(
        B_p=jnp.asarray(B_p),
        ff_B=jnp.asarray(ff_B),
        ff_weight=jnp.asarray(ff_weight),
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
#: tuple: The two inelastic L23 scenarios -- X=2 adds Raman scattering to the
#: elastic X=1, X=4 adds Raman *and* chlorophyll-a fluorescence.
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
        """Serve one zenith as an :data:`INELASTIC_RAW_FIELDS` dict, joining
        the sibling fixture to the elastic one and checking that the two
        describe the same scenes before returning."""
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


# --------------------------------------------------------- depth profiles (M7) --
#
# The ``Hydrolight1YY_profile.nc`` companions of the elastic files: Ed, Eu, Eo,
# Eod, Eou, Lu and the K's at 61 depths (-1 = above the surface, 0 = just below,
# then 0.1 m steps to 1 m and 1 m steps to 50 m). No loader read them before M7
# task 1 (rt_fit_ztt.md). ~727 MB each, so every read here asks for named fields
# only and nothing is held beyond what the caller requested.

#: The profile fields :func:`build_profile_product` consumes. A profile ``reader``
#: must supply each as a ``(z, scene, wave)`` array, plus ``z`` and ``wave``.
PROFILE_FIELDS = ("Ed_z", "Eu_z", "Eo_z", "md_z", "KLu_z")

#: Smallest irradiance (W m^-2 nm^-1) treated as data. Below it the float32
#: storage is **denormal or exactly zero** -- measured: at 50 m, 1.2% of
#: ``Eo_z`` values at 60 deg are exactly 0 (350-370 nm and 730-750 nm, every
#: scene), and the smallest positive ``Ed_z`` is 1e-45. A ratio of denormals is
#: noise, so a depth counts only where Ed, Eu and Eo all exceed this.
PROFILE_E_MIN = 1e-30

#: Relative spread across the three zeniths, ``(max - min) / mean`` of µ at the
#: deepest usable depth, below which the light field is called **asymptotic**.
#: µ∞ is an IOP, independent of the incident field, so three suns that agree to
#: within this have reached it; Q&A Design step 1 set the 1% figure.
ASYMPTOTIC_SPREAD = 0.01

#: A (scene, λ) whose deepest usable depth is shallower than this (m) is
#: ``unusable``: there is no "deep" to speak of.
PROFILE_MIN_DEPTH = 10.0

#: Quality flags of :class:`L23ProfileProduct`. ``not_asymptotic`` is the
#: honest name for what the plan called "extrapolated": the profiles stop at
#: 50 m, and M7 task 1 found no extrapolation of them trustworthy (see
#: :func:`build_profile_product`).
FLAG_UNUSABLE = 0
FLAG_NOT_ASYMPTOTIC = 1
FLAG_ASYMPTOTIC = 2
#: dict: Flag value -> name, for reports.
PROFILE_FLAG_NAMES = {
    FLAG_UNUSABLE: "unusable",
    FLAG_NOT_ASYMPTOTIC: "not_asymptotic",
    FLAG_ASYMPTOTIC: "asymptotic",
}

#: Bumped whenever the product's definition changes, so a stale cache is refused
#: rather than silently reused.
PROFILE_PRODUCT_VERSION = 1

#: Depths (m) kept by :func:`write_profile_fixture`: the surface, a spread
#: through the column, and the bottom, so the deepest-usable-depth logic has
#: something to choose between.
PROFILE_FIXTURE_DEPTHS = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0)

#: Scenes kept by :func:`write_profile_fixture` -- spread over the release so
#: the fixture holds both flags the release actually contains (asserted when it
#: is written). ``unusable`` never occurs in the full release (the shallowest
#: usable depth is 18 m), so tests inject it synthetically.
PROFILE_FIXTURE_SCENES = tuple(int(i) for i in np.linspace(0, N_SCENES - 1, 16))


def _read_profile(x: int, zenith: int, fields, depths=None) -> dict[str, np.ndarray]:
    """Read named fields of one ``_profile.nc`` file -- the default profile reader.

    Parameters
    ----------
    x : int
        L23 scenario index (1 = elastic).
    zenith : int
        Solar zenith angle in degrees.
    fields : sequence of str
        Depth-resolved fields to read, each ``(z, scene, wave)``.
    depths : sequence of float, optional
        Keep only these depths (m, matched exactly); ``None`` keeps all 61.

    Returns
    -------
    dict
        ``wave``, ``z``, and each requested field, as NumPy (float32, as stored).
    """
    from ocpy.hydrolight import loisel23

    ds = loisel23.load_ds(x, zenith, profile=True)
    z = np.asarray(ds.z.data, dtype=float)
    index = slice(None) if depths is None else _depth_index(z, depths)
    out = {"wave": np.asarray(ds.Lambda.data, dtype=float), "z": z[index]}
    out["scene"] = np.arange(ds.sizes["IOP_Scenario"])
    for field in fields:
        # isel before .data: only the requested depths are ever read from disk.
        out[field] = np.asarray(ds[field].isel(z=index).data)
    ds.close()
    return out


def _depth_index(z: np.ndarray, depths) -> np.ndarray:
    """Indices of ``depths`` in the profile grid ``z``, raising on any miss."""
    index = []
    for depth in depths:
        hit = np.flatnonzero(z == float(depth))
        if hit.size != 1:
            raise ValueError(f"L23 profiles have no depth {depth} m (grid: {z})")
        index.append(int(hit[0]))
    return np.asarray(index)


def load_mu_profiles(
    zenith: int,
    *,
    x: int = ELASTIC_X,
    scenes=None,
    depths=None,
    reader=None,
) -> tuple[np.ndarray, np.ndarray]:
    """The mean cosine ``µ(z) = (Ed − Eu) / Eo`` at every depth, for one zenith.

    Item (a) of M7 task 1. By Gershun's law ``µ(z) = a / K_net(z)``, and it tends
    to the IOP ``µ∞`` as the light field becomes asymptotic. Points whose
    irradiances fall below :data:`PROFILE_E_MIN` are returned as NaN rather than
    as a ratio of denormals.

    Parameters
    ----------
    zenith : int
        Solar zenith (degrees).
    x : int, optional
        L23 scenario index.
    scenes : array_like or slice, optional
        Scene subset; ``None`` keeps all.
    depths : sequence of float, optional
        Depth subset (m); ``None`` keeps all 61, including z = -1 (above water),
        where µ is that of the sky and sun, not of the ocean.
    reader : callable, optional
        ``(x, zenith, fields, depths) -> dict`` with ``wave``, ``z``, ``scene``
        (release indices) and the fields; defaults to the netCDF.

    Returns
    -------
    z : numpy.ndarray
        Depths (m), shape ``(n_z,)``.
    mu : numpy.ndarray
        float64, shape ``(n_z, n_scene, n_wave)``.
    """
    read = _read_profile if reader is None else reader
    raw = read(x, zenith, ("Ed_z", "Eu_z", "Eo_z"), depths)
    index = slice(None) if scenes is None else scenes
    Ed, Eu, Eo = (raw[f][:, index].astype(float) for f in ("Ed_z", "Eu_z", "Eo_z"))
    good = (Ed > PROFILE_E_MIN) & (Eu > PROFILE_E_MIN) & (Eo > PROFILE_E_MIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.where(good, (Ed - Eu) / Eo, np.nan)
    return raw["z"], mu


@dataclass(frozen=True)
class L23ProfileProduct:
    """Per-(scene, λ) quantities derived from the L23 depth profiles (M7 task 1).

    Host-side NumPy throughout: this is fitting data for ``design/py`` scripts,
    never an argument of the differentiable model.

    Attributes
    ----------
    wave : numpy.ndarray
        Wavelengths (nm), ``(n_wave,)``.
    zeniths : numpy.ndarray
        Solar zeniths (degrees), ``(n_zenith,)``; the leading axis of the
        per-zenith arrays.
    scene : numpy.ndarray
        Scene indices, ``(n_scene,)``.
    z_deep : numpy.ndarray
        The deepest depth (m) usable at **every** zenith, ``(n_scene, n_wave)``;
        NaN where there is none. One common depth, so the zeniths are compared
        like with like.
    mu_deep : numpy.ndarray
        µ = (Ed − Eu)/Eo at ``z_deep``, per zenith, ``(n_zenith, n_scene,
        n_wave)``, float64.
    spread : numpy.ndarray
        ``(max − min) / mean`` of ``mu_deep`` over zeniths, ``(n_scene, n_wave)``.
    flag : numpy.ndarray
        int8 quality flag, :data:`PROFILE_FLAG_NAMES`.
    mu_inf : numpy.ndarray
        The zenith mean of ``mu_deep`` where ``flag == FLAG_ASYMPTOTIC``, else
        NaN. **Only** there is the profile a measurement of µ∞.
    mu_d0 : numpy.ndarray
        µd at z = 0⁻, HydroLight's ``md_z`` (= Ed/Eod), per zenith, float32 as
        stored.
    KLu0 : numpy.ndarray
        ``KLu_z`` at z = 0⁻ (m^-1), per zenith, float32 as stored.
    provenance : dict
        How the product was made: version, thresholds, source files. No
        timestamp, so a rebuild is bit-identical, file included.
    """

    wave: np.ndarray
    zeniths: np.ndarray
    scene: np.ndarray
    z_deep: np.ndarray
    mu_deep: np.ndarray
    spread: np.ndarray
    flag: np.ndarray
    mu_inf: np.ndarray
    mu_d0: np.ndarray
    KLu0: np.ndarray
    provenance: dict

    #: The array fields, in a fixed order -- what save/load and equality use.
    ARRAYS = (
        "wave",
        "zeniths",
        "scene",
        "z_deep",
        "mu_deep",
        "spread",
        "flag",
        "mu_inf",
        "mu_d0",
        "KLu0",
    )

    def flag_fractions(self) -> dict[str, float]:
        """Fraction of (scene, λ) points carrying each flag, by name."""
        return {
            name: float(np.mean(self.flag == value))
            for value, name in PROFILE_FLAG_NAMES.items()
        }


def build_profile_product(
    zeniths: tuple[int, ...] = ZENITHS,
    *,
    x: int = ELASTIC_X,
    scenes=None,
    depths=None,
    reader=None,
) -> L23ProfileProduct:
    """Derive the µ∞ / µd / KLu product from the L23 profiles (M7 task 1).

    Parameters
    ----------
    zeniths : tuple of int, optional
        Zeniths to combine; µ∞ needs at least two to be judged.
    x : int, optional
        L23 scenario index (elastic by default).
    scenes : array_like, optional
        Scene indices to keep; ``None`` keeps every scene the reader serves.
    depths : sequence of float, optional
        Depths to consider (m); ``None`` uses every depth the reader serves.
    reader : callable, optional
        ``(x, zenith, fields, depths) -> dict``; defaults to the netCDF.
        :func:`profile_npz_reader` serves the committed fixture.

    Returns
    -------
    L23ProfileProduct

    Notes
    -----
    **What µ∞ is here, and why there is no extrapolation.** The plan (Q&A Design
    step 1) proposed extrapolating µ(z) to depth, jointly across the zeniths.
    Inspecting real profiles ruled it out:

    - the approach is **not monotonic** -- at 60° µ typically falls and then
      rises, while 0° falls from above -- so no single exponential describes it,
      and a multi-mode fit has more freedom than 50 m of data constrains;
    - it is **slow**: what sets it is the number of *scattering* lengths, and
      only points with ``b · 50 m ≳ 20`` reach agreement; that is **2.5%** of
      (scene, λ), almost none of them in clear blue water, where 50 m is
      ~1.5 optical depths.

    So the product reports what the profiles *measure* -- µ at the deepest
    usable depth, per zenith -- and calls it µ∞ only where three suns agree to
    :data:`ASYMPTOTIC_SPREAD`. Everywhere else ``mu_inf`` is NaN and the flag
    says why. Where µ∞ comes from instead is Q&A "Fit L23" Q13.
    """
    if len(zeniths) < 2:
        raise ValueError("build_profile_product: need at least two zeniths")
    read = _read_profile if reader is None else reader

    wave = z = scene_ids = None
    mu_all, md0, klu0 = [], [], []
    for zenith in zeniths:
        raw = read(x, zenith, PROFILE_FIELDS, depths)
        if wave is None:
            wave, z = raw["wave"], raw["z"]
        elif not (np.array_equal(raw["wave"], wave) and np.array_equal(raw["z"], z)):
            raise ValueError(f"build_profile_product: Y={zenith} grid differs")
        index = slice(None) if scenes is None else np.asarray(scenes)
        if scene_ids is None:
            scene_ids = np.asarray(raw["scene"], dtype=int)[index]
        Ed, Eu, Eo = (raw[f][:, index].astype(float) for f in ("Ed_z", "Eu_z", "Eo_z"))
        good = (Ed > PROFILE_E_MIN) & (Eu > PROFILE_E_MIN) & (Eo > PROFILE_E_MIN)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_all.append(np.where(good, (Ed - Eu) / Eo, np.nan))
        surface = _depth_index(z, (0.0,))[0]
        md0.append(raw["md_z"][surface, index])
        klu0.append(raw["KLu_z"][surface, index])

    mu_all = np.stack(mu_all)  # (zenith, z, scene, wave)
    n_scene, n_wave = mu_all.shape[2:]

    # The deepest depth below the surface usable at every zenith at once.
    below = z > 0.0
    usable = np.all(np.isfinite(mu_all), axis=0) & below[:, None, None]
    any_usable = usable.any(axis=0)
    last = usable.shape[0] - 1 - np.argmax(usable[::-1], axis=0)
    z_deep = np.where(any_usable, z[last], np.nan)

    mu_deep = np.take_along_axis(mu_all, last[None, None], axis=1)[:, 0]
    mu_deep = np.where(any_usable, mu_deep, np.nan)

    with np.errstate(invalid="ignore"):
        spread = (mu_deep.max(0) - mu_deep.min(0)) / mu_deep.mean(0)
        deep_enough = any_usable & (z_deep >= PROFILE_MIN_DEPTH)
        asymptotic = deep_enough & (spread < ASYMPTOTIC_SPREAD)
    flag = np.full((n_scene, n_wave), FLAG_UNUSABLE, dtype=np.int8)
    flag[deep_enough] = FLAG_NOT_ASYMPTOTIC
    flag[asymptotic] = FLAG_ASYMPTOTIC

    provenance = {
        "version": PROFILE_PRODUCT_VERSION,
        "generator": "robust.rt.data.l23.build_profile_product",
        "x": int(x),
        "zeniths": [int(zz) for zz in zeniths],
        "sources": [f"Hydrolight{x}{int(zz):02d}_profile.nc" for zz in zeniths],
        "depths": [float(d) for d in z],
        "E_min": PROFILE_E_MIN,
        "asymptotic_spread": ASYMPTOTIC_SPREAD,
        "min_depth": PROFILE_MIN_DEPTH,
        "mu_definition": "(Ed_z - Eu_z) / Eo_z at z_deep, float64 from float32",
    }
    return L23ProfileProduct(
        wave=np.asarray(wave),
        zeniths=np.asarray(zeniths, dtype=int),
        scene=scene_ids,
        z_deep=z_deep,
        mu_deep=mu_deep,
        spread=spread,
        flag=flag,
        mu_inf=np.where(asymptotic, mu_deep.mean(0), np.nan),
        mu_d0=np.stack(md0),
        KLu0=np.stack(klu0),
        provenance=provenance,
    )


def default_profile_product_path() -> Path:
    """Where the full-release product is cached: next to the data, not in git.

    ``$OS_COLOR/Loisel2023/derived/l23_profile_product_v{N}.npz`` -- about
    15 MB, too large to commit, and derived only from files that live there.

    Returns
    -------
    pathlib.Path
    """
    from ocpy.hydrolight import loisel23

    name = f"l23_profile_product_v{PROFILE_PRODUCT_VERSION}.npz"
    return Path(loisel23.l23_path) / "derived" / name


def save_profile_product(product: L23ProfileProduct, path) -> None:
    """Write a product to ``.npz`` atomically, provenance as a JSON header.

    Parameters
    ----------
    product : L23ProfileProduct
    path : str or pathlib.Path
        Destination; parent directories are created.
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(product, name) for name in L23ProfileProduct.ARRAYS}
    arrays["provenance"] = np.asarray(json.dumps(product.provenance, sort_keys=True))
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        load_profile_product(tmp)  # prove it loads before it replaces anything
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_profile_product(path) -> L23ProfileProduct:
    """Read a product written by :func:`save_profile_product`.

    Raises
    ------
    ValueError
        If the file was written under a different
        :data:`PROFILE_PRODUCT_VERSION` -- a stale cache is refused, not reused.
    """
    import json

    with np.load(path) as data:
        provenance = json.loads(str(data["provenance"]))
        if provenance.get("version") != PROFILE_PRODUCT_VERSION:
            raise ValueError(
                f"{path}: profile product version {provenance.get('version')}, "
                f"expected {PROFILE_PRODUCT_VERSION}; rebuild it with "
                "design/py/build_l23_profile_product.py"
            )
        arrays = {name: data[name] for name in L23ProfileProduct.ARRAYS}
    return L23ProfileProduct(**arrays, provenance=provenance)


def write_profile_fixture(
    path,
    *,
    zeniths: tuple[int, ...] = ZENITHS,
    x: int = ELASTIC_X,
    scenes=PROFILE_FIXTURE_SCENES,
    depths=PROFILE_FIXTURE_DEPTHS,
) -> None:
    """Snapshot raw profile fields for a few scenes and depths to an ``.npz``.

    Like :func:`write_fixture`, it stores the reader's *input*, so
    :func:`build_profile_product` really runs on it in CI. Written to a
    temporary file, then loaded back through :func:`profile_npz_reader` and
    :func:`build_profile_product`, and checked to hold both the ``asymptotic``
    and ``not_asymptotic`` flags before it replaces anything.

    Parameters
    ----------
    path : str or pathlib.Path
    zeniths, x, scenes, depths
        What to keep. ``scenes`` index the full release.
    """
    scenes = np.asarray(scenes, dtype=int)
    arrays: dict[str, np.ndarray] = {
        "zeniths": np.asarray(zeniths),
        "x": np.asarray(x),
        "scenes": scenes,
    }
    for zenith in zeniths:
        raw = _read_profile(x, zenith, PROFILE_FIELDS, depths)
        arrays["wave"], arrays["z"] = raw["wave"], raw["z"]
        for field in PROFILE_FIELDS:
            arrays[f"{field}_{zenith}"] = raw[field][:, scenes]

    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        product = build_profile_product(zeniths, x=x, reader=profile_npz_reader(tmp))
        needed = {FLAG_ASYMPTOTIC, FLAG_NOT_ASYMPTOTIC}
        missing = needed - set(np.unique(product.flag).tolist())
        if missing:
            raise ValueError(
                f"write_profile_fixture: snapshot lacks flags "
                f"{[PROFILE_FLAG_NAMES[m] for m in sorted(missing)]}; choose "
                f"other scenes. {path} left untouched"
            )
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def profile_npz_reader(path):
    """A profile ``reader`` backed by a :func:`write_profile_fixture` file.

    The fixture's scenes are served in the order they were written, and the
    reader's ``scene`` entry carries their release indices (the file's
    ``scenes`` array), so a product built from it is labelled correctly.

    Returns
    -------
    callable
        ``(x, zenith, fields, depths) -> dict``.

    Raises
    ------
    ValueError
        On a scenario, zenith, field or depth the fixture does not hold.
    """
    data = np.load(path)
    available = tuple(int(zz) for zz in data["zeniths"])
    stored_x = int(data["x"])
    z = np.asarray(data["z"], dtype=float)

    def read(x, zenith, fields, depths=None):
        """Serve one zenith's requested fields from the fixture."""
        if x != stored_x or int(zenith) not in available:
            raise ValueError(
                f"profile_npz_reader: fixture holds X={stored_x}, zeniths "
                f"{available}; asked for X={x}, Y={zenith}"
            )
        index = slice(None) if depths is None else _depth_index(z, depths)
        out = {"wave": np.asarray(data["wave"], dtype=float), "z": z[index]}
        out["scene"] = np.asarray(data["scenes"], dtype=int)
        for field in fields:
            key = f"{field}_{int(zenith)}"
            if key not in data:
                raise ValueError(f"profile_npz_reader: fixture lacks {field}")
            out[field] = np.asarray(data[key])[index]
        return out

    return read
