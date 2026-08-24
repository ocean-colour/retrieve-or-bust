"""
Shared pytest configuration for the retrieve-or-bust suite.

Two concerns live here, both of which every later test module needs.

**The reference data is not in the repo.** The Loisel+2023 (L23) Hydrolight
files are ~17 MB each and sit outside the tree; ``ocpy`` resolves their directory
from ``$OS_COLOR``. Anything from M1 onward that touches the RT reference needs
them, so :data:`needs_l23` exists to declare that dependency once and skip
cleanly when the data mount is absent — a skip, not a failure, so ``pytest -q``
stays meaningful on a machine without the dataset (``-ra`` prints why). This
follows BING's ``conftest.py``, which solves the same problem for the same files.

**Float64 is off by default in JAX.** The M2 gradient gate compares ``jax.grad``
against central finite differences, and in float32 the differencing noise swamps
the comparison. :func:`jax_x64` turns x64 on for a single test and restores the
previous setting afterwards, so a float64 test cannot silently change the dtype
regime other tests run under. Note that ``jax.experimental.enable_x64`` — the
context manager the older JAX docs suggest — was removed by JAX 0.11, hence the
explicit set/restore.
"""

import os
import pathlib

import pytest

#: The L23 *elastic* files (X=1, no inelastic) at the three solar-zenith angles
#: Y = 0/30/60 deg. The prototype needs all three (coding plan M1).
L23_ELASTIC_FILES = ("Hydrolight100.nc", "Hydrolight130.nc", "Hydrolight160.nc")

#: The L23 *inelastic* files: X=2 (+ Raman) and X=4 (+ Raman and chlorophyll
#: fluorescence) at the same three zeniths. Guarded separately from
#: :data:`L23_ELASTIC_FILES` because a machine can legitimately hold the elastic
#: three without these six — the inelastic effort's raw-netCDF tests must then
#: skip, not fail (inelastic coding plan, M1).
L23_INELASTIC_FILES = (
    "Hydrolight200.nc",
    "Hydrolight230.nc",
    "Hydrolight260.nc",
    "Hydrolight400.nc",
    "Hydrolight430.nc",
    "Hydrolight460.nc",
)

#: Fixtures directory (BING layout).
FILES = pathlib.Path(__file__).parent / "files"

#: A committed 50-scene x 3-zenith snapshot of the **raw** L23 fields, ~213 kB.
#: Because it holds the loader's *input* rather than its output, the real
#: :func:`robust.rt.data.l23.load_batch` runs against real numbers wherever this
#: file is present -- CI included, with no ``$OS_COLOR`` mount. Regenerate with
#: ``l23.write_fixture(path, n_scene=50)``.
L23_SMALL_FIXTURE = FILES / "l23_small.npz"

#: The inelastic **sibling** fixture (coding plan CQ4): the same 50 scenes and
#: zeniths, adding ``aph`` and the X=2/X=4 ``Rrs`` channels (plus ``a``/``bb``/
#: ``Rrs1`` copies the reader uses to prove the two files describe the same
#: water). The elastic fixture's bytes are untouched -- a test pins its hash.
#: Regenerate with ``design/py/gen_inelastic_fixture.py``.
L23_INELASTIC_FIXTURE = FILES / "l23_inelastic_fixture.npz"


def l23_available():
    """Whether the L23 elastic dataset is on disk.

    Returns
    -------
    bool
        True when ``ocpy`` imports *and* every file in
        :data:`L23_ELASTIC_FILES` is present in its L23 directory.
    """
    try:
        from ocpy.hydrolight import loisel23
    except Exception:  # noqa: BLE001 - any failure here means "no data", by design
        return False
    return all(
        os.path.isfile(os.path.join(loisel23.l23_path, name))
        for name in L23_ELASTIC_FILES
    )


#: Skip a test that needs the L23 reference data.
needs_l23 = pytest.mark.skipif(
    not l23_available(), reason="L23 elastic Hydrolight data not available ($OS_COLOR)"
)


def l23_inelastic_available():
    """Whether the L23 inelastic (X=2/X=4) dataset is on disk.

    Returns
    -------
    bool
        True when ``ocpy`` imports *and* every file in
        :data:`L23_INELASTIC_FILES` is present in its L23 directory.
    """
    try:
        from ocpy.hydrolight import loisel23
    except Exception:  # noqa: BLE001 - any failure here means "no data", by design
        return False
    return all(
        os.path.isfile(os.path.join(loisel23.l23_path, name))
        for name in L23_INELASTIC_FILES
    )


#: Skip a test that needs the L23 inelastic (X=2/X=4) reference data.
needs_l23_inelastic = pytest.mark.skipif(
    not l23_inelastic_available(),
    reason="L23 inelastic Hydrolight data (X=2/X=4) not available ($OS_COLOR)",
)


@pytest.fixture(scope="session")
def l23_small_batch():
    """A 50-scene L23 batch, loaded through the real loader from the committed fixture.

    Needs no ``$OS_COLOR``: :data:`L23_SMALL_FIXTURE` stores the loader's *input*,
    so ``load_batch`` genuinely runs here rather than a snapshot of its output
    being checked for staleness.

    Returns
    -------
    robust.rt.data.l23.L23Batch
    """
    if not L23_SMALL_FIXTURE.is_file():
        pytest.skip(f"cached L23 fixture missing: {L23_SMALL_FIXTURE}")

    from robust.rt.data import l23

    return l23.load_batch(reader=l23.npz_reader(L23_SMALL_FIXTURE))


@pytest.fixture(scope="session")
def l23_batch():
    """The full L23 elastic batch (3 zeniths x 3320 scenes), loaded once.

    Session-scoped because reading the three netCDFs costs ~0.3 s each and every
    data test wants the same batch. Skips rather than errors when the dataset is
    absent, so it is safe even if a test forgets the :data:`needs_l23` marker.

    Yields
    ------
    robust.rt.data.l23.L23Batch
    """
    if not l23_available():
        pytest.skip("L23 elastic Hydrolight data not available ($OS_COLOR)")

    from robust.rt.data import l23

    return l23.load_batch()


@pytest.fixture(scope="session")
def l23_small_inelastic_batch():
    """A 50-scene inelastic batch, loaded through the real loader from fixtures.

    Needs no ``$OS_COLOR``: the sibling fixture supplies ``aph`` and the
    X=2/X=4 channels, the elastic fixture supplies ``bbnw``/``bnw``, and
    :func:`robust.rt.data.l23.load_inelastic_batch` genuinely runs.

    Returns
    -------
    robust.rt.data.l23.L23InelasticBatch
    """
    for fixture in (L23_INELASTIC_FIXTURE, L23_SMALL_FIXTURE):
        if not fixture.is_file():
            pytest.skip(f"cached L23 fixture missing: {fixture}")

    from robust.rt.data import l23

    return l23.load_inelastic_batch(
        reader=l23.inelastic_npz_reader(L23_INELASTIC_FIXTURE, L23_SMALL_FIXTURE)
    )


def tiny_args():
    """Minimal synthetic :func:`robust.rt.forward` inputs (2 wavelengths).

    A plain function, not a fixture, so tests can call it several times and
    mutate copies freely. Lives here because two modules were maintaining
    byte-identical private copies (PR #14 review) — when these inputs must
    change, there is now exactly one place.
    """
    import jax.numpy as jnp

    from robust.rt import conventions
    from robust.rt.types import Geometry, IOPs, PhaseParams

    wave = jnp.asarray([440.0, 550.0])
    iops = IOPs(
        a=jnp.asarray([0.15, 0.12]),
        bb_w=conventions.bb_w(wave),
        bb_p=jnp.asarray([0.003, 0.003]),
    )
    return (
        iops,
        PhaseParams(B_p=jnp.asarray(0.0126)),
        Geometry.nadir(jnp.asarray(30.0)),
        wave,
    )


@pytest.fixture
def jax_x64():
    """Enable JAX float64 for one test, restoring the prior setting after.

    Yields
    ------
    module
        ``jax``, with ``jax_enable_x64`` on.
    """
    import jax

    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax
    finally:
        jax.config.update("jax_enable_x64", previous)
