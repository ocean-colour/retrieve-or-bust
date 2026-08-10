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

#: Fixtures directory (BING layout).
FILES = pathlib.Path(__file__).parent / "files"

#: A committed 50-scene x 3-zenith snapshot of the **raw** L23 fields, ~213 kB.
#: Because it holds the loader's *input* rather than its output, the real
#: :func:`robust.rt.data.l23.load_batch` runs against real numbers wherever this
#: file is present -- CI included, with no ``$OS_COLOR`` mount. Regenerate with
#: ``l23.write_fixture(path, n_scene=50)``.
L23_SMALL_FIXTURE = FILES / "l23_small.npz"


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


#: A committed 3-realisation snapshot of the **raw** PB24 fields, ~460 kB, keeping
#: each realisation's *whole* 1300-geometry grid so the angle window and its shell
#: are both exercisable. Realisation 993 is in there deliberately: it carries the
#: only defect in the OLCI set -- two exactly-zero ``rrs`` values at 753 nm,
#: ``theta_s = 80``, ``theta = 87.5`` -- so the filter that removes them has
#: something to bite on. Regenerate with
#: ``pb24.write_fixture(path, realisations=(1, 993, 2500))``.
PB24_SMALL_FIXTURE = FILES / "pb24_small.npz"

#: The realisations in :data:`PB24_SMALL_FIXTURE`, in order.
PB24_FIXTURE_REALISATIONS = (1, 993, 2500)


def pb24_available():
    """Whether the PB24 release is on disk under ``$OS_COLOR``.

    Returns
    -------
    bool
        True when ``$OS_COLOR/SD/v5`` exists and holds the first OLCI file.
    """
    root = os.environ.get("OS_COLOR")
    if not root:
        return False
    return os.path.isfile(os.path.join(root, "SD", "v5", "SD_OLCI_no_R_0001.nc"))


#: Skip a test that needs the PB24 reference data.
needs_pb24 = pytest.mark.skipif(
    not pb24_available(), reason="PB24 release not available ($OS_COLOR/SD/v5)"
)


@pytest.fixture(scope="session")
def pb24_reader():
    """A reader over the committed PB24 fixture -- no ``$OS_COLOR`` required.

    Returns
    -------
    callable
        Suitable for ``pb24.load_batch(reader=...)``.
    """
    if not PB24_SMALL_FIXTURE.is_file():
        pytest.skip(f"cached PB24 fixture missing: {PB24_SMALL_FIXTURE}")

    from robust.rt.data import pb24

    return pb24.npz_reader(PB24_SMALL_FIXTURE)


@pytest.fixture(scope="session")
def pb24_small_batch(pb24_reader):
    """A 3-realisation PB24 batch on the Q14 window, through the real loader.

    Returns
    -------
    robust.rt.data.pb24.PB24Batch
    """
    from robust.rt.data import pb24

    return pb24.load_batch(realisations=PB24_FIXTURE_REALISATIONS, reader=pb24_reader)


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
