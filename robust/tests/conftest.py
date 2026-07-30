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

import pytest

#: The L23 *elastic* files (X=1, no inelastic) at the three solar-zenith angles
#: Y = 0/30/60 deg. The prototype needs all three (coding plan M1).
L23_ELASTIC_FILES = ('Hydrolight100.nc', 'Hydrolight130.nc', 'Hydrolight160.nc')


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
    return all(os.path.isfile(os.path.join(loisel23.l23_path, name))
               for name in L23_ELASTIC_FILES)


#: Skip a test that needs the L23 reference data.
needs_l23 = pytest.mark.skipif(
    not l23_available(),
    reason='L23 elastic Hydrolight data not available ($OS_COLOR)')


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
    jax.config.update('jax_enable_x64', True)
    try:
        yield jax
    finally:
        jax.config.update('jax_enable_x64', previous)
