
# Standard imports
import glob, os, re
from setuptools import setup, find_packages


def get_version():
    """Read ``__version__`` out of ``robust/__init__.py`` without importing it.

    One source of truth: the literal lives in ``robust/__init__.py`` and is read
    here (and by ``docs/conf.py``, as ``robust.__version__``), so the packaging
    version and the documented version can never drift.

    It is parsed rather than imported deliberately -- ``import robust`` at build
    time would pull the JAX stack into the packaging environment, and pip builds
    in an isolated env where jax is not installed.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'robust', '__init__.py')) as init_file:
        match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
                          init_file.read(), re.M)
    if match is None:
        raise RuntimeError('Could not find __version__ in robust/__init__.py')
    return match.group(1)


# Begin setup
setup_keywords = dict()
setup_keywords['name'] = 'retrieve-or-bust'
setup_keywords['description'] = 'Our last best effort at IOP (inherent optical properties) retrievals, with AI'
setup_keywords['author'] = 'J. Xavier Prochaska'
setup_keywords['author_email'] = 'jxp@ucsc.edu'
setup_keywords['license'] = 'BSD'
setup_keywords['url'] = 'https://github.com/ocean-colour/retrieve-or-bust'
setup_keywords['version'] = get_version()
# Use README.md as long_description.
setup_keywords['long_description'] = ''
if os.path.exists('README.md'):
    with open('README.md') as readme:
        setup_keywords['long_description'] = readme.read()
# NB: no `provides` key. It is legacy distutils metadata (superseded by
# Provides-Dist) and it must be a *module* name, so the hyphen in
# 'retrieve-or-bust' made it illegal: `pip install .` died with
# "ValueError: illegal provides specification". bing/ocpy carry the same line
# harmlessly because their names have no hyphen.
setup_keywords['python_requires'] = '>=3.12'
setup_keywords['install_requires'] = [
    'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
    'xarray', 'h5netcdf', 'cftime', 'scikit-learn',
    'tqdm', 'IPython', 'pytest',
    # Retrieval / inference engine and plotting
    'emcee', 'corner', 'bokeh']
# The JAX stack for the differentiable RT forward model (robust/rt) is declared
# in requirements.txt only, deliberately: `pip install -e .` for non-RT work
# should not have to pull jaxlib. Install it with `pip install -r requirements.txt`.
# The sibling packages BING and ocpy are not on PyPI; install them from
# source / GitHub via requirements.txt (git+https://github.com/ocean-colour/...).
setup_keywords['zip_safe'] = False
setup_keywords['packages'] = find_packages()
# Ship the trained emulator weights (robust/rt/files/*.npz, ~7 KB) and the small
# L23 test fixture. Without this, an installed copy imports fine and then fails at
# the first forward(mode='hybrid') with a missing-weights error, because
# find_packages() collects modules only -- data files need saying out loud.
setup_keywords['package_data'] = {
    'robust': ['rt/files/*.npz', 'rt/data/*.npz', 'tests/files/*.npz'],
}

if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if not os.path.basename(fname).endswith('.rst')]

setup(**setup_keywords)
