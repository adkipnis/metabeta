from __future__ import annotations

from email.parser import Parser
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / 'dist'
VERSION = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
SDIST_ROOT = f'metabeta-{VERSION}'
WHEEL = DIST / f'metabeta-{VERSION}-py3-none-any.whl'
SDIST = DIST / f'metabeta-{VERSION}.tar.gz'

WHEEL_EXCLUDED_PREFIXES = (
    'metabeta/analytical/gaussian_local.py',
    'metabeta/posthoc/conformal.py',
    'metabeta/posthoc/coordinate.py',
    'metabeta/posthoc/gaussian_local.py',
    'metabeta/posthoc/laplace.py',
    'metabeta/simulation/',
    'metabeta/training/',
    'metabeta/outputs/',
    'tests/',
    'experiments/',
    'archive/',
    'benchmarks/',
    'scripts/',
    'demos/',
)

SDIST_EXCLUDED_PREFIXES = (
    f'{SDIST_ROOT}/metabeta/analytical/gaussian_local.py',
    f'{SDIST_ROOT}/metabeta/posthoc/conformal.py',
    f'{SDIST_ROOT}/metabeta/posthoc/coordinate.py',
    f'{SDIST_ROOT}/metabeta/posthoc/gaussian_local.py',
    f'{SDIST_ROOT}/metabeta/posthoc/laplace.py',
    f'{SDIST_ROOT}/metabeta/simulation/',
    f'{SDIST_ROOT}/metabeta/training/',
    f'{SDIST_ROOT}/metabeta/outputs/',
    f'{SDIST_ROOT}/tests/',
    f'{SDIST_ROOT}/experiments/',
    f'{SDIST_ROOT}/archive/',
    f'{SDIST_ROOT}/benchmarks/',
    f'{SDIST_ROOT}/scripts/',
)

RESEARCH_ONLY_DEPS = {
    'bambi',
    'datasets',
    'pmlb',
    'pymc',
    'rpy2',
    'schedulefree',
    'scamd',
    'statsmodels',
    'ucimlrepo',
    'wandb',
}

DEV_ONLY_DEPS = {
    'pytest',
}


def _require_artifacts() -> None:
    missing = [str(path.relative_to(ROOT)) for path in (WHEEL, SDIST) if not path.exists()]
    assert not missing, f'missing release artifacts; run `uv build` first: {missing}'


def _metadata() -> Parser:
    with zipfile.ZipFile(WHEEL) as wheel:
        metadata_name = next(
            name for name in wheel.namelist() if name.endswith('.dist-info/METADATA')
        )
        return Parser().parsestr(wheel.read(metadata_name).decode())


def test_release_artifact_contents() -> None:
    _require_artifacts()

    with zipfile.ZipFile(WHEEL) as wheel:
        wheel_names = wheel.namelist()
    for prefix in WHEEL_EXCLUDED_PREFIXES:
        assert not any(name.startswith(prefix) for name in wheel_names), prefix
    assert 'metabeta/posthoc/warmnuts.py' in wheel_names
    assert 'metabeta/utils/pymc.py' in wheel_names

    with tarfile.open(SDIST) as sdist:
        sdist_names = sdist.getnames()
    for prefix in SDIST_EXCLUDED_PREFIXES:
        assert not any(name.startswith(prefix) for name in sdist_names), prefix
    assert f'{SDIST_ROOT}/demos/intro.ipynb' in sdist_names
    assert f'{SDIST_ROOT}/demos/priors.ipynb' in sdist_names


def test_release_metadata_dependencies() -> None:
    _require_artifacts()

    metadata = _metadata()
    requirements = metadata.get_all('Requires-Dist') or []
    base_requirements = [req for req in requirements if 'extra ==' not in req]
    research_requirements = [req for req in requirements if 'extra == "research"' in req]

    for dep in RESEARCH_ONLY_DEPS:
        assert not any(req.lower().startswith(dep) for req in base_requirements), dep
    for dep in DEV_ONLY_DEPS:
        assert not any(req.lower().startswith(dep) for req in base_requirements), dep

    assert not metadata.get_all('Provides-Extra') or 'simulation' not in metadata.get_all(
        'Provides-Extra'
    )
    assert any(req.startswith('scamd') for req in research_requirements)
    assert any(req.startswith('pymc') for req in research_requirements)


def test_release_metadata_public_surface() -> None:
    _require_artifacts()

    with zipfile.ZipFile(WHEEL) as wheel:
        entrypoint_names = [
            name for name in wheel.namelist() if name.endswith('.dist-info/entry_points.txt')
        ]
        entrypoints = wheel.read(entrypoint_names[0]).decode() if entrypoint_names else ''

    assert 'metabeta-evaluate' not in entrypoints


def test_installed_wheel_import_smoke(tmp_path: Path) -> None:
    _require_artifacts()

    target = tmp_path / 'site'
    with zipfile.ZipFile(WHEEL) as wheel:
        wheel.extractall(target)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(target)
    subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'from metabeta import Api; '
                'from metabeta.models.api import Api as Api2; '
                'import metabeta.evaluation.predictive; '
                'import metabeta.posthoc.warmnuts; '
                'import metabeta.utils.pymc; '
                'assert Api is Api2'
            ),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )
