from __future__ import annotations

from email.parser import Parser
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / 'dist'
WHEEL = DIST / 'metabeta-0.4-py3-none-any.whl'
SDIST = DIST / 'metabeta-0.4.tar.gz'

WHEEL_EXCLUDED_PREFIXES = (
    'metabeta/analytical/gaussian_local.py',
    'metabeta/posthoc/gaussian_local.py',
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
    'metabeta-0.4/metabeta/analytical/gaussian_local.py',
    'metabeta-0.4/metabeta/posthoc/gaussian_local.py',
    'metabeta-0.4/metabeta/simulation/',
    'metabeta-0.4/metabeta/training/',
    'metabeta-0.4/metabeta/outputs/',
    'metabeta-0.4/tests/',
    'metabeta-0.4/experiments/',
    'metabeta-0.4/archive/',
    'metabeta-0.4/benchmarks/',
    'metabeta-0.4/scripts/',
)

RESEARCH_ONLY_DEPS = {
    'bambi',
    'datasets',
    'pmlb',
    'rpy2',
    'schedulefree',
    'scamd',
    'statsmodels',
    'ucimlrepo',
    'wandb',
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

    with tarfile.open(SDIST) as sdist:
        sdist_names = sdist.getnames()
    for prefix in SDIST_EXCLUDED_PREFIXES:
        assert not any(name.startswith(prefix) for name in sdist_names), prefix
    assert 'metabeta-0.4/demos/intro.ipynb' in sdist_names
    assert 'metabeta-0.4/demos/priors.ipynb' in sdist_names


def test_release_metadata_dependencies() -> None:
    _require_artifacts()

    metadata = _metadata()
    requirements = metadata.get_all('Requires-Dist') or []
    base_requirements = [req for req in requirements if 'extra ==' not in req]
    research_requirements = [req for req in requirements if 'extra == "research"' in req]

    for dep in RESEARCH_ONLY_DEPS:
        assert not any(req.lower().startswith(dep) for req in base_requirements), dep

    assert not metadata.get_all('Provides-Extra') or 'simulation' not in metadata.get_all(
        'Provides-Extra'
    )
    assert any(req.startswith('scamd') for req in research_requirements)


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
                'assert Api is Api2'
            ),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
    )
