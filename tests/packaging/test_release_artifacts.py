from __future__ import annotations

from email.parser import Parser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement
import yaml


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
    'statsmodels',
    'ucimlrepo',
    'wandb',
}

DEV_ONLY_DEPS = {
    'pytest',
}

REPO_ONLY_DEPS = {
    'scamd',
}

# Colab's preinstalled versions (Python 3.13); a higher floor makes pip replace them, and
# arviz 1.x breaks Colab's pymc, which needs arviz<1
COLAB_FLOORS = {
    'numpy': '>=2.1.3',
    'pandas': '>=2.2.3',
    'scikit-learn': '>=1.6.1',
    'matplotlib': '>=3.10.0',
    'arviz-stats': '>=1.3.0',
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


def _notebook_cells() -> list[tuple[str, dict]]:
    notebooks = sorted((ROOT / 'demos').glob('*.ipynb'))
    assert notebooks, 'no notebooks under demos/'
    return [
        (notebook.name, cell)
        for notebook in notebooks
        for cell in json.loads(notebook.read_text())['cells']
    ]


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
    assert f'{SDIST_ROOT}/demos/01_quickstart.ipynb' in sdist_names
    assert f'{SDIST_ROOT}/demos/02_families.ipynb' in sdist_names
    assert f'{SDIST_ROOT}/demos/intro.ipynb' not in sdist_names
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
    for dep in REPO_ONLY_DEPS:
        assert not any(req.lower().startswith(dep) for req in requirements), dep

    assert not metadata.get_all('Provides-Extra') or 'simulation' not in metadata.get_all(
        'Provides-Extra'
    )
    assert any(req.startswith('pymc') for req in research_requirements)


def test_release_metadata_installs_next_to_colab() -> None:
    _require_artifacts()

    metadata = _metadata()
    requirements = [Requirement(req) for req in metadata.get_all('Requires-Dist')]
    core = {req.name: str(req.specifier) for req in requirements if req.marker is None}
    research = {req.name for req in requirements if str(req.marker) == 'extra == "research"'}

    assert metadata['Requires-Python'] == '>=3.12'
    for name, floor in COLAB_FLOORS.items():
        assert core.get(name) == floor, name
    assert 'arviz' not in core
    assert 'arviz' in research


def test_release_metadata_license() -> None:
    _require_artifacts()

    metadata = _metadata()
    assert metadata['License-Expression'] == 'Apache-2.0'
    assert metadata.get_all('License-File') == ['LICENSE']


def test_citation_version_matches_pyproject() -> None:
    citation = yaml.safe_load((ROOT / 'CITATION.cff').read_text())
    assert citation['version'] == VERSION


def test_notebook_pins_match_pyproject() -> None:
    for name, cell in _notebook_cells():
        for pin in re.findall(r'metabeta==([^\s\'"]+)', ''.join(cell['source'])):
            assert pin == VERSION, f'{name} pins metabeta=={pin}'


def test_notebook_images_render_on_colab() -> None:
    # Colab renders neither cell attachments nor images at paths relative to the notebook
    for name, cell in _notebook_cells():
        assert 'attachments' not in cell, f'{name} has a cell attachment'
        source = ''.join(cell['source'])
        images = re.findall(r'!\[[^\]]*\]\(([^)\s]+)', source)
        images += re.findall(r'<img[^>]*\ssrc=["\']([^"\']+)', source)
        for image in images:
            assert image.startswith('https://'), f'{name} shows the image {image}'


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
