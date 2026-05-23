#!/bin/bash
set -e

NERFSTUDIO_DIR="/workspace/nerfstudio"
PYTHON=/opt/conda/envs/nerfstudio/bin/python
PIP=/opt/conda/envs/nerfstudio/bin/pip

# Prevent pip build isolation from using system Python 3.13 instead of env Python
# (affects gsplat-rade and any other CUDA extension that imports torch at build time)
export PIP_NO_BUILD_ISOLATION=1

echo "=== Patching nerfstudio pyproject.toml for cu121 ==="

# Apply 5 compatibility patches inline via Python.
$PYTHON - <<'PYEOF'
import re, pathlib

p = pathlib.Path("/workspace/nerfstudio/pyproject.toml")
text = p.read_text()

# 1. gsplat → floor-only plain version specifier (idempotent from any prior state)
# Do NOT use a git URL: pip always rebuilds git-URL deps even if already installed,
# which fails because gsplat's setup.py imports torch (needs --no-build-isolation).
# gsplat-rade (version 1.4.0) is pre-installed before this step with --no-build-isolation.
text = re.sub(r'"gsplat(?:==1\.4\.0| @ git\+[^"]+)"', '"gsplat>=1.4.0"', text)

# 2. timm exact pin → floor-only (uniception needs >=0.9)
text = text.replace('"timm==0.6.7"', '"timm>=0.6.7"')

# 3. viser exact pin → floor-only (feedforward needs >=0.2.23)
text = text.replace('"viser==1.0.0"', '"viser>=0.2.0"')

# 4. Remove opencv-python-headless (collab-splats uses opencv-python which is a superset)
text = re.sub(r'\s*"opencv-python-headless==4\.10\.0\.84",?\n', '\n', text)

# 5. nerfacc exact pin → floor-only (CUDA extension; exact pin may fail with torch 2.4)
text = text.replace('"nerfacc==0.5.2"', '"nerfacc>=0.5.2"')

p.write_text(text)
print("Patches applied.")
PYEOF

echo "=== Pre-installing open3d (nerfstudio dep; no wheel via dep resolution, direct install works) ==="
$PIP install --no-cache-dir open3d

echo "=== Pre-installing gsplat-rade with --no-build-isolation (setup.py imports torch at build time) ==="
$PIP install --no-build-isolation git+https://github.com/brian-xu/gsplat-rade.git

echo "=== Installing nerfstudio from local repo (gsplat-rade already installed, skips rebuild) ==="
$PIP install -e "$NERFSTUDIO_DIR" --no-cache-dir

echo "=== Verifying gsplat-rade is still installed (nerfstudio must not have overwritten it) ==="
$PYTHON -c "import gsplat; print('gsplat present:', gsplat.__file__)"

echo "=== Smoke test ==="
$PYTHON -c "
import nerfstudio
import gsplat
print('nerfstudio:', nerfstudio.__file__)
print('gsplat:', gsplat.__file__)
# pip installs both forks as 'gsplat' so path can't distinguish; just verify import works
gsplat_path = gsplat.__file__
assert gsplat_path, 'gsplat.__file__ is empty'
print('OK: nerfstudio + gsplat-rade installed')
"

# TODO: hloc install deferred.
# pycolmap 4.0.4 vs hloc compatibility unresolved.
# hloc v1.4 requires pycolmap ~0.4; latest hloc master may support 4.x but needs validation.
# Track as separate task before enabling localization pipeline.
