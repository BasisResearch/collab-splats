#!/usr/bin/env bash
# hloc (cvg/Hierarchical-Localization) clone for the `hloc` extra
# - editable uv path source (pyproject [tool.uv.sources]); the clone must exist before any resolve
# - --recursive: hloc's superpoint/superglue modules sys.path-append third_party/ submodules
# - re-pinned every run, as setup/loger.sh does; installing is uv's job, never pip's
set -euo pipefail

HLOC_DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/hloc"
COMMIT="c13273bd0ecc2917a35910fd843712a1c6243193"

echo "==> hloc @ ${COMMIT:0:7}"
if [ ! -d "$HLOC_DIR/.git" ]; then
    git clone --recursive https://github.com/cvg/Hierarchical-Localization "$HLOC_DIR"
fi
git -C "$HLOC_DIR" fetch --all --quiet
git -C "$HLOC_DIR" checkout --quiet "$COMMIT"
git -C "$HLOC_DIR" submodule update --init --recursive --quiet

# Weight prefetch: netvlad + LightGlue download on first use otherwise, mid-run
# - best-effort: a deps-only Docker pass has no venv yet; run again after the sync
PYTHON=/opt/venv/reconstruction/bin/python
if [ "${1:-}" = "--prefetch" ]; then
    "$PYTHON" - <<'EOF' || echo "WARN: hloc weight prefetch failed — weights will download on first use."
from hloc import extract_features, extractors, match_features, matchers
from hloc.utils.base_model import dynamic_load

for conf, pkg in (
    (extract_features.confs["netvlad"], extractors),
    (extract_features.confs["superpoint_max"], extractors),
    (match_features.confs["superpoint+lightglue"], matchers),
):
    dynamic_load(pkg, conf["model"]["name"])(conf["model"])
print("hloc weights cached")
EOF
fi
echo "==> hloc done"
