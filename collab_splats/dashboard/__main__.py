# dashboard/__main__.py
"""collab_splats dashboard launcher.

Usage:
    python -m collab_splats.dashboard app
    collab-dashboard app
    collab-dashboard app --base-dir /workspace/outputs --port 7860

    # Legacy alias (deprecated — redirects to app):
    collab-dashboard semantics
"""

from __future__ import annotations

import argparse
import importlib
import warnings

DASHBOARDS = {
    "app": "collab_splats.dashboard.app:run_app",
    "semantics": "collab_splats.dashboard.app:run_app",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="collab-dashboard",
        description="Launch a collab-splats interactive dashboard.",
    )
    parser.add_argument("mode", choices=list(DASHBOARDS.keys()))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-dir", default="/workspace/outputs")
    args = parser.parse_args()

    if args.mode == "semantics":
        warnings.warn(
            "'collab-dashboard semantics' is deprecated — use 'collab-dashboard app'",
            DeprecationWarning,
            stacklevel=2,
        )

    module_path, func_name = DASHBOARDS[args.mode].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    run_fn = getattr(mod, func_name)
    run_fn(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
