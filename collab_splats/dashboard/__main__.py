"""
collab_splats dashboard launcher.

Usage:
    python -m collab_splats.dashboard semantics
    collab-dashboard semantics
    collab-dashboard semantics --base-dir /workspace/fieldwork-data
"""

import argparse


DASHBOARDS = {
    "semantics": "collab_splats.dashboard.semantics:run_app",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="collab-dashboard",
        description="Launch a collab-splats interactive dashboard.",
    )
    parser.add_argument(
        "mode",
        choices=list(DASHBOARDS.keys()),
        help="Dashboard to launch.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to listen on (default: 7860)"
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root directory for video discovery (default: current directory)",
    )
    args = parser.parse_args()

    module_path, func_name = DASHBOARDS[args.mode].rsplit(":", 1)
    import importlib

    mod = importlib.import_module(module_path)
    run_fn = getattr(mod, func_name)
    run_fn(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
