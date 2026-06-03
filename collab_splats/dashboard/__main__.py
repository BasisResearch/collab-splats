# collab_splats/dashboard/__main__.py
"""CLI entry point for the splats dashboard."""

import argparse

from collab_splats.dashboard.app import run_app


def main() -> None:
    """Parse args and serve the dashboard."""
    parser = argparse.ArgumentParser(
        prog="collab-dashboard", description="Launch the collab-splats dashboard."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-dir", default="/workspace/outputs")
    args = parser.parse_args()
    run_app(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
