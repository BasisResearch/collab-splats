import argparse

import uvicorn

from collab_splats.webapp.app import create_app


def main() -> None:
    """Entry point for running the webapp via python -m collab_splats.webapp."""
    parser = argparse.ArgumentParser(description="collab-splats webapp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
