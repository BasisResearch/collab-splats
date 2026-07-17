# collab_splats/dashboard/__main__.py
"""CLI entry point for the splats dashboard."""

import argparse
import logging
import subprocess
import sys
import time
import urllib.request

# NB: collab_splats.dashboard.app is imported inside main() — it pulls panel and the
# dashboard stack, and --smoke must be able to spawn the server subprocess without
# paying (or double-paying) that import in the parent.


def _smoke(port: int, base_dir: str, timeout_s: float) -> int:
    """Launch a server subprocess and verify the page and its JS actually serve.

    Health gate for CI/manual checks: returns 0 when the root page and the bokeh
    bundle both come back, 1 otherwise. Exercises the real entry point (imports,
    Xvfb, session factory) rather than an in-process approximation.
    """
    cmd = [sys.executable, "-m", "collab_splats.dashboard", "--port", str(port), "--base-dir", base_dir]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    page = b""
    try:
        # Poll the root page until the server binds (imports take ~20-60s cold).
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                tail = proc.stdout.read()[-2000:] if proc.stdout else ""
                print(tail)
                print(f"SMOKE FAIL: server exited early (code {proc.returncode})")
                return 1
            try:
                page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read()
                break
            except Exception:
                time.sleep(3)
        if not page:
            print(f"SMOKE FAIL: no page within {timeout_s:.0f}s")
            return 1
        bind_s = time.monotonic() - t0
        # The page must reference the bokeh bundle and the bundle must be fetchable —
        # a 200 skeleton with broken static routes still renders as a blank browser tab.
        if b"bokeh" not in page:
            print("SMOKE FAIL: page served but contains no bokeh script reference")
            return 1
        js = urllib.request.urlopen(f"http://127.0.0.1:{port}/static/js/bokeh.min.js", timeout=60).read()
        if len(js) < 100_000:
            print(f"SMOKE FAIL: bokeh.min.js truncated ({len(js)} bytes)")
            return 1
        print(f"SMOKE PASS: page ({len(page)} B) + bokeh.min.js ({len(js)} B) served; bind took {bind_s:.0f}s")
        return 0
    finally:
        proc.terminate()


def main() -> None:
    """Parse args and serve the dashboard."""
    parser = argparse.ArgumentParser(prog="collab-dashboard", description="Launch the collab-splats dashboard.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-dir", default="/workspace/outputs")
    parser.add_argument(
        "--websocket-origin",
        action="append",
        default=None,
        help="Allowed websocket Origin host:port (repeatable). Required when browsing via an IP "
        "or forwarded hostname — without it the page loads but never renders (ws rejected). "
        "Use '*' to allow any origin.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Health check: start a server subprocess, verify the page serves, exit 0/1.",
    )
    parser.add_argument("--smoke-timeout", type=float, default=300.0, help="Seconds to wait for --smoke bind.")
    args = parser.parse_args()

    if args.smoke:
        sys.exit(_smoke(args.port, args.base_dir, args.smoke_timeout))

    # INFO logging so startup progress (Xvfb, warm imports, sessions) reaches the terminal.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    origin = args.websocket_origin
    if origin == ["*"]:
        origin = "*"
    # serve.run_app binds immediately (light imports only) and streams the heavy-stack
    # import progress to the browser's status strip until the dashboard is ready.
    from collab_splats.dashboard.serve import run_app

    run_app(host=args.host, port=args.port, base_dir=args.base_dir, websocket_origin=origin)


if __name__ == "__main__":
    main()
