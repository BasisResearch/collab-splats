"""
Run the pipeline on local inputs or on curated-bucket scenes.

- local: each video or frame directory writes to <output-root>/<name>
- remote: each scene id is fetched, run, pushed, verified, then deleted locally
- a leaf-only --stages set re-runs remote scenes from their processed outputs
- exit codes: 0 ok, 1 a scene failed, 2 nothing to do, 3 rclone unreachable (batch aborted)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml
from mergedeep import merge

from collab_splats.reconstructor import LEAF_STAGES, STAGES, Reconstructor
from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE, SceneSource

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

EXIT_OK = 0
EXIT_SCENE_FAILED = 1
EXIT_NOTHING_TO_DO = 2
EXIT_REMOTE_UNAVAILABLE = 3


########################################
# Shared helpers
########################################


def _parse_overrides(pairs: list[str]) -> dict:
    """
    Nested dict from `key.sub=value` strings; each value is parsed as YAML.
    """
    out: dict = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")

        # Walk the dotted key, creating each parent dict
        key, value = pair.split("=", 1)
        *parents, leaf = key.split(".")
        node = out

        for part in parents:
            node = node.setdefault(part, {})

            # A parent already set to a scalar cannot take a sub-key
            if not isinstance(node, dict):
                raise ValueError(
                    f"--set {key}: '{part}' is already set to a value, not a section"
                )

        node[leaf] = yaml.safe_load(value)

    return out


def _run_scene(config: dict, args: argparse.Namespace) -> Reconstructor:
    """
    Build one Reconstructor, record its config beside the outputs, run it.
    """
    # Build the reconstructor and record its config
    recon = Reconstructor(config, base_config=args.base_config)
    recon.write_run_config()

    recon.run(args.stages, overwrite=args.overwrite)
    return recon


def _summarize(results: list[tuple[str, str, str]]) -> None:
    """
    Log one line per scene.
    """
    logger.info("==== Summary ====")

    for name, status, info in results:
        logger.info("%s: %s (%s)", status, name, info)


########################################
# Local
########################################


def _scene_name(path: Path) -> str:
    """
    Output dir name for a local input: a video's stem, a frame directory's folder name.
    """
    return path.name if path.is_dir() else path.stem


def _run_local(args: argparse.Namespace) -> int:
    """
    Run every input; one failure does not stop the rest.
    """
    results = []
    last = None

    for path in args.inputs:
        output = args.output_root / _scene_name(path)
        paths = {"input_path": str(path), "output_path": str(output)}
        config = merge({}, args.overrides, paths)

        # Isolate one input's failure from the batch
        try:
            last = _run_scene(config, args)
            results.append((path.name, "OK", paths["output_path"]))
        except Exception as exc:
            logger.exception("input failed: %s", path)
            reason = str(exc)
            results.append((path.name, "FAIL", reason))

    _summarize(results)

    # Keep the last viser viewer up for inspection
    if args.keep_viewer and last is not None and last.viewer is not None:
        logger.info("--keep-viewer: viser server staying up (Ctrl-C to exit)")
        last.viewer.serve_forever()
    elif args.keep_viewer:
        logger.info("--keep-viewer: no viewer was created; nothing to keep alive")

    return EXIT_SCENE_FAILED if any(s == "FAIL" for _, s, _ in results) else EXIT_OK


########################################
# Remote
########################################


def _is_rerun(stages: list[str] | None) -> bool:
    """
    True when every stage is a leaf, so processed outputs are enough to run them.
    """
    if not stages:
        return False

    return set(stages) <= LEAF_STAGES


def _prepare_scene(
    source: SceneSource, scene: str, scene_dir: Path, args: argparse.Namespace
) -> dict:
    """
    Fetch a scene's inputs and return its config.

    - full runs fetch the curated video
    - leaf re-runs pull the processed scene and reuse one backend's run_config.yaml minus the re-run sections
    - the backend is --set pointcloud.backend, else the only backend with a recorded config
    """
    paths = {"output_path": str(scene_dir)}
    config = merge({}, args.overrides, paths)

    # Full run: start from the curated video
    if not _is_rerun(args.stages):
        video = source.fetch_video(scene, scene_dir, on_line=logger.info)
        config["input_path"] = str(video)
        return config

    # Leaf re-run: pull the processed scene, every member included
    if not source.has_processed(scene):
        raise FileNotFoundError(
            f"{scene} has no processed outputs; run the full pipeline first"
        )

    source.pull_processed(scene, scene_dir, on_line=logger.info)

    # Backend: the typed one, else the only one the pulled scene recorded
    backend = args.overrides.get("pointcloud", {}).get("backend")

    if backend is None:
        dirs = [d.name for d in scene_dir.iterdir()]
        recorded = sorted(
            name
            for name in dirs
            if Reconstructor.run_config_path(scene_dir, name).exists()
        )

        if len(recorded) != 1:
            raise ValueError(
                f"{scene}: recorded backends {recorded}; pick one with --set pointcloud.backend=<name>"
            )

        backend = recorded[0]

    # Read that backend's recorded config
    run_cfg = Reconstructor.run_config_path(scene_dir, backend)

    if not run_cfg.exists():
        raise FileNotFoundError(
            f"{scene}: pulled scene has no {run_cfg.relative_to(scene_dir)}"
        )

    pulled = yaml.safe_load(run_cfg.read_text()) or {}

    # Re-run sections come fresh from base.yaml + overrides; stage name is the section except localize
    for stage in args.stages:
        pulled.pop("localization" if stage == "localize" else stage, None)

    # Log the re-run and layer the fresh config over the pulled one
    stages = ",".join(args.stages)
    logger.info("%s: re-run %s from processed (backend=%s)", scene, stages, backend)
    return merge(pulled, config)


def _remove_scene_dir(scene_dir: Path) -> bool:
    """
    Delete a scene dir; False with a warning when anything survived.
    """
    # ignore_errors hides a partial delete, so check the dir is gone
    shutil.rmtree(scene_dir, ignore_errors=True)

    if scene_dir.exists():
        logger.warning("failed to remove local scene dir: %s", scene_dir)
        return False

    logger.info("removed local scene dir %s", scene_dir)
    return True


def _list_scenes(source: SceneSource, args: argparse.Namespace) -> list[str]:
    """
    Every scene id in the bucket the run reads: processed for a leaf re-run, curated otherwise.
    """
    if _is_rerun(args.stages):
        return source.list_processed_scenes()

    return source.list_scenes()


def _run_remote(
    source: SceneSource, scenes: list[str], args: argparse.Namespace
) -> int:
    """
    Fetch, run, push, verify and delete each scene; abort when rclone dies.

    - an empty scenes list runs every scene in the bucket
    """
    # Work list: named ids, else the whole bucket; a listing failure means rclone is down
    if not scenes:
        try:
            scenes = _list_scenes(source, args)
        except Exception as exc:
            logger.error("cannot list scenes; rclone is not working: %s", exc)
            return EXIT_REMOTE_UNAVAILABLE

    if not scenes:
        logger.error("no scenes to process")
        return EXIT_NOTHING_TO_DO

    # Per-scene results and the batch-abort flag
    results = []
    aborted = False

    for i, scene in enumerate(scenes):
        # Scene header, local dir and no failure yet
        logger.info("=== Scene: %s ===", scene)
        scene_dir = args.output_root / scene
        failure = None

        # Isolate one scene's failure; delete only after a verified push
        try:
            config = _prepare_scene(source, scene, scene_dir, args)
            _run_scene(config, args)
            source.push_outputs(scene_dir, scene, on_line=logger.info)

            if not source.verify_push(scene_dir, scene):
                failure = "push verification failed; local data kept"
            elif args.keep_local:
                logger.info("--keep-local: leaving %s in place", scene_dir)
            elif not _remove_scene_dir(scene_dir):
                failure = f"push verified but local dir remains: {scene_dir}"
        except Exception as exc:
            logger.exception("scene failed: %s", scene)
            logger.warning("local scene dir retained for %s: %s", scene, scene_dir)
            failure = str(exc)

        # Record a success and move on; a failure falls through to the abort check
        if failure is None:
            info = str(scene_dir)
            results.append((scene, "OK", info))
            continue

        results.append((scene, "FAIL", failure))

        # A dead remote aborts the batch; the rest are reported SKIPPED, never FAIL
        if not source.check_available():
            aborted = True
            logger.error("rclone is not working; aborting the batch after %s", scene)
            results.extend(
                (s, "SKIPPED", "not attempted; rclone unreachable")
                for s in scenes[i + 1 :]
            )
            break

    # Summarize the batch and what the push left out
    _summarize(results)
    excluded = ", ".join(PUSH_EXCLUDES)
    logger.info("push excluded: %s", excluded)

    # An aborted batch outranks scene failures
    if aborted:
        return EXIT_REMOTE_UNAVAILABLE

    return EXIT_SCENE_FAILED if any(s == "FAIL" for _, s, _ in results) else EXIT_OK


########################################
# Entry point
########################################


def _parser() -> argparse.ArgumentParser:
    """
    `reconstruct local ...` and `reconstruct remote ...` with shared run options.
    """
    # Options every command takes
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="each scene writes to <output-root>/<name>",
    )
    shared.add_argument(
        "--config", type=Path, help="override YAML merged over base.yaml"
    )
    shared.add_argument(
        "--base-config",
        type=Path,
        help="base.yaml to merge over; default configs/base.yaml",
    )
    shared.add_argument(
        "--stages", help="comma-separated stages; default: every enabled stage"
    )
    shared.add_argument(
        "--overwrite", action="store_true", help="rebuild stages whose output exists"
    )
    shared.add_argument(
        "--set",
        action="append",
        default=[],
        dest="sets",
        metavar="KEY=VALUE",
        help="dotted config override",
    )

    # Top-level parser with one subcommand per mode
    parser = argparse.ArgumentParser(
        prog="reconstruct",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # local: video files or frame directories
    local = sub.add_parser(
        "local", parents=[shared], help="run on video files or frame directories"
    )
    local.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="VIDEO|DIR",
        help="video files or frame directories",
    )
    local.add_argument(
        "--keep-viewer", action="store_true", help="keep the last viser viewer alive"
    )

    # remote: curated scene ids
    remote = sub.add_parser("remote", parents=[shared], help="run on curated scene ids")
    remote.add_argument(
        "scenes", nargs="*", metavar="SCENE", help="curated scene ids; omit with --all"
    )
    remote.add_argument("--all", action="store_true", help="every scene in the bucket")
    remote.add_argument(
        "--keep-local", action="store_true", help="skip the post-push delete"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and run.

    Args:
        argv: arguments after the program name; None reads sys.argv.

    Returns:
        Process exit code.
    """
    # Configure logging and parse the command line
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    parser = _parser()
    args = parser.parse_args(argv)

    # Split --stages into a list and find names that are not stages
    args.stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    unknown = sorted(set(args.stages or []) - set(STAGES))

    # An unknown stage is a usage error, caught before any scene is fetched
    if unknown:
        listed = ", ".join(unknown)
        valid = ", ".join(STAGES)
        parser.error(f"unknown stage(s): {listed}; valid: {valid}")

    # Overrides shared by both commands; --set wins over --config
    overrides = {}

    # A missing --config is a usage error, not a traceback
    if args.config and not args.config.is_file():
        parser.error(f"--config does not exist: {args.config}")

    # Load the --config overrides
    if args.config:
        text = args.config.read_text()
        overrides = yaml.safe_load(text) or {}

    # A malformed --set is a usage error, not a traceback
    try:
        sets = _parse_overrides(args.sets)
    except ValueError as exc:
        parser.error(str(exc))

    # Merge --set over --config
    args.overrides = merge({}, overrides, sets)

    # Local: refuse missing inputs before any run starts
    if args.command == "local":
        missing = [str(p) for p in args.inputs if not p.exists()]

        if missing:
            listed = ", ".join(missing)
            parser.error(f"inputs do not exist: {listed}")

        # Two inputs sharing a name would write one output dir
        names = [_scene_name(p) for p in args.inputs]
        duplicates = {n for n in names if names.count(n) > 1}

        if duplicates:
            ordered = sorted(duplicates)
            listed = ", ".join(ordered)
            parser.error(f"inputs share an output name: {listed}")

        return _run_local(args)

    # Remote: exactly one of scene ids or --all
    if not args.scenes and not args.all:
        parser.error("give one or more SCENE ids, or --all")

    if args.scenes and args.all:
        parser.error("pass scenes or --all, not both")

    # Ids are joined onto --output-root, so each must be one safe path segment
    bad = [s for s in args.scenes if not SCENE_ID_RE.match(s)]

    if bad:
        listed = ", ".join(bad)
        parser.error(
            f"not safe scene ids (one path segment, no leading '.' or '-'): {listed}"
        )

    # Run the remote batch
    source = SceneSource()
    return _run_remote(source, args.scenes, args)


if __name__ == "__main__":
    code = main()
    sys.exit(code)
