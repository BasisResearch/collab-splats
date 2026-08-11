#!/usr/bin/env python3
"""Reconstruct scenes straight out of the environments-curated GCS bucket.

For each scene: pull its video, run the same pipeline as run_pipeline.py, push the
outputs to environments-processed/<scene>/, verify the push with `rclone check
--one-way`, then delete the local copy. The curated video stays in the bucket, so a
deleted scene is always re-fetchable.

Scene ids are the curated directory names: YYYY_MM_DD-PARENTFOLDER-VIDEONAME.

Usage:
    # Named scenes
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \\
        2026_07_20-birds-C0043 2026_07_21-rats-C0100

    # Everything in the bucket
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all

    # Keep the local copy for inspection (skips the delete, not the push)
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs --all --keep-local

    # Re-run only the leaf stages (mesh, semantics, localize) against already-processed
    # scenes: the scene is pulled back out of environments-processed instead of being
    # rebuilt from video. --all then lists environments-processed, not curated.
    python docs/examples/run_pipeline_remote.py --output-root /workspace/outputs \\
        --stages mesh --overwrite --config remesh.yaml --all

A --stages set that includes preproc or pointcloud always rebuilds from the curated video,
so a re-run can never leave a stale downstream artifact behind. A leaf re-run of a scene with
no processed outputs fails that scene; the config is authoritative, so a --config backend that
disagrees with the pulled run_config.yaml is an error rather than a silent retarget. Naming a
stage whose output already exists is refused — pass --overwrite to replace it.

Credentials come from the existing rclone remote (`collab-data`) — nothing is read
from the environment or passed on the command line. One scene's failure does not abort
the batch. A scene whose reconstruction fails is not pushed and keeps its local dir on
disk (named in a warning) for inspection or retry.

A failure that turns out to be rclone itself is different: the batch re-probes the
remote after any scene failure and, if rclone is unreachable, stops rather than marching
the rest of the list into the same fault. Scenes that never ran are reported SKIPPED and
can be re-run unchanged.

Exit codes:
    0  every scene succeeded
    1  one or more scenes failed on their own merits; the rest still ran
    2  nothing to do (no scenes named, none curated)
    3  aborted early — rclone became unreachable, so later scenes never ran

3 outranks 1: a run where a scene failed and the remote then died exits 3, because the
infrastructure fault is the actionable cause and the unrun scenes are safe to retry as-is.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

from collab_splats.remote import PUSH_EXCLUDES, SCENE_ID_RE, SceneSource, discover_scenes, prepare_scene
from collab_splats.wrapper import batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

########
# Exit codes — an unattended runner needs to tell these apart without parsing the log
########

EXIT_OK = 0
EXIT_SCENE_FAILED = 1  # one or more scenes failed on their own merits; the rest still ran
EXIT_NOTHING_TO_DO = 2  # no scenes named and none curated
EXIT_REMOTE_UNAVAILABLE = 3  # batch aborted early: rclone stopped working, later scenes never ran


def run_remote(
    source,
    scenes,
    output_root,
    config_dir,
    override_config,
    stages,
    overwrite,
    keep_local: bool,
) -> int:
    """Fetch → reconstruct → push → verify → delete for each scene. Returns exit code."""
    output_root = Path(output_root)
    if scenes:
        scene_ids = list(scenes)
    else:
        # The whole work list comes from here, and list_scenes raises on any rclone failure by design
        # (absence on GCS is an empty listing, never an exit code). Uncaught, a dead or unconfigured
        # remote exited 1 with a traceback — telling an unattended runner "some scenes failed" when
        # nothing had even been attempted, and leaving EXIT_REMOTE_UNAVAILABLE unreachable at start.
        try:
            # A leaf-only --stages set re-runs from environments-processed, so that is the bucket
            # the work list must come from; anything else starts from curated video.
            scene_ids = discover_scenes(source, stages)
        except Exception as exc:
            logger.error("cannot list curated scenes — rclone is not working: %s", exc)
            return EXIT_REMOTE_UNAVAILABLE
    if not scene_ids:
        logger.error("No scenes to process")
        return EXIT_NOTHING_TO_DO

    results = []
    aborted = False
    for idx, scene in enumerate(scene_ids):
        logger.info("=== Scene: %s ===", scene)
        scene_dir = output_root / scene
        # None until the pipeline returns one; the except handler below can fire before that, which
        # is why the retention warning names scene_dir (the fetch target) rather than out.
        out = None
        failure = None
        try:
            # 1. Fetch inputs: the curated video, or the processed scene pulled back for a
            # leaf-stage re-run (which returns video=None and a config carrying its provenance)
            video, scene_config = prepare_scene(
                source, scene, scene_dir, stages, override_config, on_line=logger.info
            )

            # 2. Same pipeline as the local driver; name= pins the output dir to the scene id
            out, _ = batch.run_scene(
                video, output_root, config_dir, scene_config, stages, overwrite, name=scene
            )

            # 3. Push, excluding regenerable artifacts and the fetched video (see PUSH_EXCLUDES)
            source.push_outputs(out, scene, on_line=logger.info)

            # 4. Verify before touching anything local — a failed check leaves the data put
            if not source.verify_push(out, scene, on_line=logger.info):
                failure = "push verification failed; local data kept"
            # 5. Reclaim the disk (video included — it lives in the curated bucket). Deleting `out`,
            # the very path verify_push passed on: output_root/scene is equal today only because
            # scene_output_dir(name=) builds it identically, and that is not a contract to bet a
            # destructive delete on.
            elif keep_local:
                logger.info("--keep-local: leaving %s in place", out)
            elif not _remove_scene_dir(out):
                failure = f"push verified but local dir remains: {out}"
        except Exception as exc:  # isolate one scene's failure from the batch
            logger.exception("Scene failed: %s", scene)
            # The fetched video and any partial outputs stay behind for inspection/retry
            logger.warning("local scene dir retained for %s: %s", scene, scene_dir)
            failure = str(exc)

        if failure is None:
            results.append((scene, "OK", str(out)))
            continue

        # Every failure path arrives here, exception or not, because a transport fault can develop on
        # any of them and looks identical on all three: verify_push swallows its own RuntimeError/
        # OSError and returns a bare False, and a cleanup failure is only ever a warning. A scene may
        # equally fail on its own merits (bad video, OOM, mesh failure) and that must cost only that
        # scene — so re-probe the remote instead of guessing. Only a dead remote aborts.
        results.append((scene, "FAIL", failure))
        if not source.check_available():
            aborted = True
            skipped = scene_ids[idx + 1 :]  # excludes this scene, which already has its FAIL row
            logger.error(
                "rclone is not working — ABORTING the batch after %s; %d scene(s) will not be attempted",
                scene,
                len(skipped),
            )
            # Recorded as SKIPPED, never FAIL: they never ran, so blaming them would send the
            # operator hunting causes for scenes that are simply safe to retry.
            results.extend((s, "SKIPPED", "not attempted — batch aborted, rclone unreachable") for s in skipped)
            break

    logger.info("==== Summary ====")
    for scene, status, info in results:
        logger.info("%s: %s (%s)", status, scene, info)
    logger.info("push excluded: %s", ", ".join(PUSH_EXCLUDES))

    if aborted:
        logger.error("batch ABORTED because rclone was unreachable; SKIPPED scenes never ran and can be re-run as-is")
        return EXIT_REMOTE_UNAVAILABLE
    return EXIT_SCENE_FAILED if any(s == "FAIL" for _, s, _ in results) else EXIT_OK


def _remove_scene_dir(scene_dir: Path) -> bool:
    """Delete a scene dir; False (with a warning) when anything survived the attempt."""
    # ignore_errors keeps a partial delete from aborting the batch, but it also hides the
    # failure — so confirm the dir is gone rather than trusting the call returned
    shutil.rmtree(scene_dir, ignore_errors=True)
    if scene_dir.exists():
        logger.warning("failed to remove local scene dir; disk not reclaimed: %s", scene_dir)
        return False
    logger.info("removed local scene dir %s", scene_dir)
    return True


def main():
    """Parse args and process the requested curated scenes."""
    parser = argparse.ArgumentParser(
        description="Reconstruct scenes from environments-curated and push to environments-processed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("scenes", nargs="*", metavar="SCENE", help="Curated scene ids. Omit with --all.")
    parser.add_argument("--all", action="store_true", help="Process every scene in environments-curated.")
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        dest="output_root",
        help="Working dir; each scene lands in <output-root>/<scene>/ and is removed after a verified push.",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional shared override YAML merged over base.yaml."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=batch.DEFAULT_CONFIG_DIR,
        dest="config_dir",
        help=f"Directory holding base.yaml. Default: {batch.DEFAULT_CONFIG_DIR}",
    )
    parser.add_argument(
        "--stages",
        default=None,
        metavar="STAGE[,STAGE,...]",
        help=(
            "Steps to run: preproc,pointcloud,semantics,mesh,localize. Default: config-enabled steps. "
            "A leaf-only set (mesh,semantics,localize) re-runs against scenes pulled back from "
            "environments-processed instead of rebuilding from curated video; any set including "
            "preproc or pointcloud rebuilds from video as before."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run steps even if outputs already exist.")
    parser.add_argument(
        "--keep-local",
        action="store_true",
        dest="keep_local",
        help="Skip the post-push delete and leave the scene dir on disk.",
    )
    args = parser.parse_args()

    if not args.scenes and not args.all:
        parser.error("give one or more SCENE ids, or --all")

    # --all ids come from list_scenes, which filters on SCENE_ID_RE; explicit ids skipped that filter
    # entirely and are joined straight onto --output-root, so "../x" would write outside it. One
    # shared regex, or discovery could yield an id an explicit re-run of the same scene then refuses.
    bad = [s for s in args.scenes if not SCENE_ID_RE.match(s)]
    if bad:
        parser.error(f"not curated scene ids (expected YYYY_MM_DD-PARENT-VIDEO): {', '.join(bad)}")

    override_config = None
    if args.config:
        with open(args.config) as f:
            override_config = yaml.safe_load(f)

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    code = run_remote(
        source=SceneSource(),
        scenes=args.scenes or None,
        output_root=args.output_root,
        config_dir=args.config_dir,
        override_config=override_config,
        stages=stages,
        overwrite=args.overwrite,
        keep_local=args.keep_local,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
