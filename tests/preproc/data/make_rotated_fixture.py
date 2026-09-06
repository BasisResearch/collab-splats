"""
Generate tests/preproc/data/rotated_90.mp4 — a landscape clip carrying a 90-degree
display matrix, so its display dimensions are portrait.

Run once, commit the .mp4. Regenerate with:
    /opt/venv/reconstruction/bin/python tests/preproc/data/make_rotated_fixture.py
"""

import json
import subprocess
import tempfile
from pathlib import Path

OUT = Path(__file__).parent / "rotated_90.mp4"

# ffmpeg 4.4's mov muxer writes the display matrix from the `rotate` tag, but its
# demuxer reads the tag back NEGATED (write 270 -> read 90). Ask for 270 so the
# committed file reports the 90-degree CW rotation its name claims.
MUX_ROTATE = "270"


def main() -> None:
    """
    Encode a landscape testsrc clip, remux it with a display matrix, verify, write.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # A 320x180 landscape clip with a moving bar, so a transposed decode is obvious
        landscape = Path(tmp) / "landscape.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x180:rate=10:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(landscape),
            ],
            check=True,
            capture_output=True,
        )

        # The rotation has to land on a REMUX: ffmpeg 4.4 silently drops
        # -metadata:s:v:0 rotate= when it is transcoding in the same pass.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(landscape),
                "-c",
                "copy",
                "-metadata:s:v:0",
                f"rotate={MUX_ROTATE}",
                str(OUT),
            ],
            check=True,
            capture_output=True,
        )

    # Verify the display matrix actually landed — -metadata rotate= is silently
    # ignored by some muxers, in which case the fixture tests nothing
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0", "-show_streams", str(OUT)],
        capture_output=True,
        text=True,
        check=True,
    )
    if "Display Matrix" not in probe.stdout and '"rotate"' not in probe.stdout:
        raise SystemExit(
            f"{OUT} carries no rotation metadata — the muxer dropped it. Try "
            f"`-display_rotation 90` on the input instead."
        )

    # Stored dims must stay landscape: a fixture ffmpeg already baked upright
    # would exercise nothing downstream
    stream = json.loads(probe.stdout)["streams"][0]
    if (stream["width"], stream["height"]) != (320, 180):
        raise SystemExit(f"{OUT} is stored {stream['width']}x{stream['height']}, expected 320x180")

    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
