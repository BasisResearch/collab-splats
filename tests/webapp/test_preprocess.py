from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from collab_splats.preproc.frame_store import FrameStore


@pytest.fixture
def app_with_session(tmp_path):
    (tmp_path / "run_config.yaml").write_text("video_path: /dev/null\n")
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session

    s = get_session()
    s.output_dir = tmp_path
    s.video_path = None
    return create_app()


@pytest.fixture
def app_with_frames_zarr(tmp_path):
    """Session with a seeded frames.zarr (3 tiny keyframes) for the on-demand JPEG route."""
    frames = [np.full((4, 4, 3), i * 40, dtype=np.uint8) for i in range(3)]
    records = [{"frame_idx": i} for i in range(3)]
    prov = {"video_path": "/dev/null", "video_mtime": 0.0, "method": "uniform", "max_frames": 3}
    FrameStore.create(tmp_path / "frames.zarr", frames, records, provenance=prov)

    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session

    s = get_session()
    s.output_dir = tmp_path
    s.video_path = None
    return create_app()


@pytest.mark.asyncio
async def test_video_info_no_session():
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session

    get_session().output_dir = None
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/preprocess/info")
    assert r.status_code == 200
    assert r.json()["ok"] is False


@pytest.mark.asyncio
async def test_video_info_with_session(app_with_session):
    async with AsyncClient(transport=ASGITransport(app=app_with_session), base_url="http://test") as c:
        r = await c.get("/api/preprocess/info")
    assert r.status_code == 200
    data = r.json()
    assert "output_dir" in data


@pytest.mark.asyncio
async def test_frame_route_serves_jpeg_from_store(app_with_frames_zarr):
    async with AsyncClient(transport=ASGITransport(app=app_with_frames_zarr), base_url="http://test") as c:
        r = await c.get("/api/preprocess/frame/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"  # JPEG magic bytes


@pytest.mark.asyncio
async def test_frame_route_out_of_range(app_with_frames_zarr):
    async with AsyncClient(transport=ASGITransport(app=app_with_frames_zarr), base_url="http://test") as c:
        r = await c.get("/api/preprocess/frame/99")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_frame_route_no_store(app_with_session):
    async with AsyncClient(transport=ASGITransport(app=app_with_session), base_url="http://test") as c:
        r = await c.get("/api/preprocess/frame/0")
    assert r.status_code == 404
