import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app_with_session(tmp_path):
    (tmp_path / "run_config.yaml").write_text("video_path: /dev/null\n")
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
