import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_reconstruct_no_session():
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = None
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/reconstruct/status")
    assert r.status_code == 200
    assert r.json()["ok"] is False


@pytest.mark.asyncio
async def test_reconstruct_status_with_session(tmp_path):
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = tmp_path
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/reconstruct/status")
    data = r.json()
    assert "has_frames" in data
