import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    from collab_splats.webapp.app import create_app
    return create_app()


@pytest.mark.asyncio
async def test_root_serves_index(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "collab-splats" in r.text


@pytest.mark.asyncio
async def test_session_load_bad_dir(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/session/load", json={"output_dir": "/no/such/path"})
    assert r.status_code == 200
    assert r.json()["ok"] is False


@pytest.mark.asyncio
async def test_session_current_empty(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/session/current")
    assert r.status_code == 200
    data = r.json()
    assert "output_dir" in data
