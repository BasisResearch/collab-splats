import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_visualize_status_no_session():
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    get_session().output_dir = None
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/visualize/status")
    assert r.json()["ok"] is False


@pytest.mark.asyncio
async def test_visualize_status_with_ply(tmp_path):
    from collab_splats.webapp.app import create_app
    from collab_splats.webapp.state import get_session
    s = get_session()
    s.output_dir = tmp_path
    s.creator = "vggtx"
    ply_path = tmp_path / "vggtx" / "sparse_pc.ply"
    ply_path.parent.mkdir()
    ply_path.write_text("ply\n")
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/visualize/status")
    data = r.json()
    assert data["ok"] is True
    assert data["ply_url"] is not None
