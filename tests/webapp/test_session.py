import pytest
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_root_serves_index():
    from collab_splats.webapp.app import create_app

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "collab-splats" in resp.text
