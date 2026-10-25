from fastapi.testclient import TestClient
from app.main import create_app


def test_health_check():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
