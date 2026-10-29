"""
Tests for Persistent Daemon and OpenAI Proxy Endpoints.
"""

from fastapi.testclient import TestClient
from qnm.daemon.app import app


def test_daemon_endpoints_lifecycle():
    with TestClient(app) as client:
        # Health check
        res_health = client.get("/qnm/health")
        assert res_health.status_code == 200
        data_health = res_health.json()
        assert data_health["status"] == "healthy"
        assert "execution_mode" in data_health

        # List models (OpenAI format)
        res_models = client.get("/v1/models")
        assert res_models.status_code == 200
        assert res_models.json()["object"] == "list"

        # Native QNM query
        res_query = client.post("/qnm/query", json={
            "prompt": "Test query with symbol C:\\test\\path.txt and flag --verbose",
            "session_id": "test_daemon_sess"
        })
        assert res_query.status_code == 200
        data_query = res_query.json()
        assert "expanded_response" in data_query
        assert "telemetry" in data_query

        # OpenAI chat completions format
        res_openai = client.post("/v1/chat/completions", json={
            "model": "qnm-dual-bottleneck",
            "messages": [
                {"role": "user", "content": "What is QNM compression?"}
            ],
            "session_id": "test_daemon_sess"
        })
        assert res_openai.status_code == 200
        data_openai = res_openai.json()
        assert data_openai["object"] == "chat.completion"
        assert len(data_openai["choices"]) == 1
        assert "usage" in data_openai
        assert "context_saved_percent" in data_openai["usage"]

        # Stats
        res_stats = client.get("/qnm/stats")
        assert res_stats.status_code == 200
        assert "total_requests" in res_stats.json()

        # Clear
        res_clear = client.post("/qnm/clear", json={"session_id": "test_daemon_sess"})
        assert res_clear.status_code == 200
        assert res_clear.json()["status"] == "success"
