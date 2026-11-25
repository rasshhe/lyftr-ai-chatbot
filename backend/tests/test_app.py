from types import SimpleNamespace
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import create_app


def _fake_litellm_completion(model, messages, api_key, caching=True):
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    reply = f"Echo ({api_key}): {last_user}"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
        usage=SimpleNamespace(prompt_tokens=25, completion_tokens=11),
    )


def _make_client(tmp_path, enable_sim: bool = True):
    db_file = tmp_path / "test.db"
    env_vars = {
        "LLM_API_KEYS": "gsk-test-key-1,gsk-test-key-2",
        "LLM_MODEL": "gemini/gemini-2.0-flash",
        "DATABASE_URL": f"sqlite:///{db_file}",
        "ALLOWED_ORIGINS": "*",
        "ENABLE_FAILURE_SIMULATION": "true" if enable_sim else "false",
    }
    with patch.dict("os.environ", env_vars, clear=False):
        app = create_app()
        client = TestClient(app)
        return client


@patch("app.llm_service.litellm.completion", side_effect=_fake_litellm_completion)
def test_health_and_chat_and_cache_hit(mock_llm, tmp_path):
    client = _make_client(tmp_path)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }

    r1 = client.post("/chat", json={"session_id": "s-1", "message": "What is 2+2?"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["session_id"] == "s-1"
    assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert d1["metadata"]["key_id"] == "key_1"
    assert d1["metadata"]["fallback_attempts"] == 0
    assert d1["metadata"]["input_tokens"] == 25
    assert d1["metadata"]["output_tokens"] == 11

    r2 = client.post("/chat", json={"session_id": "s-2", "message": "What is 2+2?"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["reply"] == d1["reply"]
    assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
    assert d2["metadata"]["key_id"] is None
    assert d2["metadata"]["fallback_attempts"] == 0


@patch("app.llm_service.litellm.completion", side_effect=_fake_litellm_completion)
def test_failover_and_all_failed_503(mock_llm, tmp_path):
    client = _make_client(tmp_path, enable_sim=True)

    r_failover = client.post(
        "/chat",
        json={"session_id": "s-failover", "message": "Test failover"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert r_failover.status_code == 200
    meta = r_failover.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] == 1
    assert meta["resolution_source"] == "LIVE_INFERENCE"

    r_all_fail = client.post(
        "/chat",
        json={"session_id": "s-all-fail", "message": "Should fail"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert r_all_fail.status_code == 503
    assert r_all_fail.json() == {"detail": "all providers unavailable"}

    hist_check = client.get("/sessions/s-all-fail/messages")
    assert hist_check.status_code == 404


def test_validation_422_and_history_404(tmp_path):
    client = _make_client(tmp_path)

    bad_empty = client.post("/chat", json={"session_id": "s-bad", "message": "   "})
    assert bad_empty.status_code == 422

    bad_long = client.post("/chat", json={"session_id": "s-bad", "message": "x" * 4001})
    assert bad_long.status_code == 422

    missing_session = client.get("/sessions/does-not-exist/messages")
    assert missing_session.status_code == 404
    assert missing_session.json() == {"detail": "session not found"}
