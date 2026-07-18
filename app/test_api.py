from fastapi.testclient import TestClient
from main import app
client = TestClient(app)

TEST_AUDIO_PATH = "./test_fixtures/sample.wav"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_transcribe():
    with open(TEST_AUDIO_PATH, "rb") as f:
        response = client.post(
            "/transcribe",
            files = {"file": ("sample.wav", f, "audio/wav")}
        )
    assert response.status_code == 200
    assert "text" in response.json() and "latency_ms" in response.json()