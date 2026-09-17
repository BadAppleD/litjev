import pytest
from fastapi.testclient import TestClient

from litjev.games.preview import create_preview_app


def test_random_preview_runs_real_doom_without_model():
    with TestClient(create_preview_app(seed=7, max_steps=2, resolution="160x120")) as client:
        assert client.get("/").status_code == 200
        initial = client.get("/state").json()
        assert initial["policy"] == "uniform_random"
        assert initial["step"] == 0
        assert initial["frame"].startswith("data:image/png;base64,")
        first = client.post("/step").json()
        assert first["action"] in first["actions"]
        assert first["step"] == 1
        assert first["forward_calls"] == 0
        assert sum(first["probabilities"]) == pytest.approx(1)
        assert client.post("/step").json()["done"]
        assert client.post("/step").json()["episode"] == 2
        reset = client.post("/reset").json()
        assert reset["step"] == 0 and reset["episode"] == 3
