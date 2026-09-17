from fastapi.testclient import TestClient

from litjev.api import create_app


class FakeEngine:
    def decide(self, state, schema):
        from litjev.decision import DecisionAnswer, DecisionResponse, Usage

        del state
        answers = {
            name: DecisionAnswer(
                field_type=field.field_type,
                value=field.choices[0],
                probabilities={
                    choice: 1.0 if index == 0 else 0.0 for index, choice in enumerate(field.choices)
                },
                gamma=1.0,
            )
            for name, field in schema.items()
        }
        return DecisionResponse("fake", answers, Usage(42, 0))


def test_schema_endpoint_returns_typed_answers() -> None:
    client = TestClient(create_app(lambda: FakeEngine()))

    response = client.post(
        "/v1/calibrated-schema",
        json={
            "state": "Choose.",
            "schema": {
                "answer": {
                    "type": "enum",
                    "description": "Answer label",
                    "choices": ["A", "B"],
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["answers"]["answer"]["value"] == "A"
    timing = response.json()["timing"]
    assert timing["total_seconds"] >= timing["decision_seconds"] >= 0
    assert timing["model_setup_seconds"] >= 0


def test_playground_serves_without_loading_model():
    def forbidden_load():
        raise AssertionError("Opening the UI must not load weights")

    client = TestClient(create_app(forbidden_load))
    assert client.get("/").status_code == 200
    assert "schema-input" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert not client.get("/health").json()["model_loaded"]


def test_invalid_field_spec_returns_validation_error():
    client = TestClient(create_app(lambda: FakeEngine()))
    response = client.post("/v1/calibrated-schema", json={"state": "", "schema": {"q": 1}})
    assert response.status_code == 422


def test_mcq_endpoint_requires_ten_questions() -> None:
    client = TestClient(create_app(lambda: FakeEngine()))

    response = client.post(
        "/v1/batch-mcq",
        json={
            "questions": [{"question_id": "q1", "prompt": "1+1?", "options": {"A": "2", "B": "3"}}]
        },
    )

    assert response.status_code == 422


def test_mcq_ten_questions_make_one_engine_call():
    calls = []

    class CountingEngine(FakeEngine):
        def decide(self, state, schema):
            calls.append(schema.names)
            return super().decide(state, schema)

    client = TestClient(create_app(lambda: CountingEngine()))
    questions = [
        {"question_id": f"q{i}", "prompt": "Pick", "options": {"A": "one", "B": "two"}}
        for i in range(10)
    ]
    response = client.post("/v1/batch-mcq", json={"questions": questions})
    assert response.status_code == 200
    assert len(response.json()["answers"]) == 10
    assert len(calls) == 1 and len(calls[0]) == 10
    questions[-1]["question_id"] = "q0"
    assert client.post("/v1/batch-mcq", json={"questions": questions}).status_code == 422
