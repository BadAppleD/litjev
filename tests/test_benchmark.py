from litjev.benchmark import evaluate
from litjev.decision import DecisionAnswer, DecisionResponse, Usage
from litjev.schema import DecisionSchema


def test_sequential_uses_ten_separate_single_question_calls():
    calls = []

    class Engine:
        def decide(self, state, schema):
            from litjev.prompting import build_decision_messages

            prompt = build_decision_messages(state, schema)[0]["content"]
            # Every single-question invocation contains only its own question text.
            for i in range(10):
                assert (f"QUESTION_{i}_ONLY" in prompt) == (f"q{i}" in schema)
            calls.append(schema.names)
            return DecisionResponse(
                "fake",
                {name: DecisionAnswer("enum", "A", {"A": 1.0}, 1.0) for name in schema},
                Usage(10),
            )

    schema = DecisionSchema.from_mapping(
        {f"q{i}": {"description": f"QUESTION_{i}_ONLY", "choices": ["A", "B"]} for i in range(10)}
    )
    response, durations = evaluate(Engine(), schema, sequential=True)
    assert len(calls) == 10 and all(len(call) == 1 for call in calls)
    assert response.usage.forward_calls == 20
    assert len(response.answers) == len(durations) == 10
    calls.clear()
    response, durations = evaluate(Engine(), schema)
    assert len(calls) == 1 and len(calls[0]) == 10
    assert response.usage.forward_calls == 2
    assert durations == {}
