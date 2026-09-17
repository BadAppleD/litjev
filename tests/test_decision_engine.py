from dataclasses import dataclass

import numpy as np

from litjev.decision import RawFieldScores, SchemaDecisionEngine
from litjev.schema import DecisionSchema


@dataclass
class FakeLogitProvider:
    def score(self, state, schema):
        del state
        assert schema.names == ("q1", "q2")
        return (
            RawFieldScores("q1", np.array([3.0, 1.0]), 12),
            RawFieldScores("q2", np.array([0.0, 4.0]), 12),
        )


def test_decision_engine_assembles_typed_values_and_probabilities() -> None:
    schema = DecisionSchema.from_mapping(
        {
            "q1": {"type": "enum", "description": "First", "choices": ["A", "B"]},
            "q2": {"type": "boolean", "description": "Second"},
        }
    )
    engine = SchemaDecisionEngine(FakeLogitProvider(), temperature=2.0)

    response = engine.decide("state", schema)

    assert response.answers["q1"].value == "A"
    assert response.answers["q2"].value is False
    assert response.answers["q1"].gamma < 0.8
    assert np.isclose(sum(response.answers["q2"].probabilities.values()), 1.0)
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 0
