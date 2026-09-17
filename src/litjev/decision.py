from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from litjev.scoring import calibrated_distribution


@dataclass(frozen=True)
class RawFieldScores:
    name: str
    logits: np.ndarray
    input_tokens: int
    provenance: dict = field(default_factory=dict)


class LogitProvider(Protocol):
    def score(self, state, schema) -> tuple[RawFieldScores, ...]: ...


@dataclass(frozen=True)
class DecisionAnswer:
    field_type: str
    value: str | bool
    probabilities: dict[str, float]
    gamma: float
    logits: tuple[float, ...] = ()
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int = 0
    forward_calls: int = 2


@dataclass(frozen=True)
class DecisionResponse:
    model: str
    answers: dict[str, DecisionAnswer]
    usage: Usage
    calibration_fitted: bool = False


class SchemaDecisionEngine:
    def __init__(self, provider, temperature=1.0, model_id="unknown", calibration_fitted=False):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.provider = provider
        self.temperature = temperature
        self.model_id = model_id
        self.calibration_fitted = calibration_fitted

    def decide(self, state, schema):
        scores = self.provider.score(state, schema)
        if tuple(row.name for row in scores) != schema.names:
            raise RuntimeError("Scorer returned mismatched fields")
        answers = {}
        for row in scores:
            field = schema[row.name]
            if len(row.logits) != len(field.choices):
                raise RuntimeError("Scorer returned mismatched candidates")
            distribution = calibrated_distribution(
                row.logits, list(range(len(field.choices))), self.temperature
            )
            selected = field.choices[distribution.winner_index]
            value = selected == "true" if field.field_type == "boolean" else selected
            answers[row.name] = DecisionAnswer(
                field.field_type,
                value,
                dict(zip(field.choices, distribution.probabilities, strict=True)),
                distribution.winner_probability,
                tuple(float(value) for value in row.logits),
                {**row.provenance, "temperature": self.temperature},
            )
        return DecisionResponse(
            self.model_id, answers, Usage(scores[0].input_tokens), self.calibration_fitted
        )
