"""Typed answers are assembled from logits, not generated text."""

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from litjev.routing import (
    OUTCOME_CLASSES,
    RoutingPolicy,
    escalation_gain,
    fast_confidence,
    should_escalate,
)
from litjev.scoring import calibrated_distribution

CONFIDENCE_METHOD = "normalized_gini_concentration_v1"
HEAD_CONFIDENCE_METHOD = "decision_head_outcome_v1"


@dataclass(frozen=True)
class RawFieldScores:
    name: str
    logits: np.ndarray
    input_tokens: int
    provenance: dict = field(default_factory=dict)
    hidden: np.ndarray | None = None
    generated_tokens: int = 0


class LogitProvider(Protocol):
    def score(self, state, schema) -> tuple[RawFieldScores, ...]: ...


@dataclass(frozen=True)
class ChoiceAnswer:
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    type: Literal["score"]
    score: float
    legend: dict
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    type: Literal["noul"]
    noul: float


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int = 0


@dataclass(frozen=True)
class DecisionResponse:
    model: str
    answers: dict[str, ChoiceAnswer | ScoreAnswer | NoulAnswer]
    usage: Usage


@dataclass(frozen=True)
class Evaluation:
    result: DecisionResponse
    diagnostics: dict


def concentration(probabilities):
    """LitJev statistic, not a reproduction of Jev's unpublished formula."""
    count = len(probabilities)
    if count == 1:
        return 1.0
    return float(np.clip((count * sum(p * p for p in probabilities) - 1) / (count - 1), 0, 1))


class SchemaDecisionEngine:
    def __init__(
        self,
        provider,
        temperature=1.0,
        model_id="unknown",
        calibration_fitted=False,
        head=None,
        routing=None,
    ):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.provider = provider
        self.temperature = temperature
        self.model_id = model_id
        self.calibration_fitted = calibration_fitted
        self.head = head
        self.routing = routing if routing is not None else RoutingPolicy()

    def decide(self, state, schema, routing=None):
        return self.evaluate(state, schema, routing).result

    def _distribution(self, row, question):
        if len(row.logits) != len(question.choices):
            raise RuntimeError("Scorer returned mismatched candidates")
        distribution = calibrated_distribution(
            row.logits, list(range(len(question.choices))), self.temperature
        )
        probabilities = dict(zip(question.choices, distribution.probabilities, strict=True))
        return distribution, probabilities

    @staticmethod
    def _answer(question, distribution, probabilities, confidence):
        if question.type == "noul":
            return NoulAnswer("noul", probabilities["true"])
        if question.type == "score":
            return ScoreAnswer(
                "score",
                sum(int(k) * p for k, p in probabilities.items()),
                dict(zip(question.choices, question.descriptions, strict=True)),
                probabilities,
                confidence,
            )
        return ChoiceAnswer(
            "choice", question.choices[distribution.winner_index], probabilities, confidence
        )

    def _outcome(self, row, question, distribution):
        """Decision head prediction for one fast readout, or None without a head."""
        if self.head is None:
            return None
        from litjev.heads import build_features

        hidden = row.hidden
        if not self.head.metadata.feature_layers:
            hidden = np.zeros(0, dtype=np.float32)  # stats-only head
        elif hidden is None:
            raise RuntimeError("Decision head requires readout hidden states from the scorer")
        features = build_features(hidden, distribution.probabilities, question.type)
        return self.head.predict(features)[0]

    def evaluate(self, state, schema, routing=None):
        policy = routing if routing is not None else self.routing
        if policy.enabled and self.head is None:
            raise ValueError("Routing to the slow path requires a decision head")
        scores = self.provider.score(state, schema)
        if tuple(row.name for row in scores) != schema.names:
            raise RuntimeError("Scorer returned mismatched fields")
        answers, fields, outcomes, escalate = {}, {}, {}, []
        for row in scores:
            question = schema[row.name]
            distribution, probabilities = self._distribution(row, question)
            outcome = self._outcome(row, question, distribution)
            if outcome is None:
                confidence = concentration(distribution.probabilities)
            else:
                confidence = fast_confidence(outcome)
                outcomes[row.name] = outcome
                if should_escalate(outcome, policy):
                    escalate.append(row.name)
            answers[row.name] = self._answer(question, distribution, probabilities, confidence)
            fields[row.name] = {
                "logits": tuple(float(value) for value in row.logits),
                "probabilities": probabilities,
                "max_probability": distribution.winner_probability,
                "concentration": concentration(distribution.probabilities),
                "system": "one",
                "provenance": {**row.provenance, "temperature": self.temperature},
            }
            if outcome is not None:
                fields[row.name]["outcome"] = dict(
                    zip(OUTCOME_CLASSES, (float(p) for p in outcome), strict=True)
                )
                fields[row.name]["escalation_gain"] = escalation_gain(outcome)
        output_tokens = 0
        forward_calls = 2
        if escalate:
            slow = self.provider.think(state, schema, tuple(escalate), policy.budget)
            if tuple(row.name for row in slow) != tuple(escalate):
                raise RuntimeError("Slow path returned mismatched fields")
            for row in slow:
                question = schema[row.name]
                distribution, probabilities = self._distribution(row, question)
                outcome = outcomes[row.name]
                slow_confidence = float(outcome[0] + outcome[2])
                answers[row.name] = self._answer(
                    question, distribution, probabilities, slow_confidence
                )
                output_tokens += row.generated_tokens
                fields[row.name].update(
                    {
                        "fast_logits": fields[row.name]["logits"],
                        "fast_probabilities": fields[row.name]["probabilities"],
                        "logits": tuple(float(value) for value in row.logits),
                        "probabilities": probabilities,
                        "max_probability": distribution.winner_probability,
                        "concentration": concentration(distribution.probabilities),
                        "system": "two",
                        "slow_provenance": {**row.provenance, "temperature": self.temperature},
                    }
                )
            # One prefill, one forward per generated token, one final readout.
            forward_calls += 2 + max(row.generated_tokens for row in slow)
        return Evaluation(
            DecisionResponse(self.model_id, answers, Usage(scores[0].input_tokens, output_tokens)),
            {
                "fields": fields,
                "forward_calls": forward_calls,
                "calibration_fitted": self.calibration_fitted,
                "confidence_method": HEAD_CONFIDENCE_METHOD if self.head else CONFIDENCE_METHOD,
                "routing": {
                    "enabled": policy.enabled,
                    "lambda": policy.lambda_,
                    "budget": policy.budget,
                    "escalated": escalate,
                    "generated_tokens": output_tokens,
                },
            },
        )
