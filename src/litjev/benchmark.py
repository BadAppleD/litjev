"""Comparable direct-scoring runs, without exposing evaluation labels to the engine."""

from time import perf_counter

from litjev.decision import DecisionResponse, Usage
from litjev.schema import DecisionSchema

STATE = "Answer each question using its listed options."


def evaluate(engine, schema, sequential=False, synchronize=lambda: None):
    if not sequential:
        return engine.decide(STATE, schema), {}
    answers, durations = {}, {}
    input_tokens = forward_calls = 0
    for name, field in schema.items():
        single = DecisionSchema({name: field})
        synchronize()
        started = perf_counter()
        response = engine.decide(STATE, single)
        synchronize()
        durations[name] = perf_counter() - started
        answers.update(response.answers)
        input_tokens += response.usage.input_tokens
        forward_calls += response.usage.forward_calls
    return DecisionResponse(
        response.model,
        answers,
        Usage(input_tokens, forward_calls=forward_calls),
        response.calibration_fitted,
    ), durations
