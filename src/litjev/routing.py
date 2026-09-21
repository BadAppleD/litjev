"""Fast-or-slow routing from the decision head's joint outcome distribution.

The head predicts P(fast correct, slow correct) as four classes:
0: both right, 1: fast right / slow wrong, 2: fast wrong / slow right, 3: both wrong.
Escalating to the slow path is worth it when the expected accuracy gain exceeds
the cost `lambda_`, a runtime knob that never enters training.
"""

from dataclasses import dataclass

import numpy as np

OUTCOME_CLASSES = (
    "fast_right_slow_right",
    "fast_right_slow_wrong",
    "fast_wrong_slow_right",
    "fast_wrong_slow_wrong",
)


@dataclass(frozen=True)
class RoutingPolicy:
    lambda_: float = 0.0
    budget: int = 0

    def __post_init__(self):
        if not np.isfinite(self.lambda_):
            raise ValueError("lambda must be finite")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")

    @property
    def enabled(self):
        return self.budget > 0


def fast_confidence(outcome):
    """Calibrated P(fast answer is correct)."""
    return float(outcome[0] + outcome[1])


def escalation_gain(outcome):
    """Expected accuracy gained by thinking: P(fast wrong, slow right) - P(fast right, slow wrong)."""
    return float(outcome[2] - outcome[1])


def should_escalate(outcome, policy):
    return bool(policy.enabled and escalation_gain(outcome) > policy.lambda_)
