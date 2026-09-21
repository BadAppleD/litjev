"""Collect decision-head training data: fast readout features plus slow-path outcomes.

Gold labels are compared locally after both passes; they never enter any prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from litjev.heads import STATS_DIM, distribution_stats
from litjev.scoring import calibrated_distribution
from litjev.slots import SLOT_FORMAT

RECORD_FORMAT = "head_records_v1"


@dataclass(frozen=True)
class Record:
    question_id: str
    category: str
    question_type: str
    hidden: np.ndarray  # [layers, hidden_size]
    stats: np.ndarray  # [STATS_DIM]
    fast_choice: str
    slow_choice: str
    label: str
    fast_probability: float
    slow_probability: float
    thinking_tokens: int

    @property
    def fast_correct(self):
        return self.fast_choice == self.label

    @property
    def slow_correct(self):
        return self.slow_choice == self.label


def _winner(row, question):
    distribution = calibrated_distribution(row.logits, list(range(len(question.choices))), 1.0)
    return question.choices[distribution.winner_index], distribution


def collect_batch(provider, request, valid_ids, labels, budget, categories=None):
    """Run the fast path and the slow path on one request; return one record per scored ID."""
    schema = request.to_schema()
    fast = {row.name: row for row in provider.score(request.state, schema)}
    slow = {
        row.name: row for row in provider.think(request.state, schema, tuple(valid_ids), budget)
    }
    records = []
    for name in valid_ids:
        question = schema[name]
        fast_row, slow_row = fast[name], slow[name]
        if fast_row.hidden is None:
            raise ValueError("Collection requires a scorer configured with feature_layers")
        fast_choice, fast_distribution = _winner(fast_row, question)
        slow_choice, slow_distribution = _winner(slow_row, question)
        records.append(
            Record(
                question_id=name,
                category=(categories or {}).get(name, ""),
                question_type=question.type,
                hidden=np.asarray(fast_row.hidden, dtype=np.float16),
                stats=distribution_stats(fast_distribution.probabilities, question.type),
                fast_choice=fast_choice,
                slow_choice=slow_choice,
                label=labels[name],
                fast_probability=fast_distribution.winner_probability,
                slow_probability=slow_distribution.winner_probability,
                thinking_tokens=slow_row.generated_tokens,
            )
        )
    return records


def save_records(path, records, metadata):
    if not records:
        raise ValueError("No records to save")
    hidden = np.stack([record.hidden for record in records])
    if hidden.ndim != 3:
        raise ValueError("hidden must be [N, layers, hidden_size]")
    payload = {
        "hidden": hidden,
        "stats": np.stack([record.stats for record in records]).astype(np.float32),
        "fast_correct": np.array([record.fast_correct for record in records]),
        "slow_correct": np.array([record.slow_correct for record in records]),
        "fast_probability": np.array([record.fast_probability for record in records]),
        "thinking_tokens": np.array([record.thinking_tokens for record in records]),
        "question_id": np.array([record.question_id for record in records]),
        "category": np.array([record.category for record in records]),
        "question_type": np.array([record.question_type for record in records]),
        "metadata": np.array(
            json.dumps(
                {
                    **metadata,
                    "record_format": RECORD_FORMAT,
                    "slot_format": SLOT_FORMAT,
                    "stats_dim": STATS_DIM,
                    "count": len(records),
                }
            )
        ),
    }
    np.savez_compressed(path, **payload)


def load_records(paths):
    """Concatenate one or more record files; metadata must agree on model, layers and formats."""
    parts, metadata = [], None
    for path in paths:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))
        if meta.get("record_format") != RECORD_FORMAT or meta.get("slot_format") != SLOT_FORMAT:
            raise ValueError(f"{path}: records use an incompatible format; re-collect")
        keys = ("model_id", "revision", "feature_layers", "hidden_size")
        if metadata is None:
            metadata = meta
        elif any(meta.get(key) != metadata.get(key) for key in keys):
            raise ValueError(f"{path}: records were collected with a different model or layers")
        parts.append({key: data[key] for key in data.files if key != "metadata"})
    if not parts:
        raise ValueError("At least one record file is required")
    merged = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    return merged, metadata
