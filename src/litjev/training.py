"""Train and evaluate the decision head from collected records.

Split by category so the reported numbers measure transfer to unseen subjects,
never memorization of the training questions.
"""

from __future__ import annotations

import hashlib

import numpy as np

from litjev.heads import (
    HeadMetadata,
    auroc,
    coverage_accuracy_curve,
    outcome_labels,
    train_head,
)
from litjev.routing import escalation_gain, fast_confidence

DEFAULT_LAMBDAS = (-0.2, -0.1, -0.05, 0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def _hash_mask(keys, fraction, seed):
    scores = np.array(
        [
            int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            for key in keys
        ]
    )
    return scores < fraction


def holdout_mask(categories, fraction, seed=0):
    """Deterministic category-level holdout.

    Returns (mask, level). Level is "category" when whole categories are held out,
    or "example" when too few categories exist for that to leave both sides populated;
    the per-example split then measures within-subject generalization only.
    """
    categories = np.asarray(categories)
    if categories.size and any(categories):
        mask = _hash_mask(categories, fraction, seed)
        if mask.any() and not mask.all():
            return mask, "category"
    mask = _hash_mask(np.arange(len(categories)).astype(str), fraction, seed)
    if mask.all() or not mask.any():
        raise ValueError("Holdout split left one side empty; adjust --holdout-fraction or seed")
    return mask, "example"


def features_for(records, layer_positions):
    hidden = records["hidden"][:, list(layer_positions), :].astype(np.float32)
    return np.concatenate([hidden.reshape(len(hidden), -1), records["stats"]], axis=1)


def evaluate_head(head, features, fast_correct, slow_correct, lambdas=DEFAULT_LAMBDAS):
    outcome = head.predict(features)
    confidence = np.array([fast_confidence(row) for row in outcome])
    gain = np.array([escalation_gain(row) for row in outcome])
    return {
        "auroc_fast_correct": auroc(confidence, fast_correct),
        "auroc_gain_vs_helps": auroc(gain, ~fast_correct & slow_correct),
        "curve": coverage_accuracy_curve(gain, fast_correct, slow_correct, lambdas),
    }


def run_training(
    records,
    metadata,
    holdout_fraction=0.25,
    select_layers=1,
    epochs=30,
    probe_epochs=10,
    seed=0,
    lambdas=DEFAULT_LAMBDAS,
):
    """Layer sweep, then a final head on the best layers. Returns (head, report)."""
    layers = tuple(metadata["feature_layers"])
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    labels = outcome_labels(fast, slow)
    test, split_level = holdout_mask(records["category"], holdout_fraction, seed)
    train = ~test
    base = {
        "model_id": metadata["model_id"],
        "revision": metadata["revision"],
        "hidden_size": int(metadata["hidden_size"]),
    }
    sweep = []
    for position, layer in enumerate(layers):
        meta = HeadMetadata(**base, feature_layers=(layer,), hidden_width=256)
        features = features_for(records, (position,))
        probe, _ = train_head(meta, features[train], labels[train], epochs=probe_epochs, seed=seed)
        result = evaluate_head(probe, features[test], fast[test], slow[test], lambdas)
        sweep.append({"layer": layer, "position": position, **result})
    ranked = sorted(sweep, key=lambda row: -np.nan_to_num(row["auroc_fast_correct"]))
    chosen = ranked[: max(1, min(select_layers, len(ranked)))]
    positions = tuple(sorted(row["position"] for row in chosen))
    chosen_layers = tuple(layers[p] for p in positions)
    meta = HeadMetadata(**base, feature_layers=chosen_layers)
    features = features_for(records, positions)
    head, history = train_head(meta, features[train], labels[train], epochs=epochs, seed=seed)
    final = evaluate_head(head, features[test], fast[test], slow[test], lambdas)
    max_probability = records["fast_probability"][test]
    concentration = records["stats"][test, 3]
    report = {
        "train_count": int(train.sum()),
        "test_count": int(test.sum()),
        "holdout_fraction": holdout_fraction,
        "split_level": split_level,
        "categories_seen": sorted({str(c) for c in np.asarray(records["category"])}),
        "held_out_categories": sorted({str(c) for c in np.asarray(records["category"])[test]}),
        "fast_accuracy": float(fast[test].mean()),
        "slow_accuracy": float(slow[test].mean()),
        "thinking_helps_rate": float((~fast[test] & slow[test]).mean()),
        "thinking_hurts_rate": float((fast[test] & ~slow[test]).mean()),
        "layer_sweep": sweep,
        "chosen_layers": list(chosen_layers),
        "baseline_auroc_max_probability": auroc(max_probability, fast[test]),
        "baseline_auroc_concentration": auroc(concentration, fast[test]),
        "head": final,
        "loss_history": history,
    }
    head.metadata = HeadMetadata(
        **base,
        feature_layers=chosen_layers,
        training={
            key: report[key]
            for key in (
                "train_count",
                "test_count",
                "held_out_categories",
                "fast_accuracy",
                "slow_accuracy",
                "chosen_layers",
            )
        }
        | {"auroc_fast_correct": final["auroc_fast_correct"]},
    )
    return head, report
