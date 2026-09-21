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


def group_folds(categories, k, seed=0):
    """Fold id per example: whole categories share a fold; per-example when categories are few."""
    categories = np.asarray(categories)
    unique = sorted({str(c) for c in categories if str(c)})
    if len(unique) >= k:
        fold_of = {
            name: int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) % k
            for name in unique
        }
        folds = np.array([fold_of[str(c)] for c in categories])
        if len(set(folds.tolist())) == k:
            return folds
    return np.arange(len(categories)) % k


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


def _selection_score(result):
    """Routing objective first; fall back to fast-correct AUROC when no escalation helped."""
    gain = result["auroc_gain_vs_helps"]
    return float(gain) if np.isfinite(gain) else float(np.nan_to_num(result["auroc_fast_correct"]))


def run_training(
    records,
    metadata,
    holdout_fraction=0.25,
    select_layers=1,
    epochs=30,
    probe_epochs=10,
    seed=0,
    lambdas=DEFAULT_LAMBDAS,
    selection_folds=4,
    pca_dim=64,
):
    """Select a feature set by grouped cross-validation over training categories,
    retrain it on all training data, report on held-out categories. Returns (head, report).

    Candidates are the stats-only head (no hidden states, the floor), one head per
    collected layer with raw hidden states, and one per layer through a whitened PCA
    bottleneck. Selection never sees the test split.
    """
    layers = tuple(metadata["feature_layers"])
    fast = np.asarray(records["fast_correct"], dtype=bool)
    slow = np.asarray(records["slow_correct"], dtype=bool)
    labels = outcome_labels(fast, slow)
    test, split_level = holdout_mask(records["category"], holdout_fraction, seed)
    train = ~test
    # Grouped folds over the training side only, for model selection.
    train_index = np.flatnonzero(train)
    folds = group_folds(np.asarray(records["category"])[train], selection_folds, seed + 1)
    base = {
        "model_id": metadata["model_id"],
        "revision": metadata["revision"],
        "hidden_size": int(metadata["hidden_size"]),
    }
    candidates = [{"name": "stats_only", "layers": (), "positions": (), "width": 32, "pca": 0}]
    for position, layer in enumerate(layers):
        candidates.append(
            {
                "name": f"layer_{layer}",
                "layers": (layer,),
                "positions": (position,),
                "width": 256,
                "pca": 0,
            }
        )
        candidates.append(
            {
                "name": f"layer_{layer}_pca{pca_dim}",
                "layers": (layer,),
                "positions": (position,),
                "width": 64,
                "pca": pca_dim,
            }
        )
    if select_layers > 1 and len(layers) > 1:
        candidates.append(
            {
                "name": "all_layers",
                "layers": layers,
                "positions": tuple(range(len(layers))),
                "width": 256,
                "pca": 0,
            }
        )
    sweep = []
    for candidate in candidates:
        meta = HeadMetadata(
            **base,
            feature_layers=candidate["layers"],
            hidden_width=candidate["width"],
            pca_dim=candidate["pca"],
        )
        features = features_for(records, candidate["positions"])
        fold_scores, fold_fast, fold_gain = [], [], []
        for fold in range(selection_folds):
            val_index = train_index[folds == fold]
            fit_index = train_index[folds != fold]
            if len(val_index) == 0 or len(fit_index) < 2:
                continue
            probe, _ = train_head(
                meta, features[fit_index], labels[fit_index], epochs=probe_epochs, seed=seed
            )
            validation = evaluate_head(
                probe, features[val_index], fast[val_index], slow[val_index], lambdas
            )
            fold_scores.append(_selection_score(validation))
            fold_fast.append(validation["auroc_fast_correct"])
            fold_gain.append(validation["auroc_gain_vs_helps"])
        probe, _ = train_head(
            meta, features[train_index], labels[train_index], epochs=probe_epochs, seed=seed
        )
        held_out = evaluate_head(probe, features[test], fast[test], slow[test], lambdas)
        sweep.append(
            {
                "name": candidate["name"],
                "layers": list(candidate["layers"]),
                "pca_dim": candidate["pca"],
                "validation": {
                    "folds": len(fold_scores),
                    "auroc_fast_correct": float(np.nanmean(fold_fast)) if fold_fast else None,
                    "auroc_gain_vs_helps": float(np.nanmean(fold_gain)) if fold_gain else None,
                },
                "test": held_out,
                "selection_score": float(np.mean(fold_scores)) if fold_scores else -1.0,
            }
        )
    best = max(sweep, key=lambda row: row["selection_score"])
    chosen = next(c for c in candidates if c["name"] == best["name"])
    chosen_layers = tuple(chosen["layers"])
    meta = HeadMetadata(
        **base, feature_layers=chosen_layers, hidden_width=chosen["width"], pca_dim=chosen["pca"]
    )
    features = features_for(records, chosen["positions"])
    head, history = train_head(meta, features[train], labels[train], epochs=epochs, seed=seed)
    final = evaluate_head(head, features[test], fast[test], slow[test], lambdas)
    stats_only = next(row for row in sweep if row["name"] == "stats_only")["test"]
    max_probability = records["fast_probability"][test]
    concentration = records["stats"][test, 3]
    report = {
        "train_count": int(train.sum()),
        "test_count": int(test.sum()),
        "holdout_fraction": holdout_fraction,
        "split_level": split_level,
        "selection_folds": selection_folds,
        "categories_seen": sorted({str(c) for c in np.asarray(records["category"])}),
        "held_out_categories": sorted({str(c) for c in np.asarray(records["category"])[test]}),
        "fast_accuracy": float(fast[test].mean()),
        "slow_accuracy": float(slow[test].mean()),
        "thinking_helps_rate": float((~fast[test] & slow[test]).mean()),
        "thinking_hurts_rate": float((fast[test] & ~slow[test]).mean()),
        "candidates": sweep,
        "chosen": best["name"],
        "chosen_layers": list(chosen_layers),
        "baseline_auroc_max_probability": auroc(max_probability, fast[test]),
        "baseline_auroc_concentration": auroc(concentration, fast[test]),
        "baseline_stats_probe": stats_only,
        "head": final,
        "loss_history": history,
    }
    head.metadata = HeadMetadata(
        **base,
        feature_layers=chosen_layers,
        hidden_width=chosen["width"],
        pca_dim=chosen["pca"],
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
