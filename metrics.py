from __future__ import annotations

# 轻量二分类指标实现，供 raw-score 层分析使用。

from statistics import median
from typing import Any, Dict, Sequence


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores)]
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(label) for label in labels)
    if positives == 0:
        return None

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    ap = 0.0
    for score, label in pairs:
        if int(label) == 1:
            tp += 1
            ap += tp / max(tp + fp, 1)
        else:
            fp += 1
    return float(ap / positives)


def best_f1(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float | None]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(int(label) for _, label in pairs)
    if positives == 0:
        return {"best_f1": None, "best_threshold": None, "precision": None, "recall": None}

    tp = 0
    fp = 0
    best = {"best_f1": 0.0, "best_threshold": None, "precision": 0.0, "recall": 0.0}
    for score, label in pairs:
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / positives
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        if f1 > best["best_f1"]:
            best = {
                "best_f1": float(f1),
                "best_threshold": float(score),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def direction_free_auroc(auroc: float | None) -> float | None:
    if auroc is None:
        return None
    # 这里只用于 raw-score 分析：
    # 它衡量的是“能不能分开”，不预先假设高分还是低分对应幻觉。
    return float(max(auroc, 1.0 - auroc))


def preferred_direction(auroc: float | None) -> str | None:
    if auroc is None:
        return None
    if auroc >= 0.5:
        return "higher_is_hallucination"
    return "lower_is_hallucination"


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def summarize_binary_scores(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    labels = [int(label) for label in labels]
    scores = [float(score) for score in scores]
    ap = average_precision(labels, scores)
    return {
        "count": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "auroc": binary_auroc(labels, scores),
        "aupr": ap,
        "average_precision": ap,
        **best_f1(labels, scores),
    }


def summarize_directional_scores(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Any]:
    labels = [int(label) for label in labels]
    scores = [float(score) for score in scores]
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    auroc = binary_auroc(labels, scores)
    ap = average_precision(labels, scores)
    return {
        "count": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "auroc": auroc,
        "aupr": ap,
        "direction_free_auroc": direction_free_auroc(auroc),
        "preferred_direction": preferred_direction(auroc),
        "positive_mean": _mean(positives),
        "positive_median": _median(positives),
        "negative_mean": _mean(negatives),
        "negative_median": _median(negatives),
        "average_precision": ap,
    }


def summarize_layerwise_scores(
    labels: Sequence[int],
    score_vectors: Sequence[Sequence[float]],
    layer_ids: Sequence[int] | None = None,
) -> list[Dict[str, Any]]:
    if not score_vectors:
        return []

    width = len(score_vectors[0])
    for vector in score_vectors:
        if len(vector) != width:
            raise ValueError("All score vectors must have the same layer width.")

    if layer_ids is None:
        layer_ids = list(range(1, width + 1))
    if len(layer_ids) != width:
        raise ValueError("Layer ids and score vector width must match.")

    summaries: list[Dict[str, Any]] = []
    for layer_offset, layer_id in enumerate(layer_ids):
        # 每一层都单独当成一个标量检测器来评估，
        # 这样在上 probe 之前就能先看出哪几层的迁移信号最强。
        layer_scores = [float(vector[layer_offset]) for vector in score_vectors]
        summary = summarize_directional_scores(labels, layer_scores)
        summary["layer"] = int(layer_id)
        summaries.append(summary)
    return summaries


def best_layer_by_direction_free_auroc(layer_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    ranked = [item for item in layer_summaries if item.get("direction_free_auroc") is not None]
    if not ranked:
        return None
    best = max(
        ranked,
        key=lambda item: (
            float(item["direction_free_auroc"]),
            -float(item["layer"]),
        ),
    )
    return dict(best)
