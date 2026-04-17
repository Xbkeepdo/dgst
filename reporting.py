from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class DGSTObjectRecord:
    hallucinated: int
    canonical_name: str | None
    surface: str | None
    phrase: str | None
    layer_ids: List[int]
    layer_values: List[float]
    final_score: float


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


def extract_dgst_layer_vector(item: Dict[str, Any]) -> tuple[list[int], list[float]]:
    layer_scores = item.get("dgst_layer_scores", [])
    if layer_scores:
        layer_ids = [int(layer["layer"]) for layer in layer_scores]
        layer_values = [float(layer["risk"]) for layer in layer_scores]
        return layer_ids, layer_values

    layer_values = [float(value) for value in item.get("object_layer_dgst_risk", [])]
    layer_ids = list(range(1, len(layer_values) + 1))
    return layer_ids, layer_values


def build_dgst_object_record(item: Dict[str, Any], label: int) -> DGSTObjectRecord:
    layer_ids, layer_values = extract_dgst_layer_vector(item)
    return DGSTObjectRecord(
        hallucinated=int(label),
        canonical_name=item.get("canonical_name"),
        surface=item.get("surface"),
        phrase=item.get("phrase"),
        layer_ids=layer_ids,
        layer_values=layer_values,
        final_score=float(item["dgst_final_score"]),
    )


def collect_dgst_object_records(row: Dict[str, Any]) -> List[DGSTObjectRecord]:
    records: List[DGSTObjectRecord] = []
    for item in row.get("dgst_object_scores", []):
        if "hallucinated" not in item:
            continue
        records.append(build_dgst_object_record(item, int(item["hallucinated"])))
    return records


def _select_top_categories(
    records: Sequence[DGSTObjectRecord],
    top_k: int,
    min_count: int,
) -> List[str]:
    grouped: dict[str, list[DGSTObjectRecord]] = defaultdict(list)
    for record in records:
        canonical_name = str(record.canonical_name or "").strip()
        if canonical_name:
            grouped[canonical_name].append(record)

    eligible: list[tuple[int, str]] = []
    for canonical_name, items in grouped.items():
        positives = sum(int(item.hallucinated) for item in items)
        negatives = len(items) - positives
        if len(items) < min_count or positives == 0 or negatives == 0:
            continue
        eligible.append((len(items), canonical_name))

    eligible.sort(key=lambda item: (-item[0], item[1]))
    return [canonical_name for _, canonical_name in eligible[:top_k]]


def summarize_dgst_category_metrics(
    records: Sequence[DGSTObjectRecord],
    layer_ids: Sequence[int],
    top_k: int,
    min_count: int,
) -> List[Dict[str, Any]]:
    grouped: dict[str, list[DGSTObjectRecord]] = defaultdict(list)
    for record in records:
        canonical_name = str(record.canonical_name or "").strip()
        if canonical_name:
            grouped[canonical_name].append(record)

    summaries: List[Dict[str, Any]] = []
    for canonical_name in _select_top_categories(records, top_k=top_k, min_count=min_count):
        items = grouped[canonical_name]
        labels = [int(item.hallucinated) for item in items]
        vectors = [item.layer_values for item in items]
        final_scores = [float(item.final_score) for item in items]
        summaries.append(
            {
                "canonical_name": canonical_name,
                "count": len(items),
                "positives": sum(labels),
                "negatives": len(labels) - sum(labels),
                "global_mean_metrics": summarize_directional_scores(labels, final_scores),
                "layer_wise_metrics": summarize_layerwise_scores(labels, vectors, layer_ids=layer_ids),
            }
        )
    return summaries


def summarize_dgst_dataset_metrics(
    records: Sequence[DGSTObjectRecord],
    category_top_k: int,
    category_min_count: int,
) -> Dict[str, Any]:
    if not records:
        empty_global = {
            "count": 0,
            "positives": 0,
            "negatives": 0,
            "auroc": None,
            "aupr": None,
            "direction_free_auroc": None,
            "preferred_direction": None,
            "positive_mean": None,
            "positive_median": None,
            "negative_mean": None,
            "negative_median": None,
            "average_precision": None,
        }
        return {
            "layer_ids": [],
            "global_mean_metrics": empty_global,
            "layer_wise_metrics": [],
            "best_layer_by_direction_free_auroc": None,
            "top_category_layer_wise_metrics": [],
        }

    layer_ids = records[0].layer_ids
    labels = [int(item.hallucinated) for item in records]
    final_scores = [float(item.final_score) for item in records]
    layer_vectors = [item.layer_values for item in records]

    global_mean_metrics = summarize_directional_scores(labels, final_scores)
    layer_wise_metrics = summarize_layerwise_scores(labels, layer_vectors, layer_ids=layer_ids)

    return {
        "layer_ids": list(layer_ids),
        "global_mean_metrics": global_mean_metrics,
        "layer_wise_metrics": layer_wise_metrics,
        "best_layer_by_direction_free_auroc": best_layer_by_direction_free_auroc(layer_wise_metrics),
        "top_category_layer_wise_metrics": summarize_dgst_category_metrics(
            records,
            layer_ids=layer_ids,
            top_k=category_top_k,
            min_count=category_min_count,
        ),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_all_objects_layerwise(
    layer_metrics: Sequence[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> None:
    if not layer_metrics:
        raise ValueError("Cannot plot empty layer metrics.")

    path = Path(output_path)
    _ensure_parent(path)

    layers = [int(item["layer"]) for item in layer_metrics]
    auroc = [item.get("auroc") for item in layer_metrics]
    direction_free = [item.get("direction_free_auroc") for item in layer_metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, auroc, marker="o", linewidth=2, label="AUROC")
    plt.plot(layers, direction_free, marker="s", linewidth=2, label="Direction-free AUROC")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(layers)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_category_layerwise(
    category_metrics: Sequence[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> None:
    if not category_metrics:
        raise ValueError("Cannot plot empty category metrics.")

    path = Path(output_path)
    _ensure_parent(path)

    plt.figure(figsize=(11, 6))
    for entry in category_metrics:
        metrics = entry.get("layer_wise_metrics", [])
        if not metrics:
            continue
        layers = [int(item["layer"]) for item in metrics]
        scores = [item.get("direction_free_auroc") for item in metrics]
        label = f"{entry['canonical_name']} (n={entry['count']})"
        plt.plot(layers, scores, marker="o", linewidth=2, label=label)

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Layer")
    plt.ylabel("Direction-free AUROC")
    plt.title(title)
    if category_metrics and category_metrics[0].get("layer_wise_metrics"):
        plt.xticks([int(item["layer"]) for item in category_metrics[0]["layer_wise_metrics"]])
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    auroc: float,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    _ensure_parent(path)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}", linewidth=2.0)
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.2, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Probe ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_pr_curve(
    recall: Sequence[float],
    precision: Sequence[float],
    aupr: float,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    _ensure_parent(path)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPR = {aupr:.4f}", linewidth=2.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Probe Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
