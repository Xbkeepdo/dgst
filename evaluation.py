from __future__ import annotations

# DGST 对象级评测记录与 dataset/category 汇总逻辑。

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .metrics import (
    best_layer_by_direction_free_auroc,
    summarize_directional_scores,
    summarize_layerwise_scores,
)


@dataclass
class DGSTObjectRecord:
    hallucinated: int
    canonical_name: str | None
    surface: str | None
    phrase: str | None
    layer_ids: List[int]
    layer_values: List[float]
    final_score: float


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
