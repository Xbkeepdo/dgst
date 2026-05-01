from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .reporting import build_dgst_object_record

PROBE_FEATURE_SET_RISK = "risk"
PROBE_FEATURE_SET_PROBE6 = "probe6"
PROBE_FEATURE_SET_PROMPT = "prompt"
PROBE_FEATURE_SET_RISK_PROMPT = "risk_prompt"
PROBE_FEATURE_SET_PROBE6_PROMPT = "probe6_prompt"
PROBE_FEATURE_SET_CHOICES = (
    PROBE_FEATURE_SET_RISK,
    PROBE_FEATURE_SET_PROBE6,
    PROBE_FEATURE_SET_PROMPT,
    PROBE_FEATURE_SET_RISK_PROMPT,
    PROBE_FEATURE_SET_PROBE6_PROMPT,
)
PROBE_SPLIT_BY_GROUP = "group"
PROBE_SPLIT_BY_RESPONSE = "response"
PROBE_SPLIT_BY_IMAGE = "image"
PROBE_SPLIT_BY_SAMPLE = "sample"
PROBE_SPLIT_BY_CHOICES = (
    PROBE_SPLIT_BY_GROUP,
    PROBE_SPLIT_BY_RESPONSE,
    PROBE_SPLIT_BY_IMAGE,
    PROBE_SPLIT_BY_SAMPLE,
)
_PROBE_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?:\n\s*)+|(?<=[.!?])\s+")


@dataclass
class DGSTProbeSample:
    sample_id: str
    image_id: int
    image: str | None
    image_path: str | None
    caption: str | None
    canonical_name: str | None
    surface: str | None
    phrase: str | None
    source_surface_word: str | None
    hallucinated: int
    word_index: int | None
    alignment_strategy: str | None
    alignment_status: str | None
    token_aligned: bool
    layer_ids: List[int]
    object_layer_dgst_risk: List[float]
    dgst_probe6_feature_names: List[str]
    dgst_probe6_layer_stats: List[Dict[str, Any]]
    object_layer_dgst_probe6: List[float]
    dgst_final_score: float
    dgst_prompt_feature_names: List[str] = field(default_factory=list)
    dgst_prompt_layer_stats: List[Dict[str, Any]] = field(default_factory=list)
    object_layer_prompt_token_cos: List[float] = field(default_factory=list)
    split_group_id: int | None = None
    target_aggregation: str = "mean"
    metadata: Dict[str, Any] = field(default_factory=dict)


def dgst_probe_sample_to_dict(sample: DGSTProbeSample) -> Dict[str, Any]:
    return asdict(sample)


def write_dgst_probe_dataset(path: Path, samples: Sequence[DGSTProbeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(dgst_probe_sample_to_dict(sample), ensure_ascii=False) + "\n")


def read_dgst_probe_dataset(path: Path) -> List[DGSTProbeSample]:
    samples: List[DGSTProbeSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(
                DGSTProbeSample(
                    sample_id=str(payload["sample_id"]),
                    image_id=int(payload["image_id"]),
                    image=payload.get("image"),
                    image_path=payload.get("image_path"),
                    caption=payload.get("caption"),
                    canonical_name=payload.get("canonical_name"),
                    surface=payload.get("surface"),
                    phrase=payload.get("phrase"),
                    source_surface_word=payload.get("source_surface_word"),
                    hallucinated=int(payload["hallucinated"]),
                    word_index=int(payload["word_index"]) if payload.get("word_index") is not None else None,
                    alignment_strategy=payload.get("alignment_strategy"),
                    alignment_status=payload.get("alignment_status"),
                    token_aligned=bool(payload.get("token_aligned", True)),
                    layer_ids=[int(value) for value in payload["layer_ids"]],
                    object_layer_dgst_risk=[float(value) for value in payload["object_layer_dgst_risk"]],
                    dgst_probe6_feature_names=[str(value) for value in payload.get("dgst_probe6_feature_names", [])],
                    dgst_probe6_layer_stats=list(payload.get("dgst_probe6_layer_stats", [])),
                    object_layer_dgst_probe6=[float(value) for value in payload.get("object_layer_dgst_probe6", [])],
                    dgst_final_score=float(payload["dgst_final_score"]),
                    dgst_prompt_feature_names=[str(value) for value in payload.get("dgst_prompt_feature_names", [])],
                    dgst_prompt_layer_stats=list(payload.get("dgst_prompt_layer_stats", [])),
                    object_layer_prompt_token_cos=[
                        float(value) for value in payload.get("object_layer_prompt_token_cos", [])
                    ],
                    split_group_id=(
                        int(payload["split_group_id"]) if payload.get("split_group_id") is not None else None
                    ),
                    target_aggregation=str(payload.get("target_aggregation") or "mean"),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
    return samples


def build_dgst_probe_samples_from_row(row: Dict[str, Any]) -> List[DGSTProbeSample]:
    samples: List[DGSTProbeSample] = []
    image_id = int(row["image_id"])
    metadata = {
        str(key): value
        for key, value in row.items()
        if str(key).startswith(("amber_", "mhaldetect_", "pope_", "coco_"))
    }
    score_items: List[Dict[str, Any]] = []
    score_items.extend(dict(item) for item in row.get("dgst_object_scores", []))
    score_items.extend(dict(item) for item in row.get("dgst_sentence_scores", []))
    for item in score_items:
        if "hallucinated" not in item:
            continue
        item_kind = str(item.get("kind") or "object")
        mention_index = int(item.get("sentence_index", item.get("mention_index", len(samples))))
        sample_id = f"{image_id}:sentence:{mention_index}" if item_kind == "sentence" else f"{image_id}:{mention_index}"
        record = build_dgst_object_record(item, int(item["hallucinated"]))
        item_metadata = {
            str(key): value
            for key, value in item.items()
            if str(key).startswith("sentence_") or str(key) in {"kind", "label_rule", "aggregation_unit"}
        }
        samples.append(
            DGSTProbeSample(
                sample_id=sample_id,
                image_id=image_id,
                image=row.get("image"),
                image_path=row.get("image_path"),
                caption=row.get("caption"),
                canonical_name=record.canonical_name,
                surface=record.surface,
                phrase=record.phrase,
                source_surface_word=item.get("source_surface_word"),
                hallucinated=record.hallucinated,
                word_index=int(item["word_index"]) if item.get("word_index") is not None else None,
                alignment_strategy=item.get("alignment_strategy"),
                alignment_status=item.get("alignment_status"),
                token_aligned=bool(item.get("token_aligned", True)),
                layer_ids=list(record.layer_ids),
                object_layer_dgst_risk=list(record.layer_values),
                dgst_probe6_feature_names=[str(value) for value in item.get("dgst_probe6_feature_names", [])],
                dgst_probe6_layer_stats=list(item.get("dgst_probe6_layer_stats", [])),
                object_layer_dgst_probe6=[float(value) for value in item.get("object_layer_dgst_probe6", [])],
                dgst_final_score=float(record.final_score),
                dgst_prompt_feature_names=[str(value) for value in item.get("dgst_prompt_feature_names", [])],
                dgst_prompt_layer_stats=list(item.get("dgst_prompt_layer_stats", [])),
                object_layer_prompt_token_cos=[float(value) for value in item.get("object_layer_prompt_token_cos", [])],
                split_group_id=int(row["split_group_id"]) if row.get("split_group_id") is not None else None,
                target_aggregation=str(item.get("target_aggregation") or row.get("target_aggregation") or "mean"),
                metadata={**metadata, **item_metadata},
            )
        )
    return samples


def build_dgst_probe_samples(rows: Iterable[Dict[str, Any]]) -> List[DGSTProbeSample]:
    samples: List[DGSTProbeSample] = []
    for row in rows:
        samples.extend(build_dgst_probe_samples_from_row(row))
    return samples


def summarize_dgst_probe_samples(samples: Sequence[DGSTProbeSample]) -> Dict[str, Any]:
    positives = sum(int(sample.hallucinated) for sample in samples)
    image_ids = {int(sample.image_id) for sample in samples}
    split_group_ids = {_split_group_id(sample) for sample in samples}
    layer_widths = sorted({len(sample.object_layer_dgst_risk) for sample in samples})
    probe6_widths = sorted({len(sample.object_layer_dgst_probe6) for sample in samples if sample.object_layer_dgst_probe6})
    prompt_widths = sorted(
        {len(sample.object_layer_prompt_token_cos) for sample in samples if sample.object_layer_prompt_token_cos}
    )
    token_aligned = sum(int(sample.token_aligned) for sample in samples)
    return {
        "count": len(samples),
        "positives": positives,
        "negatives": len(samples) - positives,
        "images": len(image_ids),
        "split_groups": len(split_group_ids),
        "target_aggregations": sorted({str(sample.target_aggregation) for sample in samples}),
        "layer_widths": layer_widths,
        "probe6_widths": probe6_widths,
        "prompt_widths": prompt_widths,
        "token_aligned": token_aligned,
    }


def _caption_sentence_spans(caption: str | None) -> List[Dict[str, Any]]:
    text = str(caption or "").strip()
    if not text:
        return [{"index": 0, "start_token": 0, "end_token": 1, "char_start": 0, "char_end": 0, "text": ""}]

    spans: List[Dict[str, Any]] = []
    token_cursor = 0
    part_start = 0
    boundaries = list(_SENTENCE_SPLIT_RE.finditer(text))
    parts = []
    for boundary in boundaries:
        parts.append((part_start, boundary.start()))
        part_start = boundary.end()
    parts.append((part_start, len(text)))

    for start, end in parts:
        raw_sentence = text[start:end]
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        leading = len(raw_sentence) - len(raw_sentence.lstrip())
        trailing = len(raw_sentence) - len(raw_sentence.rstrip())
        char_start = start + leading
        char_end = end - trailing
        token_count = len(_PROBE_WORD_RE.findall(sentence.lower()))
        if token_count <= 0:
            continue
        spans.append(
            {
                "index": len(spans),
                "start_token": token_cursor,
                "end_token": token_cursor + token_count,
                "char_start": int(char_start),
                "char_end": int(char_end),
                "text": sentence,
            }
        )
        token_cursor += token_count

    if not spans:
        token_count = max(1, len(_PROBE_WORD_RE.findall(text.lower())))
        spans.append({"index": 0, "start_token": 0, "end_token": token_count, "char_start": 0, "char_end": len(text), "text": text})
    return spans


def _sentence_for_word_index(word_index: int | None, spans: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not spans:
        return {"index": 0, "start_token": 0, "end_token": 1, "text": ""}
    if word_index is None:
        return spans[0]
    token_index = int(word_index)
    for span in spans:
        if int(span["start_token"]) <= token_index < int(span["end_token"]):
            return span
    return spans[-1] if token_index >= int(spans[-1]["end_token"]) else spans[0]


def build_sentence_entries_from_mentions(
    caption: str | None,
    object_mentions: Sequence[Dict[str, Any]],
    *,
    include_empty_sentences: bool = True,
) -> List[Dict[str, Any]]:
    spans = _caption_sentence_spans(caption)
    grouped: Dict[int, List[Dict[str, Any]]] = {int(span["index"]): [] for span in spans}
    for mention in object_mentions:
        sentence = _sentence_for_word_index(
            int(mention["word_index"]) if mention.get("word_index") is not None else None,
            spans,
        )
        grouped.setdefault(int(sentence["index"]), []).append(dict(mention))

    entries: List[Dict[str, Any]] = []
    for span in spans:
        sentence_index = int(span["index"])
        mentions = grouped.get(sentence_index, [])
        if not mentions and not include_empty_sentences:
            continue
        hallucinated_mentions = [mention for mention in mentions if int(mention.get("hallucinated", 0))]
        surfaces = [str(mention.get("surface") or mention.get("phrase") or "") for mention in mentions]
        canonicals = [str(mention.get("canonical_name") or "") for mention in mentions]
        sentence_text = str(span["text"])
        entries.append(
            {
                "kind": "sentence",
                "phrase": sentence_text,
                "surface": sentence_text,
                "surface_word": sentence_text,
                "canonical_name": "sentence",
                "mention_index": sentence_index,
                "word_index": int(span["start_token"]),
                "hallucinated": int(bool(hallucinated_mentions)),
                "sentence_index": sentence_index,
                "sentence_text": sentence_text,
                "sentence_start_token": int(span["start_token"]),
                "sentence_end_token": int(span["end_token"]),
                "sentence_char_start": int(span["char_start"]),
                "sentence_char_end": int(span["char_end"]),
                "sentence_mention_count": int(len(mentions)),
                "sentence_hallucinated_mention_count": int(len(hallucinated_mentions)),
                "sentence_surfaces": surfaces,
                "sentence_canonical_names": canonicals,
            }
        )
    return entries


def _aggregate_vectors(vectors: Sequence[Sequence[float]], aggregation: str) -> List[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("Cannot aggregate vectors with different widths.")
    if aggregation == "mean":
        return [float(sum(float(vector[index]) for vector in vectors) / len(vectors)) for index in range(width)]
    if aggregation == "max":
        return [float(max(float(vector[index]) for vector in vectors)) for index in range(width)]
    raise ValueError(f"Unsupported sentence feature aggregation: {aggregation}")


def _aggregate_layer_stats(layer_stats: Sequence[Sequence[Dict[str, Any]]], aggregation: str) -> List[Dict[str, Any]]:
    if not layer_stats:
        return []
    width = len(layer_stats[0])
    if any(len(stats) != width for stats in layer_stats):
        return [dict(item) for item in layer_stats[0]]
    aggregated: List[Dict[str, Any]] = []
    for layer_index in range(width):
        keys = sorted({key for stats in layer_stats for key in stats[layer_index].keys()})
        row: Dict[str, Any] = {}
        for key in keys:
            values = [stats[layer_index].get(key) for stats in layer_stats]
            if all(isinstance(value, (int, float)) for value in values):
                numeric_values = [float(value) for value in values]
                row[key] = (
                    float(sum(numeric_values) / len(numeric_values))
                    if aggregation == "mean"
                    else float(max(numeric_values))
                )
            else:
                row[key] = values[0]
        aggregated.append(row)
    return aggregated


def aggregate_dgst_probe_samples_by_sentence(
    samples: Sequence[DGSTProbeSample],
    *,
    feature_aggregation: str = "mean",
) -> List[DGSTProbeSample]:
    if feature_aggregation not in {"mean", "max"}:
        raise ValueError("feature_aggregation must be either 'mean' or 'max'.")

    caption_spans: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
    grouped: Dict[tuple[int, int, str], List[DGSTProbeSample]] = {}
    sentence_meta: Dict[tuple[int, int, str], Dict[str, Any]] = {}

    for sample in samples:
        caption = sample.caption or ""
        caption_key = (int(sample.image_id), caption)
        spans = caption_spans.setdefault(caption_key, _caption_sentence_spans(caption))
        sentence = _sentence_for_word_index(sample.word_index, spans)
        key = (int(sample.image_id), int(sentence["index"]), caption)
        grouped.setdefault(key, []).append(sample)
        sentence_meta[key] = sentence

    sentence_samples: List[DGSTProbeSample] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2])):
        group = grouped[key]
        first = group[0]
        sentence = sentence_meta[key]
        sentence_index = int(sentence["index"])
        hallucinated = int(any(int(sample.hallucinated) for sample in group))
        surfaces = [str(sample.surface or sample.phrase or sample.canonical_name or "") for sample in group]
        canonicals = [str(sample.canonical_name or "") for sample in group]
        metadata = dict(first.metadata)
        metadata.update(
            {
                "aggregation_unit": "sentence",
                "feature_aggregation": feature_aggregation,
                "label_rule": "positive_if_any_object_mention_hallucinated",
                "source_caption": first.caption,
                "sentence_index": sentence_index,
                "sentence_text": sentence["text"],
                "sentence_start_token": int(sentence["start_token"]),
                "sentence_end_token": int(sentence["end_token"]),
                "sentence_mention_count": int(len(group)),
                "sentence_hallucinated_mention_count": int(sum(int(sample.hallucinated) for sample in group)),
                "sentence_surfaces": surfaces,
                "sentence_canonical_names": canonicals,
                "source_sample_ids": [sample.sample_id for sample in group],
            }
        )
        sentence_samples.append(
            DGSTProbeSample(
                sample_id=f"{first.image_id}:sent{sentence_index}",
                image_id=int(first.image_id),
                image=first.image,
                image_path=first.image_path,
                caption=str(sentence["text"]),
                canonical_name="; ".join(dict.fromkeys(value for value in canonicals if value)) or "sentence",
                surface=str(sentence["text"]),
                phrase=str(sentence["text"]),
                source_surface_word="; ".join(value for value in surfaces if value) or None,
                hallucinated=hallucinated,
                word_index=int(sentence["start_token"]),
                alignment_strategy=f"sentence_object_mention_{feature_aggregation}",
                alignment_status="sentence_aggregated",
                token_aligned=all(bool(sample.token_aligned) for sample in group),
                layer_ids=list(first.layer_ids),
                object_layer_dgst_risk=_aggregate_vectors(
                    [sample.object_layer_dgst_risk for sample in group],
                    feature_aggregation,
                ),
                dgst_probe6_feature_names=list(first.dgst_probe6_feature_names),
                dgst_probe6_layer_stats=_aggregate_layer_stats(
                    [sample.dgst_probe6_layer_stats for sample in group if sample.dgst_probe6_layer_stats],
                    feature_aggregation,
                ),
                object_layer_dgst_probe6=_aggregate_vectors(
                    [sample.object_layer_dgst_probe6 for sample in group if sample.object_layer_dgst_probe6],
                    feature_aggregation,
                ),
                dgst_final_score=float(
                    _aggregate_vectors([[sample.dgst_final_score] for sample in group], feature_aggregation)[0]
                ),
                dgst_prompt_feature_names=list(first.dgst_prompt_feature_names),
                dgst_prompt_layer_stats=_aggregate_layer_stats(
                    [sample.dgst_prompt_layer_stats for sample in group if sample.dgst_prompt_layer_stats],
                    feature_aggregation,
                ),
                object_layer_prompt_token_cos=_aggregate_vectors(
                    [sample.object_layer_prompt_token_cos for sample in group if sample.object_layer_prompt_token_cos],
                    feature_aggregation,
                ),
                split_group_id=int(first.image_id),
                target_aggregation=f"sentence_{feature_aggregation}",
                metadata=metadata,
            )
        )
    return sentence_samples


def _require_prompt_features(sample: DGSTProbeSample) -> List[float]:
    if not sample.object_layer_prompt_token_cos:
        raise ValueError(
            "Probe sample does not contain object_layer_prompt_token_cos. "
            "Re-export the probe dataset after enabling prompt-aware feature export."
        )
    return list(sample.object_layer_prompt_token_cos)


def _require_probe6_features(sample: DGSTProbeSample) -> List[float]:
    if not sample.object_layer_dgst_probe6:
        raise ValueError(
            "Probe sample does not contain object_layer_dgst_probe6. "
            "Re-export the probe dataset after enabling the probe6 feature implementation."
        )
    return list(sample.object_layer_dgst_probe6)


def get_probe_feature_vector(sample: DGSTProbeSample, feature_set: str) -> List[float]:
    if feature_set == PROBE_FEATURE_SET_RISK:
        return list(sample.object_layer_dgst_risk)
    if feature_set == PROBE_FEATURE_SET_PROBE6:
        return _require_probe6_features(sample)
    if feature_set == PROBE_FEATURE_SET_PROMPT:
        return _require_prompt_features(sample)
    if feature_set == PROBE_FEATURE_SET_RISK_PROMPT:
        return list(sample.object_layer_dgst_risk) + _require_prompt_features(sample)
    if feature_set == PROBE_FEATURE_SET_PROBE6_PROMPT:
        return _require_probe6_features(sample) + _require_prompt_features(sample)
    raise ValueError(f"Unsupported probe feature set: {feature_set}")


def _split_group_id(sample: DGSTProbeSample) -> int:
    return int(sample.split_group_id) if sample.split_group_id is not None else int(sample.image_id)


def _source_image_key(sample: DGSTProbeSample) -> str:
    return str(sample.image_path or sample.image or sample.image_id)


def _split_key(sample: DGSTProbeSample, split_by: str) -> str:
    if split_by == PROBE_SPLIT_BY_GROUP:
        return str(_split_group_id(sample))
    if split_by == PROBE_SPLIT_BY_RESPONSE:
        return str(sample.image_id)
    if split_by == PROBE_SPLIT_BY_IMAGE:
        return _source_image_key(sample)
    if split_by == PROBE_SPLIT_BY_SAMPLE:
        return str(sample.sample_id)
    raise ValueError(f"Unsupported probe split_by: {split_by}")


def split_dgst_probe_samples_by_image(
    samples: Sequence[DGSTProbeSample],
    test_size: float,
    seed: int,
    split_by: str = PROBE_SPLIT_BY_GROUP,
) -> tuple[List[DGSTProbeSample], List[DGSTProbeSample], Dict[str, Any]]:
    grouped: dict[str, List[DGSTProbeSample]] = {}
    for sample in samples:
        grouped.setdefault(_split_key(sample, split_by), []).append(sample)

    if len(grouped) < 2:
        raise ValueError("At least two split groups are required to split DGST probe data.")

    split_keys = sorted(grouped.keys())
    image_labels = [int(any(sample.hallucinated for sample in grouped[key])) for key in split_keys]

    positives = sum(image_labels)
    negatives = len(image_labels) - positives
    stratify = image_labels if positives >= 2 and negatives >= 2 else None

    train_ids, val_ids = train_test_split(
        split_keys,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_id_set = set(str(value) for value in train_ids)
    val_id_set = set(str(value) for value in val_ids)

    train_samples = [sample for sample in samples if _split_key(sample, split_by) in train_id_set]
    val_samples = [sample for sample in samples if _split_key(sample, split_by) in val_id_set]
    if not train_samples or not val_samples:
        raise ValueError("The DGST probe split produced an empty train or validation set.")

    split_summary = {
        "seed": int(seed),
        "test_size": float(test_size),
        "split_by": split_by,
        "stratified_by_image_has_hallucination": bool(stratify is not None),
        "train_split_keys": sorted(train_id_set),
        "val_split_keys": sorted(val_id_set),
        "train_image_ids": sorted(train_id_set),
        "val_image_ids": sorted(val_id_set),
        "train_image_count": len(train_id_set),
        "val_image_count": len(val_id_set),
        "train_sample_summary": summarize_dgst_probe_samples(train_samples),
        "val_sample_summary": summarize_dgst_probe_samples(val_samples),
    }
    return train_samples, val_samples, split_summary


def split_dgst_probe_samples_with_fixed_image_ids(
    samples: Sequence[DGSTProbeSample],
    train_image_ids: Sequence[Any],
    val_image_ids: Sequence[Any],
    split_by: str = PROBE_SPLIT_BY_GROUP,
) -> tuple[List[DGSTProbeSample], List[DGSTProbeSample], Dict[str, Any]]:
    train_id_set = {str(value) for value in train_image_ids}
    val_id_set = {str(value) for value in val_image_ids}
    if not train_id_set or not val_id_set:
        raise ValueError("Both train_image_ids and val_image_ids must be non-empty.")
    overlap = train_id_set & val_id_set
    if overlap:
        raise ValueError(f"Train/val image ids overlap: {len(overlap)} images.")

    image_ids_in_dataset = {_split_key(sample, split_by) for sample in samples}
    missing_train = sorted(train_id_set - image_ids_in_dataset)
    missing_val = sorted(val_id_set - image_ids_in_dataset)
    if missing_train:
        train_id_set -= set(missing_train)
    if missing_val:
        val_id_set -= set(missing_val)
    if not train_id_set or not val_id_set:
        raise ValueError(
            "After dropping absent image ids from the fixed split, train or validation ids became empty."
        )

    train_samples = [sample for sample in samples if _split_key(sample, split_by) in train_id_set]
    val_samples = [sample for sample in samples if _split_key(sample, split_by) in val_id_set]
    if not train_samples or not val_samples:
        raise ValueError("The fixed DGST split produced an empty train or validation set.")

    split_summary = {
        "split_source": "fixed_image_ids",
        "split_by": split_by,
        "missing_train_image_ids": missing_train,
        "missing_val_image_ids": missing_val,
        "train_split_keys": sorted(train_id_set),
        "val_split_keys": sorted(val_id_set),
        "train_image_ids": sorted(train_id_set),
        "val_image_ids": sorted(val_id_set),
        "train_image_count": len(train_id_set),
        "val_image_count": len(val_id_set),
        "train_sample_summary": summarize_dgst_probe_samples(train_samples),
        "val_sample_summary": summarize_dgst_probe_samples(val_samples),
    }
    return train_samples, val_samples, split_summary


class DGSTProbeDataset(Dataset):
    def __init__(self, samples: Sequence[DGSTProbeSample], feature_set: str = PROBE_FEATURE_SET_RISK):
        if not samples:
            raise ValueError("DGSTProbeDataset requires at least one sample.")
        first_vector = get_probe_feature_vector(samples[0], feature_set)
        width = len(first_vector)
        for sample in samples:
            if len(get_probe_feature_vector(sample, feature_set)) != width:
                raise ValueError("All DGST probe samples must share the same feature width.")
        self.samples = list(samples)
        self.feature_set = str(feature_set)
        self.features = torch.tensor(
            [get_probe_feature_vector(sample, feature_set) for sample in samples],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([sample.hallucinated for sample in samples], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class DGSTProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any():
            raise ValueError("DGST probe input contains NaN values.")
        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        out = self.dropout1(out)
        out = F.leaky_relu(self.bn2(self.fc2(out)), negative_slope=0.01)
        out = self.dropout2(out)
        out = F.leaky_relu(self.bn3(self.fc3(out)), negative_slope=0.01)
        out = self.dropout3(out)
        return self.fc4(out)


@dataclass
class DGSTProbeConfig:
    input_dim: int
    feature_set: str = PROBE_FEATURE_SET_RISK
    positive_class: str = "hallucination"
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 5
    seed: int = 0


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    train_samples: Sequence[DGSTProbeSample],
    val_samples: Sequence[DGSTProbeSample],
    batch_size: int,
    feature_set: str = PROBE_FEATURE_SET_RISK,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = DGSTProbeDataset(train_samples, feature_set=feature_set)
    val_dataset = DGSTProbeDataset(val_samples, feature_set=feature_set)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _train_epoch(
    model: DGSTProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / len(losses)) if losses else 0.0


def _validate_epoch(
    model: DGSTProbe,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    hallucination_probabilities: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            logits = model(features)
            loss = criterion(logits, batch_labels)
            losses.append(float(loss.item()))
            hallucination_probabilities.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
            labels.extend(int(value) for value in batch_labels.squeeze(1).cpu().tolist())

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Validation split must contain both positive and negative samples.")

    roc_auc = float(roc_auc_score(labels, hallucination_probabilities))
    aupr = float(average_precision_score(labels, hallucination_probabilities))
    fpr, tpr, roc_thresholds = roc_curve(labels, hallucination_probabilities)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, hallucination_probabilities)
    return {
        "val_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "auroc": roc_auc,
        "aupr": aupr,
        "labels": labels,
        "hallucination_probabilities": hallucination_probabilities,
        "roc_curve": {
            "fpr": [float(value) for value in fpr.tolist()],
            "tpr": [float(value) for value in tpr.tolist()],
            "thresholds": [float(value) for value in roc_thresholds.tolist()],
        },
        "pr_curve": {
            "precision": [float(value) for value in pr_precision.tolist()],
            "recall": [float(value) for value in pr_recall.tolist()],
            "thresholds": [float(value) for value in pr_thresholds.tolist()],
        },
    }


def compute_subset_macro_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    subsets: Sequence[str | None],
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[float] | List[int]]] = {}
    for label, score, subset in zip(labels, scores, subsets):
        if subset is None or not str(subset).strip():
            continue
        key = str(subset)
        bucket = grouped.setdefault(key, {"labels": [], "scores": []})
        bucket["labels"].append(int(label))  # type: ignore[union-attr]
        bucket["scores"].append(float(score))  # type: ignore[union-attr]

    per_subset: Dict[str, Dict[str, Any]] = {}
    aurocs: List[float] = []
    auprs: List[float] = []
    for subset, bucket in sorted(grouped.items()):
        subset_labels = [int(value) for value in bucket["labels"]]  # type: ignore[index]
        subset_scores = [float(value) for value in bucket["scores"]]  # type: ignore[index]
        positives = int(sum(subset_labels))
        negatives = int(len(subset_labels) - positives)
        if positives == 0 or negatives == 0:
            continue
        auroc = float(roc_auc_score(subset_labels, subset_scores))
        aupr = float(average_precision_score(subset_labels, subset_scores))
        per_subset[subset] = {
            "count": int(len(subset_labels)),
            "positives": positives,
            "negatives": negatives,
            "auroc": auroc,
            "aupr": aupr,
        }
        aurocs.append(auroc)
        auprs.append(aupr)

    if not per_subset:
        return {}
    return {
        "macro_auroc": float(sum(aurocs) / len(aurocs)),
        "macro_aupr": float(sum(auprs) / len(auprs)),
        "subset_count": int(len(per_subset)),
        "subsets": per_subset,
    }


def train_dgst_probe(
    train_samples: Sequence[DGSTProbeSample],
    val_samples: Sequence[DGSTProbeSample],
    config: DGSTProbeConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    if not train_samples or not val_samples:
        raise ValueError("DGST probe training requires non-empty train and validation samples.")

    set_torch_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_data_loaders(
        train_samples,
        val_samples,
        batch_size=config.batch_size,
        feature_set=config.feature_set,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGSTProbe(input_dim=config.input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: Dict[str, Any] | None = None
    history: List[Dict[str, float]] = []

    epoch_progress = tqdm(range(config.num_epochs), desc="Training DGST probe", unit="epoch", leave=True)
    for epoch in epoch_progress:
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["val_loss"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["val_loss"]),
                "auroc": float(val_metrics["auroc"]),
                "aupr": float(val_metrics["aupr"]),
            }
        )
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{float(val_metrics['val_loss']):.4f}",
            auroc=f"{float(val_metrics['auroc']):.4f}",
            aupr=f"{float(val_metrics['aupr']):.4f}",
        )
        if float(val_metrics["val_loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["val_loss"])
            best_epoch = int(epoch)
            best_metrics = dict(val_metrics)
            torch.save(model.state_dict(), output_dir / "model.pth")

    epoch_progress.close()
    if best_metrics is None:
        raise RuntimeError("DGST probe training finished without validation metrics.")

    amber_subset_macro = compute_subset_macro_metrics(
        [int(sample.hallucinated) for sample in val_samples],
        [float(value) for value in best_metrics["hallucination_probabilities"]],
        [
            str(sample.metadata.get("amber_discriminative_type"))
            if sample.metadata.get("amber_discriminative_type") is not None
            else None
            for sample in val_samples
        ],
    )
    if amber_subset_macro:
        best_metrics["amber_subset_macro"] = amber_subset_macro

    pope_subset_macro = compute_subset_macro_metrics(
        [int(sample.hallucinated) for sample in val_samples],
        [float(value) for value in best_metrics["hallucination_probabilities"]],
        [
            str(sample.metadata.get("pope_subset"))
            if sample.metadata.get("pope_subset") is not None
            else None
            for sample in val_samples
        ],
    )
    if pope_subset_macro:
        best_metrics["pope_subset_macro"] = pope_subset_macro

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "best_metrics": best_metrics,
        "device": str(device),
    }


def build_dgst_val_prediction_rows(
    val_samples: Sequence[DGSTProbeSample],
    hallucination_probabilities: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample, probability in zip(val_samples, hallucination_probabilities):
        row = {
            "sample_id": sample.sample_id,
            "image_id": int(sample.image_id),
            "split_group_id": _split_group_id(sample),
            "canonical_name": sample.canonical_name,
            "surface": sample.surface,
            "phrase": sample.phrase,
            "hallucinated": int(sample.hallucinated),
            "hallucination_label": int(sample.hallucinated),
            "hallucination_probability": float(probability),
            "non_hallucination_probability": float(1.0 - probability),
            "dgst_final_score": float(sample.dgst_final_score),
        }
        row.update(sample.metadata)
        rows.append(row)
    return rows
