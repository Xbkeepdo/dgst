from __future__ import annotations

# DGST 对象级 probe 样本的数据结构、读写和按 image 划分逻辑。

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sklearn.model_selection import train_test_split

from .evaluation import build_dgst_object_record


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
    dgst_final_score: float


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
                    dgst_final_score=float(payload["dgst_final_score"]),
                )
            )
    return samples


def build_dgst_probe_samples_from_row(row: Dict[str, Any]) -> List[DGSTProbeSample]:
    samples: List[DGSTProbeSample] = []
    image_id = int(row["image_id"])
    for item in row.get("dgst_object_scores", []):
        if "hallucinated" not in item:
            continue
        mention_index = int(item.get("mention_index", len(samples)))
        record = build_dgst_object_record(item, int(item["hallucinated"]))
        samples.append(
            DGSTProbeSample(
                sample_id=f"{image_id}:{mention_index}",
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
                dgst_final_score=float(record.final_score),
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
    layer_widths = sorted({len(sample.object_layer_dgst_risk) for sample in samples})
    token_aligned = sum(int(sample.token_aligned) for sample in samples)
    return {
        "count": len(samples),
        "positives": positives,
        "negatives": len(samples) - positives,
        "images": len(image_ids),
        "layer_widths": layer_widths,
        "token_aligned": token_aligned,
    }


def split_dgst_probe_samples_by_image(
    samples: Sequence[DGSTProbeSample],
    test_size: float,
    seed: int,
) -> tuple[List[DGSTProbeSample], List[DGSTProbeSample], Dict[str, Any]]:
    grouped: dict[int, List[DGSTProbeSample]] = {}
    for sample in samples:
        grouped.setdefault(int(sample.image_id), []).append(sample)

    if len(grouped) < 2:
        raise ValueError("At least two images are required to split DGST probe data.")

    image_ids = sorted(grouped.keys())
    image_labels = [int(any(sample.hallucinated for sample in grouped[image_id])) for image_id in image_ids]

    positives = sum(image_labels)
    negatives = len(image_labels) - positives
    stratify = image_labels if positives >= 2 and negatives >= 2 else None

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_id_set = set(int(value) for value in train_ids)
    val_id_set = set(int(value) for value in val_ids)

    train_samples = [sample for sample in samples if int(sample.image_id) in train_id_set]
    val_samples = [sample for sample in samples if int(sample.image_id) in val_id_set]
    if not train_samples or not val_samples:
        raise ValueError("The DGST probe split produced an empty train or validation set.")

    split_summary = {
        "seed": int(seed),
        "test_size": float(test_size),
        "stratified_by_image_has_hallucination": bool(stratify is not None),
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
    train_image_ids: Sequence[int],
    val_image_ids: Sequence[int],
) -> tuple[List[DGSTProbeSample], List[DGSTProbeSample], Dict[str, Any]]:
    train_id_set = {int(value) for value in train_image_ids}
    val_id_set = {int(value) for value in val_image_ids}
    if not train_id_set or not val_id_set:
        raise ValueError("Both train_image_ids and val_image_ids must be non-empty.")
    overlap = train_id_set & val_id_set
    if overlap:
        raise ValueError(f"Train/val image ids overlap: {len(overlap)} images.")

    image_ids_in_dataset = {int(sample.image_id) for sample in samples}
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

    train_samples = [sample for sample in samples if int(sample.image_id) in train_id_set]
    val_samples = [sample for sample in samples if int(sample.image_id) in val_id_set]
    if not train_samples or not val_samples:
        raise ValueError("The fixed DGST split produced an empty train or validation set.")

    split_summary = {
        "split_source": "fixed_image_ids",
        "missing_train_image_ids": missing_train,
        "missing_val_image_ids": missing_val,
        "train_image_ids": sorted(train_id_set),
        "val_image_ids": sorted(val_id_set),
        "train_image_count": len(train_id_set),
        "val_image_count": len(val_id_set),
        "train_sample_summary": summarize_dgst_probe_samples(train_samples),
        "val_sample_summary": summarize_dgst_probe_samples(val_samples),
    }
    return train_samples, val_samples, split_summary
