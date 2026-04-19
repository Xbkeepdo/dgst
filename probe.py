from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
PROBE_FEATURE_SET_CHOICES = (PROBE_FEATURE_SET_RISK, PROBE_FEATURE_SET_PROBE6)


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
                dgst_probe6_feature_names=[str(value) for value in item.get("dgst_probe6_feature_names", [])],
                dgst_probe6_layer_stats=list(item.get("dgst_probe6_layer_stats", [])),
                object_layer_dgst_probe6=[float(value) for value in item.get("object_layer_dgst_probe6", [])],
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
    probe6_widths = sorted({len(sample.object_layer_dgst_probe6) for sample in samples if sample.object_layer_dgst_probe6})
    token_aligned = sum(int(sample.token_aligned) for sample in samples)
    return {
        "count": len(samples),
        "positives": positives,
        "negatives": len(samples) - positives,
        "images": len(image_ids),
        "layer_widths": layer_widths,
        "probe6_widths": probe6_widths,
        "token_aligned": token_aligned,
    }


def get_probe_feature_vector(sample: DGSTProbeSample, feature_set: str) -> List[float]:
    if feature_set == PROBE_FEATURE_SET_RISK:
        return list(sample.object_layer_dgst_risk)
    if feature_set == PROBE_FEATURE_SET_PROBE6:
        if not sample.object_layer_dgst_probe6:
            raise ValueError(
                "Probe sample does not contain object_layer_dgst_probe6. "
                "Re-export the probe dataset after enabling the probe6 feature implementation."
            )
        return list(sample.object_layer_dgst_probe6)
    raise ValueError(f"Unsupported probe feature set: {feature_set}")


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
        self.labels = torch.tensor([1 - sample.hallucinated for sample in samples], dtype=torch.float32)

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
    non_hallucination_probabilities: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for features, batch_labels in loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            logits = model(features)
            loss = criterion(logits, batch_labels)
            losses.append(float(loss.item()))
            non_hallucination_probabilities.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
            labels.extend(int(value) for value in batch_labels.squeeze(1).cpu().tolist())

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Validation split must contain both positive and negative samples.")

    roc_auc = float(roc_auc_score(labels, non_hallucination_probabilities))
    aupr = float(average_precision_score(labels, non_hallucination_probabilities))
    fpr, tpr, roc_thresholds = roc_curve(labels, non_hallucination_probabilities)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, non_hallucination_probabilities)
    return {
        "val_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        "auroc": roc_auc,
        "aupr": aupr,
        "labels": labels,
        "non_hallucination_probabilities": non_hallucination_probabilities,
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
    non_hallucination_probabilities: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample, probability in zip(val_samples, non_hallucination_probabilities):
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_id": int(sample.image_id),
                "canonical_name": sample.canonical_name,
                "surface": sample.surface,
                "phrase": sample.phrase,
                "hallucinated": int(sample.hallucinated),
                "non_hallucination_label": int(1 - sample.hallucinated),
                "non_hallucination_probability": float(probability),
                "hallucination_probability": float(1.0 - probability),
                "dgst_final_score": float(sample.dgst_final_score),
            }
        )
    return rows
