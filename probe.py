from __future__ import annotations

# DGST 对象级 probe 的模型定义、训练循环和验证逻辑。

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .probe_data import DGSTProbeSample


class DGSTProbeDataset(Dataset):
    def __init__(self, samples: Sequence[DGSTProbeSample]):
        if not samples:
            raise ValueError("DGSTProbeDataset requires at least one sample.")
        width = len(samples[0].object_layer_dgst_risk)
        for sample in samples:
            if len(sample.object_layer_dgst_risk) != width:
                raise ValueError("All DGST probe samples must share the same layer width.")
        self.samples = list(samples)
        self.features = torch.tensor([sample.object_layer_dgst_risk for sample in samples], dtype=torch.float32)
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
) -> tuple[DataLoader, DataLoader]:
    train_dataset = DGSTProbeDataset(train_samples)
    val_dataset = DGSTProbeDataset(val_samples)
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

    train_loader, val_loader = create_data_loaders(train_samples, val_samples, batch_size=config.batch_size)
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
