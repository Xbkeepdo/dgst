from __future__ import annotations

# 统一维护 raw-score 和 probe 两类绘图函数。

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    # 统一保证输出目录存在，避免保存图片时报路径错误。
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_all_objects_layerwise(
    layer_metrics: Sequence[dict[str, Any]],
    output_path: str | Path,
    title: str,
) -> None:
    # 画“全体对象”的 layer-wise AUROC 主图。
    # 一条线是原始 AUROC，另一条线是方向无关的 AUROC。
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
    # 画高频类别的对比图。
    # 每个类别一条曲线，展示该类别在不同层上的 direction-free AUROC。
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
