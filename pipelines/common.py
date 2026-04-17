from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable, Sequence

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

from ..config import DGSTConfig, DGSTEvalConfig
from ..data.dataset_adapters import create_dataset_adapter


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    if input_path.suffix != ".json":
        return read_jsonl(input_path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("entries")
        if isinstance(rows, list):
            return rows
        return [payload]
    raise ValueError(f"Expected JSON list or JSONL in {input_path}")


def split_round_robin(items: Sequence[int], num_shards: int) -> list[list[int]]:
    shards = [[] for _ in range(num_shards)]
    for index, item in enumerate(items):
        shards[index % num_shards].append(int(item))
    return shards


def write_lines(path: str | Path, values: Iterable[int]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(value) for value in values), encoding="utf-8")


def _pretty_log_path(path: Path) -> Path:
    return path.with_name("experiment_log.pretty.json")


def append_experiment_log(path: str | Path, payload: dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())
    append_jsonl(log_path, row)
    pretty_path = _pretty_log_path(log_path)
    pretty_path.write_text(json.dumps(read_jsonl(log_path), ensure_ascii=False, indent=2), encoding="utf-8")


def build_dgst_analyzer(dgst_config: DGSTConfig):
    from ..core.analyzer import LlavaDGSTAnalyzer

    return LlavaDGSTAnalyzer(
        model_id=dgst_config.lvlm,
        torch_dtype=dgst_config.dtype,
        device_map=dgst_config.device_map,
        max_memory=dgst_config.max_memory,
        local_files_only=dgst_config.local_files_only,
        attn_implementation=dgst_config.attn_implementation,
    )


def wait_for_processes(
    processes: Sequence[subprocess.Popen],
    *,
    desc: str,
    labels: Sequence[str] | None = None,
    progress_paths: Sequence[Path] | None = None,
    total_items: int | None = None,
    item_desc: str = "item",
    poll_interval: float = 1.0,
) -> list[int]:
    pending = set(range(len(processes)))
    exit_codes: list[int | None] = [None] * len(processes)
    progress = tqdm(total=len(processes), desc=desc, unit="shard", leave=True)
    item_progress = (
        tqdm(total=total_items, desc=f"{desc} ({item_desc})", unit=item_desc, leave=True)
        if progress_paths is not None and total_items is not None
        else None
    )
    processed_items = 0
    while pending:
        finished_this_round: list[int] = []
        for index in list(pending):
            code = processes[index].poll()
            if code is None:
                continue
            exit_codes[index] = int(code)
            finished_this_round.append(index)
        if item_progress is not None and progress_paths is not None:
            current_items = 0
            for path in progress_paths:
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as handle:
                    current_items += sum(1 for line in handle if line.strip())
            current_items = min(current_items, int(total_items or current_items))
            if current_items > processed_items:
                item_progress.update(current_items - processed_items)
                processed_items = current_items
                item_progress.set_postfix(done=f"{processed_items}/{int(total_items or processed_items)}")
        for index in finished_this_round:
            pending.remove(index)
            progress.update(1)
            label = labels[index] if labels is not None and index < len(labels) else f"shard_{index}"
            progress.set_postfix(last=label, code=exit_codes[index])
        if pending:
            time.sleep(poll_interval)
    if item_progress is not None:
        remaining = int(total_items or 0) - processed_items
        if remaining > 0:
            item_progress.update(remaining)
        item_progress.close()
    progress.close()
    return [int(code) for code in exit_codes if code is not None]


def load_sampled_image_ids(eval_config: DGSTEvalConfig) -> tuple[list[int], Any]:
    spec = eval_config.dataset_spec
    adapter = create_dataset_adapter(
        dataset=spec.dataset,
        dataset_root=spec.dataset_root,
        annotation_path=spec.annotation_path,
        split=spec.split,
        cache_path=spec.adapter_cache,
        lexicon_path=spec.lexicon_file,
    )
    if eval_config.image_ids_file is not None:
        image_ids = adapter.list_image_ids(
            split=spec.split,
            max_samples=eval_config.num_data,
            image_ids_file=eval_config.image_ids_file,
        )
        return image_ids, adapter

    image_ids = adapter.list_image_ids(split=spec.split)
    if eval_config.num_data is None or eval_config.num_data <= 0 or eval_config.num_data >= len(image_ids):
        return image_ids, adapter
    rng = random.Random(eval_config.seed)
    return rng.sample(image_ids, eval_config.num_data), adapter


def load_ground_truth_entries(adapter: Any, eval_config: DGSTEvalConfig) -> list[dict[str, Any]]:
    spec = eval_config.dataset_spec
    metadata = adapter.protocol_metadata(spec.protocol)
    desired_image_ids: list[int] | None = None
    regenerate = not spec.ground_truth_file.exists()
    existing_entries: list[dict[str, Any]] = []
    if not regenerate:
        existing_entries = read_json_or_jsonl(spec.ground_truth_file)
        first_entry = existing_entries[0] if existing_entries else {}
        regenerate = any(
            (
                first_entry.get("dataset") != metadata["dataset"],
                first_entry.get("dataset_version") != metadata["dataset_version"],
                first_entry.get("split") != metadata["split"],
                first_entry.get("protocol") != metadata["protocol"],
                first_entry.get("taxonomy_space") != metadata["taxonomy_space"],
            )
        )
    if not regenerate:
        desired_image_ids, _ = load_sampled_image_ids(eval_config)
        existing_image_ids = [int(entry["image_id"]) for entry in existing_entries]
        regenerate = existing_image_ids != desired_image_ids
    if regenerate:
        if desired_image_ids is None:
            desired_image_ids, _ = load_sampled_image_ids(eval_config)
        adapter.save_ground_truth_jsonl(
            spec.ground_truth_file,
            image_ids=desired_image_ids,
            protocol=spec.protocol,
        )
    entries = read_json_or_jsonl(spec.ground_truth_file)
    if eval_config.image_ids_file is not None:
        wanted = set(
            adapter.list_image_ids(
                split=spec.split,
                image_ids_file=eval_config.image_ids_file,
            )
        )
        entries = [entry for entry in entries if int(entry["image_id"]) in wanted]
    if eval_config.num_data is not None:
        entries = entries[: eval_config.num_data]
    return entries


def choose_device(device_arg: str | None):
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    positives = int(sum(labels))
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Evaluation samples must contain both positive and negative labels.")
    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, scores)
    positive_scores = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    negative_scores = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    return {
        "count": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "auroc": auroc,
        "aupr": aupr,
        "positive_mean": float(sum(positive_scores) / len(positive_scores)),
        "negative_mean": float(sum(negative_scores) / len(negative_scores)),
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
