from __future__ import annotations

# DGST 的独立任务编排层。

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

from tqdm import tqdm

from .config import (
    DGSTConfig,
    DGSTEvalConfig,
    DGSTExportConfig,
    ProbeEvalConfig,
    ProbeTrainConfig,
    apply_paper_probe_defaults,
    default_export_run_name,
    project_paths,
)
from .dataset_adapters import create_dataset_adapter
from .evaluation import collect_dgst_object_records, summarize_dgst_dataset_metrics
from .probe_data import (
    DGSTProbeSample,
    build_dgst_probe_samples,
    read_dgst_probe_dataset,
    split_dgst_probe_samples_by_image,
    split_dgst_probe_samples_with_fixed_image_ids,
    summarize_dgst_probe_samples,
    write_dgst_probe_dataset,
)
from .io import (
    append_experiment_log,
    append_jsonl,
    read_json,
    read_json_or_jsonl,
    read_jsonl,
    split_round_robin,
    write_json,
    write_jsonl,
    write_lines,
)
from .plots import plot_all_objects_layerwise, plot_category_layerwise, plot_pr_curve, plot_roc_curve


def _build_dgst_analyzer(dgst_config: DGSTConfig):
    from .analyzer import LlavaDGSTAnalyzer

    return LlavaDGSTAnalyzer(
        model_id=dgst_config.lvlm,
        torch_dtype=dgst_config.dtype,
        device_map=dgst_config.device_map,
        max_memory=dgst_config.max_memory,
        local_files_only=dgst_config.local_files_only,
        attn_implementation=dgst_config.attn_implementation,
    )


def _wait_for_processes(
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


def _load_sampled_image_ids(
    eval_config: DGSTEvalConfig,
) -> tuple[list[int], Any]:
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


def _load_ground_truth_entries(
    adapter: Any,
    eval_config: DGSTEvalConfig,
) -> list[dict[str, Any]]:
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
        desired_image_ids, _ = _load_sampled_image_ids(eval_config)
        existing_image_ids = [int(entry["image_id"]) for entry in existing_entries]
        regenerate = existing_image_ids != desired_image_ids
    if regenerate:
        if desired_image_ids is None:
            desired_image_ids, _ = _load_sampled_image_ids(eval_config)
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


def _choose_device(device_arg: str | None):
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

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


def _compose_dgst_summary(
    *,
    metadata: dict[str, Any],
    eval_config: DGSTEvalConfig,
    processed_images: int,
    total_mentions: int,
    aligned_mentions: int,
    dropped_unaligned_mentions: int,
    total_gt_categories: int,
    total_evaluated_gt_categories: int,
    started_at: datetime,
    finished_at: datetime,
    metric_summary: dict[str, Any],
    notes: list[str],
) -> dict[str, Any]:
    global_metrics = metric_summary["global_mean_metrics"]
    cfg = eval_config.dgst_config
    spec = eval_config.dataset_spec
    return {
        "run_type": "dgst_zero_shot",
        "dataset": metadata["dataset"],
        "dataset_version": metadata["dataset_version"],
        "split": metadata["split"],
        "protocol": metadata["protocol"],
        "taxonomy_space": metadata["taxonomy_space"],
        "lexicon_version": metadata["lexicon_version"],
        "mention_linker_backend": metadata["mention_linker_backend"],
        "taxonomy_backend": metadata["taxonomy_backend"],
        "lvlm": cfg.lvlm,
        "prompt": cfg.prompt,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float((finished_at - started_at).total_seconds()),
        "dataset_root": str(spec.dataset_root),
        "annotation_source": str(spec.annotation_path),
        "ground_truth_file": str(spec.ground_truth_file),
        "caption_cache_file": str(eval_config.caption_file) if eval_config.caption_file is not None else None,
        "detail_result_file": str(eval_config.result_file) if eval_config.result_file is not None else None,
        "processed_images": int(processed_images),
        "chair_backend": f"{metadata['taxonomy_backend']}:{metadata['mention_linker_backend']}",
        "chair_word_count_total": int(total_mentions),
        "total_detected_mentions": int(total_mentions),
        "aligned_mentions": int(aligned_mentions),
        "aligned_object_sample_count": int(aligned_mentions),
        "dropped_unaligned_count": int(dropped_unaligned_mentions),
        "alignment_rate": float(aligned_mentions / total_mentions) if total_mentions else 0.0,
        "alignment_rate_after_realign": float(aligned_mentions / total_mentions) if total_mentions else 0.0,
        "gt_category_count": int(total_gt_categories),
        "evaluated_gt_category_count": int(total_evaluated_gt_categories),
        "gt_category_coverage": float(total_evaluated_gt_categories / total_gt_categories) if total_gt_categories else 0.0,
        "evaluated_sample_count": int(global_metrics["count"]),
        "positive_count": int(global_metrics["positives"]),
        "negative_count": int(global_metrics["negatives"]),
        "global_mean_metrics": global_metrics,
        "layer_wise_metrics": metric_summary["layer_wise_metrics"],
        "best_layer_by_direction_free_auroc": metric_summary["best_layer_by_direction_free_auroc"],
        "top_category_layer_wise_metrics": metric_summary["top_category_layer_wise_metrics"],
        "dgst_config": {
            "tau": cfg.tau,
            "transport_top_k": cfg.transport_top_k,
            "gamma1": cfg.gamma1,
            "gamma2": cfg.gamma2,
            "baseline_layers": cfg.baseline_layers,
            "risk_start_layer": cfg.risk_start_layer,
            "alpha": cfg.alpha,
            "ot_solver": cfg.ot_solver,
        },
        "plot_files": {
            "plot_dir": str(eval_config.plot_dir) if eval_config.plot_dir is not None else None,
            "all_objects_layerwise_auroc": str(Path(eval_config.plot_dir) / "dgst_all_objects_layerwise_auroc.png")
            if eval_config.plot_dir is not None and metric_summary["layer_wise_metrics"]
            else None,
            "top_categories_layerwise_auroc": str(Path(eval_config.plot_dir) / "dgst_top_categories_layerwise_auroc.png")
            if eval_config.plot_dir is not None and metric_summary["top_category_layer_wise_metrics"]
            else None,
        },
        "notes": notes,
    }


def _finalize_dgst_plots(eval_config: DGSTEvalConfig, metric_summary: dict[str, Any], notes: list[str]) -> None:
    if eval_config.plot_dir is None:
        return
    plot_dir = Path(eval_config.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    layer_wise_metrics = metric_summary["layer_wise_metrics"]
    category_metrics = metric_summary["top_category_layer_wise_metrics"]
    all_objects_plot = plot_dir / "dgst_all_objects_layerwise_auroc.png"
    category_plot = plot_dir / "dgst_top_categories_layerwise_auroc.png"
    if layer_wise_metrics:
        plot_all_objects_layerwise(
            layer_metrics=layer_wise_metrics,
            output_path=all_objects_plot,
            title="DGST All Objects Layer-wise AUROC",
        )
    else:
        notes.append("all_objects plot skipped: no aligned object records")
    if category_metrics:
        plot_category_layerwise(
            category_metrics=category_metrics,
            output_path=category_plot,
            title="DGST Top Categories Layer-wise Direction-free AUROC",
        )
    else:
        notes.append("category plot skipped: not enough categories with both positive and negative samples")


def run_dgst_single_analysis(
    *,
    image_path: str | Path,
    output_path: str | Path,
    dgst_config: DGSTConfig,
    answer: str | None = None,
    target_mode: str = "objects",
    object_phrases: Sequence[str] | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    analyzer = _build_dgst_analyzer(dgst_config)
    if answer is None:
        result = analyzer.generate_and_analyze(
            image_path=image_path,
            question=dgst_config.prompt,
            max_new_tokens=dgst_config.max_new_tokens,
            do_sample=dgst_config.inference_temp > 0.0,
            temperature=dgst_config.inference_temp,
            top_p=dgst_config.top_p,
            target_mode=target_mode,
            object_phrases=object_phrases,
            tau=dgst_config.tau,
            transport_top_k=dgst_config.transport_top_k,
            gamma1=dgst_config.gamma1,
            gamma2=dgst_config.gamma2,
            baseline_layers=dgst_config.baseline_layers,
            risk_start_layer=dgst_config.risk_start_layer,
            alpha=dgst_config.alpha,
            ot_solver=dgst_config.ot_solver,
        )
    else:
        result = analyzer.analyze(
            image_path=image_path,
            question=dgst_config.prompt,
            answer=answer,
            target_mode=target_mode,
            object_phrases=object_phrases,
            tau=dgst_config.tau,
            transport_top_k=dgst_config.transport_top_k,
            gamma1=dgst_config.gamma1,
            gamma2=dgst_config.gamma2,
            baseline_layers=dgst_config.baseline_layers,
            risk_start_layer=dgst_config.risk_start_layer,
            alpha=dgst_config.alpha,
            ot_solver=dgst_config.ot_solver,
        )
    analyzer.save_json(result, output_path)
    print(f"Saved DGST single-image result to: {Path(output_path).resolve()}")
    print(f"Answer: {result['answer']}")
    print(f"Mean DGST Final Score: {float(result.get('dgst_final_score_mean', 0.0)):.6f}")
    return result


def _run_dgst_evaluation_single(eval_config: DGSTEvalConfig) -> dict[str, Any]:
    spec = eval_config.dataset_spec
    cfg = eval_config.dgst_config
    started_at = datetime.now(timezone.utc)
    adapter = create_dataset_adapter(
        dataset=spec.dataset,
        dataset_root=spec.dataset_root,
        annotation_path=spec.annotation_path,
        split=spec.split,
        cache_path=spec.adapter_cache,
        lexicon_path=spec.lexicon_file,
    )
    metadata = adapter.protocol_metadata(spec.protocol)
    ground_truth_entries = _load_ground_truth_entries(adapter, eval_config)

    caption_data = (
        {int(row["image_id"]): row for row in read_jsonl(eval_config.caption_file)}
        if eval_config.caption_file is not None
        else {}
    )
    result_data = (
        {int(row["image_id"]): row for row in read_jsonl(eval_config.result_file)}
        if eval_config.result_file is not None
        else {}
    )
    analyzer = _build_dgst_analyzer(cfg)

    object_records = []
    processed_images = 0
    total_mentions = 0
    aligned_mentions = 0
    dropped_unaligned_mentions = 0
    total_gt_categories = 0
    total_evaluated_gt_categories = 0
    notes: list[str] = ["dgst zero-shot validation"]

    progress = tqdm(ground_truth_entries, total=len(ground_truth_entries), desc=f"Evaluating DGST {spec.dataset}:{spec.protocol}")
    for entry in progress:
        image_id = int(entry["image_id"])
        image_name = str(entry.get("image") or adapter.image_filename(image_id))
        image_path = spec.dataset_root / image_name
        if not image_path.exists():
            print(f"Skip missing image: {image_path}")
            continue

        if image_id in result_data:
            row = result_data[image_id]
            cached_records = collect_dgst_object_records(row)
            object_records.extend(cached_records)
            processed_images += 1
            total_mentions += int(row.get("chair_word_count_total", len(row.get("object_mentions", []))))
            aligned_mentions += len(cached_records)
            dropped_unaligned_mentions += int(
                row.get(
                    "dropped_unaligned_count",
                    max(0, int(row.get("chair_word_count_total", len(row.get("object_mentions", [])))) - len(cached_records)),
                )
            )
            total_gt_categories += int(row.get("gt_category_count", 0))
            total_evaluated_gt_categories += int(row.get("evaluated_gt_category_count", row.get("gt_category_count", 0)))
            progress.set_postfix(image_id=image_id, cached="result", aligned=len(cached_records))
            continue

        cached_caption = caption_data.get(image_id)
        if cached_caption is not None and eval_config.reuse_captions:
            caption = str(cached_caption["caption"])
        else:
            caption = analyzer.generate_answer(
                image_path=image_path,
                question=cfg.prompt,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.inference_temp > 0.0,
                temperature=cfg.inference_temp,
                top_p=cfg.top_p,
            )
            if eval_config.caption_file is not None:
                cached_caption = {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "prompt": cfg.prompt,
                    "caption": caption,
                }
                append_jsonl(eval_config.caption_file, cached_caption)
                caption_data[image_id] = cached_caption

        adapter_eval = adapter.evaluate_caption(image_id, caption, protocol=spec.protocol)
        result = analyzer.analyze(
            image_path=image_path,
            question=cfg.prompt,
            answer=caption,
            target_mode="objects",
            object_mentions=adapter_eval["object_mentions"],
            tau=cfg.tau,
            transport_top_k=cfg.transport_top_k,
            gamma1=cfg.gamma1,
            gamma2=cfg.gamma2,
            baseline_layers=cfg.baseline_layers,
            risk_start_layer=cfg.risk_start_layer,
            alpha=cfg.alpha,
            ot_solver=cfg.ot_solver,
        )

        gt_objects = set(adapter_eval["ground_truth_objects"])
        mention_labels = {
            idx: int(mention["canonical_name"] not in gt_objects)
            for idx, mention in enumerate(adapter_eval["object_mentions"])
        }

        dgst_object_scores = []
        for item in result.get("dgst_object_scores", []):
            mention_index = item.get("mention_index")
            if mention_index is None:
                continue
            score_item = dict(item)
            score_item["hallucinated"] = int(item.get("hallucinated", mention_labels.get(int(mention_index), 0)))
            dgst_object_scores.append(score_item)

        row = {
            "image_id": image_id,
            "image": image_name,
            "image_path": str(image_path),
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "taxonomy_space": metadata["taxonomy_space"],
            "lexicon_version": metadata["lexicon_version"],
            "mention_linker_backend": metadata["mention_linker_backend"],
            "taxonomy_backend": metadata["taxonomy_backend"],
            "prompt": cfg.prompt,
            "caption": caption,
            "chair_s": adapter_eval["chair_s"],
            "chair_i": adapter_eval["chair_i"],
            "chair_backend": adapter_eval["chair_backend"],
            "chair_word_count_total": adapter_eval["chair_word_count_total"],
            "ground_truth_objects": adapter_eval["ground_truth_objects"],
            "gt_category_count": adapter_eval["gt_category_count"],
            "evaluated_gt_category_count": adapter_eval["evaluated_gt_category_count"],
            "gt_category_coverage": adapter_eval["gt_category_coverage"],
            "object_mentions": adapter_eval["object_mentions"],
            "hallucinated_mentions": adapter_eval["hallucinated_mentions"],
            "dgst_object_alignment": result.get("object_alignment", []),
            "dgst_object_scores": dgst_object_scores,
            "aligned_object_sample_count": len(dgst_object_scores),
            "dropped_unaligned_count": max(0, int(adapter_eval["chair_word_count_total"]) - len(dgst_object_scores)),
            "alignment_rate_after_realign": float(len(dgst_object_scores) / int(adapter_eval["chair_word_count_total"]))
            if int(adapter_eval["chair_word_count_total"])
            else 0.0,
            "dgst_final_score_mean": result.get("dgst_final_score_mean"),
            "dgst_config": result.get("dgst_config"),
        }
        row_records = collect_dgst_object_records(row)
        object_records.extend(row_records)
        processed_images += 1
        total_mentions += int(adapter_eval["chair_word_count_total"])
        aligned_mentions += len(row_records)
        dropped_unaligned_mentions += max(0, int(adapter_eval["chair_word_count_total"]) - len(row_records))
        total_gt_categories += int(adapter_eval["gt_category_count"])
        total_evaluated_gt_categories += int(adapter_eval["evaluated_gt_category_count"])

        if eval_config.result_file is not None:
            append_jsonl(eval_config.result_file, row)
            result_data[image_id] = row
        progress.set_postfix(image_id=image_id, aligned=f"{len(row_records)}/{int(adapter_eval['chair_word_count_total'])}")

    metric_summary = summarize_dgst_dataset_metrics(
        object_records,
        category_top_k=eval_config.category_top_k,
        category_min_count=eval_config.category_min_count,
    )
    finished_at = datetime.now(timezone.utc)
    _finalize_dgst_plots(eval_config, metric_summary, notes)
    summary = _compose_dgst_summary(
        metadata=metadata,
        eval_config=eval_config,
        processed_images=processed_images,
        total_mentions=total_mentions,
        aligned_mentions=aligned_mentions,
        dropped_unaligned_mentions=dropped_unaligned_mentions,
        total_gt_categories=total_gt_categories,
        total_evaluated_gt_categories=total_evaluated_gt_categories,
        started_at=started_at,
        finished_at=finished_at,
        metric_summary=metric_summary,
        notes=notes,
    )
    if eval_config.evaluation_file is not None:
        write_json(eval_config.evaluation_file, summary)
    append_experiment_log(
        project_paths().experiment_log_file,
        {
            "run_type": "dgst_zero_shot",
            "run_name": Path(eval_config.evaluation_file).stem if eval_config.evaluation_file is not None else None,
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "lvlm": cfg.lvlm,
            "processed_images": processed_images,
            "aligned_mentions": aligned_mentions,
            "alignment_rate": summary["alignment_rate"],
            "metrics_file": str(eval_config.evaluation_file) if eval_config.evaluation_file is not None else None,
        },
    )
    return summary


def _build_dgst_eval_subprocess_command(eval_config: DGSTEvalConfig) -> list[str]:
    spec = eval_config.dataset_spec
    cfg = eval_config.dgst_config
    command = [
        sys.executable,
        "-m",
        "dgst.cli",
        "evaluate",
        "--dataset",
        spec.dataset,
        "--split",
        spec.split,
        "--protocol",
        spec.protocol,
        "--dataset-root",
        str(spec.dataset_root),
        "--annotation-path",
        str(spec.annotation_path),
        "--adapter-cache",
        str(spec.adapter_cache),
        "--ground-truth-file",
        str(spec.ground_truth_file),
        "--lvlm",
        cfg.lvlm,
        "--dtype",
        cfg.dtype,
        "--device-map",
        cfg.device_map,
        "--prompt",
        cfg.prompt,
        "--max-new-tokens",
        str(cfg.max_new_tokens),
        "--inference-temp",
        str(cfg.inference_temp),
        "--top-p",
        str(cfg.top_p),
        "--tau",
        str(cfg.tau),
        "--transport-top-k",
        str(cfg.transport_top_k),
        "--gamma1",
        str(cfg.gamma1),
        "--gamma2",
        str(cfg.gamma2),
        "--baseline-layers",
        str(cfg.baseline_layers),
        "--risk-start-layer",
        str(cfg.risk_start_layer),
        "--alpha",
        str(cfg.alpha),
        "--ot-solver",
        cfg.ot_solver,
        "--gpus",
        ",".join(eval_config.gpus),
        "--category-top-k",
        str(eval_config.category_top_k),
        "--category-min-count",
        str(eval_config.category_min_count),
        "--seed",
        str(eval_config.seed),
    ]
    if eval_config.num_data is not None:
        command.extend(["--num-data", str(eval_config.num_data)])
    if spec.lexicon_file is not None:
        command.extend(["--lexicon-file", str(spec.lexicon_file)])
    if cfg.local_files_only:
        command.append("--local-files-only")
    else:
        command.append("--no-local-files-only")
    if cfg.attn_implementation is not None:
        command.extend(["--attn-implementation", str(cfg.attn_implementation)])
    if cfg.max_memory:
        memory = ",".join(f"{key}={value}" for key, value in cfg.max_memory.items())
        command.extend(["--max-memory", memory])
    if eval_config.caption_file is not None:
        command.extend(["--caption-file", str(eval_config.caption_file)])
    if eval_config.result_file is not None:
        command.extend(["--result-file", str(eval_config.result_file)])
    if eval_config.evaluation_file is not None:
        command.extend(["--evaluation-file", str(eval_config.evaluation_file)])
    if eval_config.plot_dir is not None:
        command.extend(["--plot-dir", str(eval_config.plot_dir)])
    if eval_config.image_ids_file is not None:
        command.extend(["--image-ids-file", str(eval_config.image_ids_file)])
    if not eval_config.reuse_captions:
        command.append("--no-reuse-captions")
    return command


def run_dgst_dataset_evaluation(eval_config: DGSTEvalConfig) -> dict[str, Any]:
    if len(eval_config.gpus) <= 1:
        return _run_dgst_evaluation_single(eval_config)

    sampled_image_ids, adapter = _load_sampled_image_ids(eval_config)
    spec = eval_config.dataset_spec
    metadata = adapter.protocol_metadata(spec.protocol)
    adapter.save_ground_truth_jsonl(spec.ground_truth_file, image_ids=sampled_image_ids, protocol=spec.protocol)

    started_at = datetime.now(timezone.utc)
    shards = split_round_robin(sampled_image_ids, len(eval_config.gpus))
    with tempfile.TemporaryDirectory(prefix="dgst_parallel_eval_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        processes: list[subprocess.Popen] = []
        shard_meta: list[dict[str, Any]] = []

        for shard_index, (gpu_id, shard_ids) in enumerate(zip(eval_config.gpus, shards)):
            ids_file = tmpdir / f"shard_{shard_index}_ids.txt"
            caption_file = tmpdir / f"shard_{shard_index}_captions.jsonl"
            result_file = tmpdir / f"shard_{shard_index}_results.jsonl"
            summary_file = tmpdir / f"shard_{shard_index}_summary.json"
            plot_dir = tmpdir / f"shard_{shard_index}_plots"
            write_lines(ids_file, shard_ids)
            worker_config = DGSTEvalConfig(
                dataset_spec=spec,
                dgst_config=eval_config.dgst_config,
                num_data=len(shard_ids),
                seed=eval_config.seed,
                gpus=(gpu_id,),
                reuse_captions=eval_config.reuse_captions,
                caption_file=caption_file,
                result_file=result_file,
                evaluation_file=summary_file,
                plot_dir=plot_dir,
                image_ids_file=ids_file,
                category_top_k=eval_config.category_top_k,
                category_min_count=eval_config.category_min_count,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            process = subprocess.Popen(
                _build_dgst_eval_subprocess_command(worker_config),
                cwd=str(project_paths().repo_root),
                env=env,
            )
            processes.append(process)
            shard_meta.append(
                {
                    "shard_index": shard_index,
                    "gpu": gpu_id,
                    "image_count": len(shard_ids),
                    "ids_file": str(ids_file),
                    "caption_file": str(caption_file),
                    "result_file": str(result_file),
                    "summary_file": str(summary_file),
                }
            )

        exit_codes = _wait_for_processes(
            processes,
            desc="Waiting for DGST evaluation shards",
            labels=[f"gpu{meta['gpu']}" for meta in shard_meta],
            progress_paths=[Path(meta["result_file"]) for meta in shard_meta],
            total_items=sum(int(meta["image_count"]) for meta in shard_meta),
            item_desc="image",
        )
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"One or more DGST shard processes failed: {exit_codes}")

        all_rows: list[dict[str, Any]] = []
        all_captions: list[dict[str, Any]] = []
        processed_images = 0
        total_mentions = 0
        aligned_mentions = 0
        dropped_unaligned_mentions = 0
        total_gt_categories = 0
        total_evaluated_gt_categories = 0
        for meta in shard_meta:
            summary = read_json(meta["summary_file"])
            processed_images += int(summary["processed_images"])
            total_mentions += int(summary["total_detected_mentions"])
            aligned_mentions += int(summary["aligned_mentions"])
            dropped_unaligned_mentions += int(summary.get("dropped_unaligned_count", 0))
            total_gt_categories += int(summary.get("gt_category_count", 0))
            total_evaluated_gt_categories += int(summary.get("evaluated_gt_category_count", 0))
            all_rows.extend(read_jsonl(meta["result_file"]))
            all_captions.extend(read_jsonl(meta["caption_file"]))

        if eval_config.caption_file is not None:
            write_jsonl(eval_config.caption_file, all_captions)
        if eval_config.result_file is not None:
            write_jsonl(eval_config.result_file, all_rows)

        object_records = []
        for row in all_rows:
            object_records.extend(collect_dgst_object_records(row))
        metric_summary = summarize_dgst_dataset_metrics(
            object_records,
            category_top_k=eval_config.category_top_k,
            category_min_count=eval_config.category_min_count,
        )
        notes: list[str] = ["dgst zero-shot validation", "sample-parallel execution across GPUs"]
        finished_at = datetime.now(timezone.utc)
        _finalize_dgst_plots(eval_config, metric_summary, notes)
        summary = _compose_dgst_summary(
            metadata=metadata,
            eval_config=eval_config,
            processed_images=processed_images,
            total_mentions=total_mentions,
            aligned_mentions=aligned_mentions,
            dropped_unaligned_mentions=dropped_unaligned_mentions,
            total_gt_categories=total_gt_categories,
            total_evaluated_gt_categories=total_evaluated_gt_categories,
            started_at=started_at,
            finished_at=finished_at,
            metric_summary=metric_summary,
            notes=notes,
        )
        if eval_config.evaluation_file is not None:
            write_json(eval_config.evaluation_file, summary)
        append_experiment_log(
            project_paths().experiment_log_file,
            {
                "run_type": "dgst_zero_shot",
                "run_name": Path(eval_config.evaluation_file).stem if eval_config.evaluation_file is not None else None,
                "dataset": metadata["dataset"],
                "dataset_version": metadata["dataset_version"],
                "split": metadata["split"],
                "protocol": metadata["protocol"],
                "lvlm": eval_config.dgst_config.lvlm,
                "processed_images": processed_images,
                "aligned_mentions": aligned_mentions,
                "alignment_rate": summary["alignment_rate"],
                "metrics_file": str(eval_config.evaluation_file) if eval_config.evaluation_file is not None else None,
            },
        )
        return summary


def export_dgst_probe_dataset(export_config: DGSTExportConfig) -> dict[str, Any]:
    spec = export_config.dataset_spec
    run_name = export_config.run_name or default_export_run_name(spec.dataset, spec.protocol, export_config.num_data)
    base_dir = project_paths().probe_data_dir / run_name
    output_file = export_config.output_file or (base_dir / "probe_dataset.jsonl")
    manifest_file = export_config.manifest_file or (base_dir / "probe_dataset_manifest.json")
    work_dir = export_config.work_dir or (base_dir / "shards")
    work_dir.mkdir(parents=True, exist_ok=True)

    eval_config = DGSTEvalConfig(
        dataset_spec=spec,
        dgst_config=export_config.dgst_config,
        num_data=export_config.num_data,
        seed=export_config.seed,
        gpus=export_config.gpus,
        reuse_captions=export_config.reuse_captions,
        caption_file=None,
        result_file=None,
        evaluation_file=None,
        plot_dir=None,
        image_ids_file=None,
    )
    sampled_image_ids, adapter = _load_sampled_image_ids(eval_config)
    adapter.save_ground_truth_jsonl(spec.ground_truth_file, image_ids=sampled_image_ids, protocol=spec.protocol)

    started_at = datetime.now(timezone.utc)
    shards = split_round_robin(sampled_image_ids, len(export_config.gpus))
    shard_meta: list[dict[str, Any]] = []
    processes: list[subprocess.Popen] = []

    for shard_index, (gpu_id, shard_ids) in enumerate(zip(export_config.gpus, shards)):
        ids_file = work_dir / f"shard_{shard_index}_ids.txt"
        caption_file = work_dir / f"shard_{shard_index}_captions.jsonl"
        result_file = work_dir / f"shard_{shard_index}_results.jsonl"
        summary_file = work_dir / f"shard_{shard_index}_summary.json"
        plot_dir = work_dir / f"shard_{shard_index}_plots"
        write_lines(ids_file, shard_ids)
        worker_config = DGSTEvalConfig(
            dataset_spec=spec,
            dgst_config=export_config.dgst_config,
            num_data=len(shard_ids),
            seed=export_config.seed,
            gpus=(gpu_id,),
            reuse_captions=export_config.reuse_captions,
            caption_file=caption_file,
            result_file=result_file,
            evaluation_file=summary_file,
            plot_dir=plot_dir,
            image_ids_file=ids_file,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(
            _build_dgst_eval_subprocess_command(worker_config),
            cwd=str(project_paths().repo_root),
            env=env,
        )
        processes.append(process)
        shard_meta.append(
            {
                "shard_index": shard_index,
                "gpu": gpu_id,
                "image_count": len(shard_ids),
                "ids_file": str(ids_file),
                "caption_file": str(caption_file),
                "result_file": str(result_file),
                "summary_file": str(summary_file),
            }
        )

    exit_codes = _wait_for_processes(
        processes,
        desc="Waiting for DGST export shards",
        labels=[f"gpu{meta['gpu']}" for meta in shard_meta],
        progress_paths=[Path(meta["result_file"]) for meta in shard_meta],
        total_items=sum(int(meta["image_count"]) for meta in shard_meta),
        item_desc="image",
    )
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more DGST shard processes failed: {exit_codes}")

    all_rows: list[dict[str, Any]] = []
    processed_images = 0
    total_mentions = 0
    aligned_mentions = 0
    dropped_unaligned = 0
    for meta in shard_meta:
        summary = read_json(meta["summary_file"])
        processed_images += int(summary["processed_images"])
        total_mentions += int(summary.get("chair_word_count_total", summary["total_detected_mentions"]))
        aligned_mentions += int(summary.get("aligned_object_sample_count", summary["aligned_mentions"]))
        dropped_unaligned += int(summary.get("dropped_unaligned_count", 0))
        all_rows.extend(read_jsonl(meta["result_file"]))

    dgst_probe_samples = build_dgst_probe_samples(all_rows)
    write_dgst_probe_dataset(output_file, dgst_probe_samples)

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "run_name": run_name,
        "dataset": spec.dataset,
        "split": spec.split,
        "protocol": spec.protocol,
        "lvlm": export_config.dgst_config.lvlm,
        "execution_mode": "sample_parallel_export" if len(export_config.gpus) > 1 else "single_export",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float((finished_at - started_at).total_seconds()),
        "dataset_root": str(spec.dataset_root),
        "annotation_source": str(spec.annotation_path),
        "adapter_cache": str(spec.adapter_cache),
        "ground_truth_file": str(spec.ground_truth_file),
        "output_file": str(output_file),
        "processed_images": processed_images,
        "chair_backend": f"{adapter.protocol_metadata(spec.protocol)['taxonomy_backend']}:{adapter.mention_linker_backend}",
        "chair_word_count_total": total_mentions,
        "total_detected_mentions": total_mentions,
        "aligned_mentions": aligned_mentions,
        "aligned_object_sample_count": aligned_mentions,
        "dropped_unaligned_count": dropped_unaligned,
        "alignment_rate": float(aligned_mentions / total_mentions) if total_mentions else 0.0,
        "alignment_rate_after_realign": float(aligned_mentions / total_mentions) if total_mentions else 0.0,
        "probe_sample_summary": summarize_dgst_probe_samples(dgst_probe_samples),
        "dgst_config": {
            "tau": export_config.dgst_config.tau,
            "transport_top_k": export_config.dgst_config.transport_top_k,
            "gamma1": export_config.dgst_config.gamma1,
            "gamma2": export_config.dgst_config.gamma2,
            "baseline_layers": export_config.dgst_config.baseline_layers,
            "risk_start_layer": export_config.dgst_config.risk_start_layer,
            "alpha": export_config.dgst_config.alpha,
            "ot_solver": export_config.dgst_config.ot_solver,
        },
        "shards": shard_meta,
        "notes": [
            "dgst object-level probe dataset export",
            "shard result files are persistent and support resume",
        ],
    }
    write_json(manifest_file, manifest)
    print(f"Saved DGST probe dataset to: {output_file.resolve()}")
    print(f"Saved DGST dataset manifest to: {manifest_file.resolve()}")
    return manifest


def _build_dgst_split(
    samples: Sequence[DGSTProbeSample],
    split_file: Path | None,
    test_size: float,
    seed: int,
) -> tuple[list[DGSTProbeSample], list[DGSTProbeSample], dict[str, Any]]:
    if split_file is not None:
        split_payload = read_json(split_file)
        train_samples, val_samples, split_summary = split_dgst_probe_samples_with_fixed_image_ids(
            samples,
            train_image_ids=split_payload["train_image_ids"],
            val_image_ids=split_payload["val_image_ids"],
        )
        split_summary["seed"] = split_payload.get("seed")
        split_summary["test_size"] = split_payload.get("test_size")
        split_summary["stratified_by_image_has_hallucination"] = split_payload.get(
            "stratified_by_image_has_hallucination"
        )
        split_summary["split_file"] = str(split_file)
        return train_samples, val_samples, split_summary
    return split_dgst_probe_samples_by_image(samples, test_size=test_size, seed=seed)


def _run_single_dgst_probe_training(
    samples: Sequence[DGSTProbeSample],
    train_config: ProbeTrainConfig,
    *,
    run_seed: int,
    single_output_dir: Path,
) -> dict[str, Any]:
    from .probe import DGSTProbeConfig, build_dgst_val_prediction_rows, train_dgst_probe

    train_samples, val_samples, split_summary = _build_dgst_split(
        samples,
        train_config.split_file,
        train_config.test_size,
        run_seed,
    )
    probe_config = DGSTProbeConfig(
        input_dim=len(samples[0].object_layer_dgst_risk),
        batch_size=train_config.batch_size,
        num_epochs=train_config.num_epochs,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        lr_factor=train_config.lr_factor,
        lr_patience=train_config.lr_patience,
        seed=run_seed,
    )
    started_at = datetime.now(timezone.utc)
    training_result = train_dgst_probe(train_samples, val_samples, config=probe_config, output_dir=single_output_dir)
    finished_at = datetime.now(timezone.utc)

    best_metrics = training_result["best_metrics"]
    roc_plot = single_output_dir / "roc_curve.png"
    pr_plot = single_output_dir / "pr_curve.png"
    plot_roc_curve(
        fpr=best_metrics["roc_curve"]["fpr"],
        tpr=best_metrics["roc_curve"]["tpr"],
        auroc=float(best_metrics["auroc"]),
        output_path=roc_plot,
    )
    plot_pr_curve(
        recall=best_metrics["pr_curve"]["recall"],
        precision=best_metrics["pr_curve"]["precision"],
        aupr=float(best_metrics["aupr"]),
        output_path=pr_plot,
    )
    write_jsonl(
        single_output_dir / "val_predictions.jsonl",
        build_dgst_val_prediction_rows(val_samples, best_metrics["non_hallucination_probabilities"]),
    )
    write_json(single_output_dir / "split.json", split_summary)
    metrics = {
        "run_type": "dgst_probe",
        "run_name": single_output_dir.name,
        "dataset_file": str(train_config.dataset_file),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float((finished_at - started_at).total_seconds()),
        "device": training_result["device"],
        "positive_class": "non_hallucination",
        "label_definition": {"1": "non_hallucination", "0": "hallucination"},
        "train_sample_summary": summarize_dgst_probe_samples(train_samples),
        "val_sample_summary": summarize_dgst_probe_samples(val_samples),
        "best_epoch": int(training_result["best_epoch"]),
        "best_val_loss": float(training_result["best_val_loss"]),
        "auroc": float(best_metrics["auroc"]),
        "aupr": float(best_metrics["aupr"]),
        "roc_curve": best_metrics["roc_curve"],
        "pr_curve": best_metrics["pr_curve"],
        "artifacts": {
            "model": str(single_output_dir / "model.pth"),
            "config": str(single_output_dir / "config.json"),
            "history": str(single_output_dir / "history.json"),
            "split": str(single_output_dir / "split.json"),
            "val_predictions": str(single_output_dir / "val_predictions.jsonl"),
            "roc_curve_plot": str(roc_plot),
            "pr_curve_plot": str(pr_plot),
        },
        "notes": [
            "object-level DGST probe",
            "fixed image-level split reused from prior run" if train_config.split_file is not None else "image-level 80/20 split",
            "positive class is non_hallucination",
        ],
    }
    write_json(single_output_dir / "metrics.json", metrics)
    return metrics


def train_dgst_probe_run(train_config: ProbeTrainConfig) -> dict[str, Any]:
    train_config = apply_paper_probe_defaults(train_config)
    if not train_config.dataset_file.exists():
        raise FileNotFoundError(f"DGST probe dataset file not found: {train_config.dataset_file}")
    samples = read_dgst_probe_dataset(train_config.dataset_file)
    if not samples:
        raise ValueError("DGST probe dataset is empty.")

    started_at = datetime.now(timezone.utc)
    run_metrics: list[dict[str, Any]] = []
    run_seeds = [train_config.seed + index for index in range(train_config.num_runs)]
    for run_seed in tqdm(run_seeds, desc="DGST probe runs", unit="run", leave=True):
        single_output_dir = train_config.output_dir if train_config.num_runs == 1 else train_config.output_dir / f"seed_{run_seed}"
        single_output_dir.mkdir(parents=True, exist_ok=True)
        metrics = _run_single_dgst_probe_training(samples, train_config, run_seed=run_seed, single_output_dir=single_output_dir)
        metrics["seed"] = run_seed
        run_metrics.append(metrics)
    finished_at = datetime.now(timezone.utc)

    if train_config.num_runs == 1:
        metrics = run_metrics[0]
    else:
        mean_auroc = sum(item["auroc"] for item in run_metrics) / len(run_metrics)
        mean_aupr = sum(item["aupr"] for item in run_metrics) / len(run_metrics)
        std_auroc = (sum((item["auroc"] - mean_auroc) ** 2 for item in run_metrics) / len(run_metrics)) ** 0.5
        std_aupr = (sum((item["aupr"] - mean_aupr) ** 2 for item in run_metrics) / len(run_metrics)) ** 0.5
        metrics = {
            "run_type": "dgst_probe",
            "run_name": train_config.output_dir.name,
            "dataset_file": str(train_config.dataset_file),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": float((finished_at - started_at).total_seconds()),
            "positive_class": "non_hallucination",
            "num_runs": int(train_config.num_runs),
            "seeds": run_seeds,
            "mean_auroc": float(mean_auroc),
            "std_auroc": float(std_auroc),
            "mean_aupr": float(mean_aupr),
            "std_aupr": float(std_aupr),
            "runs": [
                {
                    "seed": int(item["seed"]),
                    "run_dir": str((train_config.output_dir / f"seed_{item['seed']}").resolve()),
                    "auroc": float(item["auroc"]),
                    "aupr": float(item["aupr"]),
                    "best_epoch": int(item["best_epoch"]),
                    "best_val_loss": float(item["best_val_loss"]),
                }
                for item in run_metrics
            ],
            "notes": ["object-level DGST probe", "reported metrics are averaged across multiple random seeds"],
        }
        write_json(train_config.output_dir / "metrics.json", metrics)

    append_experiment_log(
        project_paths().experiment_log_file,
        {
            "run_type": "dgst_probe",
            "run_name": train_config.output_dir.name,
            "dataset_file": str(train_config.dataset_file),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": float((finished_at - started_at).total_seconds()),
            "positive_class": metrics.get("positive_class", "non_hallucination"),
            "auroc": float(metrics.get("auroc", metrics.get("mean_auroc"))),
            "aupr": float(metrics.get("aupr", metrics.get("mean_aupr"))),
            "metrics_file": str(train_config.output_dir / "metrics.json"),
        },
    )
    print(f"Saved DGST probe run to: {train_config.output_dir.resolve()}")
    return metrics


def _resolve_dgst_probe_artifacts(eval_config: ProbeEvalConfig) -> tuple[Path, Path, str]:
    paths = project_paths()
    if eval_config.probe_run:
        run_dir = paths.probe_runs_dir / eval_config.probe_run
        model_file = run_dir / "model.pth"
        config_file = run_dir / "config.json"
        probe_name = eval_config.probe_run
        if not model_file.exists() or not config_file.exists():
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                metrics = read_json(metrics_file)
                candidate_runs = metrics.get("runs", [])
                ranked_candidates = sorted(
                    (
                        item
                        for item in candidate_runs
                        if item.get("run_dir")
                    ),
                    key=lambda item: (
                        float(item.get("auroc", float("-inf"))),
                        float(item.get("aupr", float("-inf"))),
                    ),
                    reverse=True,
                )
                for candidate in ranked_candidates:
                    candidate_dir = Path(candidate["run_dir"])
                    candidate_model = candidate_dir / "model.pth"
                    candidate_config = candidate_dir / "config.json"
                    if candidate_model.exists() and candidate_config.exists():
                        model_file = candidate_model
                        config_file = candidate_config
                        probe_name = f"{eval_config.probe_run}:{candidate_dir.name}"
                        break
    else:
        if eval_config.model_file is None or eval_config.config_file is None:
            raise ValueError("Provide either probe_run or both model_file and config_file.")
        model_file = eval_config.model_file
        config_file = eval_config.config_file
        probe_name = model_file.parent.name
    if not model_file.exists():
        raise FileNotFoundError(f"DGST probe model not found: {model_file}")
    if not config_file.exists():
        raise FileNotFoundError(f"DGST probe config not found: {config_file}")
    return model_file, config_file, probe_name


def _load_dgst_probe_eval_samples(eval_config: ProbeEvalConfig) -> tuple[list[DGSTProbeSample], str]:
    if eval_config.input_format == "probe_dataset":
        if eval_config.dataset_file is None:
            raise ValueError("dataset_file is required when input_format=probe_dataset")
        if not eval_config.dataset_file.exists():
            raise FileNotFoundError(f"DGST probe dataset file not found: {eval_config.dataset_file}")
        return read_dgst_probe_dataset(eval_config.dataset_file), str(eval_config.dataset_file)
    if eval_config.result_file is None:
        raise ValueError("result_file is required when input_format=results_jsonl")
    if not eval_config.result_file.exists():
        raise FileNotFoundError(f"Result file not found: {eval_config.result_file}")
    rows = read_jsonl(eval_config.result_file)
    return build_dgst_probe_samples(rows), str(eval_config.result_file)


def _build_dgst_probe_model(config_file: Path, model_file: Path, device):
    import torch

    from .probe import DGSTProbe, DGSTProbeConfig

    config_payload = read_json(config_file)
    config = DGSTProbeConfig(**config_payload)
    model = DGSTProbe(input_dim=config.input_dim).to(device)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config_payload


def _predict_dgst_non_hallucination_probabilities(model, samples: Sequence[DGSTProbeSample], device) -> list[float]:
    import torch

    if not samples:
        raise ValueError("No DGST probe samples found in evaluation input.")
    features = torch.tensor([sample.object_layer_dgst_risk for sample in samples], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(features).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().tolist()
    return [float(value) for value in probabilities]


def _build_dgst_probe_prediction_rows(
    samples: Sequence[DGSTProbeSample],
    non_hall_probs: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, probability in zip(samples, non_hall_probs):
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_id": int(sample.image_id),
                "image": sample.image,
                "image_path": sample.image_path,
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


def _summarize_dgst_probe_eval_samples(samples: Sequence[DGSTProbeSample]) -> dict[str, Any]:
    image_ids = {int(sample.image_id) for sample in samples}
    token_aligned = sum(int(sample.token_aligned) for sample in samples)
    return {
        "count": int(len(samples)),
        "positives_hallucination": int(sum(int(sample.hallucinated) for sample in samples)),
        "negatives_hallucination": int(sum(int(1 - sample.hallucinated) for sample in samples)),
        "images": int(len(image_ids)),
        "token_aligned": int(token_aligned),
        "layer_widths": sorted({len(sample.object_layer_dgst_risk) for sample in samples}),
    }


def _infer_dgst_dataset_metadata(result_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not result_rows:
        return {}
    first = result_rows[0]
    return {
        "dataset": first.get("dataset"),
        "dataset_version": first.get("dataset_version"),
        "split": first.get("split"),
        "protocol": first.get("protocol"),
        "taxonomy_space": first.get("taxonomy_space"),
        "lexicon_version": first.get("lexicon_version"),
        "mention_linker_backend": first.get("mention_linker_backend"),
        "taxonomy_backend": first.get("taxonomy_backend"),
    }


def evaluate_dgst_probe_run(eval_config: ProbeEvalConfig) -> dict[str, Any]:
    model_file, config_file, probe_name = _resolve_dgst_probe_artifacts(eval_config)
    samples, source_path = _load_dgst_probe_eval_samples(eval_config)
    if not samples:
        raise ValueError("DGST evaluation input produced zero samples.")
    result_rows = read_jsonl(source_path) if eval_config.input_format == "results_jsonl" else []
    dataset_metadata = _infer_dgst_dataset_metadata(result_rows)

    paths = project_paths()
    if eval_config.output_dir is not None:
        output_dir = eval_config.output_dir
    else:
        source_stem = Path(source_path).stem
        run_name = eval_config.run_name or f"{probe_name}_on_{source_stem}"
        output_dir = paths.probe_evals_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _choose_device(eval_config.device)
    started_at = datetime.now(timezone.utc)
    model, config_payload = _build_dgst_probe_model(config_file, model_file, device)
    non_hall_probs = _predict_dgst_non_hallucination_probabilities(model, samples, device)
    finished_at = datetime.now(timezone.utc)

    non_hall_labels = [int(1 - sample.hallucinated) for sample in samples]
    hall_labels = [int(sample.hallucinated) for sample in samples]
    hallucination_probs = [float(1.0 - value) for value in non_hall_probs]
    non_hall_metrics = _compute_binary_metrics(non_hall_labels, non_hall_probs)
    hallucination_metrics = _compute_binary_metrics(hall_labels, hallucination_probs)
    predictions = _build_dgst_probe_prediction_rows(samples, non_hall_probs)
    prediction_file = output_dir / "predictions.jsonl"
    metrics_file = output_dir / "metrics.json"
    write_jsonl(prediction_file, predictions)

    metrics = {
        "run_type": "dgst_probe_eval",
        "probe_run": probe_name,
        "probe_model_file": str(model_file),
        "probe_config_file": str(config_file),
        "probe_config": config_payload,
        "evaluation_source": source_path,
        "input_format": eval_config.input_format,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float((finished_at - started_at).total_seconds()),
        "device": str(device),
        "evaluation_sample_summary": _summarize_dgst_probe_eval_samples(samples),
        "positive_class": "non_hallucination",
        "label_definition": {
            "non_hallucination": {"label": 1, "score": "non_hallucination_probability"},
            "hallucination": {"label": 1, "score": "hallucination_probability"},
        },
        "non_hallucination_metrics": non_hall_metrics,
        "hallucination_metrics": hallucination_metrics,
        "artifacts": {"predictions": str(prediction_file), "metrics": str(metrics_file)},
        "notes": [
            "DGST probe was trained with non_hallucination as the positive class.",
            "Both non-hallucination-positive and hallucination-positive metrics are reported for clarity.",
        ],
    }
    metrics.update({key: value for key, value in dataset_metadata.items() if value is not None})
    write_json(metrics_file, metrics)
    append_experiment_log(
        paths.experiment_log_file,
        {
            "run_type": "dgst_probe_eval",
            "run_name": output_dir.name,
            "probe_run": probe_name,
            "evaluation_source": source_path,
            "dataset": dataset_metadata.get("dataset"),
            "dataset_version": dataset_metadata.get("dataset_version"),
            "split": dataset_metadata.get("split"),
            "protocol": dataset_metadata.get("protocol"),
            "positive_class": "non_hallucination",
            "count": non_hall_metrics["count"],
            "auroc": float(non_hall_metrics["auroc"]),
            "aupr": float(non_hall_metrics["aupr"]),
            "hallucination_auroc": float(hallucination_metrics["auroc"]),
            "hallucination_aupr": float(hallucination_metrics["aupr"]),
            "metrics_file": str(metrics_file),
        },
    )
    print(f"Saved DGST probe evaluation to: {output_dir.resolve()}")
    return metrics


def build_dataset_cache(dataset_spec) -> dict[str, Any]:
    stage_progress = tqdm(total=3, desc=f"Building cache for {dataset_spec.dataset}", unit="step", leave=True)
    adapter = create_dataset_adapter(
        dataset=dataset_spec.dataset,
        dataset_root=dataset_spec.dataset_root,
        annotation_path=dataset_spec.annotation_path,
        split=dataset_spec.split,
        cache_path=None,
        lexicon_path=dataset_spec.lexicon_file,
    )
    stage_progress.update(1)
    adapter.save_cache(dataset_spec.adapter_cache)
    stage_progress.update(1)
    adapter.save_ground_truth_jsonl(dataset_spec.ground_truth_file, protocol=dataset_spec.protocol)
    stage_progress.update(1)
    stage_progress.close()
    summary = {
        "dataset": dataset_spec.dataset,
        "split": dataset_spec.split,
        "protocol": dataset_spec.protocol,
        "annotation_source": str(dataset_spec.annotation_path),
        "cache_file": str(dataset_spec.adapter_cache),
        "ground_truth_file": str(dataset_spec.ground_truth_file),
        "image_count": len(adapter.split_image_ids),
        "category_count": len(adapter.category_id_to_name),
        "reference_caption_images": len(adapter.image_id_to_references),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
