from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from tqdm import tqdm

from ..config import DGSTConfig, DGSTEvalConfig, project_paths
from ..data.dataset_adapters import create_dataset_adapter
from ..reporting import (
    collect_dgst_object_records,
    plot_all_objects_layerwise,
    plot_category_layerwise,
    summarize_dgst_dataset_metrics,
)
from .common import (
    append_experiment_log,
    append_jsonl,
    build_dgst_analyzer,
    load_ground_truth_entries,
    load_sampled_image_ids,
    read_json,
    read_jsonl,
    split_round_robin,
    wait_for_processes,
    write_json,
    write_jsonl,
    write_lines,
)


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
    analyzer = build_dgst_analyzer(dgst_config)
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
    ground_truth_entries = load_ground_truth_entries(adapter, eval_config)

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
    analyzer = build_dgst_analyzer(cfg)

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

    sampled_image_ids, adapter = load_sampled_image_ids(eval_config)
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

        exit_codes = wait_for_processes(
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
