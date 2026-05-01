from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from tqdm import tqdm

from ..config import (
    DGSTEvalConfig,
    DGSTExportConfig,
    ProbeEvalConfig,
    ProbeTrainConfig,
    apply_paper_probe_defaults,
    default_export_run_name,
    project_paths,
)
from ..probe import (
    DGSTProbe,
    DGSTProbeConfig,
    DGSTProbeSample,
    aggregate_dgst_probe_samples_by_sentence,
    build_sentence_entries_from_mentions,
    build_dgst_probe_samples,
    build_dgst_val_prediction_rows,
    get_probe_feature_vector,
    read_dgst_probe_dataset,
    split_dgst_probe_samples_by_image,
    split_dgst_probe_samples_with_fixed_image_ids,
    summarize_dgst_probe_samples,
    train_dgst_probe,
    write_dgst_probe_dataset,
)
from ..reporting import plot_pr_curve, plot_roc_curve
from .common import (
    append_experiment_log,
    append_jsonl,
    build_dgst_analyzer,
    choose_device,
    compute_binary_metrics,
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
from .run import _build_dgst_eval_subprocess_command


def export_dgst_probe_dataset(export_config: DGSTExportConfig) -> dict[str, Any]:
    spec = export_config.dataset_spec
    run_name = export_config.run_name or default_export_run_name(spec.dataset, spec.protocol, export_config.num_data)
    base_dir = project_paths().probe_data_dir / run_name
    output_file = export_config.output_file or (base_dir / "probe_dataset.jsonl")
    manifest_file = export_config.manifest_file or (base_dir / "probe_dataset_manifest.json")
    work_dir = export_config.work_dir or (base_dir / "shards")
    work_dir.mkdir(parents=True, exist_ok=True)

    from ..config import DGSTEvalConfig

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
    sampled_image_ids, adapter = load_sampled_image_ids(eval_config)
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

    exit_codes = wait_for_processes(
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
            "target_aggregation": export_config.dgst_config.target_aggregation,
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


def build_sentence_level_dgst_probe_dataset(
    *,
    input_file: Path,
    output_file: Path,
    manifest_file: Path,
    feature_aggregation: str = "mean",
) -> dict[str, Any]:
    source_samples = read_dgst_probe_dataset(input_file)
    sentence_samples = aggregate_dgst_probe_samples_by_sentence(
        source_samples,
        feature_aggregation=feature_aggregation,
    )
    write_dgst_probe_dataset(output_file, sentence_samples)

    manifest = {
        "run_type": "dgst_sentence_probe_dataset",
        "source_file": str(input_file),
        "output_file": str(output_file),
        "aggregation_unit": "sentence",
        "feature_aggregation": feature_aggregation,
        "label_definition": {
            "1": "sentence contains at least one hallucinated object mention",
            "0": "all detected object mentions in sentence are non-hallucinated",
        },
        "source_sample_summary": summarize_dgst_probe_samples(source_samples),
        "probe_sample_summary": summarize_dgst_probe_samples(sentence_samples),
        "notes": [
            "derived from object-level DGST probe samples without another LVLM forward pass",
            "features are aggregated across object mentions inside each generated sentence",
        ],
    }
    write_json(manifest_file, manifest)
    print(f"Saved sentence-level DGST probe dataset to: {output_file.resolve()}")
    print(f"Saved sentence-level manifest to: {manifest_file.resolve()}")
    return manifest


def _build_sentence_last_worker_command(eval_config: DGSTEvalConfig) -> list[str]:
    spec = eval_config.dataset_spec
    cfg = eval_config.dgst_config
    command = [
        sys.executable,
        "-m",
        "dgst.cli",
        "sentence-last-worker",
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
        "--target-aggregation",
        cfg.target_aggregation,
        "--gpus",
        ",".join(eval_config.gpus),
        "--seed",
        str(eval_config.seed),
    ]
    if eval_config.num_data is not None:
        command.extend(["--num-data", str(eval_config.num_data)])
    if spec.lexicon_file is not None:
        command.extend(["--lexicon-file", str(spec.lexicon_file)])
    command.append("--local-files-only" if cfg.local_files_only else "--no-local-files-only")
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
    if eval_config.image_ids_file is not None:
        command.extend(["--image-ids-file", str(eval_config.image_ids_file)])
    if not eval_config.reuse_captions:
        command.append("--no-reuse-captions")
    if eval_config.sentence_last_generation_time:
        command.append("--generation-time")
    return command


def run_dgst_sentence_last_worker(eval_config: DGSTEvalConfig) -> dict[str, Any]:
    spec = eval_config.dataset_spec
    cfg = eval_config.dgst_config
    started_at = datetime.now(timezone.utc)
    image_ids, adapter = load_sampled_image_ids(eval_config)
    metadata = adapter.protocol_metadata(spec.protocol)
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

    processed_images = 0
    total_sentences = 0
    aligned_sentences = 0
    dropped_sentences = 0
    total_mentions = 0
    rows: list[dict[str, Any]] = []
    progress = tqdm(image_ids, total=len(image_ids), desc=f"Exporting DGST sentence-last {spec.dataset}:{spec.protocol}")
    for image_id in progress:
        image_id = int(image_id)
        expected_feature_timing = (
            "generation_time" if eval_config.sentence_last_generation_time else "teacher_forced_replay"
        )
        if (
            image_id in result_data
            and str(result_data[image_id].get("target_unit")) == "sentence_last_token"
            and str(result_data[image_id].get("feature_timing", "teacher_forced_replay")) == expected_feature_timing
        ):
            row = result_data[image_id]
            rows.append(row)
            processed_images += 1
            total_sentences += int(row.get("sentence_count_total", len(row.get("sentence_mentions", []))))
            aligned_sentences += int(row.get("aligned_sentence_sample_count", len(row.get("dgst_sentence_scores", []))))
            dropped_sentences += int(row.get("dropped_unaligned_sentence_count", 0))
            total_mentions += int(row.get("chair_word_count_total", len(row.get("object_mentions", []))))
            progress.set_postfix(image_id=image_id, cached="result")
            continue

        image_name = str(adapter.image_filename(image_id))
        image_path = spec.dataset_root / image_name
        if not image_path.exists():
            print(f"Skip missing image: {image_path}")
            continue
        question = (
            adapter.question_for_image_id(image_id)
            if hasattr(adapter, "question_for_image_id")
            else cfg.prompt
        )
        caption_from_dataset = (
            adapter.caption_for_image_id(image_id)
            if hasattr(adapter, "caption_for_image_id")
            else None
        )
        cached_caption = caption_data.get(image_id)
        generated_answer: dict[str, Any] | None = None
        use_generation_time = bool(eval_config.sentence_last_generation_time and caption_from_dataset is None)
        if use_generation_time:
            generated_answer = analyzer.generate_answer_with_ids(
                image_path=image_path,
                question=question,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.inference_temp > 0.0,
                temperature=cfg.inference_temp,
                top_p=cfg.top_p,
            )
            caption = str(generated_answer["answer"])
            if eval_config.caption_file is not None:
                cached_caption = {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "prompt": question,
                    "caption": caption,
                    "feature_timing": "generation_time",
                }
                append_jsonl(eval_config.caption_file, cached_caption)
                caption_data[image_id] = cached_caption
        elif caption_from_dataset is not None:
            caption = str(caption_from_dataset)
        elif cached_caption is not None and eval_config.reuse_captions:
            caption = str(cached_caption["caption"])
        else:
            caption = analyzer.generate_answer(
                image_path=image_path,
                question=question,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.inference_temp > 0.0,
                temperature=cfg.inference_temp,
                top_p=cfg.top_p,
            )
            if eval_config.caption_file is not None:
                cached_caption = {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "prompt": question,
                    "caption": caption,
                }
                append_jsonl(eval_config.caption_file, cached_caption)
                caption_data[image_id] = cached_caption

        adapter_eval = adapter.evaluate_caption(image_id, caption, protocol=spec.protocol)
        sentence_entries = build_sentence_entries_from_mentions(
            caption,
            adapter_eval["object_mentions"],
            include_empty_sentences=True,
        )
        if generated_answer is not None:
            result = analyzer.analyze_generation_time_sentence_last(
                image_path=image_path,
                question=question,
                answer=caption,
                answer_token_ids=generated_answer["answer_token_ids"],
                sentence_mentions=sentence_entries,
                generation=generated_answer.get("generation"),
                tau=cfg.tau,
                transport_top_k=cfg.transport_top_k,
                gamma1=cfg.gamma1,
                gamma2=cfg.gamma2,
                baseline_layers=cfg.baseline_layers,
                risk_start_layer=cfg.risk_start_layer,
                alpha=cfg.alpha,
                ot_solver=cfg.ot_solver,
            )
        else:
            result = analyzer.analyze(
                image_path=image_path,
                question=question,
                answer=caption,
                target_mode="sentence_last",
                sentence_mentions=sentence_entries,
                tau=cfg.tau,
                transport_top_k=cfg.transport_top_k,
                gamma1=cfg.gamma1,
                gamma2=cfg.gamma2,
                baseline_layers=cfg.baseline_layers,
                risk_start_layer=cfg.risk_start_layer,
                alpha=cfg.alpha,
                ot_solver=cfg.ot_solver,
                target_aggregation="last",
            )
        sentence_scores = []
        label_by_sentence = {
            int(entry["sentence_index"]): int(entry.get("hallucinated", 0))
            for entry in sentence_entries
        }
        for item in result.get("dgst_sentence_scores", []):
            score_item = dict(item)
            sentence_index = int(score_item.get("sentence_index", score_item.get("mention_index", 0)))
            score_item["hallucinated"] = int(label_by_sentence.get(sentence_index, score_item.get("hallucinated", 0)))
            score_item["aggregation_unit"] = "sentence"
            score_item["label_rule"] = "positive_if_sentence_contains_any_hallucinated_object_mention"
            sentence_scores.append(score_item)

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
            "prompt": question,
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
            "sentence_mentions": sentence_entries,
            "dgst_sentence_alignment": result.get("sentence_alignment", []),
            "dgst_sentence_scores": sentence_scores,
            "sentence_count_total": len(sentence_entries),
            "aligned_sentence_sample_count": len(sentence_scores),
            "dropped_unaligned_sentence_count": max(0, len(sentence_entries) - len(sentence_scores)),
            "dgst_final_score_mean": result.get("dgst_final_score_mean"),
            "target_aggregation": "sentence_last",
            "target_unit": "sentence_last_token",
            "feature_timing": result.get("feature_timing", "teacher_forced_replay"),
            "dgst_config": result.get("dgst_config"),
        }
        row.update(
            {
                key: value
                for key, value in adapter_eval.items()
                if str(key).startswith(("amber_", "mhaldetect_", "pope_", "coco_"))
            }
        )
        rows.append(row)
        if eval_config.result_file is not None:
            append_jsonl(eval_config.result_file, row)
            result_data[image_id] = row
        processed_images += 1
        total_sentences += len(sentence_entries)
        aligned_sentences += len(sentence_scores)
        dropped_sentences += max(0, len(sentence_entries) - len(sentence_scores))
        total_mentions += int(adapter_eval["chair_word_count_total"])
        progress.set_postfix(image_id=image_id, aligned=f"{len(sentence_scores)}/{len(sentence_entries)}")
    progress.close()

    finished_at = datetime.now(timezone.utc)
    summary = {
        "run_type": "dgst_sentence_last_worker",
        "dataset": metadata["dataset"],
        "dataset_version": metadata["dataset_version"],
        "split": metadata["split"],
        "protocol": metadata["protocol"],
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float((finished_at - started_at).total_seconds()),
        "processed_images": int(processed_images),
        "sentence_count_total": int(total_sentences),
        "aligned_sentence_sample_count": int(aligned_sentences),
        "dropped_unaligned_sentence_count": int(dropped_sentences),
        "chair_word_count_total": int(total_mentions),
        "alignment_rate": float(aligned_sentences / total_sentences) if total_sentences else 0.0,
        "feature_timing": "generation_time" if eval_config.sentence_last_generation_time else "teacher_forced_replay",
    }
    if eval_config.evaluation_file is not None:
        write_json(eval_config.evaluation_file, summary)
    return summary


def export_dgst_sentence_last_probe_dataset(
    export_config: DGSTExportConfig,
    *,
    caption_file: Path | None = None,
) -> dict[str, Any]:
    spec = export_config.dataset_spec
    run_name = export_config.run_name or f"{default_export_run_name(spec.dataset, spec.protocol, export_config.num_data)}_sentence_last"
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
        caption_file=caption_file,
        result_file=None,
        evaluation_file=None,
        plot_dir=None,
        image_ids_file=None,
    )
    sampled_image_ids, adapter = load_sampled_image_ids(eval_config)
    adapter.save_ground_truth_jsonl(spec.ground_truth_file, image_ids=sampled_image_ids, protocol=spec.protocol)

    started_at = datetime.now(timezone.utc)
    shards = split_round_robin(sampled_image_ids, len(export_config.gpus))
    shard_meta: list[dict[str, Any]] = []
    processes: list[subprocess.Popen] = []
    for shard_index, (gpu_id, shard_ids) in enumerate(zip(export_config.gpus, shards)):
        ids_file = work_dir / f"shard_{shard_index}_ids.txt"
        shard_caption_file = caption_file or (work_dir / f"shard_{shard_index}_captions.jsonl")
        result_file = work_dir / f"shard_{shard_index}_results.jsonl"
        summary_file = work_dir / f"shard_{shard_index}_summary.json"
        write_lines(ids_file, shard_ids)
        worker_config = DGSTEvalConfig(
            dataset_spec=spec,
            dgst_config=export_config.dgst_config,
            num_data=len(shard_ids),
            seed=export_config.seed,
            gpus=(gpu_id,),
            reuse_captions=export_config.reuse_captions and not export_config.sentence_last_generation_time,
            caption_file=shard_caption_file,
            result_file=result_file,
            evaluation_file=summary_file,
            plot_dir=None,
            image_ids_file=ids_file,
            sentence_last_generation_time=export_config.sentence_last_generation_time,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        process = subprocess.Popen(
            _build_sentence_last_worker_command(worker_config),
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
                "caption_file": str(shard_caption_file),
                "result_file": str(result_file),
                "summary_file": str(summary_file),
            }
        )

    exit_codes = wait_for_processes(
        processes,
        desc="Waiting for DGST sentence-last shards",
        labels=[f"gpu{meta['gpu']}" for meta in shard_meta],
        progress_paths=[Path(meta["result_file"]) for meta in shard_meta],
        total_items=sum(int(meta["image_count"]) for meta in shard_meta),
        item_desc="image",
    )
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more DGST sentence-last shard processes failed: {exit_codes}")

    all_rows: list[dict[str, Any]] = []
    processed_images = 0
    total_sentences = 0
    aligned_sentences = 0
    dropped_sentences = 0
    total_mentions = 0
    for meta in shard_meta:
        summary = read_json(meta["summary_file"])
        processed_images += int(summary["processed_images"])
        total_sentences += int(summary.get("sentence_count_total", 0))
        aligned_sentences += int(summary.get("aligned_sentence_sample_count", 0))
        dropped_sentences += int(summary.get("dropped_unaligned_sentence_count", 0))
        total_mentions += int(summary.get("chair_word_count_total", 0))
        all_rows.extend(read_jsonl(meta["result_file"]))

    dgst_probe_samples = build_dgst_probe_samples(all_rows)
    write_dgst_probe_dataset(output_file, dgst_probe_samples)

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "run_name": run_name,
        "run_type": "dgst_sentence_last_probe_export",
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
        "caption_file": str(caption_file) if caption_file is not None else None,
        "output_file": str(output_file),
        "processed_images": int(processed_images),
        "chair_word_count_total": int(total_mentions),
        "sentence_count_total": int(total_sentences),
        "aligned_sentence_sample_count": int(aligned_sentences),
        "dropped_unaligned_sentence_count": int(dropped_sentences),
        "alignment_rate": float(aligned_sentences / total_sentences) if total_sentences else 0.0,
        "probe_sample_summary": summarize_dgst_probe_samples(dgst_probe_samples),
        "label_definition": {
            "1": "sentence contains at least one hallucinated object mention",
            "0": "sentence contains no hallucinated object mentions",
        },
        "target_unit": "sentence_last_token",
        "feature_timing": "generation_time" if export_config.sentence_last_generation_time else "teacher_forced_replay",
        "dgst_config": {
            "tau": export_config.dgst_config.tau,
            "transport_top_k": export_config.dgst_config.transport_top_k,
            "gamma1": export_config.dgst_config.gamma1,
            "gamma2": export_config.dgst_config.gamma2,
            "baseline_layers": export_config.dgst_config.baseline_layers,
            "risk_start_layer": export_config.dgst_config.risk_start_layer,
            "alpha": export_config.dgst_config.alpha,
            "ot_solver": export_config.dgst_config.ot_solver,
            "target_aggregation": "sentence_last",
        },
        "shards": shard_meta,
        "notes": [
            "sentence-level probe dataset following the VIB-style last-token unit",
            (
                "one sample per generated sentence; feature is captured at the generation step that predicts the "
                "sentence final token"
                if export_config.sentence_last_generation_time
                else "one sample per generated sentence; feature is the sentence final token from replay forward"
            ),
        ],
    }
    write_json(manifest_file, manifest)
    print(f"Saved DGST sentence-last probe dataset to: {output_file.resolve()}")
    print(f"Saved DGST sentence-last manifest to: {manifest_file.resolve()}")
    return manifest


def _build_dgst_split(
    samples: Sequence[DGSTProbeSample],
    split_file: Path | None,
    test_size: float,
    seed: int,
    split_by: str,
) -> tuple[list[DGSTProbeSample], list[DGSTProbeSample], dict[str, Any]]:
    if split_file is not None:
        split_payload = read_json(split_file)
        resolved_split_by = str(split_payload.get("split_by") or split_by)
        train_samples, val_samples, split_summary = split_dgst_probe_samples_with_fixed_image_ids(
            samples,
            train_image_ids=split_payload.get("train_split_keys", split_payload["train_image_ids"]),
            val_image_ids=split_payload.get("val_split_keys", split_payload["val_image_ids"]),
            split_by=resolved_split_by,
        )
        split_summary["seed"] = split_payload.get("seed")
        split_summary["test_size"] = split_payload.get("test_size")
        split_summary["stratified_by_image_has_hallucination"] = split_payload.get(
            "stratified_by_image_has_hallucination"
        )
        split_summary["split_file"] = str(split_file)
        return train_samples, val_samples, split_summary
    return split_dgst_probe_samples_by_image(samples, test_size=test_size, seed=seed, split_by=split_by)


def _run_single_dgst_probe_training(
    samples: Sequence[DGSTProbeSample],
    train_config: ProbeTrainConfig,
    *,
    run_seed: int,
    single_output_dir: Path,
) -> dict[str, Any]:
    train_samples, val_samples, split_summary = _build_dgst_split(
        samples,
        train_config.split_file,
        train_config.test_size,
        run_seed,
        train_config.split_by,
    )
    sample_level = (
        "sentence-level"
        if any(str(sample.target_aggregation).startswith("sentence_") for sample in samples)
        else "object-level"
    )
    probe_config = DGSTProbeConfig(
        input_dim=len(get_probe_feature_vector(samples[0], train_config.feature_set)),
        feature_set=train_config.feature_set,
        positive_class="hallucination",
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
        build_dgst_val_prediction_rows(val_samples, best_metrics["hallucination_probabilities"]),
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
        "feature_set": train_config.feature_set,
        "feature_dim": int(probe_config.input_dim),
        "split_by": train_config.split_by,
        "positive_class": "hallucination",
        "label_definition": {"1": "hallucination", "0": "non_hallucination"},
        "train_sample_summary": summarize_dgst_probe_samples(train_samples),
        "val_sample_summary": summarize_dgst_probe_samples(val_samples),
        "best_epoch": int(training_result["best_epoch"]),
        "best_val_loss": float(training_result["best_val_loss"]),
        "auroc": float(best_metrics["auroc"]),
        "aupr": float(best_metrics["aupr"]),
        "primary_metric": (
            {
                "aggregation": "amber_subset_macro",
                "auroc": float(best_metrics["amber_subset_macro"]["macro_auroc"]),
                "aupr": float(best_metrics["amber_subset_macro"]["macro_aupr"]),
            }
            if best_metrics.get("amber_subset_macro")
            else (
                {
                    "aggregation": "pope_subset_macro",
                    "auroc": float(best_metrics["pope_subset_macro"]["macro_auroc"]),
                    "aupr": float(best_metrics["pope_subset_macro"]["macro_aupr"]),
                }
                if best_metrics.get("pope_subset_macro")
                else {
                    "aggregation": "pooled",
                    "auroc": float(best_metrics["auroc"]),
                    "aupr": float(best_metrics["aupr"]),
                }
            )
        ),
        **(
            {"amber_subset_macro": best_metrics["amber_subset_macro"]}
            if best_metrics.get("amber_subset_macro")
            else {}
        ),
        **(
            {"pope_subset_macro": best_metrics["pope_subset_macro"]}
            if best_metrics.get("pope_subset_macro")
            else {}
        ),
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
            f"{sample_level} DGST probe",
            "fixed split reused from prior run"
            if train_config.split_file is not None
            else f"{train_config.split_by}-level train/validation split with test_size={float(train_config.test_size):.6g}",
            f"split_by={train_config.split_by}",
            "positive class is hallucination",
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
            "positive_class": "hallucination",
            "feature_set": train_config.feature_set,
            "split_by": train_config.split_by,
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
            "positive_class": metrics.get("positive_class", "hallucination"),
            "feature_set": train_config.feature_set,
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
                    (item for item in candidate_runs if item.get("run_dir")),
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

    config_payload = read_json(config_file)
    config = DGSTProbeConfig(**config_payload)
    model = DGSTProbe(input_dim=config.input_dim).to(device)
    model._dgst_feature_set = str(getattr(config, "feature_set", "risk"))
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config_payload


def _predict_dgst_hallucination_probabilities(model, samples: Sequence[DGSTProbeSample], device) -> list[float]:
    import torch

    if not samples:
        raise ValueError("No DGST probe samples found in evaluation input.")
    feature_set = getattr(model, "_dgst_feature_set", "risk")
    features = torch.tensor(
        [get_probe_feature_vector(sample, feature_set) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits = model(features).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().tolist()
    return [float(value) for value in probabilities]


def _build_dgst_probe_prediction_rows(
    samples: Sequence[DGSTProbeSample],
    hall_probs: Sequence[float],
    *,
    feature_set: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, probability in zip(samples, hall_probs):
        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_id": int(sample.image_id),
                "split_group_id": int(sample.split_group_id) if sample.split_group_id is not None else int(sample.image_id),
                "image": sample.image,
                "image_path": sample.image_path,
                "canonical_name": sample.canonical_name,
                "surface": sample.surface,
                "phrase": sample.phrase,
                "hallucinated": int(sample.hallucinated),
                "hallucination_label": int(sample.hallucinated),
                "hallucination_probability": float(probability),
                "non_hallucination_probability": float(1.0 - probability),
                "dgst_final_score": float(sample.dgst_final_score),
                "feature_set": feature_set,
            }
        )
    return rows


def _summarize_dgst_probe_eval_samples(samples: Sequence[DGSTProbeSample]) -> dict[str, Any]:
    image_ids = {int(sample.image_id) for sample in samples}
    split_group_ids = {
        int(sample.split_group_id) if sample.split_group_id is not None else int(sample.image_id)
        for sample in samples
    }
    token_aligned = sum(int(sample.token_aligned) for sample in samples)
    return {
        "count": int(len(samples)),
        "positives_hallucination": int(sum(int(sample.hallucinated) for sample in samples)),
        "negatives_hallucination": int(sum(int(1 - sample.hallucinated) for sample in samples)),
        "images": int(len(image_ids)),
        "split_groups": int(len(split_group_ids)),
        "token_aligned": int(token_aligned),
        "layer_widths": sorted({len(sample.object_layer_dgst_risk) for sample in samples}),
        "probe6_widths": sorted({len(sample.object_layer_dgst_probe6) for sample in samples if sample.object_layer_dgst_probe6}),
        "prompt_widths": sorted(
            {len(sample.object_layer_prompt_token_cos) for sample in samples if sample.object_layer_prompt_token_cos}
        ),
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

    device = choose_device(eval_config.device)
    started_at = datetime.now(timezone.utc)
    model, config_payload = _build_dgst_probe_model(config_file, model_file, device)
    feature_set = str(config_payload.get("feature_set", "risk"))
    hall_probs = _predict_dgst_hallucination_probabilities(model, samples, device)
    finished_at = datetime.now(timezone.utc)

    non_hall_labels = [int(1 - sample.hallucinated) for sample in samples]
    hall_labels = [int(sample.hallucinated) for sample in samples]
    non_hall_probs = [float(1.0 - value) for value in hall_probs]
    hallucination_metrics = compute_binary_metrics(hall_labels, hall_probs)
    non_hall_metrics = compute_binary_metrics(non_hall_labels, non_hall_probs)
    predictions = _build_dgst_probe_prediction_rows(samples, hall_probs, feature_set=feature_set)
    prediction_file = output_dir / "predictions.jsonl"
    metrics_file = output_dir / "metrics.json"
    write_jsonl(prediction_file, predictions)

    metrics = {
        "run_type": "dgst_probe_eval",
        "probe_run": probe_name,
        "probe_model_file": str(model_file),
        "probe_config_file": str(config_file),
        "probe_config": config_payload,
        "feature_set": feature_set,
        "evaluation_source": source_path,
        "input_format": eval_config.input_format,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": float((finished_at - started_at).total_seconds()),
        "device": str(device),
        "evaluation_sample_summary": _summarize_dgst_probe_eval_samples(samples),
        "positive_class": "hallucination",
        "label_definition": {
            "hallucination": {"label": 1, "score": "hallucination_probability"},
            "non_hallucination": {"label": 1, "score": "non_hallucination_probability"},
        },
        "non_hallucination_metrics": non_hall_metrics,
        "hallucination_metrics": hallucination_metrics,
        "artifacts": {"predictions": str(prediction_file), "metrics": str(metrics_file)},
        "notes": [
            "DGST probe was trained with hallucination as the positive class.",
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
            "positive_class": "hallucination",
            "count": non_hall_metrics["count"],
            "auroc": float(hallucination_metrics["auroc"]),
            "aupr": float(hallucination_metrics["aupr"]),
            "hallucination_auroc": float(hallucination_metrics["auroc"]),
            "hallucination_aupr": float(hallucination_metrics["aupr"]),
            "metrics_file": str(metrics_file),
        },
    )
    print(f"Saved DGST probe evaluation to: {output_dir.resolve()}")
    return metrics
