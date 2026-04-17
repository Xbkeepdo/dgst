from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any, Sequence

from tqdm import tqdm

from ..config import (
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
    build_dgst_probe_samples,
    build_dgst_val_prediction_rows,
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
    choose_device,
    compute_binary_metrics,
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

    device = choose_device(eval_config.device)
    started_at = datetime.now(timezone.utc)
    model, config_payload = _build_dgst_probe_model(config_file, model_file, device)
    non_hall_probs = _predict_dgst_non_hallucination_probabilities(model, samples, device)
    finished_at = datetime.now(timezone.utc)

    non_hall_labels = [int(1 - sample.hallucinated) for sample in samples]
    hall_labels = [int(sample.hallucinated) for sample in samples]
    hallucination_probs = [float(1.0 - value) for value in non_hall_probs]
    non_hall_metrics = compute_binary_metrics(non_hall_labels, non_hall_probs)
    hallucination_metrics = compute_binary_metrics(hall_labels, hallucination_probs)
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
