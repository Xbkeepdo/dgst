from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    DGSTConfig,
    DGSTEvalConfig,
    DGSTExportConfig,
    ProbeEvalConfig,
    ProbeTrainConfig,
    default_dgst_config,
    default_export_run_name,
    parse_csv_list,
    parse_gpus,
    parse_max_memory,
    project_paths,
    resolve_dataset_spec,
)


def _default_eval_run_name(*, prefix: str, dataset: str, protocol: str, num_data: int, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{dataset}_{protocol}_n{num_data}_s{seed}_{timestamp}"


def _resolve_eval_output_paths(
    *,
    args: argparse.Namespace,
    default_root: Path,
    default_evaluation_file: Path,
    default_plot_dir: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    result_file = Path(args.result_file) if args.result_file else None
    evaluation_file = Path(args.evaluation_file) if args.evaluation_file else None
    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    if evaluation_file == default_evaluation_file:
        evaluation_file = None
    if plot_dir == default_plot_dir:
        plot_dir = None

    if not getattr(args, "auto_name", False) and not getattr(args, "run_name", None) and not getattr(args, "experiment_dir", None):
        return result_file, Path(args.evaluation_file) if args.evaluation_file else None, Path(args.plot_dir) if args.plot_dir else None

    if args.experiment_dir:
        base_dir = Path(args.experiment_dir)
        if args.run_name:
            base_dir = base_dir / args.run_name
        elif args.auto_name:
            base_dir = base_dir / _default_eval_run_name(
                prefix="eval",
                dataset=args.dataset,
                protocol=args.protocol,
                num_data=args.num_data,
                seed=args.seed,
            )
    else:
        run_name = args.run_name or _default_eval_run_name(
            prefix="eval",
            dataset=args.dataset,
            protocol=args.protocol,
            num_data=args.num_data,
            seed=args.seed,
        )
        base_dir = default_root / run_name

    return (
        result_file or (base_dir / "results.jsonl"),
        evaluation_file or (base_dir / "evaluation.json"),
        plot_dir or (base_dir / "plots"),
    )


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--split", default="val")
    parser.add_argument("--protocol", choices=["native", "coco_overlap"], default="native")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--adapter-cache", default=None)
    parser.add_argument("--ground-truth-file", default=None)
    parser.add_argument("--lexicon-file", default=None)


def _add_dgst_args(parser: argparse.ArgumentParser) -> None:
    defaults = default_dgst_config()
    parser.add_argument("--lvlm", default=defaults.lvlm)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=defaults.dtype)
    parser.add_argument("--device-map", default=defaults.device_map)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=defaults.local_files_only,
    )
    parser.add_argument("--attn-implementation", default=defaults.attn_implementation)
    parser.add_argument("--prompt", default=defaults.prompt)
    parser.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--inference-temp", type=float, default=defaults.inference_temp)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--tau", type=float, default=defaults.tau)
    parser.add_argument("--transport-top-k", type=int, default=defaults.transport_top_k)
    parser.add_argument("--gamma1", type=float, default=defaults.gamma1)
    parser.add_argument("--gamma2", type=float, default=defaults.gamma2)
    parser.add_argument("--baseline-layers", type=int, default=defaults.baseline_layers)
    parser.add_argument("--risk-start-layer", type=int, default=defaults.risk_start_layer)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--ot-solver", default=defaults.ot_solver)


def _build_dgst_config(args: argparse.Namespace) -> DGSTConfig:
    return DGSTConfig(
        lvlm=args.lvlm,
        dtype=args.dtype,
        device_map=args.device_map,
        max_memory=parse_max_memory(args.max_memory),
        local_files_only=bool(args.local_files_only),
        attn_implementation=args.attn_implementation,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        inference_temp=args.inference_temp,
        top_p=args.top_p,
        tau=args.tau,
        transport_top_k=args.transport_top_k,
        gamma1=args.gamma1,
        gamma2=args.gamma2,
        baseline_layers=args.baseline_layers,
        risk_start_layer=args.risk_start_layer,
        alpha=args.alpha,
        ot_solver=args.ot_solver,
    )


def _build_dataset_spec(args: argparse.Namespace):
    return resolve_dataset_spec(
        args.dataset,
        split=args.split,
        protocol=args.protocol,
        dataset_root=args.dataset_root,
        annotation_path=args.annotation_path,
        adapter_cache=args.adapter_cache,
        ground_truth_file=args.ground_truth_file,
        lexicon_file=args.lexicon_file,
    )


def build_parser() -> argparse.ArgumentParser:
    paths = project_paths()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    _add_dgst_args(single)
    single.add_argument("--image", default=str(paths.default_single_image))
    single.add_argument("--answer", default=None)
    single.add_argument("--target-mode", choices=["all", "objects"], default="objects")
    single.add_argument("--object-phrases", default="")
    single.add_argument("--output", default=str(paths.single_output_file))

    evaluate = subparsers.add_parser("evaluate")
    _add_dataset_args(evaluate)
    _add_dgst_args(evaluate)
    evaluate.add_argument("--num-data", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--gpus", default="0")
    evaluate.add_argument("--caption-file", default=None)
    evaluate.add_argument("--result-file", default=None)
    evaluate.add_argument("--evaluation-file", default=str(paths.evaluation_file))
    evaluate.add_argument("--plot-dir", default=str(paths.plots_dir))
    evaluate.add_argument("--image-ids-file", default=None)
    evaluate.add_argument("--category-top-k", type=int, default=5)
    evaluate.add_argument("--category-min-count", type=int, default=8)
    evaluate.add_argument("--experiment-dir", default=None)
    evaluate.add_argument("--run-name", default=None)
    evaluate.add_argument("--auto-name", action="store_true")
    evaluate.add_argument("--no-reuse-captions", action="store_true")

    export_probe = subparsers.add_parser("export-probe")
    _add_dataset_args(export_probe)
    _add_dgst_args(export_probe)
    export_probe.add_argument("--num-data", type=int, default=None)
    export_probe.add_argument("--seed", type=int, default=0)
    export_probe.add_argument("--gpus", default="0,1")
    export_probe.add_argument("--run-name", default=None)
    export_probe.add_argument("--output-file", default=None)
    export_probe.add_argument("--manifest-file", default=None)
    export_probe.add_argument("--work-dir", default=None)
    export_probe.add_argument("--no-reuse-captions", action="store_true")

    train_probe = subparsers.add_parser("train-probe")
    train_probe.add_argument("--dataset-file", default=None)
    train_probe.add_argument("--run-name", default=None)
    train_probe.add_argument("--output-dir", default=None)
    train_probe.add_argument("--batch-size", type=int, default=16)
    train_probe.add_argument("--num-epochs", type=int, default=100)
    train_probe.add_argument("--learning-rate", type=float, default=1e-3)
    train_probe.add_argument("--weight-decay", type=float, default=1e-5)
    train_probe.add_argument("--lr-factor", type=float, default=0.5)
    train_probe.add_argument("--lr-patience", type=int, default=5)
    train_probe.add_argument("--test-size", type=float, default=0.2)
    train_probe.add_argument("--seed", type=int, default=0)
    train_probe.add_argument("--split-file", default=None)
    train_probe.add_argument("--num-runs", type=int, default=1)
    train_probe.add_argument("--paper-config", action="store_true")

    eval_probe = subparsers.add_parser("eval-probe")
    eval_probe.add_argument("--probe-run", default=None)
    eval_probe.add_argument("--model-file", default=None)
    eval_probe.add_argument("--config-file", default=None)
    eval_probe.add_argument("--result-file", default=None)
    eval_probe.add_argument("--dataset-file", default=None)
    eval_probe.add_argument("--input-format", choices=["results_jsonl", "probe_dataset"], default="results_jsonl")
    eval_probe.add_argument("--run-name", default=None)
    eval_probe.add_argument("--output-dir", default=None)
    eval_probe.add_argument("--device", default=None)

    build_cache = subparsers.add_parser("build-cache")
    _add_dataset_args(build_cache)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = project_paths()

    if args.command == "single":
        from .tasks import run_dgst_single_analysis

        run_dgst_single_analysis(
            image_path=args.image,
            output_path=args.output,
            dgst_config=_build_dgst_config(args),
            answer=args.answer,
            target_mode=args.target_mode,
            object_phrases=parse_csv_list(args.object_phrases),
        )
        return

    if args.command == "evaluate":
        from .tasks import run_dgst_dataset_evaluation

        result_file, evaluation_file, plot_dir = _resolve_eval_output_paths(
            args=args,
            default_root=paths.outputs_dir,
            default_evaluation_file=paths.evaluation_file,
            default_plot_dir=paths.plots_dir,
        )
        eval_config = DGSTEvalConfig(
            dataset_spec=_build_dataset_spec(args),
            dgst_config=_build_dgst_config(args),
            num_data=args.num_data,
            seed=args.seed,
            gpus=parse_gpus(args.gpus),
            reuse_captions=not args.no_reuse_captions,
            caption_file=Path(args.caption_file) if args.caption_file else None,
            result_file=result_file,
            evaluation_file=evaluation_file,
            plot_dir=plot_dir,
            image_ids_file=Path(args.image_ids_file) if args.image_ids_file else None,
            category_top_k=args.category_top_k,
            category_min_count=args.category_min_count,
        )
        summary = run_dgst_dataset_evaluation(eval_config)
        if eval_config.evaluation_file is not None:
            print(f"Saved DGST evaluation summary to: {eval_config.evaluation_file.resolve()}")
        print(summary["global_mean_metrics"])
        return

    if args.command == "export-probe":
        from .tasks import export_dgst_probe_dataset

        dataset_spec = _build_dataset_spec(args)
        default_run_name = default_export_run_name(dataset_spec.dataset, dataset_spec.protocol, args.num_data)
        export_config = DGSTExportConfig(
            dataset_spec=dataset_spec,
            dgst_config=_build_dgst_config(args),
            num_data=args.num_data,
            seed=args.seed,
            gpus=parse_gpus(args.gpus),
            reuse_captions=not args.no_reuse_captions,
            run_name=args.run_name or default_run_name,
            output_file=Path(args.output_file) if args.output_file else None,
            manifest_file=Path(args.manifest_file) if args.manifest_file else None,
            work_dir=Path(args.work_dir) if args.work_dir else None,
        )
        manifest = export_dgst_probe_dataset(export_config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    if args.command == "train-probe":
        from .tasks import train_dgst_probe_run

        dataset_file = Path(args.dataset_file) if args.dataset_file else paths.probe_data_dir / "probe_dataset.jsonl"
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else paths.probe_runs_dir / (args.run_name or datetime.now(timezone.utc).strftime("probe_%Y%m%d_%H%M%S"))
        )
        metrics = train_dgst_probe_run(
            ProbeTrainConfig(
                dataset_file=dataset_file,
                output_dir=output_dir,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                lr_factor=args.lr_factor,
                lr_patience=args.lr_patience,
                test_size=args.test_size,
                seed=args.seed,
                split_file=Path(args.split_file) if args.split_file else None,
                num_runs=args.num_runs,
                paper_config=bool(args.paper_config),
            )
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "eval-probe":
        from .tasks import evaluate_dgst_probe_run

        metrics = evaluate_dgst_probe_run(
            ProbeEvalConfig(
                probe_run=args.probe_run,
                model_file=Path(args.model_file) if args.model_file else None,
                config_file=Path(args.config_file) if args.config_file else None,
                result_file=Path(args.result_file) if args.result_file else None,
                dataset_file=Path(args.dataset_file) if args.dataset_file else None,
                input_format=args.input_format,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                run_name=args.run_name,
                device=args.device,
            )
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "build-cache":
        from .tasks import build_dataset_cache

        summary = build_dataset_cache(_build_dataset_spec(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
