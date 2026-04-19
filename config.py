from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
USERDATA_ROOT = Path("/home/apulis-dev/userdata")


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = PROJECT_ROOT
    repo_root: Path = REPO_ROOT
    userdata_root: Path = USERDATA_ROOT
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    plots_dir: Path = PROJECT_ROOT / "analysis_plots"
    probe_data_dir: Path = PROJECT_ROOT / "probe_data"
    probe_runs_dir: Path = PROJECT_ROOT / "probe_runs"
    probe_evals_dir: Path = PROJECT_ROOT / "probe_evals"
    experiment_log_file: Path = PROJECT_ROOT / "experiment_log.jsonl"
    coco_ground_truth_file: Path = PROJECT_ROOT / "coco_ground_truth.json"
    objects365_ground_truth_file: Path = PROJECT_ROOT / "objects365_ground_truth.jsonl"
    chair_cache_file: Path = PROJECT_ROOT / "chair.pkl"
    objects365_cache_file: Path = PROJECT_ROOT / "objects365_v2_val.cache.json"
    evaluation_file: Path = PROJECT_ROOT / "evaluation_results.json"
    single_output_file: Path = PROJECT_ROOT / "outputs" / "dgst_result.json"

    @property
    def local_llava_model(self) -> Path:
        return self.userdata_root / "models" / "llava" / "llava-hf" / "llava-1___5-7b-hf"

    @property
    def default_single_image(self) -> Path:
        candidates = [
            self.project_root / "examples" / "sample.jpg",
            self.repo_root / "examples" / "sample.jpg",
            self.repo_root / "vicr" / "examples" / "sample.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    split: str
    protocol: str
    dataset_root: Path
    annotation_path: Path
    adapter_cache: Path
    ground_truth_file: Path
    lexicon_file: Path | None = None


@dataclass(frozen=True)
class DGSTConfig:
    lvlm: str
    dtype: str = "float16"
    device_map: str = "auto"
    max_memory: Mapping[int | str, str] | None = None
    local_files_only: bool = True
    attn_implementation: str | None = None
    prompt: str = "Describe the given image in detail."
    max_new_tokens: int = 512
    inference_temp: float = 0.0
    top_p: float = 1.0
    tau: float = 0.07
    transport_top_k: int = 32
    gamma1: float = 1.0
    gamma2: float = 1.0
    baseline_layers: int = 10
    risk_start_layer: int = 15
    alpha: float = 2.0
    ot_solver: str = "linprog"


@dataclass(frozen=True)
class DGSTEvalConfig:
    dataset_spec: DatasetSpec
    dgst_config: DGSTConfig
    num_data: int | None = 100
    seed: int = 0
    gpus: tuple[str, ...] = ("0",)
    reuse_captions: bool = True
    caption_file: Path | None = None
    result_file: Path | None = None
    evaluation_file: Path | None = None
    plot_dir: Path | None = None
    image_ids_file: Path | None = None
    category_top_k: int = 5
    category_min_count: int = 8


@dataclass(frozen=True)
class DGSTExportConfig:
    dataset_spec: DatasetSpec
    dgst_config: DGSTConfig
    num_data: int | None = None
    seed: int = 0
    gpus: tuple[str, ...] = ("0", "1")
    reuse_captions: bool = True
    run_name: str | None = None
    output_file: Path | None = None
    manifest_file: Path | None = None
    work_dir: Path | None = None


@dataclass(frozen=True)
class ProbeTrainConfig:
    dataset_file: Path
    output_dir: Path
    feature_set: str = "risk"
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 5
    test_size: float = 0.2
    seed: int = 0
    split_file: Path | None = None
    num_runs: int = 1
    paper_config: bool = False


@dataclass(frozen=True)
class ProbeEvalConfig:
    probe_run: str | None = None
    model_file: Path | None = None
    config_file: Path | None = None
    result_file: Path | None = None
    dataset_file: Path | None = None
    input_format: str = "results_jsonl"
    output_dir: Path | None = None
    run_name: str | None = None
    device: str | None = None


def project_paths() -> ProjectPaths:
    return ProjectPaths()


def normalize_dataset_name(dataset: str) -> str:
    normalized = str(dataset).strip().lower()
    if normalized == "mscoco":
        return "coco"
    return normalized


def default_dataset_registry(paths: ProjectPaths | None = None) -> dict[str, dict[str, Path]]:
    paths = paths or project_paths()
    userdata = paths.userdata_root
    objects365_annotation = userdata / "objects365v2" / "annotations" / "zhiyuan_objv2_val_fixname.json"
    if not objects365_annotation.exists():
        objects365_annotation = userdata / "objects365v2" / "annotations" / "zhiyuan_objv2_val.json"
    return {
        "coco": {
            "dataset_root": userdata / "val2014",
            "annotation_path": userdata / "annotations",
            "adapter_cache": paths.chair_cache_file,
            "ground_truth_file": paths.coco_ground_truth_file,
        },
        "objects365": {
            "dataset_root": userdata / "objects365v2" / "images" / "val",
            "annotation_path": objects365_annotation,
            "adapter_cache": paths.objects365_cache_file,
            "ground_truth_file": paths.objects365_ground_truth_file,
            "lexicon_file": PROJECT_ROOT / "resources" / "objects365_aliases.json",
        },
    }


def resolve_dataset_spec(
    dataset: str,
    *,
    split: str = "val",
    protocol: str = "native",
    dataset_root: str | Path | None = None,
    annotation_path: str | Path | None = None,
    adapter_cache: str | Path | None = None,
    ground_truth_file: str | Path | None = None,
    lexicon_file: str | Path | None = None,
    paths: ProjectPaths | None = None,
) -> DatasetSpec:
    paths = paths or project_paths()
    normalized = normalize_dataset_name(dataset)
    registry = default_dataset_registry(paths)
    if normalized not in registry:
        raise ValueError(f"Unsupported dataset: {dataset}")
    defaults = registry[normalized]
    return DatasetSpec(
        dataset=normalized,
        split=str(split),
        protocol=str(protocol),
        dataset_root=Path(dataset_root) if dataset_root is not None else defaults["dataset_root"],
        annotation_path=Path(annotation_path) if annotation_path is not None else defaults["annotation_path"],
        adapter_cache=Path(adapter_cache) if adapter_cache is not None else defaults["adapter_cache"],
        ground_truth_file=Path(ground_truth_file) if ground_truth_file is not None else defaults["ground_truth_file"],
        lexicon_file=Path(lexicon_file) if lexicon_file is not None else defaults.get("lexicon_file"),
    )


def default_dgst_config(paths: ProjectPaths | None = None) -> DGSTConfig:
    paths = paths or project_paths()
    return DGSTConfig(
        lvlm=str(paths.local_llava_model),
        local_files_only=True,
    )


def parse_max_memory(text: str | None) -> dict[int | str, str] | None:
    if text is None or not text.strip():
        return None
    parsed: dict[int | str, str] = {}
    for item in text.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid max-memory item: {entry!r}")
        key_text, value = entry.split("=", 1)
        key_text = key_text.strip()
        value = value.strip()
        if not key_text or not value:
            raise ValueError(f"Invalid max-memory item: {entry!r}")
        key: int | str = int(key_text) if key_text.isdigit() else key_text
        parsed[key] = value
    return parsed or None


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_gpus(text: str | None) -> tuple[str, ...]:
    if text is None:
        return tuple()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def default_export_run_name(dataset: str, protocol: str, num_data: int | None) -> str:
    normalized = normalize_dataset_name(dataset)
    if num_data is None or num_data <= 0:
        return f"{normalized}_{protocol}_full"
    return f"{normalized}_{protocol}_sample_{num_data}"


def apply_paper_probe_defaults(config: ProbeTrainConfig) -> ProbeTrainConfig:
    if not config.paper_config:
        return config
    return ProbeTrainConfig(
        dataset_file=config.dataset_file,
        output_dir=config.output_dir,
        feature_set=config.feature_set,
        batch_size=32,
        num_epochs=100,
        learning_rate=5e-4,
        weight_decay=0.0,
        lr_factor=0.5,
        lr_patience=5,
        test_size=config.test_size,
        seed=config.seed,
        split_file=config.split_file,
        num_runs=config.num_runs,
        paper_config=True,
    )
