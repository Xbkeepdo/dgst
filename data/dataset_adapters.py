from __future__ import annotations

# 统一封装 COCO / Objects365 的对象词表、同义词规则和 caption 评测接口。

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from tqdm import tqdm

try:
    import nltk
except Exception:
    nltk = None

from .chair import (
    ANIMAL_WORDS,
    COCO_ALIASES,
    COCO_DOUBLE_WORDS,
    IRREGULAR_SINGULARS,
    NLTK_DATA_CANDIDATES,
    VEHICLE_WORDS,
    _WORD_RE,
    _dedupe_preserve_order,
    _get_pattern_singularize,
)

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickled_adapter(cache: Path):
    try:
        with cache.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def _normalize_surface_text(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_canonical_name(text: str) -> str:
    return _normalize_surface_text(text)


def _normalize_token(token: str) -> str:
    pattern_singularize = _get_pattern_singularize()
    if pattern_singularize is not None:
        normalized = pattern_singularize(token)
        if normalized.endswith("nni") and token.endswith("nnis"):
            return token
        return normalized
    if token in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ves") and len(token) > 4:
        return token[:-3] + "f"
    if token.endswith("es") and len(token) > 3 and not token.endswith(("ses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _normalize_alias_text(text: str) -> str:
    surface = _normalize_surface_text(text)
    tokens = [_normalize_token(token) for token in surface.split() if token]
    return " ".join(tokens).strip()


def _auto_aliases_for_canonical(canonical: str) -> List[str]:
    normalized = _normalize_alias_text(canonical)
    if not normalized:
        return []

    aliases = {normalized}
    parts = [part.strip() for part in _normalize_surface_text(canonical).split("/") if part.strip()]
    if len(parts) > 1:
        aliases.add(_normalize_alias_text(" ".join(parts)))
        for part in parts:
            aliases.add(_normalize_alias_text(part))

    words = normalized.split()
    if len(words) > 1:
        aliases.add(" ".join(words))
    return sorted(alias for alias in aliases if alias)


def _coco_alias_groups() -> List[List[str]]:
    groups: List[List[str]] = []
    for canonical, aliases in COCO_ALIASES.items():
        normalized_canonical = _normalize_canonical_name(canonical)
        values = [normalized_canonical]
        values.extend(_normalize_alias_text(item) for item in ([canonical] + list(aliases)) if item)
        groups.append(_dedupe_preserve_order([item for item in values if item]))

    special_mapping = {phrase: phrase for phrase in COCO_DOUBLE_WORDS}
    for animal_word in ANIMAL_WORDS:
        special_mapping[f"baby {animal_word}"] = animal_word
        special_mapping[f"adult {animal_word}"] = animal_word
    for vehicle_word in VEHICLE_WORDS:
        special_mapping[f"passenger {vehicle_word}"] = vehicle_word
    special_mapping["bow tie"] = "tie"
    special_mapping["toilet seat"] = "toilet"
    special_mapping["wine glas"] = "wine glass"

    canonical_to_aliases: Dict[str, List[str]] = {group[0]: list(group) for group in groups if group}
    for alias, canonical in special_mapping.items():
        normalized_alias = _normalize_alias_text(alias)
        normalized_canonical = _normalize_canonical_name(canonical)
        canonical_to_aliases.setdefault(normalized_canonical, [normalized_canonical])
        canonical_to_aliases[normalized_canonical].append(normalized_alias)

    return [_dedupe_preserve_order(values) for values in canonical_to_aliases.values()]


def _objects365_default_resource_paths() -> tuple[Path, Path, Path]:
    return (
        RESOURCE_DIR / "objects365_canonical.json",
        RESOURCE_DIR / "objects365_aliases.json",
        RESOURCE_DIR / "objects365_to_coco_overlap.json",
    )


class RuleAliasDatasetAdapter:
    dataset_name = "generic"
    dataset_version = "unknown"
    mention_linker_backend = "rule_alias_linker"
    native_taxonomy_space = "generic"
    lexicon_version = "generic_v1"

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.annotation_path = Path(annotation_path)
        self.annotation_source = self.annotation_path
        self.split = str(split)
        self.category_id_to_name: Dict[int, str] = {}
        self.image_id_to_objects: Dict[int, set[str]] = {}
        self.image_id_to_filename: Dict[int, str] = {}
        self.image_id_to_references: Dict[int, List[str]] = {}
        self.split_image_ids: List[int] = []
        self._ensure_runtime_state()
        self._load_annotations()

    @classmethod
    def from_cache(
        cls,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        cache_path: str | Path | None = None,
    ) -> "RuleAliasDatasetAdapter":
        cache = Path(cache_path) if cache_path is not None else None
        root = Path(dataset_root)
        source = Path(annotation_path)
        if cache is not None and cache.exists():
            if cache.suffix == ".pkl":
                adapter = _load_pickled_adapter(cache)
                if adapter is None:
                    adapter = cls(dataset_root=root, annotation_path=source, split=split)
                    adapter.save_cache(cache)
                    return adapter
            else:
                payload = _read_json(cache)
                adapter = cls.__new__(cls)
                adapter.category_id_to_name = {int(k): str(v) for k, v in payload["category_id_to_name"].items()}
                adapter.image_id_to_objects = {
                    int(k): set(str(value) for value in values)
                    for k, values in payload["image_id_to_objects"].items()
                }
                adapter.image_id_to_filename = {int(k): str(v) for k, v in payload["image_id_to_filename"].items()}
                adapter.image_id_to_references = {
                    int(k): [str(value) for value in values]
                    for k, values in payload.get("image_id_to_references", {}).items()
                }
                adapter.split_image_ids = [int(value) for value in payload["split_image_ids"]]
            adapter.dataset_root = root
            adapter.annotation_path = source
            adapter.annotation_source = source
            adapter.split = str(split)
            adapter._ensure_runtime_state()
            return adapter

        adapter = cls(dataset_root=root, annotation_path=source, split=split)
        if cache is not None:
            adapter.save_cache(cache)
        return adapter

    def save_cache(self, cache_path: str | Path) -> None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.suffix == ".pkl":
            with cache.open("wb") as handle:
                pickle.dump(self, handle)
            return
        cache.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "category_id_to_name": {str(k): v for k, v in self.category_id_to_name.items()},
            "image_id_to_objects": {str(k): sorted(v) for k, v in self.image_id_to_objects.items()},
            "image_id_to_filename": {str(k): v for k, v in self.image_id_to_filename.items()},
            "image_id_to_references": {str(k): v for k, v in self.image_id_to_references.items()},
            "split_image_ids": self.split_image_ids,
        }

    def _ensure_runtime_state(self) -> None:
        self.native_alias_groups = self._build_native_alias_groups()
        self.native_alias_to_canonical = self._build_alias_map(self.native_alias_groups)
        self.native_canonical_to_aliases = {
            group[0]: _dedupe_preserve_order(group) for group in self.native_alias_groups if group
        }
        self.native_max_alias_len = self._compute_max_alias_len(self.native_alias_to_canonical)
        self.overlap_mapping = self._build_overlap_mapping()
        self.overlap_alias_groups = self._build_overlap_alias_groups()
        self.overlap_alias_to_canonical = self._build_alias_map(self.overlap_alias_groups)
        self.overlap_canonical_to_aliases = {
            group[0]: _dedupe_preserve_order(group) for group in self.overlap_alias_groups if group
        }
        self.overlap_max_alias_len = self._compute_max_alias_len(self.overlap_alias_to_canonical)

    def _build_alias_map(self, groups: Sequence[Sequence[str]]) -> Dict[str, str]:
        alias_to_canonical: Dict[str, str] = {}
        for group in groups:
            if not group:
                continue
            canonical = str(group[0])
            for alias in group:
                normalized_alias = _normalize_alias_text(alias)
                if normalized_alias:
                    alias_to_canonical[normalized_alias] = canonical
        return alias_to_canonical

    def _compute_max_alias_len(self, alias_map: Dict[str, str]) -> int:
        return max((len(alias.split()) for alias in alias_map), default=1)

    def _build_overlap_alias_groups(self) -> List[List[str]]:
        if not self.overlap_mapping:
            return []
        source_aliases = self.native_canonical_to_aliases
        grouped: Dict[str, List[str]] = {}
        for source_canonical, target_canonical in self.overlap_mapping.items():
            grouped.setdefault(target_canonical, [target_canonical])
            grouped[target_canonical].extend(source_aliases.get(source_canonical, []))
            grouped[target_canonical].extend(COCO_ALIASES.get(target_canonical, []))
            grouped[target_canonical].append(target_canonical)
        return [_dedupe_preserve_order([_normalize_alias_text(item) for item in values if item]) for values in grouped.values()]

    def _tokenize(self, text: str) -> List[str]:
        if nltk is not None:
            for path in NLTK_DATA_CANDIDATES:
                if path.exists():
                    path_str = str(path)
                    if path_str not in nltk.data.path:
                        nltk.data.path.insert(0, path_str)
            try:
                raw_tokens = nltk.word_tokenize(str(text).lower())
            except Exception:
                raw_tokens = _WORD_RE.findall(str(text).lower())
        else:
            raw_tokens = _WORD_RE.findall(str(text).lower())
        return [_normalize_token(token) for token in raw_tokens if _WORD_RE.fullmatch(token)]

    def _protocol_alias_map(self, protocol: str) -> Dict[str, str]:
        if protocol == "native":
            return self.native_alias_to_canonical
        if protocol == "coco_overlap":
            return self.overlap_alias_to_canonical
        raise ValueError(f"Unsupported protocol: {protocol}")

    def _protocol_canonical_aliases(self, protocol: str) -> Dict[str, List[str]]:
        if protocol == "native":
            return self.native_canonical_to_aliases
        if protocol == "coco_overlap":
            return self.overlap_canonical_to_aliases
        raise ValueError(f"Unsupported protocol: {protocol}")

    def _protocol_max_len(self, protocol: str) -> int:
        if protocol == "native":
            return self.native_max_alias_len
        if protocol == "coco_overlap":
            return self.overlap_max_alias_len
        raise ValueError(f"Unsupported protocol: {protocol}")

    def _build_alignment_variants(self, surface: str, canonical: str, protocol: str) -> List[str]:
        variants = [surface, canonical]
        variants.extend(self._protocol_canonical_aliases(protocol).get(canonical, []))
        return _dedupe_preserve_order([item for item in variants if item])

    def caption_to_words(self, caption: str, protocol: str = "native") -> tuple[List[str], List[str], List[int]]:
        tokens = self._tokenize(caption)
        alias_map = self._protocol_alias_map(protocol)
        max_len = self._protocol_max_len(protocol)
        matched_surfaces: List[str] = []
        matched_canonicals: List[str] = []
        matched_indices: List[int] = []

        index = 0
        while index < len(tokens):
            matched = None
            for length in range(max_len, 0, -1):
                if index + length > len(tokens):
                    continue
                phrase = " ".join(tokens[index : index + length])
                canonical = alias_map.get(phrase)
                if canonical is None:
                    continue
                matched = (phrase, canonical, length)
                break
            if matched is None:
                index += 1
                continue
            phrase, canonical, length = matched
            matched_surfaces.append(phrase)
            matched_canonicals.append(canonical)
            matched_indices.append(index)
            index += length
        return matched_surfaces, matched_canonicals, matched_indices

    def caption_to_mentions(self, caption: str, protocol: str = "native") -> List[Dict[str, Any]]:
        surfaces, canonicals, indices = self.caption_to_words(caption, protocol=protocol)
        mentions: List[Dict[str, Any]] = []
        for mention_index, (surface, canonical, idx) in enumerate(zip(surfaces, canonicals, indices)):
            token_span = len(surface.split())
            mentions.append(
                {
                    "surface": surface,
                    "surface_word": surface,
                    "phrase": surface,
                    "canonical_name": canonical,
                    "word_index": idx,
                    "mention_index": mention_index,
                    "token_start": idx,
                    "token_end": idx + token_span,
                    "alignment_variants": self._build_alignment_variants(surface, canonical, protocol=protocol),
                }
            )
        return mentions

    def get_ground_truth_objects(self, image_id: int, protocol: str = "native") -> set[str]:
        native_objects = set(self.image_id_to_objects.get(int(image_id), set()))
        if protocol == "native":
            return native_objects
        if protocol != "coco_overlap":
            raise ValueError(f"Unsupported protocol: {protocol}")
        return {
            self.overlap_mapping[canonical]
            for canonical in native_objects
            if canonical in self.overlap_mapping
        }

    def ground_truth_entry(self, image_id: int, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        metadata = self.protocol_metadata(protocol)
        return {
            "image_id": image_id,
            "image": self.image_id_to_filename.get(image_id, self.image_filename(image_id)),
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "taxonomy_space": metadata["taxonomy_space"],
            "lexicon_version": metadata["lexicon_version"],
            "objects": sorted(self.get_ground_truth_objects(image_id, protocol=protocol)),
        }

    def list_image_ids(
        self,
        split: str | None = None,
        max_samples: int | None = None,
        image_ids_file: str | Path | None = None,
    ) -> List[int]:
        if split is not None and split != self.split:
            raise ValueError(f"{self.dataset_name} adapter only loaded split={self.split!r}, got request for {split!r}")
        if image_ids_file is None:
            image_ids = list(self.split_image_ids)
        else:
            path = Path(image_ids_file)
            image_ids = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_samples is not None:
            image_ids = image_ids[:max_samples]
        return image_ids

    def iter_ground_truth_entries(
        self,
        image_ids: Iterable[int] | None = None,
        limit: int | None = None,
        protocol: str = "native",
    ) -> Iterable[Dict[str, Any]]:
        selected_ids = list(image_ids) if image_ids is not None else list(self.split_image_ids)
        if limit is not None:
            selected_ids = selected_ids[:limit]
        for image_id in selected_ids:
            yield self.ground_truth_entry(image_id, protocol=protocol)

    def save_ground_truth_jsonl(
        self,
        output_path: str | Path,
        image_ids: Iterable[int] | None = None,
        limit: int | None = None,
        protocol: str = "native",
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        selected_ids = list(image_ids) if image_ids is not None else list(self.split_image_ids)
        if limit is not None:
            selected_ids = selected_ids[:limit]
        with path.open("w", encoding="utf-8") as handle:
            progress = tqdm(
                selected_ids,
                desc=f"Writing {self.dataset_name} ground truth",
                unit="image",
                leave=False,
            )
            for image_id in progress:
                entry = self.ground_truth_entry(image_id, protocol=protocol)
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            progress.close()

    def taxonomy_space(self, protocol: str) -> str:
        return self.native_taxonomy_space if protocol == "native" else "coco_overlap"

    def evaluate_caption(self, image_id: int, caption: str, protocol: str = "native") -> Dict[str, Any]:
        gt_objects = self.get_ground_truth_objects(int(image_id), protocol=protocol)
        mentions = self.caption_to_mentions(caption, protocol=protocol)
        for mention in mentions:
            mention["hallucinated"] = int(mention["canonical_name"] not in gt_objects)
        hallucinated = [mention for mention in mentions if mention["canonical_name"] not in gt_objects]
        recall_mentions = [mention for mention in mentions if mention["canonical_name"] in gt_objects]
        native_gt_count = len(self.get_ground_truth_objects(int(image_id), protocol="native"))
        evaluated_gt_count = len(gt_objects)
        metadata = self.protocol_metadata(protocol)
        return {
            "image_id": int(image_id),
            "caption": caption,
            "ground_truth_objects": sorted(gt_objects),
            "object_mentions": mentions,
            "hallucinated_mentions": hallucinated,
            "recall_mentions": recall_mentions,
            "mention_linker_backend": self.mention_linker_backend,
            "taxonomy_backend": metadata["taxonomy_backend"],
            "protocol": protocol,
            "taxonomy_space": metadata["taxonomy_space"],
            "dataset_version": self.dataset_version,
            "lexicon_version": metadata["lexicon_version"],
            "chair_backend": f"{metadata['taxonomy_backend']}:{self.mention_linker_backend}",
            "chair_word_count_total": len(mentions),
            "chair_s": 1.0 if hallucinated else 0.0,
            "chair_i": float(len(hallucinated) / len(mentions)) if mentions else 0.0,
            "recall": float(len({item['canonical_name'] for item in recall_mentions}) / len(gt_objects)) if gt_objects else 0.0,
            "gt_category_count": native_gt_count,
            "evaluated_gt_category_count": evaluated_gt_count,
            "gt_category_coverage": float(evaluated_gt_count / native_gt_count) if native_gt_count else 0.0,
        }

    def protocol_metadata(self, protocol: str) -> Dict[str, str]:
        return {
            "dataset": self.dataset_name,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "protocol": protocol,
            "taxonomy_space": self.taxonomy_space(protocol),
            "lexicon_version": self.lexicon_version,
            "mention_linker_backend": self.mention_linker_backend,
            "taxonomy_backend": self.taxonomy_space(protocol),
        }

    def image_filename(self, image_id: int) -> str:
        return self.image_id_to_filename.get(int(image_id), "")

    def _build_native_alias_groups(self) -> List[List[str]]:
        raise NotImplementedError

    def _build_overlap_mapping(self) -> Dict[str, str]:
        return {}

    def _load_annotations(self) -> None:
        raise NotImplementedError


class CocoDatasetAdapter(RuleAliasDatasetAdapter):
    dataset_name = "coco"
    dataset_version = "2014"
    native_taxonomy_space = "coco"
    lexicon_version = "coco_htec1_aliases_v1"

    def _build_native_alias_groups(self) -> List[List[str]]:
        return _coco_alias_groups()

    def _read_optional_json(self, filename: str) -> Dict[str, Any] | None:
        path = self.annotation_path / filename
        if not path.exists():
            return None
        return _read_json(path)

    def _load_annotations(self) -> None:
        instances_train = self._read_optional_json("instances_train2014.json")
        instances_val = self._read_optional_json("instances_val2014.json")
        captions_train = self._read_optional_json("captions_train2014.json")
        captions_val = self._read_optional_json("captions_val2014.json")

        for payload in [instances_train, instances_val]:
            if not payload:
                continue
            for category in tqdm(payload.get("categories", []), desc="Loading COCO categories", unit="cat", leave=False):
                self.category_id_to_name[int(category["id"])] = _normalize_canonical_name(category["name"])

        for payload in [instances_train, instances_val]:
            if not payload:
                continue
            for annotation in tqdm(payload.get("annotations", []), desc="Loading COCO objects", unit="ann", leave=False):
                image_id = int(annotation["image_id"])
                category_id = int(annotation["category_id"])
                category_name = self.category_id_to_name.get(category_id)
                if category_name is not None:
                    self.image_id_to_objects.setdefault(image_id, set()).add(category_name)

        for payload in [captions_train, captions_val]:
            if not payload:
                continue
            for annotation in tqdm(payload.get("annotations", []), desc="Loading COCO captions", unit="cap", leave=False):
                image_id = int(annotation["image_id"])
                caption = str(annotation.get("caption", ""))
                self.image_id_to_references.setdefault(image_id, []).append(caption)
                for mention in self.caption_to_mentions(caption, protocol="native"):
                    self.image_id_to_objects.setdefault(image_id, set()).add(mention["canonical_name"])

        images_payload = instances_val or captions_val or {}
        self.split_image_ids = sorted(int(item["id"]) for item in images_payload.get("images", []))
        for item in images_payload.get("images", []):
            image_id = int(item["id"])
            self.image_id_to_filename[image_id] = str(item.get("file_name") or f"COCO_val2014_{image_id:012d}.jpg")

    def image_filename(self, image_id: int) -> str:
        image_id = int(image_id)
        return self.image_id_to_filename.get(image_id, f"COCO_val2014_{image_id:012d}.jpg")


class Objects365DatasetAdapter(RuleAliasDatasetAdapter):
    dataset_name = "objects365"
    dataset_version = "v2"
    native_taxonomy_space = "objects365_v2"

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        canonical_path: str | Path | None = None,
        alias_path: str | Path | None = None,
        overlap_path: str | Path | None = None,
    ) -> None:
        self.canonical_path = Path(canonical_path) if canonical_path is not None else _objects365_default_resource_paths()[0]
        self.alias_path = Path(alias_path) if alias_path is not None else _objects365_default_resource_paths()[1]
        self.overlap_path = Path(overlap_path) if overlap_path is not None else _objects365_default_resource_paths()[2]
        super().__init__(dataset_root=dataset_root, annotation_path=annotation_path, split=split)

    @classmethod
    def from_cache(
        cls,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        cache_path: str | Path | None = None,
        canonical_path: str | Path | None = None,
        alias_path: str | Path | None = None,
        overlap_path: str | Path | None = None,
    ) -> "Objects365DatasetAdapter":
        cache = Path(cache_path) if cache_path is not None else None
        root = Path(dataset_root)
        source = Path(annotation_path)
        canonical = Path(canonical_path) if canonical_path is not None else _objects365_default_resource_paths()[0]
        alias = Path(alias_path) if alias_path is not None else _objects365_default_resource_paths()[1]
        overlap = Path(overlap_path) if overlap_path is not None else _objects365_default_resource_paths()[2]
        if cache is not None and cache.exists():
            if cache.suffix == ".pkl":
                adapter = _load_pickled_adapter(cache)
                if adapter is None:
                    adapter = cls(
                        dataset_root=root,
                        annotation_path=source,
                        split=split,
                        canonical_path=canonical,
                        alias_path=alias,
                        overlap_path=overlap,
                    )
                    adapter.save_cache(cache)
                    return adapter
            else:
                payload = _read_json(cache)
                adapter = cls.__new__(cls)
                adapter.category_id_to_name = {int(k): str(v) for k, v in payload["category_id_to_name"].items()}
                adapter.image_id_to_objects = {
                    int(k): set(str(value) for value in values)
                    for k, values in payload["image_id_to_objects"].items()
                }
                adapter.image_id_to_filename = {int(k): str(v) for k, v in payload["image_id_to_filename"].items()}
                adapter.image_id_to_references = {
                    int(k): [str(value) for value in values]
                    for k, values in payload.get("image_id_to_references", {}).items()
                }
                adapter.split_image_ids = [int(value) for value in payload["split_image_ids"]]
            adapter.dataset_root = root
            adapter.annotation_path = source
            adapter.annotation_source = source
            adapter.split = str(split)
            adapter.canonical_path = canonical
            adapter.alias_path = alias
            adapter.overlap_path = overlap
            adapter._ensure_runtime_state()
            return adapter

        adapter = cls(
            dataset_root=root,
            annotation_path=source,
            split=split,
            canonical_path=canonical,
            alias_path=alias,
            overlap_path=overlap,
        )
        if cache is not None:
            adapter.save_cache(cache)
        return adapter

    def _build_native_alias_groups(self) -> List[List[str]]:
        canonical_payload = _read_json(self.canonical_path)
        alias_payload = _read_json(self.alias_path)
        self.lexicon_version = str(alias_payload.get("lexicon_version") or canonical_payload.get("lexicon_version") or "objects365_aliases_v1")
        manual_aliases = {
            _normalize_canonical_name(item["canonical"]): [str(value) for value in item.get("aliases", [])]
            for item in alias_payload.get("alias_groups", [])
        }

        groups: List[List[str]] = []
        for item in canonical_payload.get("categories", []):
            canonical = _normalize_canonical_name(item["canonical"])
            values = [canonical]
            values.extend(_auto_aliases_for_canonical(canonical))
            values.extend(_normalize_alias_text(value) for value in manual_aliases.get(canonical, []))
            groups.append(_dedupe_preserve_order([value for value in values if value]))
        return groups

    def _build_overlap_mapping(self) -> Dict[str, str]:
        payload = _read_json(self.overlap_path)
        return {
            _normalize_canonical_name(source): _normalize_canonical_name(target)
            for source, target in payload.get("mapping", {}).items()
        }

    def _load_annotations(self) -> None:
        payload = _read_json(self.annotation_path)
        for category in tqdm(payload.get("categories", []), desc="Loading Objects365 categories", unit="cat", leave=False):
            category_id = int(category["id"])
            canonical_name = _normalize_canonical_name(category["name"])
            self.category_id_to_name[category_id] = canonical_name

        for image in tqdm(payload.get("images", []), desc="Loading Objects365 images", unit="img", leave=False):
            image_id = int(image["id"])
            file_name = str(
                image.get("file_name")
                or image.get("filename")
                or image.get("image_name")
                or image.get("coco_url", "")
            )
            self.image_id_to_filename[image_id] = file_name
            self.split_image_ids.append(image_id)

        for annotation in tqdm(payload.get("annotations", []), desc="Loading Objects365 objects", unit="ann", leave=False):
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            category_name = self.category_id_to_name.get(category_id)
            if category_name is not None:
                self.image_id_to_objects.setdefault(image_id, set()).add(category_name)

        self.split_image_ids = sorted(set(self.split_image_ids))

    def image_filename(self, image_id: int) -> str:
        return self.image_id_to_filename.get(int(image_id), "")


def create_dataset_adapter(
    dataset: str,
    dataset_root: str | Path,
    annotation_path: str | Path,
    split: str = "val",
    cache_path: str | Path | None = None,
    lexicon_path: str | Path | None = None,
) -> RuleAliasDatasetAdapter:
    normalized_dataset = str(dataset).strip().lower()
    if normalized_dataset in {"coco", "mscoco"}:
        return CocoDatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
        )
    if normalized_dataset == "objects365":
        return Objects365DatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
            alias_path=lexicon_path,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
