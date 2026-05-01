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


class AmberDatasetAdapter(RuleAliasDatasetAdapter):
    dataset_name = "amber"
    dataset_version = "1.0"
    native_taxonomy_space = "amber_generative"
    lexicon_version = "amber_relation_v1"

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        relation_path: str | Path | None = None,
        safe_words_path: str | Path | None = None,
        query_path: str | Path | None = None,
    ) -> None:
        source = Path(annotation_path)
        data_dir = source if source.is_dir() else source.parent
        self.annotation_file = source / "annotations.json" if source.is_dir() else source
        self.relation_path = Path(relation_path) if relation_path is not None else data_dir / "relation.json"
        self.safe_words_path = Path(safe_words_path) if safe_words_path is not None else data_dir / "safe_words.txt"
        self.query_path = Path(query_path) if query_path is not None else data_dir / "query" / "query_generative.json"
        self.image_id_to_truth_objects: Dict[int, List[str]] = {}
        self.image_id_to_hallu_objects: Dict[int, List[str]] = {}
        super().__init__(dataset_root=dataset_root, annotation_path=annotation_path, split=split)

    @classmethod
    def from_cache(
        cls,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        cache_path: str | Path | None = None,
        relation_path: str | Path | None = None,
        safe_words_path: str | Path | None = None,
        query_path: str | Path | None = None,
    ) -> "AmberDatasetAdapter":
        cache = Path(cache_path) if cache_path is not None else None
        root = Path(dataset_root)
        source = Path(annotation_path)
        data_dir = source if source.is_dir() else source.parent
        annotation_file = source / "annotations.json" if source.is_dir() else source
        relation = Path(relation_path) if relation_path is not None else data_dir / "relation.json"
        safe_words = Path(safe_words_path) if safe_words_path is not None else data_dir / "safe_words.txt"
        query = Path(query_path) if query_path is not None else data_dir / "query" / "query_generative.json"

        if cache is not None and cache.exists():
            if cache.suffix == ".pkl":
                adapter = _load_pickled_adapter(cache)
                if adapter is None:
                    adapter = cls(
                        dataset_root=root,
                        annotation_path=source,
                        split=split,
                        relation_path=relation,
                        safe_words_path=safe_words,
                        query_path=query,
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
                adapter.image_id_to_truth_objects = {
                    int(k): [str(value) for value in values]
                    for k, values in payload.get("image_id_to_truth_objects", {}).items()
                }
                adapter.image_id_to_hallu_objects = {
                    int(k): [str(value) for value in values]
                    for k, values in payload.get("image_id_to_hallu_objects", {}).items()
                }
                adapter.image_id_to_filename = {int(k): str(v) for k, v in payload["image_id_to_filename"].items()}
                adapter.image_id_to_references = {
                    int(k): [str(value) for value in values]
                    for k, values in payload.get("image_id_to_references", {}).items()
                }
                adapter.split_image_ids = [int(value) for value in payload["split_image_ids"]]
            adapter.dataset_root = root
            adapter.annotation_path = source
            adapter.annotation_source = annotation_file
            adapter.annotation_file = annotation_file
            adapter.relation_path = relation
            adapter.safe_words_path = safe_words
            adapter.query_path = query
            adapter.split = str(split)
            if not getattr(adapter, "image_id_to_truth_objects", None):
                adapter.image_id_to_truth_objects = {
                    image_id: sorted(values) for image_id, values in adapter.image_id_to_objects.items()
                }
            if not getattr(adapter, "image_id_to_hallu_objects", None):
                adapter.image_id_to_hallu_objects = {}
            adapter._ensure_runtime_state()
            return adapter

        adapter = cls(
            dataset_root=root,
            annotation_path=source,
            split=split,
            relation_path=relation,
            safe_words_path=safe_words,
            query_path=query,
        )
        if cache is not None:
            adapter.save_cache(cache)
        return adapter

    def to_json(self) -> Dict[str, Any]:
        payload = super().to_json()
        payload["image_id_to_truth_objects"] = {
            str(k): list(v) for k, v in self.image_id_to_truth_objects.items()
        }
        payload["image_id_to_hallu_objects"] = {
            str(k): list(v) for k, v in self.image_id_to_hallu_objects.items()
        }
        return payload

    def _ensure_runtime_state(self) -> None:
        self.amber_relation = self._load_relation_map()
        self.global_safe_words = self._load_global_safe_words()
        self.amber_annotation_words = self._load_annotation_words()
        super()._ensure_runtime_state()
        self.amber_vocabulary = set(self.native_alias_to_canonical.keys()) | set(self.amber_annotation_words)

    def _load_relation_map(self) -> Dict[str, List[str]]:
        if not self.relation_path.exists():
            return {}
        payload = _read_json(self.relation_path)
        return {
            _normalize_alias_text(key): _dedupe_preserve_order(
                [_normalize_alias_text(value) for value in values if _normalize_alias_text(value)]
            )
            for key, values in payload.items()
            if _normalize_alias_text(key)
        }

    def _load_global_safe_words(self) -> set[str]:
        if not self.safe_words_path.exists():
            return set()
        return {
            _normalize_alias_text(line)
            for line in self.safe_words_path.read_text(encoding="utf-8").splitlines()
            if _normalize_alias_text(line)
        }

    def _load_annotation_payload(self) -> List[Dict[str, Any]]:
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"AMBER annotation file not found: {self.annotation_file}")
        payload = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"AMBER annotations must be a JSON list: {self.annotation_file}")
        return payload

    def _load_annotation_words(self) -> set[str]:
        if not self.annotation_file.exists():
            return set()
        words: set[str] = set()
        for item in self._load_annotation_payload():
            if item.get("type") != "generative":
                continue
            for key in ("truth", "hallu"):
                for value in item.get(key, []):
                    normalized = _normalize_alias_text(str(value))
                    if normalized:
                        words.add(normalized)
        return words

    def _build_native_alias_groups(self) -> List[List[str]]:
        groups: List[List[str]] = []
        relation = getattr(self, "amber_relation", None) or self._load_relation_map()
        for canonical, aliases in relation.items():
            values = [canonical]
            values.extend(aliases)
            groups.append(_dedupe_preserve_order([value for value in values if value]))

        grouped_words = {group[0] for group in groups if group}
        annotation_words = getattr(self, "amber_annotation_words", None) or self._load_annotation_words()
        for word in sorted(annotation_words):
            if word not in grouped_words:
                groups.append([word])
        return groups

    def _load_query_image_names(self) -> Dict[int, str]:
        if not self.query_path.exists():
            return {}
        payload = json.loads(self.query_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return {}
        return {
            int(item["id"]): str(item.get("image") or f"AMBER_{int(item['id'])}.jpg")
            for item in payload
            if "id" in item
        }

    def _load_annotations(self) -> None:
        query_images = self._load_query_image_names()
        annotation_words: set[str] = set()

        for item in tqdm(self._load_annotation_payload(), desc="Loading AMBER generative annotations", unit="ann", leave=False):
            if item.get("type") != "generative":
                continue
            image_id = int(item["id"])
            truth_objects = _dedupe_preserve_order(
                [_normalize_alias_text(str(value)) for value in item.get("truth", []) if _normalize_alias_text(str(value))]
            )
            hallu_objects = _dedupe_preserve_order(
                [_normalize_alias_text(str(value)) for value in item.get("hallu", []) if _normalize_alias_text(str(value))]
            )
            self.image_id_to_truth_objects[image_id] = truth_objects
            self.image_id_to_hallu_objects[image_id] = hallu_objects
            self.image_id_to_objects[image_id] = set(truth_objects)
            self.image_id_to_filename[image_id] = query_images.get(image_id, f"AMBER_{image_id}.jpg")
            self.split_image_ids.append(image_id)
            annotation_words.update(truth_objects)
            annotation_words.update(hallu_objects)

        self.split_image_ids = sorted(set(self.split_image_ids))
        for category_id, name in enumerate(sorted(annotation_words), start=1):
            self.category_id_to_name[category_id] = name

    def _relation_aliases(self, word: str) -> List[str]:
        normalized = _normalize_alias_text(word)
        aliases = [normalized]
        aliases.extend(self.amber_relation.get(normalized, []))
        return _dedupe_preserve_order([value for value in aliases if value])

    def _image_word_maps(self, image_id: int) -> tuple[Dict[str, str], Dict[str, str]]:
        safe_alias_to_canonical: Dict[str, str] = {}
        hallu_alias_to_canonical: Dict[str, str] = {}
        for canonical in self.image_id_to_truth_objects.get(int(image_id), []):
            for alias in self._relation_aliases(canonical):
                safe_alias_to_canonical.setdefault(alias, canonical)
        for canonical in self.image_id_to_hallu_objects.get(int(image_id), []):
            for alias in self._relation_aliases(canonical):
                hallu_alias_to_canonical.setdefault(alias, canonical)
        return safe_alias_to_canonical, hallu_alias_to_canonical

    def _extract_candidate_nouns(self, text: str) -> List[tuple[int, str]]:
        if nltk is not None:
            for path in NLTK_DATA_CANDIDATES:
                if path.exists():
                    path_str = str(path)
                    if path_str not in nltk.data.path:
                        nltk.data.path.insert(0, path_str)
            try:
                raw_tokens = nltk.word_tokenize(str(text).lower())
                tagged = nltk.pos_tag(raw_tokens)
                return [
                    (idx, _normalize_token(word))
                    for idx, (word, pos) in enumerate(tagged)
                    if pos.startswith("NN") and _WORD_RE.fullmatch(word) and _normalize_token(word)
                ]
            except Exception:
                pass
        return [
            (idx, _normalize_token(token))
            for idx, token in enumerate(_WORD_RE.findall(str(text).lower()))
            if _normalize_token(token)
        ]

    def caption_to_words(self, caption: str, protocol: str = "native") -> tuple[List[str], List[str], List[int]]:
        mentions = self.caption_to_mentions(caption, protocol=protocol)
        return (
            [str(mention["surface"]) for mention in mentions],
            [str(mention["canonical_name"]) for mention in mentions],
            [int(mention["word_index"]) for mention in mentions],
        )

    def caption_to_mentions(
        self,
        caption: str,
        protocol: str = "native",
        image_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        if protocol != "native":
            raise ValueError(f"AMBER only supports protocol='native', got {protocol!r}")

        safe_map: Dict[str, str] = {}
        hallu_map: Dict[str, str] = {}
        if image_id is not None:
            safe_map, hallu_map = self._image_word_maps(int(image_id))

        mentions: List[Dict[str, Any]] = []
        for word_index, noun in self._extract_candidate_nouns(caption):
            if noun in self.global_safe_words:
                continue
            if noun not in self.amber_vocabulary and noun not in safe_map and noun not in hallu_map:
                continue

            hallucinated = 0
            canonical = self.native_alias_to_canonical.get(noun, noun)
            if image_id is not None:
                if noun in safe_map:
                    canonical = safe_map[noun]
                    hallucinated = 0
                elif noun in hallu_map:
                    canonical = hallu_map[noun]
                    hallucinated = 1
                else:
                    canonical = noun
                    hallucinated = 1

            variants = [noun, canonical]
            variants.extend(self._relation_aliases(canonical))
            mentions.append(
                {
                    "surface": noun,
                    "surface_word": noun,
                    "phrase": noun,
                    "canonical_name": canonical,
                    "word_index": word_index,
                    "mention_index": len(mentions),
                    "token_start": word_index,
                    "token_end": word_index + 1,
                    "hallucinated": int(hallucinated),
                    "alignment_variants": _dedupe_preserve_order([value for value in variants if value]),
                }
            )
        return mentions

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
            "objects": list(self.image_id_to_truth_objects.get(image_id, [])),
            "amber_hallucination_candidates": list(self.image_id_to_hallu_objects.get(image_id, [])),
        }

    def evaluate_caption(self, image_id: int, caption: str, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        gt_objects = set(self.image_id_to_truth_objects.get(image_id, []))
        hallu_objects = set(self.image_id_to_hallu_objects.get(image_id, []))
        mentions = self.caption_to_mentions(caption, protocol=protocol, image_id=image_id)
        hallucinated = [mention for mention in mentions if int(mention.get("hallucinated", 0))]
        recall_mentions = [mention for mention in mentions if not int(mention.get("hallucinated", 0))]
        recalled_truth = {str(mention["canonical_name"]) for mention in recall_mentions if mention["canonical_name"] in gt_objects}
        recalled_hallu = {str(mention["canonical_name"]) for mention in hallucinated if mention["canonical_name"] in hallu_objects}
        metadata = self.protocol_metadata(protocol)
        return {
            "image_id": image_id,
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
            "recall": float(len(recalled_truth) / len(gt_objects)) if gt_objects else 0.0,
            "gt_category_count": len(gt_objects),
            "evaluated_gt_category_count": len(gt_objects),
            "gt_category_coverage": 1.0 if gt_objects else 0.0,
            "amber_truth_coverage": float(len(recalled_truth) / len(gt_objects)) if gt_objects else 0.0,
            "amber_hallu_candidate_coverage": float(len(recalled_hallu) / len(hallu_objects)) if hallu_objects else 0.0,
        }

    def image_filename(self, image_id: int) -> str:
        return self.image_id_to_filename.get(int(image_id), f"AMBER_{int(image_id)}.jpg")


class AmberDiscriminativeDatasetAdapter(RuleAliasDatasetAdapter):
    dataset_name = "amber_discriminative"
    dataset_version = "1.0"
    native_taxonomy_space = "amber_discriminative"
    lexicon_version = "amber_discriminative_truth_v1"
    mention_linker_backend = "amber_yes_no_token_linker"

    QUERY_FILES = {
        "val": "query_discriminative.json",
        "all": "query_discriminative.json",
        "d": "query_discriminative.json",
        "existence": "query_discriminative-existence.json",
        "de": "query_discriminative-existence.json",
        "attribute": "query_discriminative-attribute.json",
        "da": "query_discriminative-attribute.json",
        "relation": "query_discriminative-relation.json",
        "dr": "query_discriminative-relation.json",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        query_path: str | Path | None = None,
    ) -> None:
        source = Path(annotation_path)
        data_dir = source if source.is_dir() else source.parent
        self.annotation_file = source / "annotations.json" if source.is_dir() else source
        self.query_path = Path(query_path) if query_path is not None else self._default_query_path(data_dir, split)
        self.sample_records: Dict[int, Dict[str, Any]] = {}
        super().__init__(dataset_root=dataset_root, annotation_path=annotation_path, split=split)

    @classmethod
    def from_cache(
        cls,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        cache_path: str | Path | None = None,
        query_path: str | Path | None = None,
    ) -> "AmberDiscriminativeDatasetAdapter":
        cache = Path(cache_path) if cache_path is not None else None
        root = Path(dataset_root)
        source = Path(annotation_path)
        data_dir = source if source.is_dir() else source.parent
        annotation_file = source / "annotations.json" if source.is_dir() else source
        query = Path(query_path) if query_path is not None else cls._default_query_path(data_dir, split)

        if cache is not None and cache.exists():
            if cache.suffix == ".pkl":
                adapter = _load_pickled_adapter(cache)
                if adapter is None:
                    adapter = cls(dataset_root=root, annotation_path=source, split=split, query_path=query)
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
                adapter.sample_records = {
                    int(k): dict(value) for k, value in payload.get("sample_records", {}).items()
                }
            adapter.dataset_root = root
            adapter.annotation_path = source
            adapter.annotation_source = annotation_file
            adapter.annotation_file = annotation_file
            adapter.query_path = query
            adapter.split = str(split)
            adapter._ensure_runtime_state()
            return adapter

        adapter = cls(dataset_root=root, annotation_path=source, split=split, query_path=query)
        if cache is not None:
            adapter.save_cache(cache)
        return adapter

    @classmethod
    def _default_query_path(cls, data_dir: Path, split: str) -> Path:
        key = str(split).strip().lower()
        filename = cls.QUERY_FILES.get(key)
        if filename is None:
            raise ValueError(
                "AMBER discriminative split must be one of "
                "val/all/d, existence/de, attribute/da, or relation/dr."
            )
        return data_dir / "query" / filename

    def to_json(self) -> Dict[str, Any]:
        payload = super().to_json()
        payload["sample_records"] = {str(k): v for k, v in self.sample_records.items()}
        return payload

    def _ensure_runtime_state(self) -> None:
        super()._ensure_runtime_state()

    def _build_native_alias_groups(self) -> List[List[str]]:
        return [["yes"], ["no"], ["invalid"]]

    def _load_annotation_map(self) -> Dict[int, Dict[str, Any]]:
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"AMBER annotation file not found: {self.annotation_file}")
        payload = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"AMBER annotations must be a JSON list: {self.annotation_file}")
        return {int(item["id"]): dict(item) for item in payload}

    def _load_query_rows(self) -> List[Dict[str, Any]]:
        if not self.query_path.exists():
            raise FileNotFoundError(f"AMBER discriminative query file not found: {self.query_path}")
        payload = json.loads(self.query_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"AMBER discriminative queries must be a JSON list: {self.query_path}")
        return payload

    def _clean_question(self, question: str) -> str:
        text = str(question).replace("<image>", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _load_annotations(self) -> None:
        annotation_by_id = self._load_annotation_map()
        self.category_id_to_name = {1: "correct_yes_no", 2: "incorrect_yes_no"}

        for row in tqdm(self._load_query_rows(), desc="Loading AMBER discriminative queries", unit="query", leave=False):
            query_id = int(row["id"])
            annotation = annotation_by_id.get(query_id)
            if annotation is None:
                continue
            truth = str(annotation.get("truth", "")).strip().lower()
            if truth not in {"yes", "no"}:
                continue
            image_name = str(row.get("image") or "")
            if not image_name:
                continue
            task_type = str(annotation.get("type") or "discriminative")
            query = self._clean_question(str(row.get("query") or "Answer yes or no."))
            self.sample_records[query_id] = {
                "query_id": query_id,
                "image": image_name,
                "query": query,
                "truth": truth,
                "type": task_type,
            }
            self.image_id_to_filename[query_id] = image_name
            self.image_id_to_objects[query_id] = {"correct_yes_no", "incorrect_yes_no"}
            self.split_image_ids.append(query_id)

        self.split_image_ids = sorted(self.split_image_ids)

    def question_for_image_id(self, image_id: int) -> str:
        query = str(self.sample_records[int(image_id)]["query"])
        return f"{query} Answer with Yes or No."

    def _extract_yes_no(self, response: str) -> tuple[str | None, str | None, int | None]:
        match = re.search(r"\b(yes|no)\b", str(response), flags=re.IGNORECASE)
        if match is None:
            return None, None, None
        return match.group(1).lower(), match.group(0), int(match.start())

    def _fallback_surface(self, response: str) -> tuple[str, int]:
        match = re.search(r"\S+", str(response))
        if match is None:
            return "", 0
        return match.group(0), int(match.start())

    def _answer_variants(self, surface: str, canonical: str) -> List[str]:
        variants = [surface, surface.strip(), canonical, canonical.title(), canonical.upper()]
        return _dedupe_preserve_order([value for value in variants if str(value).strip()])

    def caption_to_mentions(
        self,
        caption: str,
        protocol: str = "native",
        image_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        if protocol != "native":
            raise ValueError(f"AMBER discriminative only supports protocol='native', got {protocol!r}")
        if image_id is None:
            return []
        record = self.sample_records[int(image_id)]
        truth = str(record["truth"])
        prediction, surface, start = self._extract_yes_no(caption)
        valid_answer = prediction is not None
        if prediction is None or surface is None or start is None:
            surface, start = self._fallback_surface(caption)
            prediction = "invalid"
        if not surface:
            return []
        hallucinated = int((prediction if valid_answer else None) != truth)
        return [
            {
                "kind": "amber_yes_no_answer",
                "surface": surface,
                "surface_word": surface,
                "phrase": surface,
                "canonical_name": prediction,
                "word_index": start,
                "mention_index": 0,
                "char_start": start,
                "char_end": start + len(surface),
                "amber_discriminative_truth": truth,
                "amber_discriminative_prediction": prediction,
                "amber_discriminative_valid_answer": bool(valid_answer),
                "amber_discriminative_correct": int(valid_answer and prediction == truth),
                "amber_discriminative_type": record.get("type"),
                "hallucinated": hallucinated,
                "alignment_variants": self._answer_variants(surface, prediction),
            }
        ]

    def caption_to_words(self, caption: str, protocol: str = "native") -> tuple[List[str], List[str], List[int]]:
        mentions = self.caption_to_mentions(caption, protocol=protocol)
        return (
            [str(mention["surface"]) for mention in mentions],
            [str(mention["canonical_name"]) for mention in mentions],
            [int(mention["word_index"]) for mention in mentions],
        )

    def ground_truth_entry(self, image_id: int, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        metadata = self.protocol_metadata(protocol)
        record = self.sample_records[image_id]
        return {
            "image_id": image_id,
            "image": self.image_id_to_filename[image_id],
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "taxonomy_space": metadata["taxonomy_space"],
            "lexicon_version": metadata["lexicon_version"],
            "objects": ["correct_yes_no", "incorrect_yes_no"],
            "amber_discriminative_query_id": image_id,
            "amber_discriminative_query": record["query"],
            "amber_discriminative_truth": record["truth"],
            "amber_discriminative_type": record["type"],
        }

    def evaluate_caption(self, image_id: int, caption: str, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        mentions = self.caption_to_mentions(caption, protocol=protocol, image_id=image_id)
        hallucinated = [mention for mention in mentions if int(mention.get("hallucinated", 0))]
        correct = [mention for mention in mentions if not int(mention.get("hallucinated", 0))]
        record = self.sample_records[image_id]
        prediction = mentions[0]["amber_discriminative_prediction"] if mentions else "invalid"
        valid_answer = bool(mentions[0]["amber_discriminative_valid_answer"]) if mentions else False
        metadata = self.protocol_metadata(protocol)
        return {
            "image_id": image_id,
            "caption": caption,
            "ground_truth_objects": ["correct_yes_no", "incorrect_yes_no"],
            "object_mentions": mentions,
            "hallucinated_mentions": hallucinated,
            "recall_mentions": correct,
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
            "recall": float(len(correct) / len(mentions)) if mentions else 0.0,
            "gt_category_count": 1,
            "evaluated_gt_category_count": len(mentions),
            "gt_category_coverage": 1.0 if mentions else 0.0,
            "amber_discriminative_query_id": image_id,
            "amber_discriminative_query": record["query"],
            "amber_discriminative_truth": record["truth"],
            "amber_discriminative_prediction": prediction,
            "amber_discriminative_valid_answer": valid_answer,
            "amber_discriminative_correct": int(valid_answer and prediction == record["truth"]),
            "amber_discriminative_type": record["type"],
        }

    def protocol_metadata(self, protocol: str) -> Dict[str, str]:
        payload = super().protocol_metadata(protocol)
        payload["dataset_version"] = f"{self.dataset_version}:discriminative:{self.split}"
        return payload

    def image_filename(self, image_id: int) -> str:
        return self.image_id_to_filename.get(int(image_id), "")


class AmberDiscriminativePairedDatasetAdapter(AmberDiscriminativeDatasetAdapter):
    dataset_name = "amber_discriminative_paired"
    dataset_version = "1.1"
    native_taxonomy_space = "amber_discriminative_paired"
    lexicon_version = "amber_discriminative_paired_truth_v1"
    mention_linker_backend = "amber_yes_no_paired_token_linker"

    ANSWER_ORDER = ("yes", "no")
    VIRTUAL_ID_OFFSET = 10_000_000

    @staticmethod
    def _virtual_image_id(query_id: int, answer: str) -> int:
        answer_bit = 1 if str(answer).strip().lower() == "yes" else 0
        return AmberDiscriminativePairedDatasetAdapter.VIRTUAL_ID_OFFSET + int(query_id) * 10 + answer_bit

    def _load_annotations(self) -> None:
        annotation_by_id = self._load_annotation_map()
        self.category_id_to_name = {1: "correct_yes_no", 2: "incorrect_yes_no"}

        for row in tqdm(
            self._load_query_rows(),
            desc="Loading AMBER discriminative paired queries",
            unit="query",
            leave=False,
        ):
            query_id = int(row["id"])
            annotation = annotation_by_id.get(query_id)
            if annotation is None:
                continue
            truth = str(annotation.get("truth", "")).strip().lower()
            if truth not in {"yes", "no"}:
                continue
            image_name = str(row.get("image") or "")
            if not image_name:
                continue
            task_type = str(annotation.get("type") or "discriminative")
            query = self._clean_question(str(row.get("query") or "Answer yes or no."))
            self.split_image_ids.append(query_id)
            for answer in self.ANSWER_ORDER:
                virtual_id = self._virtual_image_id(query_id, answer)
                self.sample_records[virtual_id] = {
                    "query_id": query_id,
                    "image": image_name,
                    "query": query,
                    "truth": truth,
                    "type": task_type,
                    "pair_answer": answer,
                }
                self.image_id_to_filename[virtual_id] = image_name
                self.image_id_to_objects[virtual_id] = {"correct_yes_no", "incorrect_yes_no"}

        self.split_image_ids = sorted(self.split_image_ids)

    def expand_sampled_image_ids(self, image_ids: Iterable[int]) -> List[int]:
        expanded: List[int] = []
        for image_id in image_ids:
            value = int(image_id)
            if value in self.sample_records:
                expanded.append(value)
                continue
            for answer in self.ANSWER_ORDER:
                virtual_id = self._virtual_image_id(value, answer)
                if virtual_id in self.sample_records:
                    expanded.append(virtual_id)
        return expanded

    def caption_for_image_id(self, image_id: int) -> str:
        answer = str(self.sample_records[int(image_id)]["pair_answer"]).strip().lower()
        return "Yes" if answer == "yes" else "No"

    def ground_truth_entry(self, image_id: int, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        metadata = self.protocol_metadata(protocol)
        record = self.sample_records[image_id]
        return {
            "image_id": image_id,
            "image": self.image_id_to_filename[image_id],
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "taxonomy_space": metadata["taxonomy_space"],
            "lexicon_version": metadata["lexicon_version"],
            "objects": ["correct_yes_no", "incorrect_yes_no"],
            "split_group_id": int(record["query_id"]),
            "amber_discriminative_query_id": int(record["query_id"]),
            "amber_discriminative_pair_answer": record["pair_answer"],
            "amber_discriminative_query": record["query"],
            "amber_discriminative_truth": record["truth"],
            "amber_discriminative_type": record["type"],
        }

    def evaluate_caption(self, image_id: int, caption: str, protocol: str = "native") -> Dict[str, Any]:
        payload = super().evaluate_caption(image_id=image_id, caption=caption, protocol=protocol)
        record = self.sample_records[int(image_id)]
        payload["split_group_id"] = int(record["query_id"])
        payload["amber_discriminative_query_id"] = int(record["query_id"])
        payload["amber_discriminative_pair_answer"] = record["pair_answer"]
        payload["amber_discriminative_query"] = record["query"]
        payload["amber_discriminative_truth"] = record["truth"]
        payload["amber_discriminative_type"] = record["type"]
        return payload

    def protocol_metadata(self, protocol: str) -> Dict[str, str]:
        payload = super().protocol_metadata(protocol)
        payload["dataset_version"] = f"{self.dataset_version}:discriminative_paired:{self.split}"
        return payload


class PopePairedDatasetAdapter(RuleAliasDatasetAdapter):
    dataset_name = "pope_paired"
    dataset_version = "official_coco"
    native_taxonomy_space = "pope_paired"
    lexicon_version = "pope_paired_truth_v1"
    mention_linker_backend = "pope_yes_no_paired_token_linker"

    ANSWER_ORDER = ("yes", "no")
    VIRTUAL_ID_OFFSET = 20_000_000
    SUBSET_OFFSETS = {
        "random": 100_000,
        "popular": 200_000,
        "adversarial": 300_000,
    }
    SUBSET_FILES = {
        "random": "coco_pope_random.json",
        "popular": "coco_pope_popular.json",
        "adversarial": "coco_pope_adversarial.json",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
    ) -> None:
        self.sample_records: Dict[int, Dict[str, Any]] = {}
        super().__init__(dataset_root=dataset_root, annotation_path=annotation_path, split=split)

    @classmethod
    def from_cache(
        cls,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        cache_path: str | Path | None = None,
    ) -> "PopePairedDatasetAdapter":
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
                adapter.sample_records = {
                    int(k): dict(value) for k, value in payload.get("sample_records", {}).items()
                }
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

    def to_json(self) -> Dict[str, Any]:
        payload = super().to_json()
        payload["sample_records"] = {str(k): v for k, v in self.sample_records.items()}
        return payload

    def _build_native_alias_groups(self) -> List[List[str]]:
        return [["yes"], ["no"], ["invalid"]]

    @classmethod
    def _virtual_image_id(cls, query_id: int, answer: str) -> int:
        answer_bit = 1 if str(answer).strip().lower() == "yes" else 0
        return cls.VIRTUAL_ID_OFFSET + int(query_id) * 10 + answer_bit

    def _subset_files_for_split(self) -> List[tuple[str, Path]]:
        source = Path(self.annotation_path)
        split_key = str(self.split).strip().lower()
        if source.is_file():
            stem = source.stem.lower()
            subset = next((name for name in self.SUBSET_FILES if name in stem), split_key or "custom")
            return [(subset, source)]

        if split_key in {"val", "all", "coco", "pope"}:
            subsets = ("random", "popular", "adversarial")
        elif split_key in self.SUBSET_FILES:
            subsets = (split_key,)
        else:
            raise ValueError(
                "POPE split must be one of val/all/coco, random, popular, or adversarial; "
                f"got {self.split!r}"
            )

        files: List[tuple[str, Path]] = []
        for subset in subsets:
            path = source / self.SUBSET_FILES[subset]
            if not path.exists():
                raise FileNotFoundError(f"POPE annotation file not found: {path}")
            files.append((subset, path))
        return files

    def _load_pope_rows(self, path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return rows
        if text.startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError(f"POPE JSON file must contain a list or JSONL rows: {path}")
            return [dict(item) for item in payload]
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _clean_question(self, question: str) -> str:
        text = str(question).replace("<image>", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _load_annotations(self) -> None:
        self.category_id_to_name = {1: "correct_yes_no", 2: "incorrect_yes_no"}
        self.split_image_ids = []

        for subset, path in self._subset_files_for_split():
            subset_offset = self.SUBSET_OFFSETS.get(subset, 900_000)
            for row in tqdm(
                self._load_pope_rows(path),
                desc=f"Loading POPE {subset}",
                unit="query",
                leave=False,
            ):
                raw_question_id = int(row["question_id"])
                query_id = subset_offset + raw_question_id
                truth = str(row.get("label", "")).strip().lower()
                if truth not in {"yes", "no"}:
                    continue
                image_name = str(row.get("image") or "")
                if not image_name:
                    continue
                query = self._clean_question(str(row.get("text") or "Answer yes or no."))
                self.split_image_ids.append(query_id)
                for answer in self.ANSWER_ORDER:
                    virtual_id = self._virtual_image_id(query_id, answer)
                    self.sample_records[virtual_id] = {
                        "query_id": query_id,
                        "pope_question_id": raw_question_id,
                        "pope_subset": subset,
                        "image": image_name,
                        "query": query,
                        "truth": truth,
                        "pair_answer": answer,
                    }
                    self.image_id_to_filename[virtual_id] = image_name
                    self.image_id_to_objects[virtual_id] = {"correct_yes_no", "incorrect_yes_no"}

        self.split_image_ids = sorted(set(self.split_image_ids))

    def expand_sampled_image_ids(self, image_ids: Iterable[int]) -> List[int]:
        expanded: List[int] = []
        for image_id in image_ids:
            value = int(image_id)
            if value in self.sample_records:
                expanded.append(value)
                continue
            for answer in self.ANSWER_ORDER:
                virtual_id = self._virtual_image_id(value, answer)
                if virtual_id in self.sample_records:
                    expanded.append(virtual_id)
        return expanded

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
        selected_ids = self.expand_sampled_image_ids(selected_ids)
        with path.open("w", encoding="utf-8") as handle:
            progress = tqdm(
                selected_ids,
                desc=f"Writing {self.dataset_name} ground truth",
                unit="sample",
                leave=False,
            )
            for image_id in progress:
                entry = self.ground_truth_entry(image_id, protocol=protocol)
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            progress.close()

    def question_for_image_id(self, image_id: int) -> str:
        query = str(self.sample_records[int(image_id)]["query"])
        return f"{query} Answer with Yes or No."

    def caption_for_image_id(self, image_id: int) -> str:
        answer = str(self.sample_records[int(image_id)]["pair_answer"]).strip().lower()
        return "Yes" if answer == "yes" else "No"

    def _extract_yes_no(self, response: str) -> tuple[str | None, str | None, int | None]:
        match = re.search(r"\b(yes|no)\b", str(response), flags=re.IGNORECASE)
        if match is None:
            return None, None, None
        return match.group(1).lower(), match.group(0), int(match.start())

    def _fallback_surface(self, response: str) -> tuple[str, int]:
        match = re.search(r"\S+", str(response))
        if match is None:
            return "", 0
        return match.group(0), int(match.start())

    def _answer_variants(self, surface: str, canonical: str) -> List[str]:
        variants = [surface, surface.strip(), canonical, canonical.title(), canonical.upper()]
        return _dedupe_preserve_order([value for value in variants if str(value).strip()])

    def caption_to_mentions(
        self,
        caption: str,
        protocol: str = "native",
        image_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        if protocol != "native":
            raise ValueError(f"POPE paired only supports protocol='native', got {protocol!r}")
        if image_id is None:
            return []
        record = self.sample_records[int(image_id)]
        truth = str(record["truth"])
        prediction, surface, start = self._extract_yes_no(caption)
        valid_answer = prediction is not None
        if prediction is None or surface is None or start is None:
            surface, start = self._fallback_surface(caption)
            prediction = "invalid"
        if not surface:
            return []
        hallucinated = int((prediction if valid_answer else None) != truth)
        return [
            {
                "kind": "pope_yes_no_answer",
                "surface": surface,
                "surface_word": surface,
                "phrase": surface,
                "canonical_name": prediction,
                "word_index": start,
                "mention_index": 0,
                "char_start": start,
                "char_end": start + len(surface),
                "pope_query_id": int(record["query_id"]),
                "pope_question_id": int(record["pope_question_id"]),
                "pope_subset": record["pope_subset"],
                "pope_truth": truth,
                "pope_prediction": prediction,
                "pope_pair_answer": record["pair_answer"],
                "pope_valid_answer": bool(valid_answer),
                "pope_correct": int(valid_answer and prediction == truth),
                "hallucinated": hallucinated,
                "alignment_variants": self._answer_variants(surface, prediction),
            }
        ]

    def caption_to_words(self, caption: str, protocol: str = "native") -> tuple[List[str], List[str], List[int]]:
        mentions = self.caption_to_mentions(caption, protocol=protocol)
        return (
            [str(mention["surface"]) for mention in mentions],
            [str(mention["canonical_name"]) for mention in mentions],
            [int(mention["word_index"]) for mention in mentions],
        )

    def ground_truth_entry(self, image_id: int, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        metadata = self.protocol_metadata(protocol)
        record = self.sample_records[image_id]
        return {
            "image_id": image_id,
            "image": self.image_id_to_filename[image_id],
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "taxonomy_space": metadata["taxonomy_space"],
            "lexicon_version": metadata["lexicon_version"],
            "objects": ["correct_yes_no", "incorrect_yes_no"],
            "split_group_id": int(record["query_id"]),
            "pope_query_id": int(record["query_id"]),
            "pope_question_id": int(record["pope_question_id"]),
            "pope_subset": record["pope_subset"],
            "pope_pair_answer": record["pair_answer"],
            "pope_query": record["query"],
            "pope_truth": record["truth"],
        }

    def evaluate_caption(self, image_id: int, caption: str, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        mentions = self.caption_to_mentions(caption, protocol=protocol, image_id=image_id)
        hallucinated = [mention for mention in mentions if int(mention.get("hallucinated", 0))]
        correct = [mention for mention in mentions if not int(mention.get("hallucinated", 0))]
        record = self.sample_records[image_id]
        prediction = mentions[0]["pope_prediction"] if mentions else "invalid"
        valid_answer = bool(mentions[0]["pope_valid_answer"]) if mentions else False
        metadata = self.protocol_metadata(protocol)
        return {
            "image_id": image_id,
            "caption": caption,
            "ground_truth_objects": ["correct_yes_no", "incorrect_yes_no"],
            "object_mentions": mentions,
            "hallucinated_mentions": hallucinated,
            "recall_mentions": correct,
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
            "recall": float(len(correct) / len(mentions)) if mentions else 0.0,
            "gt_category_count": 1,
            "evaluated_gt_category_count": len(mentions),
            "gt_category_coverage": 1.0 if mentions else 0.0,
            "split_group_id": int(record["query_id"]),
            "pope_query_id": int(record["query_id"]),
            "pope_question_id": int(record["pope_question_id"]),
            "pope_subset": record["pope_subset"],
            "pope_pair_answer": record["pair_answer"],
            "pope_query": record["query"],
            "pope_truth": record["truth"],
            "pope_prediction": prediction,
            "pope_valid_answer": valid_answer,
            "pope_correct": int(valid_answer and prediction == record["truth"]),
        }

    def protocol_metadata(self, protocol: str) -> Dict[str, str]:
        payload = super().protocol_metadata(protocol)
        payload["dataset_version"] = f"{self.dataset_version}:paired:{self.split}"
        return payload

    def image_filename(self, image_id: int) -> str:
        return self.image_id_to_filename.get(int(image_id), "")


class MHalDetectDatasetAdapter(RuleAliasDatasetAdapter):
    dataset_name = "mhaldetect"
    dataset_version = "raw"
    native_taxonomy_space = "mhaldetect_segments"
    lexicon_version = "mhaldetect_span_labels_v1"
    mention_linker_backend = "mhaldetect_span_linker"

    LABEL_ACCURATE = "ACCURATE"
    LABEL_INACCURATE = "INACCURATE"
    LABEL_ANALYSIS = "ANALYSIS"

    def __init__(
        self,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
    ) -> None:
        self.sample_records: Dict[int, Dict[str, Any]] = {}
        super().__init__(dataset_root=dataset_root, annotation_path=annotation_path, split=split)

    @classmethod
    def from_cache(
        cls,
        dataset_root: str | Path,
        annotation_path: str | Path,
        split: str = "val",
        cache_path: str | Path | None = None,
    ) -> "MHalDetectDatasetAdapter":
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
                adapter.sample_records = {
                    int(k): dict(value) for k, value in payload.get("sample_records", {}).items()
                }
            adapter.dataset_root = root
            adapter.annotation_path = source
            adapter.annotation_source = adapter._resolve_annotation_file(source, split)
            adapter.split = str(split)
            adapter._ensure_runtime_state()
            return adapter

        adapter = cls(dataset_root=root, annotation_path=source, split=split)
        if cache is not None:
            adapter.save_cache(cache)
        return adapter

    def to_json(self) -> Dict[str, Any]:
        payload = super().to_json()
        payload["sample_records"] = {str(k): v for k, v in self.sample_records.items()}
        return payload

    def _ensure_runtime_state(self) -> None:
        super()._ensure_runtime_state()
        self.segment_labels = [self.LABEL_ACCURATE, self.LABEL_INACCURATE]
        for record in getattr(self, "sample_records", {}).values():
            record["question"] = self._clean_question(str(record.get("question") or ""))

    def _build_native_alias_groups(self) -> List[List[str]]:
        return [["accurate"], ["inaccurate"]]

    def _resolve_annotation_file(self, annotation_path: str | Path, split: str) -> Path:
        source = Path(annotation_path)
        if source.is_file():
            return source
        split_name = str(split).strip().lower()
        if split_name not in {"train", "val"}:
            raise ValueError(f"M-HalDetect split must be 'train' or 'val', got {split!r}")
        return source / f"{split_name}_raw.json"

    def _load_annotation_rows(self) -> List[Dict[str, Any]]:
        path = self._resolve_annotation_file(self.annotation_path, self.split)
        if not path.exists():
            raise FileNotFoundError(f"M-HalDetect annotation file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"M-HalDetect annotations must be a JSON list: {path}")
        self.annotation_source = path
        return payload

    def _load_annotations(self) -> None:
        rows = self._load_annotation_rows()
        self.category_id_to_name = {1: "accurate", 2: "inaccurate"}

        for sample_id, row in enumerate(tqdm(rows, desc=f"Loading M-HalDetect {self.split}", unit="sample", leave=False), start=1):
            image_name = str(row["image"])
            response = str(row.get("response") or "")
            question = self._clean_question(str(row.get("question") or "Describe the given image in detail."))
            annotations = [dict(item) for item in row.get("annotations", [])]
            self.sample_records[sample_id] = {
                "sample_id": sample_id,
                "question": question,
                "response": response,
                "image": image_name,
                "annotations": annotations,
            }
            self.image_id_to_filename[sample_id] = image_name
            self.image_id_to_references[sample_id] = [response]
            self.image_id_to_objects[sample_id] = {
                _normalize_canonical_name(item.get("label", ""))
                for item in annotations
                if str(item.get("label", "")).upper() in {self.LABEL_ACCURATE, self.LABEL_INACCURATE}
            }
            self.split_image_ids.append(sample_id)

        self.split_image_ids = sorted(self.split_image_ids)

    def _clean_question(self, question: str) -> str:
        text = str(question).strip()
        text = text.replace("<image>", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text or "Describe the given image in detail."

    def _span_variants(self, text: str) -> List[str]:
        stripped = str(text).strip()
        collapsed = re.sub(r"\s+", " ", stripped).strip()
        trimmed = collapsed.strip(" \t\r\n.,;:")
        return _dedupe_preserve_order([value for value in [stripped, collapsed, trimmed] if value])

    def question_for_image_id(self, image_id: int) -> str:
        return str(self.sample_records[int(image_id)]["question"])

    def caption_for_image_id(self, image_id: int) -> str:
        return str(self.sample_records[int(image_id)]["response"])

    def caption_to_mentions(
        self,
        caption: str,
        protocol: str = "native",
        image_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        if protocol != "native":
            raise ValueError(f"M-HalDetect only supports protocol='native', got {protocol!r}")
        if image_id is None:
            return []

        record = self.sample_records[int(image_id)]
        mentions: List[Dict[str, Any]] = []
        for annotation in record.get("annotations", []):
            label = str(annotation.get("label", "")).upper()
            if label not in {self.LABEL_ACCURATE, self.LABEL_INACCURATE}:
                continue
            text = str(annotation.get("text") or "").strip()
            if not text:
                continue
            start = int(annotation.get("start", 0))
            end = int(annotation.get("end", start + len(text)))
            hallucinated = int(label == self.LABEL_INACCURATE)
            canonical = "inaccurate" if hallucinated else "accurate"
            mentions.append(
                {
                    "kind": "mhaldetect_segment",
                    "surface": text,
                    "surface_word": text,
                    "phrase": text,
                    "canonical_name": canonical,
                    "word_index": start,
                    "mention_index": len(mentions),
                    "char_start": start,
                    "char_end": end,
                    "mhaldetect_label": label,
                    "hallucinated": hallucinated,
                    "alignment_variants": self._span_variants(text),
                }
            )
        return mentions

    def caption_to_words(self, caption: str, protocol: str = "native") -> tuple[List[str], List[str], List[int]]:
        return [], [], []

    def get_ground_truth_objects(self, image_id: int, protocol: str = "native") -> set[str]:
        return set(self.image_id_to_objects.get(int(image_id), set()))

    def ground_truth_entry(self, image_id: int, protocol: str = "native") -> Dict[str, Any]:
        image_id = int(image_id)
        metadata = self.protocol_metadata(protocol)
        record = self.sample_records[image_id]
        labels = [
            str(item.get("label", "")).upper()
            for item in record.get("annotations", [])
            if str(item.get("label", "")).upper() in {self.LABEL_ACCURATE, self.LABEL_INACCURATE}
        ]
        return {
            "image_id": image_id,
            "image": self.image_id_to_filename[image_id],
            "dataset": metadata["dataset"],
            "dataset_version": metadata["dataset_version"],
            "split": metadata["split"],
            "protocol": metadata["protocol"],
            "taxonomy_space": metadata["taxonomy_space"],
            "lexicon_version": metadata["lexicon_version"],
            "mhaldetect_sample_id": image_id,
            "mhaldetect_question": record["question"],
            "mhaldetect_response": record["response"],
            "mhaldetect_segment_count": len(labels),
            "mhaldetect_inaccurate_segment_count": sum(int(label == self.LABEL_INACCURATE) for label in labels),
            "objects": sorted(self.get_ground_truth_objects(image_id, protocol=protocol)),
        }

    def evaluate_caption(self, image_id: int, caption: str, protocol: str = "native") -> Dict[str, Any]:
        mentions = self.caption_to_mentions(caption, protocol=protocol, image_id=int(image_id))
        hallucinated = [mention for mention in mentions if int(mention["hallucinated"])]
        accurate = [mention for mention in mentions if not int(mention["hallucinated"])]
        metadata = self.protocol_metadata(protocol)
        return {
            "image_id": int(image_id),
            "caption": caption,
            "ground_truth_objects": ["accurate", "inaccurate"],
            "object_mentions": mentions,
            "hallucinated_mentions": hallucinated,
            "recall_mentions": accurate,
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
            "recall": float(len(accurate) / len(mentions)) if mentions else 0.0,
            "gt_category_count": len(mentions),
            "evaluated_gt_category_count": len(mentions),
            "gt_category_coverage": 1.0 if mentions else 0.0,
            "mhaldetect_segment_count": len(mentions),
            "mhaldetect_inaccurate_segment_count": len(hallucinated),
            "mhaldetect_accurate_segment_count": len(accurate),
        }

    def protocol_metadata(self, protocol: str) -> Dict[str, str]:
        payload = super().protocol_metadata(protocol)
        payload["dataset_version"] = f"{self.dataset_version}:{self.split}"
        return payload

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
    if normalized_dataset == "amber":
        return AmberDatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
        )
    if normalized_dataset in {"amber_discriminative", "amber-yn", "amber_yesno"}:
        return AmberDiscriminativeDatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
        )
    if normalized_dataset in {
        "amber_discriminative_paired",
        "amber-discriminative-paired",
        "amber_yesno_paired",
        "amber-yn-paired",
        "amber_paired",
    }:
        return AmberDiscriminativePairedDatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
        )
    if normalized_dataset in {"pope", "pope_paired", "pope-paired", "pope_yesno", "pope-yn"}:
        return PopePairedDatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
        )
    if normalized_dataset in {"mhaldetect", "m-haldetect", "mhal"}:
        return MHalDetectDatasetAdapter.from_cache(
            dataset_root=dataset_root,
            annotation_path=annotation_path,
            split=split,
            cache_path=cache_path,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")
