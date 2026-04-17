from __future__ import annotations

# CHAIR 风格对象解析和缓存逻辑的基础模块。

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

try:
    import nltk
except Exception:
    nltk = None
from tqdm import tqdm

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
NLTK_DATA_CANDIDATES = [
    Path("/home/apulis-dev/userdata/nltk/my_nltk"),
    Path("/home/apulis-dev/userdata/nltk"),
    Path("/home/apulis-dev/nltk_data"),
]
HTEC1_CHAIR_SOURCE_CANDIDATES = [
    Path("/home/apulis-dev/code/htec1/util/chair.py"),
    Path(__file__).resolve().parents[2] / "htec1" / "util" / "chair.py",
]

COCO_ALIASES: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "player", "players", "child", "children", "baby", "babies", "kid", "kids", "skier", "skiers", "surfer", "surfers"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cycle", "cycles"],
    "car": ["car", "cars", "sedan", "sedans", "suv", "suvs", "taxi", "taxis", "automobile", "automobiles"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "moped", "mopeds"],
    "airplane": ["airplane", "airplanes", "plane", "planes", "jet", "jets", "aircraft", "aircrafts"],
    "bus": ["bus", "buses", "coach", "coaches"],
    "train": ["train", "trains", "locomotive", "locomotives"],
    "truck": ["truck", "trucks", "pickup", "pickups", "lorry", "lorries"],
    "boat": ["boat", "boats", "ship", "ships", "canoe", "canoes", "kayak", "kayaks"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
    "parking meter": ["parking meter", "parking meters"],
    "bench": ["bench", "benches"],
    "bird": ["bird", "birds", "seagull", "seagulls", "pigeon", "pigeons", "duck", "ducks"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "horse": ["horse", "horses", "pony", "ponies"],
    "sheep": ["sheep"],
    "cow": ["cow", "cows", "cattle"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "backpacks", "rucksack", "rucksacks"],
    "umbrella": ["umbrella", "umbrellas"],
    "handbag": ["handbag", "handbags", "purse", "purses"],
    "tie": ["tie", "ties", "necktie", "neckties"],
    "suitcase": ["suitcase", "suitcases", "luggage"],
    "frisbee": ["frisbee", "frisbees", "disc", "discs"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "sports balls", "ball", "balls", "soccer ball", "soccer balls", "football", "footballs", "basketball", "basketballs", "tennis ball", "tennis balls", "baseball", "baseballs"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "baseball bats", "bat", "bats"],
    "baseball glove": ["baseball glove", "baseball gloves", "glove", "gloves", "mitt", "mitts"],
    "skateboard": ["skateboard", "skateboards"],
    "surfboard": ["surfboard", "surfboards"],
    "tennis racket": ["tennis racket", "tennis rackets", "racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottle", "bottles"],
    "wine glass": ["wine glass", "wine glasses", "glass", "glasses", "goblet", "goblets"],
    "cup": ["cup", "cups", "mug", "mugs"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches", "burger", "burgers", "hamburger", "hamburgers"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes"],
    "chair": ["chair", "chairs"],
    "couch": ["couch", "couches", "sofa", "sofas"],
    "potted plant": ["potted plant", "potted plants", "plant", "plants"],
    "bed": ["bed", "beds"],
    "dining table": ["dining table", "dining tables", "table", "tables"],
    "toilet": ["toilet", "toilets"],
    "tv": ["tv", "tvs", "television", "televisions", "monitor", "monitors"],
    "laptop": ["laptop", "laptops", "notebook", "notebooks"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remotes"],
    "keyboard": ["keyboard", "keyboards"],
    "cell phone": ["cell phone", "cell phones", "phone", "phones", "smartphone", "smartphones", "mobile phone", "mobile phones"],
    "microwave": ["microwave", "microwaves"],
    "oven": ["oven", "ovens", "stove", "stoves"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges"],
    "book": ["book", "books"],
    "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy bears", "teddy", "teddies", "stuffed bear", "stuffed bears"],
    "hair drier": ["hair drier", "hair driers", "hair dryer", "hair dryers", "dryer", "dryers"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}

COCO_DOUBLE_WORDS = [
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "home plate",
    "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]

IRREGULAR_SINGULARS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "knives": "knife",
    "wives": "wife",
    "loaves": "loaf",
    "wolves": "wolf",
    "leaves": "leaf",
    "tomatoes": "tomato",
    "potatoes": "potato",
    "buses": "bus",
    "glasses": "glass",
    "batteries": "battery",
}

_PATTERN_SINGULARIZE: Callable[[str], str] | None = None
_PATTERN_IMPORT_FAILED = False


def _fallback_synonyms_text() -> str:
    lines: List[str] = []
    for canonical, aliases in COCO_ALIASES.items():
        values = [canonical] + [alias for alias in aliases if alias != canonical]
        lines.append(", ".join(values))
    return "\n".join(lines)


def _load_htec1_synonyms_text() -> str:
    for path in HTEC1_CHAIR_SOURCE_CANDIDATES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"synonyms_txt\s*=\s*'''(.*?)'''", text, flags=re.S)
        if match:
            return match.group(1)
    return _fallback_synonyms_text()


def _parse_synonym_groups(text: str) -> List[List[str]]:
    groups: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        values = [item.strip() for item in line.split(",") if item.strip()]
        if values:
            groups.append(values)
    return groups


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _get_pattern_singularize() -> Callable[[str], str] | None:
    global _PATTERN_SINGULARIZE, _PATTERN_IMPORT_FAILED
    if _PATTERN_SINGULARIZE is not None:
        return _PATTERN_SINGULARIZE
    if _PATTERN_IMPORT_FAILED:
        return None
    try:
        from pattern.en import singularize as pattern_singularize
    except Exception:
        _PATTERN_IMPORT_FAILED = True
        return None
    _PATTERN_SINGULARIZE = pattern_singularize
    return _PATTERN_SINGULARIZE


class CocoChairEvaluator:
    def __init__(self, annotation_dir: str | Path) -> None:
        self.annotation_dir = Path(annotation_dir)
        self.annotation_source = self.annotation_dir
        self._ensure_runtime_state()
        self.category_id_to_name: Dict[int, str] = {}
        self.image_id_to_objects: Dict[int, set[str]] = {}
        self.image_id_to_references: Dict[int, List[str]] = {}
        self.val_image_ids: List[int] = []
        self._load_annotations()

    @classmethod
    def from_cache(cls, annotation_dir: str | Path, cache_path: str | Path | None = None) -> "CocoChairEvaluator":
        cache = Path(cache_path) if cache_path is not None else None
        source = Path(annotation_dir)
        if cache is not None and cache.exists():
            if cache.suffix == ".pkl":
                evaluator = pickle.load(cache.open("rb"))
                evaluator.annotation_source = source
                evaluator.annotation_dir = source
                evaluator._ensure_runtime_state()
                return evaluator
            payload = json.loads(cache.read_text(encoding="utf-8"))
            obj = cls.__new__(cls)
            obj.annotation_source = source
            obj.annotation_dir = source
            obj._ensure_runtime_state()
            obj.category_id_to_name = {int(k): v for k, v in payload["category_id_to_name"].items()}
            obj.image_id_to_objects = {int(k): set(v) for k, v in payload["image_id_to_objects"].items()}
            obj.image_id_to_references = {int(k): list(v) for k, v in payload["image_id_to_references"].items()}
            obj.val_image_ids = [int(v) for v in payload["val_image_ids"]]
            return obj

        obj = cls(source)
        if cache is not None:
            obj.save_cache(cache)
        return obj

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
            "category_id_to_name": self.category_id_to_name,
            "image_id_to_objects": {str(k): sorted(v) for k, v in self.image_id_to_objects.items()},
            "image_id_to_references": {str(k): v for k, v in self.image_id_to_references.items()},
            "val_image_ids": self.val_image_ids,
        }

    def _ensure_runtime_state(self) -> None:
        # 运行时状态与缓存里的 ground truth 分开维护。
        # 这样即使 chair.pkl 是旧版本缓存，也能在加载后重新套用 htec1 风格同义词规则。
        self.synonym_groups = self._load_synonym_groups()
        self.alias_to_canonical = self._build_alias_map()
        self.mscoco_objects = set(self.alias_to_canonical.keys())
        self.inverse_synonym_dict = dict(self.alias_to_canonical)
        self.canonical_to_aliases = self._build_canonical_alias_map()
        self.category_ids = [group[0] for group in self.synonym_groups if group]
        self.double_word_dict = self._build_double_word_dict()

    def _load_synonym_groups(self) -> List[List[str]]:
        return _parse_synonym_groups(_load_htec1_synonyms_text())

    def _build_alias_map(self) -> Dict[str, str]:
        alias_to_canonical: Dict[str, str] = {}
        for group in self.synonym_groups:
            if not group:
                continue
            canonical = group[0]
            for alias in group:
                alias_to_canonical[alias] = canonical
        return alias_to_canonical

    def _build_canonical_alias_map(self) -> Dict[str, List[str]]:
        canonical_to_aliases: Dict[str, List[str]] = {}
        for group in self.synonym_groups:
            if not group:
                continue
            canonical_to_aliases[group[0]] = _dedupe_preserve_order(group)
        return canonical_to_aliases

    def _build_double_word_dict(self) -> Dict[str, str]:
        mapping = {phrase: phrase for phrase in COCO_DOUBLE_WORDS}
        for animal_word in ANIMAL_WORDS:
            mapping[f"baby {animal_word}"] = animal_word
            mapping[f"adult {animal_word}"] = animal_word
        for vehicle_word in VEHICLE_WORDS:
            mapping[f"passenger {vehicle_word}"] = vehicle_word
        mapping["bow tie"] = "tie"
        mapping["toilet seat"] = "toilet"
        mapping["wine glas"] = "wine glass"
        return mapping

    def _progress(self, iterable: Iterable[Any], desc: str, leave: bool = True) -> tqdm:
        return tqdm(
            iterable,
            desc=desc,
            leave=leave,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
        )

    def _load_annotations(self) -> None:
        print(f"Loading COCO annotations from: {self.annotation_dir}", flush=True)
        instances_train = self._read_optional_json("instances_train2014.json")
        instances_val = self._read_optional_json("instances_val2014.json")
        captions_train = self._read_optional_json("captions_train2014.json")
        captions_val = self._read_optional_json("captions_val2014.json")

        for split_name, payload in [("instances_train2014", instances_train), ("instances_val2014", instances_val)]:
            if not payload:
                continue
            for category in self._progress(payload.get("categories", []), desc=f"Loading {split_name} categories", leave=False):
                self.category_id_to_name[int(category["id"])] = str(category["name"])

        for split_name, payload in [("instances_train2014", instances_train), ("instances_val2014", instances_val)]:
            if not payload:
                continue
            for annotation in self._progress(payload.get("annotations", []), desc=f"Loading {split_name} objects"):
                image_id = int(annotation["image_id"])
                category_id = int(annotation["category_id"])
                category_name = self.category_id_to_name.get(category_id)
                if category_name is not None:
                    self.image_id_to_objects.setdefault(image_id, set()).add(category_name)

        for split_name, payload in [("captions_train2014", captions_train), ("captions_val2014", captions_val)]:
            if not payload:
                continue
            for annotation in self._progress(payload.get("annotations", []), desc=f"Loading {split_name} references"):
                image_id = int(annotation["image_id"])
                caption = str(annotation.get("caption", ""))
                self.image_id_to_references.setdefault(image_id, []).append(caption)
                for mention in self.caption_to_mentions(caption):
                    self.image_id_to_objects.setdefault(image_id, set()).add(mention["canonical_name"])

        if instances_val:
            self.val_image_ids = sorted(int(item["id"]) for item in instances_val.get("images", []))
        elif captions_val:
            self.val_image_ids = sorted(int(item["id"]) for item in captions_val.get("images", []))
        print(
            "Finished loading annotations: "
            f"val_images={len(self.val_image_ids)} "
            f"object_gt_images={len(self.image_id_to_objects)} "
            f"reference_caption_images={len(self.image_id_to_references)}"
        , flush=True)

    def _read_optional_json(self, filename: str) -> Dict[str, Any] | None:
        path = self.annotation_dir / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _normalize_token(self, token: str) -> str:
        # 尽量贴近 htec1/glsim 的 baseline：优先用 pattern.en 的 singularize。
        # 这里改成延迟导入，避免离线环境里 wordnet 缺失时在模块导入阶段直接崩掉。
        pattern_singularize = _get_pattern_singularize()
        if pattern_singularize is not None:
            normalized = pattern_singularize(token)
            # pattern.en 会把 tennis 这类词错误裁成 tenni，这里做一个很小的保护，
            # 避免明显的词形错误污染后续 COCO 物体匹配。
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

    def _tokenize(self, text: str) -> List[str]:
        if nltk is not None:
            for path in NLTK_DATA_CANDIDATES:
                if path.exists():
                    path_str = str(path)
                    if path_str not in nltk.data.path:
                        nltk.data.path.insert(0, path_str)
            try:
                raw_tokens = nltk.word_tokenize(text.lower())
            except Exception:
                raw_tokens = _WORD_RE.findall(text.lower())
        else:
            raw_tokens = _WORD_RE.findall(text.lower())
        return [self._normalize_token(token) for token in raw_tokens if _WORD_RE.fullmatch(token)]

    def _build_alignment_variants(self, surface: str, canonical: str) -> List[str]:
        variants = [surface]
        if canonical:
            variants.append(canonical)
            variants.extend(self.canonical_to_aliases.get(canonical, []))
        return _dedupe_preserve_order([item.strip() for item in variants if item and item.strip()])

    def caption_to_words(self, caption: str) -> tuple[List[str], List[str], List[int], List[str]]:
        # 这一步尽量贴近 htec1：先做词级分词与词形归一化，再做双词合并，
        # 最后把同义词映射成 canonical name，但保留 surface 词本身用于后续 token 对齐。
        words = self._tokenize(caption)
        i = 0
        double_words: List[str] = []
        idxs: List[int] = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        if "toilet" in words and "seat" in words:
            words = [word for word in words if word != "seat"]

        filtered_idxs = [idxs[idx] for idx, word in enumerate(words) if word in self.mscoco_objects]
        filtered_words = [word for word in words if word in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[word] for word in filtered_words]
        return filtered_words, node_words, filtered_idxs, double_words

    def caption_to_mentions(self, caption: str) -> List[Dict[str, Any]]:
        tokens, node_words, idxs, _ = self.caption_to_words(caption)
        mentions: List[Dict[str, Any]] = []
        for mention_index, (word, canonical, idx) in enumerate(zip(tokens, node_words, idxs)):
            token_span = word.count(" ") + 1
            mentions.append({
                "surface": word,
                "surface_word": word,
                "phrase": word,
                "canonical_name": canonical,
                "word_index": idx,
                "mention_index": mention_index,
                "token_start": idx,
                "token_end": idx + token_span,
                "alignment_variants": self._build_alignment_variants(word, canonical),
            })
        return mentions

    def compute_hallucinations(self, image_id: int, caption: str) -> Dict[str, Any]:
        gt_objects = self.get_ground_truth_objects(int(image_id))
        words, node_words, idxs, _ = self.caption_to_words(caption)
        cap_dict = {
            "mscoco_hallucinated_words": [],
            "mscoco_gt_words": sorted(gt_objects),
            "mscoco_generated_words": list(node_words),
            "hallucination_idxs": [],
            "hallucinated_words": 0,
            "recall_words": [],
            "recall_idxs": [],
            "object_mentions": [],
        }

        for mention_index, (word, node_word, idx) in enumerate(zip(words, node_words, idxs)):
            mention = {
                "surface": word,
                "surface_word": word,
                "phrase": word,
                "canonical_name": node_word,
                "word_index": idx,
                "mention_index": mention_index,
                "token_start": idx,
                "token_end": idx + word.count(" ") + 1,
                "alignment_variants": self._build_alignment_variants(word, node_word),
            }
            cap_dict["object_mentions"].append(mention)
            if node_word not in gt_objects:
                cap_dict["hallucinated_words"] += 1
                cap_dict["mscoco_hallucinated_words"].append((word, node_word))
                cap_dict["hallucination_idxs"].append(idx)
            else:
                cap_dict["recall_words"].append((word, node_word))
                cap_dict["recall_idxs"].append(idx)

        return cap_dict

    def get_ground_truth_objects(self, image_id: int) -> set[str]:
        return set(self.image_id_to_objects.get(int(image_id), set()))

    @staticmethod
    def image_filename(image_id: int) -> str:
        return f"COCO_val2014_{int(image_id):012d}.jpg"

    def ground_truth_entry(self, image_id: int) -> Dict[str, Any]:
        image_id = int(image_id)
        return {
            "image_id": image_id,
            "image": self.image_filename(image_id),
            "objects": sorted(self.get_ground_truth_objects(image_id)),
        }

    def iter_ground_truth_entries(
        self,
        image_ids: Iterable[int] | None = None,
        limit: int | None = None,
    ) -> Iterable[Dict[str, Any]]:
        selected_ids = list(image_ids) if image_ids is not None else list(self.val_image_ids)
        if limit is not None:
            selected_ids = selected_ids[:limit]
        for image_id in selected_ids:
            yield self.ground_truth_entry(image_id)

    def save_ground_truth_jsonl(
        self,
        output_path: str | Path,
        image_ids: Iterable[int] | None = None,
        limit: int | None = None,
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.iter_ground_truth_entries(image_ids=image_ids, limit=limit):
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def evaluate_caption(self, image_id: int, caption: str) -> Dict[str, Any]:
        gt_objects = self.get_ground_truth_objects(int(image_id))
        cap_dict = self.compute_hallucinations(image_id, caption)
        mentions = cap_dict["object_mentions"]
        for mention in mentions:
            mention["hallucinated"] = int(mention["canonical_name"] not in gt_objects)
        hallucinated = [mention for mention in mentions if mention["canonical_name"] not in gt_objects]
        recall_mentions = [mention for mention in mentions if mention["canonical_name"] in gt_objects]
        return {
            "image_id": int(image_id),
            "caption": caption,
            "ground_truth_objects": sorted(gt_objects),
            "object_mentions": mentions,
            "hallucinated_mentions": hallucinated,
            "recall_mentions": recall_mentions,
            "chair_backend": "htec1_aligned",
            "chair_word_count_total": len(mentions),
            "chair_s": 1.0 if hallucinated else 0.0,
            "chair_i": float(len(hallucinated) / len(mentions)) if mentions else 0.0,
            "recall": float(len({item['canonical_name'] for item in recall_mentions}) / len(gt_objects)) if gt_objects else 0.0,
        }

    def iter_val_image_ids(self, limit: int | None = None) -> Iterable[int]:
        image_ids = self.val_image_ids
        if limit is not None:
            image_ids = image_ids[:limit]
        return image_ids
