from __future__ import annotations

# 负责把对象 mention 对齐回 answer token，
# 并把 token-level vICR 聚合成 object-level 分数。

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .scoring import mean_or_zero


@dataclass
class TargetSpan:
    # 一个待分析目标在 answer 中对应到的 token 区间。
    # 对物体幻觉检测来说，这里对应 htec1 词流中的一个对象词样本。
    kind: str
    surface: str
    phrase: str
    answer_token_start: int
    answer_token_end: int
    merged_token_positions: List[int]
    canonical_name: str | None = None
    mention_index: int | None = None
    word_index: int | None = None
    hallucinated: int | None = None
    alignment_strategy: str | None = None
    source_surface: str | None = None


def _find_subsequence(sequence: Sequence[int], pattern: Sequence[int]) -> List[int]:
    # 在 token id 序列里找一个连续子序列出现的位置。
    if not pattern or len(pattern) > len(sequence):
        return []
    matches: List[int] = []
    width = len(pattern)
    for start in range(0, len(sequence) - width + 1):
        if list(sequence[start : start + width]) == list(pattern):
            matches.append(start)
    return matches


def _encode_phrase_variants(tokenizer: Any, phrase: str) -> List[List[int]]:
    # 同一个短语会同时尝试“原样”和“前面带空格”两种编码。
    stripped = phrase.strip()
    variants = [stripped, f" {stripped}"]
    encoded: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()
    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        key = tuple(token_ids)
        if token_ids and key not in seen:
            encoded.append(token_ids)
            seen.add(key)
    return encoded


def _pluralize_word(word: str) -> str:
    stripped = word.strip()
    if not stripped:
        return stripped
    lower = stripped.lower()
    if lower.endswith("s") and not lower.endswith("ss"):
        return stripped
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return stripped[:-1] + "ies"
    if lower.endswith(("ch", "sh", "x", "z", "s")):
        return stripped + "es"
    return stripped + "s"


def _singularize_word(word: str) -> str:
    stripped = word.strip()
    if not stripped:
        return stripped
    lower = stripped.lower()
    if lower.endswith("ies") and len(lower) > 3:
        return stripped[:-3] + "y"
    if lower.endswith("es") and lower[:-2].endswith(("ch", "sh", "x", "z", "s")):
        return stripped[:-2]
    if lower.endswith("s") and not lower.endswith("ss"):
        return stripped[:-1]
    return stripped


def _inflect_phrase_variants(phrase: str) -> List[str]:
    stripped = phrase.strip()
    if not stripped:
        return []
    words = stripped.split()
    variants = [stripped]
    if words:
        plural_words = list(words)
        plural_words[-1] = _pluralize_word(plural_words[-1])
        plural_phrase = " ".join(plural_words).strip()
        if plural_phrase and plural_phrase not in variants:
            variants.append(plural_phrase)

        singular_words = list(words)
        singular_words[-1] = _singularize_word(singular_words[-1])
        singular_phrase = " ".join(singular_words).strip()
        if singular_phrase and singular_phrase not in variants:
            variants.append(singular_phrase)
    return variants


def _choose_best_match(
    matches: Sequence[int],
    width: int,
    occupied: Sequence[bool],
    preferred_start: int,
) -> int | None:
    # 当一个词在 answer 里出现多次时，优先选择最靠近当前词流位置的连续 span。
    candidates: List[Tuple[int, int]] = []
    for start in matches:
        end = start + width
        if any(occupied[idx] for idx in range(start, end)):
            continue
        candidates.append((abs(start - preferred_start), start))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return int(candidates[0][1])


def _candidate_phrases(entry: Dict[str, Any]) -> List[Tuple[str, str]]:
    surface = str(entry.get("surface_word") or entry.get("surface") or entry.get("phrase") or "").strip()
    canonical = str(entry.get("canonical_name") or "").strip()
    variants = [str(item).strip() for item in entry.get("alignment_variants", []) if str(item).strip()]

    ordered: List[Tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_variants(prefix: str, phrase: str) -> None:
        for index, item in enumerate(_inflect_phrase_variants(phrase)):
            strategy = prefix if index == 0 else f"{prefix}_inflect"
            key = (strategy, item)
            if item and key not in seen:
                ordered.append((strategy, item))
                seen.add(key)

    if surface:
        add_variants("surface_exact", surface)
    if canonical and canonical != surface:
        add_variants("canonical_exact", canonical)
    for variant in variants:
        if variant in {surface, canonical}:
            continue
        add_variants("alias_exact", variant)
    return ordered


def _alignment_record(entry: Dict[str, Any], status: str, strategy: str | None, target: TargetSpan | None) -> Dict[str, Any]:
    return {
        "mention_index": entry.get("mention_index"),
        "word_index": entry.get("word_index"),
        "canonical_name": entry.get("canonical_name"),
        "surface": entry.get("surface"),
        "surface_word": entry.get("surface_word") or entry.get("surface"),
        "phrase": entry.get("phrase") or entry.get("surface"),
        "hallucinated": int(entry.get("hallucinated", 0)),
        "alignment_status": status,
        "alignment_strategy": strategy,
        "answer_token_span": [target.answer_token_start, target.answer_token_end] if target is not None else None,
        "merged_token_positions": list(target.merged_token_positions) if target is not None else [],
    }


def _align_entries_to_targets(
    tokenizer: Any,
    answer_token_ids: Sequence[int],
    answer_token_positions: Sequence[int],
    entries: Sequence[Dict[str, Any]],
) -> tuple[List[TargetSpan], List[Dict[str, Any]]]:
    # 这里的输入已经是 htec1 风格词流样本。我们尽量把它们映射回 answer token，
    # 但不再让“是否找到 token span”决定这个词在 CHAIR 里是否存在。
    occupied = [False] * len(answer_token_ids)
    targets: List[TargetSpan] = []
    alignment_rows: List[Dict[str, Any]] = []
    preferred_start = 0

    for mention_index, entry in enumerate(entries):
        matched_target: TargetSpan | None = None
        matched_strategy: str | None = None
        for strategy, phrase in _candidate_phrases(entry):
            for token_pattern in _encode_phrase_variants(tokenizer, phrase):
                match_start = _choose_best_match(
                    _find_subsequence(answer_token_ids, token_pattern),
                    width=len(token_pattern),
                    occupied=occupied,
                    preferred_start=preferred_start,
                )
                if match_start is None:
                    continue

                match_end = match_start + len(token_pattern)
                token_text = tokenizer.decode(
                    list(answer_token_ids[match_start:match_end]),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ).strip()
                matched_target = TargetSpan(
                    kind=str(entry.get("kind", "object")),
                    surface=token_text or str(entry.get("surface") or phrase),
                    phrase=str(entry.get("phrase") or entry.get("surface") or phrase),
                    answer_token_start=match_start,
                    answer_token_end=match_end,
                    merged_token_positions=list(answer_token_positions[match_start:match_end]),
                    canonical_name=entry.get("canonical_name"),
                    mention_index=int(entry.get("mention_index", mention_index)),
                    word_index=entry.get("word_index"),
                    hallucinated=int(entry.get("hallucinated", 0)),
                    alignment_strategy=strategy,
                    source_surface=str(entry.get("surface_word") or entry.get("surface") or phrase),
                )
                matched_strategy = strategy
                for idx in range(match_start, match_end):
                    occupied[idx] = True
                preferred_start = match_end
                break
            if matched_target is not None:
                break

        alignment_rows.append(
            _alignment_record(
                entry=entry,
                status="aligned" if matched_target is not None else "unaligned",
                strategy=matched_strategy,
                target=matched_target,
            )
        )
        if matched_target is not None:
            targets.append(matched_target)

    targets.sort(key=lambda item: item.answer_token_start)
    return targets, alignment_rows


def _build_targets_from_entries(
    tokenizer: Any,
    answer_token_ids: Sequence[int],
    answer_token_positions: Sequence[int],
    entries: Sequence[Dict[str, Any]],
) -> List[TargetSpan]:
    targets, _ = _align_entries_to_targets(
        tokenizer=tokenizer,
        answer_token_ids=answer_token_ids,
        answer_token_positions=answer_token_positions,
        entries=entries,
    )
    return targets


def build_object_targets(
    tokenizer: Any,
    answer_token_ids: Sequence[int],
    answer_token_positions: Sequence[int],
    object_phrases: Sequence[str],
) -> List[TargetSpan]:
    if not object_phrases:
        return []

    normalized = [phrase.strip() for phrase in object_phrases if phrase.strip()]
    entries = [
        {
            "kind": "object",
            "phrase": phrase,
            "surface": phrase,
            "surface_word": phrase,
            "canonical_name": None,
            "mention_index": idx,
        }
        for idx, phrase in enumerate(normalized)
    ]
    return _build_targets_from_entries(tokenizer, answer_token_ids, answer_token_positions, entries)


def build_object_targets_from_mentions(
    tokenizer: Any,
    answer_token_ids: Sequence[int],
    answer_token_positions: Sequence[int],
    object_mentions: Sequence[Dict[str, Any]],
) -> tuple[List[TargetSpan], List[Dict[str, Any]]]:
    # object_mentions 现在是 CHAIR 词流样本。
    # 即使某个词最后没对齐到 token，也会出现在 alignment_debug 中。
    entries: List[Dict[str, Any]] = []
    for mention_index, mention in enumerate(object_mentions):
        phrase = str(mention.get("phrase") or mention.get("surface") or "").strip()
        if not phrase:
            continue
        entries.append(
            {
                "kind": "object",
                "phrase": phrase,
                "surface": mention.get("surface"),
                "surface_word": mention.get("surface_word") or mention.get("surface"),
                "canonical_name": mention.get("canonical_name"),
                "word_index": mention.get("word_index"),
                "hallucinated": mention.get("hallucinated", 0),
                "mention_index": mention.get("mention_index", mention_index),
                "alignment_variants": list(mention.get("alignment_variants", [])),
            }
        )
    return _align_entries_to_targets(
        tokenizer=tokenizer,
        answer_token_ids=answer_token_ids,
        answer_token_positions=answer_token_positions,
        entries=entries,
    )


def summarize_targets(
    token_scores: Sequence[Dict[str, Any]],
    targets: Iterable[TargetSpan],
) -> List[Dict[str, Any]]:
    token_scores = list(token_scores)
    results: List[Dict[str, Any]] = []

    for target in targets:
        selected = token_scores[target.answer_token_start : target.answer_token_end]
        if not selected:
            continue

        layer_to_values: Dict[int, List[float]] = {}
        for token_score in selected:
            for layer_score in token_score["layer_scores"]:
                layer = int(layer_score["layer"])
                layer_to_values.setdefault(layer, []).append(float(layer_score["vicr"]))

        layer_summary = [
            {"layer": layer, "mean_vicr": mean_or_zero(values)}
            for layer, values in sorted(layer_to_values.items())
        ]
        object_layer_icr = [float(item["mean_vicr"]) for item in layer_summary]
        object_global_mean_icr = mean_or_zero(object_layer_icr)

        results.append(
            {
                "kind": target.kind,
                "phrase": target.phrase,
                "surface": target.surface,
                "source_surface_word": target.source_surface,
                "canonical_name": target.canonical_name,
                "mention_index": target.mention_index,
                "word_index": target.word_index,
                "hallucinated": int(target.hallucinated or 0),
                "alignment_status": "aligned",
                "alignment_strategy": target.alignment_strategy,
                "answer_token_span": [target.answer_token_start, target.answer_token_end],
                "merged_token_positions": target.merged_token_positions,
                "subtokens": [item["token"] for item in selected],
                "object_layer_icr": object_layer_icr,
                "object_global_mean_icr": object_global_mean_icr,
                "mean_vicr": object_global_mean_icr,
                "max_vicr": max((item["max_vicr"] for item in selected), default=0.0),
                "layer_mean_scores": layer_summary,
                "token_aligned": True,
            }
        )

    return results
