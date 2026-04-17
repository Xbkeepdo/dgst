from __future__ import annotations

# 统一维护 JSON/JSONL 读写、分片和实验日志。
# 这样编排层和算法层都不需要重复复制这些基础 helper。

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    if input_path.suffix != ".json":
        return read_jsonl(input_path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        rows = payload.get("entries")
        if isinstance(rows, list):
            return rows
        return [payload]
    raise ValueError(f"Expected JSON list or JSONL in {input_path}")


def split_round_robin(items: Sequence[int], num_shards: int) -> list[list[int]]:
    shards = [[] for _ in range(num_shards)]
    for index, item in enumerate(items):
        shards[index % num_shards].append(int(item))
    return shards


def write_lines(path: str | Path, values: Iterable[int]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(value) for value in values), encoding="utf-8")


def _pretty_log_path(path: Path) -> Path:
    return path.with_name("experiment_log.pretty.json")


def append_experiment_log(path: str | Path, payload: dict[str, Any]) -> None:
    # experiment_log.jsonl 保持程序友好的逐行结构，
    # pretty 版本则方便直接在 IDE 里浏览。
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())
    append_jsonl(log_path, row)
    pretty_path = _pretty_log_path(log_path)
    pretty_path.write_text(json.dumps(read_jsonl(log_path), ensure_ascii=False, indent=2), encoding="utf-8")
