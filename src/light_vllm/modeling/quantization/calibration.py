"""AWQ 校准语料读取与固定长度 token 构造。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from io import TextIOBase
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


class CalibrationTokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def _conversation_text(value: object) -> str | None:
    if not isinstance(value, list):
        return None
    turns: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        role = item.get("role", item.get("from", "unknown"))
        content = item.get("content", item.get("value"))
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        turns.append(f"<{role}>\n{content}")
    return "\n".join(turns) if turns else None


def _json_array_records(stream: TextIOBase, *, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    """增量解析顶层 JSON 数组，避免 ShareGPT 大文件被一次性载入内存。"""

    if chunk_size <= 0:
        raise ValueError("calibration JSON chunk_size must be positive")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False

    def read_more() -> bool:
        nonlocal buffer, position, eof
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = stream.read(chunk_size)
        if not chunk:
            eof = True
            return False
        buffer += chunk
        return True

    def skip_space() -> None:
        nonlocal position
        while position < len(buffer) and buffer[position].isspace():
            position += 1

    while True:
        skip_space()
        if position < len(buffer):
            break
        if not read_more():
            raise ValueError("calibration JSON is empty")
    if buffer[position] != "[":
        raise ValueError("calibration JSON must contain a top-level array")
    position += 1

    has_record = False
    after_comma = False
    while True:
        skip_space()
        if position >= len(buffer):
            if not read_more():
                raise ValueError("calibration JSON array is missing its closing bracket")
            continue
        if buffer[position] == "]":
            if after_comma:
                raise ValueError("calibration JSON array cannot end after a comma")
            position += 1
            break
        if has_record and not after_comma:
            raise ValueError("calibration JSON array records must be separated by a comma")
        try:
            record, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError as exc:
            if eof or not read_more():
                raise ValueError("invalid or truncated calibration JSON array") from exc
            continue
        yield record
        position = end
        has_record = True
        after_comma = False

        while True:
            skip_space()
            if position >= len(buffer):
                if not read_more():
                    raise ValueError("calibration JSON array is missing its closing bracket")
                continue
            marker = buffer[position]
            if marker == ",":
                position += 1
                after_comma = True
                break
            if marker == "]":
                position += 1
                break
            raise ValueError("calibration JSON array records must be separated by a comma")
        if not after_comma:
            break

    trailing = buffer[position:] + stream.read()
    if trailing.strip():
        raise ValueError("calibration JSON contains data after the top-level array")


def _record_text(record: object, *, text_field: str, location: str) -> str:
    if not isinstance(record, Mapping):
        raise ValueError(f"calibration {location} must contain an object")
    value = record.get(text_field)
    text = value if isinstance(value, str) else None
    if text is None:
        text = _conversation_text(record.get("conversations", record.get("messages")))
    if text is None:
        raise ValueError(
            f"calibration {location} has no string {text_field!r} or valid conversations/messages"
        )
    return text


def iter_calibration_texts(
    path: str | Path,
    *,
    text_field: str = "text",
    max_records: int | None = None,
) -> Iterator[str]:
    """流式读取纯文本、JSONL 或 JSON 数组中的校准文本。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"calibration data {source} does not exist")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("calibration text_field must be a non-empty string")
    if max_records is not None and max_records <= 0:
        raise ValueError("calibration max_records must be positive")
    yielded = 0
    suffix = source.suffix.lower()
    with source.open("r", encoding="utf-8") as stream:
        if suffix == ".json":
            for record_number, record in enumerate(_json_array_records(stream), start=1):
                text = _record_text(
                    record,
                    text_field=text_field,
                    location=f"record {record_number}",
                )
                if text.strip():
                    yield text
                    yielded += 1
                if max_records is not None and yielded >= max_records:
                    break
        else:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if suffix == ".jsonl":
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid calibration JSON at line {line_number}") from exc
                    text = _record_text(
                        record,
                        text_field=text_field,
                        location=f"line {line_number}",
                    )
                else:
                    text = line
                if text.strip():
                    yield text
                    yielded += 1
                if max_records is not None and yielded >= max_records:
                    break
    if not yielded:
        raise ValueError("calibration dataset contains no usable text")


def read_calibration_texts(
    path: str | Path,
    *,
    text_field: str = "text",
    max_records: int | None = None,
) -> tuple[str, ...]:
    """需要重复访问数据时，显式收集流式读取结果。"""

    return tuple(
        iter_calibration_texts(
            path,
            text_field=text_field,
            max_records=max_records,
        )
    )


def build_calibration_input_ids(
    tokenizer: CalibrationTokenizer,
    texts: Sequence[str] | Iterable[str],
    *,
    max_samples: int = 128,
    sequence_length: int = 512,
) -> Tensor:
    """把语料串接后切成无 padding 的等长校准块。"""

    if max_samples <= 0 or sequence_length <= 0:
        raise ValueError("calibration sample count and sequence length must be positive")
    separator = tokenizer.eos_token_id
    stream: list[int] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue
        stream.extend(token_ids)
        if separator is not None:
            stream.append(separator)
        if len(stream) >= max_samples * sequence_length:
            break
    if not stream:
        raise ValueError("calibration text produced no token IDs")
    num_full_samples = min(max_samples, len(stream) // sequence_length)
    if num_full_samples:
        used = num_full_samples * sequence_length
        return torch.tensor(stream[:used], dtype=torch.long).reshape(
            num_full_samples, sequence_length
        )
    # 小型单元测试或试跑可以不足一个完整 block；仍不引入 padding token。
    return torch.tensor(stream[:sequence_length], dtype=torch.long).unsqueeze(0)
