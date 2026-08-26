from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CODE_CORPUS = """
from dataclasses import dataclass

@dataclass
class RequestState:
    request_id: str
    prompt_tokens: int
    output_tokens: int

def reserve_capacity(state: RequestState, available_tokens: int) -> bool:
    required_tokens = state.prompt_tokens + state.output_tokens
    if required_tokens > available_tokens:
        return False
    return True

def release_capacity(state: RequestState, available_tokens: int) -> int:
    released_tokens = state.prompt_tokens + state.output_tokens
    return available_tokens + released_tokens
""".strip()


REPETITION_CORPUS = """
def normalize_user_record(record):
    normalized = {
        "user_id": record["user_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_project_record(record):
    normalized = {
        "project_id": record["project_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_team_record(record):
    normalized = {
        "team_id": record["team_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_service_record(record):
    normalized = {
""".strip()


BRANCHING_TAIL = """
def normalize_user_record(record):
    normalized = {
        "user_id": record["user_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_user_record(record):
    normalized = {
        "user_id": record["user_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_project_record(record):
    normalized = {
        "project_id": record["project_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_team_record(record):
    normalized = {
        "team_id": record["team_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_service_record(record):
    normalized = {
        "service_id": record["service_id"],
        "display_name": record["display_name"].strip(),
        "is_active": bool(record["is_active"]),
    }
    return normalized

def normalize_user_record(record):
    normalized = {
""".strip()


def _fit_tokens(tokenizer: Any, text: str, length: int) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("tokenizer produced an empty sequence")
    repeated = token_ids * (length // len(token_ids) + 1)
    return repeated[:length]


def _fit_tokens_with_tail(tokenizer: Any, tail: str, length: int) -> list[int]:
    tail_ids = tokenizer.encode(tail, add_special_tokens=False)
    if len(tail_ids) >= length:
        raise ValueError("branching tail must be shorter than the requested prompt")
    prefix_ids = tokenizer.encode(CODE_CORPUS, add_special_tokens=False)
    prefix_length = length - len(tail_ids)
    repeated_prefix = prefix_ids * (prefix_length // len(prefix_ids) + 1)
    return repeated_prefix[:prefix_length] + tail_ids


def _request(
    tokenizer: Any,
    *,
    request_id: str,
    kind: str,
    text: str,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, object]:
    return {
        "id": request_id,
        "kind": kind,
        "input_ids": _fit_tokens(tokenizer, text, prompt_tokens),
        "max_new_tokens": output_tokens,
    }


def _write(path: Path, *, name: str, requests: list[dict[str, object]]) -> None:
    prompt_tokens = sum(len(request["input_ids"]) for request in requests)  # type: ignore[arg-type]
    output_tokens = sum(int(request["max_new_tokens"]) for request in requests)
    payload = {
        "name": name,
        "num_requests": len(requests),
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
        "requests": requests,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = [
        _request(
            tokenizer,
            request_id=f"baseline-{index:03d}",
            kind="baseline",
            text=CODE_CORPUS,
            prompt_tokens=256,
            output_tokens=64,
        )
        for index in range(64)
    ]
    _write(args.output_dir / "baseline.json", name="baseline-256-64", requests=baseline)

    decode_steady = [
        _request(
            tokenizer,
            request_id=f"decode-steady-{index:03d}",
            kind="decode-steady",
            text=CODE_CORPUS,
            prompt_tokens=16,
            output_tokens=512,
        )
        for index in range(16)
    ]
    _write(
        args.output_dir / "decode-steady.json",
        name="decode-steady-16-512",
        requests=decode_steady,
    )
    correctness = [
        _request(
            tokenizer,
            request_id=f"correctness-{index:03d}",
            kind="correctness",
            text=CODE_CORPUS,
            prompt_tokens=16,
            output_tokens=8,
        )
        for index in range(4)
    ]
    _write(
        args.output_dir / "correctness.json",
        name="correctness-4x16-8",
        requests=correctness,
    )
    decode_steady_32 = [
        _request(
            tokenizer,
            request_id=f"decode-steady-32-{index:03d}",
            kind="decode-steady",
            text=CODE_CORPUS,
            prompt_tokens=16,
            output_tokens=512,
        )
        for index in range(32)
    ]
    _write(
        args.output_dir / "decode-steady-32.json",
        name="decode-steady-32x16-512",
        requests=decode_steady_32,
    )

    pressure: list[dict[str, object]] = []
    for index in range(12):
        pressure.append(
            _request(
                tokenizer,
                request_id=f"long-{index:03d}",
                kind="long",
                text=CODE_CORPUS,
                prompt_tokens=2048,
                output_tokens=256,
            )
        )
        for short_index in range(3):
            pressure.append(
                _request(
                    tokenizer,
                    request_id=f"short-{index:03d}-{short_index}",
                    kind="short",
                    text=CODE_CORPUS,
                    prompt_tokens=64,
                    output_tokens=64,
                )
            )
    _write(
        args.output_dir / "scheduler-pressure.json",
        name="scheduler-pressure-mixed",
        requests=pressure,
    )

    preemption_pressure = [
        _request(
            tokenizer,
            request_id=f"decode-heavy-{index:03d}",
            kind="decode-heavy",
            text=CODE_CORPUS,
            prompt_tokens=128,
            output_tokens=512,
        )
        for index in range(16)
    ]
    _write(
        args.output_dir / "preemption-pressure.json",
        name="preemption-pressure-128-512",
        requests=preemption_pressure,
    )

    speculative = [
        _request(
            tokenizer,
            request_id=f"spec-{index:03d}",
            kind="repetition-code",
            text=REPETITION_CORPUS,
            prompt_tokens=512,
            output_tokens=128,
        )
        for index in range(48)
    ]
    _write(
        args.output_dir / "speculative-code.json",
        name="speculative-repetition-code",
        requests=speculative,
    )

    branching = [
        {
            "id": f"branching-{index:03d}",
            "kind": "branching-code",
            "input_ids": _fit_tokens_with_tail(tokenizer, BRANCHING_TAIL, 512),
            "max_new_tokens": 128,
        }
        for index in range(48)
    ]
    _write(
        args.output_dir / "speculative-branching-code.json",
        name="speculative-branching-code",
        requests=branching,
    )


if __name__ == "__main__":
    main()
