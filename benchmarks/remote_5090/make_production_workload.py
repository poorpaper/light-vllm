from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def _iter_first_turns(path: Path) -> Iterator[tuple[str, str, str]]:
    try:
        import ijson
    except ImportError as exc:  # benchmark-only streaming dependency
        raise RuntimeError("install benchmark dependency with: pip install ijson") from exc

    with path.open("rb") as source:
        for item in ijson.items(source, "item", use_float=True):
            if not isinstance(item, dict):
                continue
            conversations = item.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                continue
            first, second = conversations[:2]
            if (
                not isinstance(first, dict)
                or first.get("from") != "human"
                or not isinstance(second, dict)
                or second.get("from") != "gpt"
            ):
                continue
            prompt = first.get("value")
            response = second.get("value")
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(response, str)
                or not response.strip()
            ):
                continue
            yield str(item.get("id", "unknown")), prompt.strip(), response.strip()


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return [int(token_id) for token_id in token_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--min-prompt-tokens", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--min-output-tokens", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--declared-max-output-tokens",
        type=int,
        help="give every request this conservative server limit instead of its source length",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise ValueError("num requests must be positive")
    if not 0 < args.min_prompt_tokens <= args.max_prompt_tokens:
        raise ValueError("prompt token bounds must be positive and ordered")
    if not 0 < args.min_output_tokens <= args.max_output_tokens:
        raise ValueError("output token bounds must be positive and ordered")
    if args.declared_max_output_tokens is not None and args.declared_max_output_tokens <= 0:
        raise ValueError("declared max output tokens must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    random_source = random.Random(args.seed)
    reservoir: list[dict[str, object]] = []
    eligible = 0
    for source_id, prompt, observed_response in _iter_first_turns(args.source):
        # Avoid spending tokenizer time on records that cannot fit the configured bounds.
        if (
            len(prompt) > args.max_prompt_tokens * 16
            or len(observed_response) > args.max_output_tokens * 16
        ):
            continue
        input_ids = _encode_prompt(tokenizer, prompt)
        if not args.min_prompt_tokens <= len(input_ids) <= args.max_prompt_tokens:
            continue
        output_tokens = len(tokenizer.encode(observed_response, add_special_tokens=False))
        if not args.min_output_tokens <= output_tokens <= args.max_output_tokens:
            continue
        eligible += 1
        request = {
            "id": f"production-{eligible:05d}",
            "kind": "sharegpt-first-turn",
            "source_id": source_id,
            "input_ids": input_ids,
            "max_new_tokens": (
                args.declared_max_output_tokens
                if args.declared_max_output_tokens is not None
                else output_tokens
            ),
            "observed_response_tokens": output_tokens,
            "eos_token_id": int(tokenizer.eos_token_id),
        }
        if len(reservoir) < args.num_requests:
            reservoir.append(request)
            continue
        replacement = random_source.randrange(eligible)
        if replacement < args.num_requests:
            reservoir[replacement] = request

    if len(reservoir) != args.num_requests:
        raise ValueError(
            f"source yielded only {len(reservoir)} eligible prompts; required {args.num_requests}"
        )

    random_source.shuffle(reservoir)
    prompt_lengths = [len(request["input_ids"]) for request in reservoir]  # type: ignore[arg-type]
    output_lengths = [int(request["observed_response_tokens"]) for request in reservoir]
    requested_output_lengths = [int(request["max_new_tokens"]) for request in reservoir]
    payload = {
        "name": "production-sharegpt-trace",
        "source": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "source_file": args.source.name,
        "seed": args.seed,
        "num_requests": len(reservoir),
        "total_prompt_tokens": sum(prompt_lengths),
        "total_requested_output_tokens": sum(requested_output_lengths),
        "total_observed_response_tokens": sum(output_lengths),
        "max_new_tokens_per_request": args.declared_max_output_tokens,
        "eos_token_id": int(tokenizer.eos_token_id),
        "prompt_tokens_min": min(prompt_lengths),
        "prompt_tokens_max": max(prompt_lengths),
        "output_tokens_min": min(output_lengths),
        "output_tokens_max": max(output_lengths),
        "requests": reservoir,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "requests"}, indent=2))


if __name__ == "__main__":
    main()
