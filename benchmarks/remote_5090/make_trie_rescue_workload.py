from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _first_successful_output(payload: dict[str, Any]) -> list[int]:
    for result in payload["results"]:
        tokens = [int(token_id) for token_id in result["generated_token_ids"]]
        if result.get("error") is None and tokens:
            return tokens
    raise ValueError("the reference result contains no successful output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-workload", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--frequent-branches", type=int, default=3)
    parser.add_argument("--suffix-tokens", type=int, default=5)
    parser.add_argument("--branch-tokens", type=int, default=7)
    parser.add_argument("--decoy-token", type=int)
    parser.add_argument("--filler-tokens", type=int, default=12)
    args = parser.parse_args()

    workload = json.loads(args.source_workload.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
    seed = [int(token_id) for token_id in workload["requests"][0]["input_ids"]]
    reference_output = _first_successful_output(reference)
    suffix = seed[-args.suffix_tokens :]
    branch = reference_output[: args.branch_tokens]
    if len(suffix) != args.suffix_tokens or len(branch) != args.branch_tokens:
        raise ValueError("source prompt and reference output are shorter than requested")

    if args.filler_tokens < 0:
        raise ValueError("filler token count must be non-negative")
    filler = seed[: args.filler_tokens]
    decoy_root = args.decoy_token if args.decoy_token is not None else (branch[0] + 1) % 151_936
    if not 0 <= decoy_root < 151_936:
        raise ValueError("decoy token must be within the Qwen2.5 vocabulary")
    if decoy_root == branch[0]:
        raise AssertionError("decoy root must differ from the target branch")
    frequent_segment = suffix + branch + filler
    decoy_segment = suffix + [decoy_root, *branch[1:]] + filler
    tail = frequent_segment * args.frequent_branches + decoy_segment + suffix
    if len(tail) > args.prompt_tokens:
        raise ValueError("configured branch evidence does not fit in the prompt")
    prefix_length = args.prompt_tokens - len(tail)
    prefix = (seed * (prefix_length // len(seed) + 1))[:prefix_length]
    prompt = prefix + tail

    requests = [
        {
            "id": f"trie-rescue-{index:03d}",
            "kind": "trie-rescue",
            "input_ids": prompt,
            "max_new_tokens": args.output_tokens,
        }
        for index in range(args.requests)
    ]
    payload = {
        "name": "trie-rescues-recent-chain-miss",
        "num_requests": len(requests),
        "total_prompt_tokens": len(requests) * len(prompt),
        "total_output_tokens": len(requests) * args.output_tokens,
        "construction": {
            "suffix": suffix,
            "frequent_branch": branch,
            "recent_decoy_branch": [decoy_root, *branch[1:]],
            "frequent_branch_occurrences": args.frequent_branches,
        },
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
