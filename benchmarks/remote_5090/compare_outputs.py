from __future__ import annotations

import argparse
import json
from pathlib import Path


def _outputs(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        result["request_id"]: result["generated_token_ids"]
        for result in payload["results"]
        if result["error"] is None
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()

    expected = _outputs(args.expected)
    actual = _outputs(args.actual)
    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    mismatched = sorted(
        request_id
        for request_id in expected.keys() & actual.keys()
        if expected[request_id] != actual[request_id]
    )
    summary = {
        "expected": len(expected),
        "actual": len(actual),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
    }
    print(json.dumps(summary, indent=2))
    if missing or extra or mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
