from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    events: list[dict[str, Any]] = json.loads(args.trace.read_text(encoding="utf-8"))["traceEvents"]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    metadata: dict[str, dict[str, str]] = defaultdict(dict)
    for event in events:
        if event.get("ph") == "M" and event.get("name") in {
            "process_name",
            "thread_name",
        }:
            metadata[str(event.get("pid"))][str(event.get("tid"))] = str(
                event.get("args", {}).get("name", "")
            )
            continue
        duration = float(event.get("dur", 0.0))
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        if duration <= 0 or category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        values = grouped[(category, name)]
        values[0] += 1
        values[1] += duration

    rows = [
        {
            "category": category,
            "name": name,
            "count": int(values[0]),
            "total_ms": values[1] / 1000,
            "mean_us": values[1] / values[0],
        }
        for (category, name), values in grouped.items()
    ]
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    payload = {
        "trace": str(args.trace),
        "metadata": metadata,
        "top_cpu_events_inclusive": rows[: args.limit],
        "user_annotations": [
            row for row in rows if row["category"] in {"user_annotation", "gpu_user_annotation"}
        ][: args.limit],
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
