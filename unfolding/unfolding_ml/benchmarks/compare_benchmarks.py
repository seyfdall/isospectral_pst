from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def run(args: argparse.Namespace) -> None:
    cpu = load_json(Path(args.cpu_summary))
    gpu = load_json(Path(args.gpu_summary))

    cpu_s = cpu.get("summary", {})
    gpu_s = gpu.get("summary", {})

    rows = [
        (
            "n_instances",
            metric(cpu_s, "n_instances"),
            metric(gpu_s, "n_instances"),
        ),
        (
            "n_success",
            metric(cpu_s, "n_success"),
            metric(gpu_s, "n_success"),
        ),
        (
            "mean_elapsed_sec",
            metric(cpu_s, "mean_elapsed_sec"),
            metric(gpu_s, "mean_elapsed_sec"),
        ),
        (
            "total_elapsed_sec",
            metric(cpu_s, "total_elapsed_sec"),
            metric(gpu_s, "total_elapsed_sec"),
        ),
        (
            "recovery_rate",
            metric(cpu_s, "recovery_rate"),
            metric(gpu_s, "recovery_rate"),
        ),
    ]

    print("\n=== Benchmark Comparison ===")
    print(f"CPU summary: {args.cpu_summary}")
    print(f"GPU summary: {args.gpu_summary}")
    print()
    print(f"{'metric':<20} {'cpu_symbolic':>16} {'gpu_numeric':>16}")
    print("-" * 54)
    for key, c, g in rows:
        print(f"{key:<20} {c:>16.6f} {g:>16.6f}")

    output = {
        "cpu_summary_path": args.cpu_summary,
        "gpu_summary_path": args.gpu_summary,
        "cpu": cpu,
        "gpu": gpu,
        "comparison_rows": [
            {"metric": key, "cpu_symbolic": c, "gpu_numeric": g}
            for key, c, g in rows
        ],
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote comparison json to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CPU symbolic and GPU numeric summaries")
    parser.add_argument("--cpu-summary", required=True, type=str)
    parser.add_argument("--gpu-summary", required=True, type=str)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
