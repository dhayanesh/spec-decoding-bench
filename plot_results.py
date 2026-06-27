import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULT_FILES = [
    RESULTS_DIR / "eagle3.json",
    RESULTS_DIR / "suffix_decoding.json",
    RESULTS_DIR / "dflash.json",
]

METHOD_LABELS = {
    "eagle3": "EAGLE3",
    "suffix_decoding": "Suffix",
    "dflash": "DFlash",
}
PHASE_LABELS = {
    "sequential": "Sequential",
    "parallel": "Parallel",
}
COLORS = {
    "sequential": "#4C78A8",
    "parallel": "#F58518",
}


def load_results() -> list[dict]:
    rows = []
    for path in RESULT_FILES:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def values_by_method(rows: list[dict], metric: str) -> dict[str, list[float]]:
    methods = list(METHOD_LABELS)
    phases = list(PHASE_LABELS)
    values = {phase: [] for phase in phases}

    for phase in phases:
        for method in methods:
            row = next(
                item for item in rows
                if item["method"] == method and item["phase"] == phase
            )
            values[phase].append(row[metric])
    return values


def latency_by_method(rows: list[dict], metric: str) -> dict[str, list[float]]:
    methods = list(METHOD_LABELS)
    phases = list(PHASE_LABELS)
    values = {phase: [] for phase in phases}

    for phase in phases:
        for method in methods:
            row = next(
                item for item in rows
                if item["method"] == method and item["phase"] == phase
            )
            values[phase].append(row["latency_seconds"][metric])
    return values


def draw_grouped_bars(ax, title: str, ylabel: str, values: dict[str, list[float]]) -> None:
    methods = list(METHOD_LABELS)
    x = np.arange(len(methods))
    width = 0.36

    for offset, phase in [(-width / 2, "sequential"), (width / 2, "parallel")]:
        bars = ax.bar(
            x + offset,
            values[phase],
            width,
            label=PHASE_LABELS[phase],
            color=COLORS[phase],
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in methods])
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    rows = load_results()
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("GPT-OSS-20B Speculative Decoding Benchmark", fontsize=16)

    draw_grouped_bars(
        axes[0, 0],
        "Request Throughput",
        "requests/sec",
        values_by_method(rows, "requests_per_second"),
    )
    draw_grouped_bars(
        axes[0, 1],
        "Completion Token Throughput",
        "tokens/sec",
        values_by_method(rows, "completion_tokens_per_second"),
    )
    draw_grouped_bars(
        axes[1, 0],
        "Total Time",
        "seconds",
        values_by_method(rows, "total_seconds"),
    )
    draw_grouped_bars(
        axes[1, 1],
        "P95 Latency",
        "seconds",
        latency_by_method(rows, "p95"),
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_path = RESULTS_DIR / "benchmark_summary.png"
    fig.savefig(out_path, dpi=180)
    print(f"chart_file={out_path}")


if __name__ == "__main__":
    main()
