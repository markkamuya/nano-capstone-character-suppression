"""Create presentation-ready summaries and SVG figures from a completed run.

The analysis uses only the Python standard library so the repository remains
easy to reproduce on a clean machine. It does not modify the raw experiment.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


CORE_METRICS = (
    ("vocabulary_match", "Recognizable-word rate", 0.0, 0.75),
    ("jensen_shannon_divergence", "Distance from held-out text", 0.0, 0.9),
    ("type_token_diversity", "Vocabulary diversity", 0.0, 1.0),
)

METRIC_LABELS = {
    "character_entropy": "Character entropy",
    "vocabulary_match": "Recognizable-word rate",
    "type_token_diversity": "Vocabulary diversity",
    "ngram_repetition": "Character-sequence repetition",
    "jensen_shannon_divergence": "Distance from held-out text",
    "rank_frequency_slope": "Rank-frequency slope",
    "rank_frequency_r_squared": "Rank-frequency fit",
}

INTERVENTION_LABELS = {
    "suppress_top": "Remove first choice",
    "random_remove": "Remove random option",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["suppression_rate"] = float(row["suppression_rate"])
        row["seed"] = int(row["seed"])
        for metric in METRIC_LABELS:
            row[metric] = float(row[metric])
    return rows


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    mean = fmean(values)
    if len(values) < 2:
        return mean, 0.0
    # t(19, .975) is 2.093. Every planned condition has 20 repetitions.
    critical = 2.093 if len(values) == 20 else 1.96
    return mean, critical * stdev(values) / math.sqrt(len(values))


def condition_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["intervention"], row["suppression_rate"])
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for (intervention, rate), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "intervention": intervention,
            "suppression_rate": rate,
            "repetitions": len(group),
        }
        for metric in METRIC_LABELS:
            mean, ci = mean_ci95(float(row[metric]) for row in group)
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci95"] = ci
        output.append(record)
    return output


def paired_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["intervention"], row["suppression_rate"], row["seed"]): row
        for row in rows
    }
    rates = sorted({row["suppression_rate"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    output: list[dict[str, Any]] = []
    for rate in rates:
        for metric in METRIC_LABELS:
            differences = [
                indexed[("suppress_top", rate, seed)][metric]
                - indexed[("random_remove", rate, seed)][metric]
                for seed in seeds
            ]
            mean, ci = mean_ci95(differences)
            output.append(
                {
                    "suppression_rate": rate,
                    "metric": metric,
                    "targeted_minus_control_mean": mean,
                    "ci95": ci,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, text: str, **attributes: Any) -> str:
    def attribute_name(key: str) -> str:
        if key.endswith("_"):
            key = key[:-1]
        return key.replace("_", "-")

    attrs = " ".join(
        f'{attribute_name(key)}="{value}"' for key, value in attributes.items()
    )
    return f'<text x="{x}" y="{y}" {attrs}>{html.escape(text)}</text>'


def write_results_figure(path: Path, stats: list[dict[str, Any]]) -> None:
    view_width, view_height = 1500, 600
    display_width, display_height = 750, 300
    margin_x, panel_width, panel_gap = 55, 420, 55
    plot_top, plot_height = 100, 350
    colors = {"suppress_top": "#C84B4B", "random_remove": "#2F6F9F"}
    by_key = {
        (row["intervention"], row["suppression_rate"]): row for row in stats
    }
    rates = [0.0, 0.25, 0.5, 0.75, 1.0]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" viewBox="0 0 {view_width} {view_height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172033}.title{font-size:18px;font-weight:700}.tick{font-size:12px}.legend{font-size:13px}</style>',
    ]
    for panel_index, (metric, title, y_min, y_max) in enumerate(CORE_METRICS):
        left = margin_x + panel_index * (panel_width + panel_gap)
        right = left + panel_width
        bottom = plot_top + plot_height
        parts.append(svg_text(left, 38, title, class_="title"))
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#667085"/>')
        parts.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{bottom}" stroke="#667085"/>')
        for tick_index in range(5):
            value = y_min + (y_max - y_min) * tick_index / 4
            y = bottom - plot_height * tick_index / 4
            parts.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#E4E7EC"/>')
            parts.append(svg_text(left - 8, y + 4, f"{value:.2f}", class_="tick", text_anchor="end"))
        for rate_index, rate in enumerate(rates):
            x = left + panel_width * rate_index / 4
            parts.append(svg_text(x, bottom + 23, f"{int(rate * 100)}%", class_="tick", text_anchor="middle"))
        parts.append(svg_text((left + right) / 2, bottom + 48, "Intervention rate", class_="tick", text_anchor="middle"))

        for intervention in ("random_remove", "suppress_top"):
            points: list[str] = []
            for rate_index, rate in enumerate(rates):
                row = by_key[(intervention, rate)]
                mean = float(row[f"{metric}_mean"])
                ci = float(row[f"{metric}_ci95"])
                x = left + panel_width * rate_index / 4
                y = bottom - plot_height * (mean - y_min) / (y_max - y_min)
                y_upper = bottom - plot_height * (mean + ci - y_min) / (y_max - y_min)
                y_lower = bottom - plot_height * (mean - ci - y_min) / (y_max - y_min)
                points.append(f"{x},{y}")
                parts.append(f'<line x1="{x}" y1="{y_upper}" x2="{x}" y2="{y_lower}" stroke="{colors[intervention]}"/>')
                parts.append(f'<line x1="{x-4}" y1="{y_upper}" x2="{x+4}" y2="{y_upper}" stroke="{colors[intervention]}"/>')
                parts.append(f'<line x1="{x-4}" y1="{y_lower}" x2="{x+4}" y2="{y_lower}" stroke="{colors[intervention]}"/>')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[intervention]}" stroke-width="3"/>')
            for point in points:
                x, y = point.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[intervention]}"/>')

    legend_y = 575
    for index, intervention in enumerate(("suppress_top", "random_remove")):
        x = 420 + index * 290
        parts.append(f'<line x1="{x}" y1="{legend_y-4}" x2="{x+34}" y2="{legend_y-4}" stroke="{colors[intervention]}" stroke-width="4"/>')
        parts.append(svg_text(x + 44, legend_y, INTERVENTION_LABELS[intervention], class_="legend"))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def excerpt(path: Path, length: int = 320) -> str:
    text = path.read_text(encoding="utf-8")[:length].replace("\n", " ")
    return " ".join(text.split())


def write_findings(
    path: Path,
    stats: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    sample_root: Path,
) -> None:
    by_key = {
        (row["intervention"], row["suppression_rate"]): row for row in stats
    }
    effect_map = {
        (row["suppression_rate"], row["metric"]): row for row in effects
    }
    lines = [
        "# Preliminary Results from the Frozen Full Run",
        "",
        "The completed run contains 200 samples: five intervention rates, two",
        "removal rules, and twenty repeated seeds. Each sample contains 2,000",
        "characters. Error bars in the figure are 95% confidence intervals.",
        "",
        "## Main pattern",
        "",
        "Random removal changed the measurements only slightly. In contrast,",
        "removing the model's first choice produced a large, orderly change as",
        "the intervention rate increased.",
        "",
    ]
    baseline = by_key[("suppress_top", 0.0)]
    full = by_key[("suppress_top", 1.0)]
    full_control = by_key[("random_remove", 1.0)]
    key_values = [
        (
            "Recognizable-word rate",
            float(baseline["vocabulary_match_mean"]),
            float(full["vocabulary_match_mean"]),
            float(full_control["vocabulary_match_mean"]),
        ),
        (
            "Distance from held-out text",
            float(baseline["jensen_shannon_divergence_mean"]),
            float(full["jensen_shannon_divergence_mean"]),
            float(full_control["jensen_shannon_divergence_mean"]),
        ),
        (
            "Vocabulary diversity",
            float(baseline["type_token_diversity_mean"]),
            float(full["type_token_diversity_mean"]),
            float(full_control["type_token_diversity_mean"]),
        ),
    ]
    for label, baseline_value, targeted_value, control_value in key_values:
        lines.append(
            f"- **{label}:** {baseline_value:.3f} at baseline, "
            f"{targeted_value:.3f} under full first-choice suppression, and "
            f"{control_value:.3f} under full random removal."
        )
    lines.extend(
        [
            "",
            "At 100% intervention, the paired targeted-minus-control difference",
            f"in statistical distance was {effect_map[(1.0, 'jensen_shannon_divergence')]['targeted_minus_control_mean']:.3f} "
            f"(95% CI ±{effect_map[(1.0, 'jensen_shannon_divergence')]['ci95']:.3f}).",
            "",
            "## Interpretation",
            "",
            "The targeted intervention did not merely add generic disturbance.",
            "It increased character uncertainty and apparent vocabulary variety",
            "while sharply reducing recognizable words and moving character",
            "patterns away from the held-out book. Diversity therefore increased",
            "without a corresponding improvement in language quality.",
            "",
            "The repetition hypothesis was not supported: targeted suppression",
            "reduced repeated four-character sequences instead of increasing them.",
            "The distributional-divergence and targeted-effect hypotheses were",
            "supported descriptively across all nonzero rates.",
            "",
            "## Important limitation",
            "",
            "Even the baseline model produces noisy, only partly recognizable",
            "English. The results therefore establish how the intervention changes",
            "this particular character model; they do not yet show how a strong",
            "modern language model would respond. A smoothing sensitivity analysis",
            "or a stronger baseline model would be needed before broader claims.",
            "",
            "## Representative excerpts (seed 0)",
            "",
        ]
    )
    examples = [
        ("Baseline (0%)", sample_root / "suppress_top/rate_0/seed_0.txt"),
        ("First-choice suppression (25%)", sample_root / "suppress_top/rate_0p25/seed_0.txt"),
        ("First-choice suppression (100%)", sample_root / "suppress_top/rate_1/seed_0.txt"),
        ("Random removal (100%)", sample_root / "random_remove/rate_1/seed_0.txt"),
    ]
    for label, sample_path in examples:
        lines.extend([f"**{label}:**", "", f"> {excerpt(sample_path)}", ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    experiment = args.run_directory / "experiment"
    rows = read_rows(experiment / "results.csv")
    stats = condition_statistics(rows)
    effects = paired_effects(rows)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_directory / "condition_statistics.csv", stats)
    write_csv(args.output_directory / "paired_effects.csv", effects)
    write_results_figure(args.output_directory / "core_results.svg", stats)
    write_findings(
        args.output_directory / "FINDINGS.md",
        stats,
        effects,
        experiment / "samples",
    )
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    (args.output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote presentation artifacts to {args.output_directory}")


if __name__ == "__main__":
    main()
