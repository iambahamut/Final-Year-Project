"""Offline analysis of metrics CSVs produced by metrics.PerformanceLogger.

Computes accuracy, precision/recall/F1 per gesture, a confusion matrix, FP/FN
rates broken down by gesture and lighting, and timing statistics (processing
time, latency, FPS).

Usage:
    python evaluate_metrics.py logs/metrics_default_20260428_120000.csv
    python evaluate_metrics.py logs/*.csv --markdown report.md

No third-party deps — stdlib only — to keep the harness portable.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
import sys
from collections import defaultdict
from typing import Iterable


GESTURE_LABELS = ("none", "pinch", "thumbs_up", "point", "flat_palm")


def _load_rows(paths: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for pattern in paths:
        matched = glob.glob(pattern) or [pattern]
        for path in matched:
            with open(path, "r", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_confusion_matrix(rows: list[dict]) -> dict:
    """Return matrix[truth][pred] = count."""
    matrix = {gt: {pred: 0 for pred in GESTURE_LABELS} for gt in GESTURE_LABELS}
    for r in rows:
        gt = r.get("ground_truth_label", "none")
        pred = r.get("predicted_label", "none")
        if gt in matrix and pred in matrix[gt]:
            matrix[gt][pred] += 1
    return matrix


def _per_gesture_metrics(matrix: dict) -> dict:
    """Compute precision/recall/F1 per gesture from a confusion matrix."""
    totals = {}
    for label in GESTURE_LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in GESTURE_LABELS if other != label)
        fn = sum(matrix[label][other] for other in GESTURE_LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )
        totals[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return totals


def _accuracy(matrix: dict) -> tuple[float, int, int]:
    correct = sum(matrix[g][g] for g in GESTURE_LABELS)
    total = sum(matrix[g][p] for g in GESTURE_LABELS for p in GESTURE_LABELS)
    return (correct / total if total else 0.0), correct, total


def _fp_fn_overall(rows: list[dict]) -> dict:
    """FP = predicted ≠ none while truth = none. FN = predicted = none while truth ≠ none."""
    totals = {"fp": 0, "fn": 0, "n_truth_none": 0, "n_truth_active": 0}
    for r in rows:
        gt = r.get("ground_truth_label", "none")
        pred = r.get("predicted_label", "none")
        if gt == "none":
            totals["n_truth_none"] += 1
            if pred != "none":
                totals["fp"] += 1
        else:
            totals["n_truth_active"] += 1
            if pred == "none":
                totals["fn"] += 1
    totals["fp_rate"] = (
        totals["fp"] / totals["n_truth_none"] if totals["n_truth_none"] else 0.0
    )
    totals["fn_rate"] = (
        totals["fn"] / totals["n_truth_active"] if totals["n_truth_active"] else 0.0
    )
    return totals


def _fp_fn_by_lighting(rows: list[dict]) -> dict:
    by_light: dict = defaultdict(lambda: {"fp": 0, "fn": 0, "n_none": 0, "n_active": 0})
    for r in rows:
        light = r.get("lighting_condition", "normal")
        gt = r.get("ground_truth_label", "none")
        pred = r.get("predicted_label", "none")
        b = by_light[light]
        if gt == "none":
            b["n_none"] += 1
            if pred != "none":
                b["fp"] += 1
        else:
            b["n_active"] += 1
            if pred == "none":
                b["fn"] += 1
    for b in by_light.values():
        b["fp_rate"] = b["fp"] / b["n_none"] if b["n_none"] else 0.0
        b["fn_rate"] = b["fn"] / b["n_active"] if b["n_active"] else 0.0
    return dict(by_light)


def _timing_stats(rows: list[dict], col: str) -> dict:
    values = [_to_float(r.get(col)) for r in rows if r.get(col) not in (None, "")]
    values = [v for v in values if not math.isnan(v) and v > 0]
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _format_confusion_matrix(matrix: dict) -> str:
    width = max(10, max(len(g) for g in GESTURE_LABELS) + 2)
    header = f"{'truth\\pred':<{width}}" + "".join(
        f"{p:>{width}}" for p in GESTURE_LABELS
    )
    lines = [header]
    for gt in GESTURE_LABELS:
        row = f"{gt:<{width}}" + "".join(
            f"{matrix[gt][p]:>{width}}" for p in GESTURE_LABELS
        )
        lines.append(row)
    return "\n".join(lines)


def _format_report(rows: list[dict]) -> str:
    if not rows:
        return "No rows loaded — nothing to report."

    matrix = _build_confusion_matrix(rows)
    accuracy, correct, total = _accuracy(matrix)
    per_gesture = _per_gesture_metrics(matrix)
    fpfn = _fp_fn_overall(rows)
    fpfn_light = _fp_fn_by_lighting(rows)
    proc = _timing_stats(rows, "processing_time_ms")
    lat = _timing_stats(rows, "gesture_latency_ms")
    fps = _timing_stats(rows, "fps_rolling")

    out = []
    out.append("=" * 72)
    out.append("Performance Report")
    out.append("=" * 72)
    out.append(f"Frames analysed: {total}")
    out.append(f"Overall accuracy: {accuracy:.4f}  ({correct}/{total})")
    out.append("")

    out.append("Confusion matrix")
    out.append("-" * 72)
    out.append(_format_confusion_matrix(matrix))
    out.append("")

    out.append("Per-gesture metrics")
    out.append("-" * 72)
    out.append(f"{'gesture':<14}{'precision':>12}{'recall':>10}{'f1':>10}"
               f"{'tp':>6}{'fp':>6}{'fn':>6}")
    for label in GESTURE_LABELS:
        m = per_gesture[label]
        out.append(
            f"{label:<14}{m['precision']:>12.4f}{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}"
        )
    out.append("")

    out.append("False positives / false negatives (overall)")
    out.append("-" * 72)
    out.append(f"FP={fpfn['fp']} / {fpfn['n_truth_none']} truth=none frames "
               f"({fpfn['fp_rate']*100:.2f}%)")
    out.append(f"FN={fpfn['fn']} / {fpfn['n_truth_active']} truth=active frames "
               f"({fpfn['fn_rate']*100:.2f}%)")
    out.append("")

    if fpfn_light:
        out.append("FP / FN by lighting condition")
        out.append("-" * 72)
        out.append(f"{'lighting':<22}{'FP':>6}{'FP%':>8}{'FN':>6}{'FN%':>8}")
        for light, b in fpfn_light.items():
            out.append(
                f"{light:<22}{b['fp']:>6}{b['fp_rate']*100:>7.2f}%"
                f"{b['fn']:>6}{b['fn_rate']*100:>7.2f}%"
            )
        out.append("")

    out.append("Timing")
    out.append("-" * 72)
    for name, stats in (
        ("processing_time_ms", proc),
        ("gesture_latency_ms", lat),
        ("fps_rolling", fps),
    ):
        out.append(
            f"{name:<22} mean={stats['mean']:.2f}  std={stats['std']:.2f}"
            f"  min={stats['min']:.2f}  max={stats['max']:.2f}  n={stats['count']}"
        )

    return "\n".join(out)


def _format_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_No rows loaded._"
    matrix = _build_confusion_matrix(rows)
    accuracy, correct, total = _accuracy(matrix)
    per_gesture = _per_gesture_metrics(matrix)
    proc = _timing_stats(rows, "processing_time_ms")
    lat = _timing_stats(rows, "gesture_latency_ms")
    fps = _timing_stats(rows, "fps_rolling")

    lines = [
        "# Performance Report",
        "",
        f"- Frames analysed: **{total}**",
        f"- Overall accuracy: **{accuracy:.4f}** ({correct}/{total})",
        "",
        "## Confusion Matrix",
        "",
        "| truth \\\\ pred | " + " | ".join(GESTURE_LABELS) + " |",
        "|" + "---|" * (len(GESTURE_LABELS) + 1),
    ]
    for gt in GESTURE_LABELS:
        lines.append("| " + gt + " | " +
                     " | ".join(str(matrix[gt][p]) for p in GESTURE_LABELS) +
                     " |")

    lines += [
        "",
        "## Per-gesture",
        "",
        "| gesture | precision | recall | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in GESTURE_LABELS:
        m = per_gesture[label]
        lines.append(
            f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['f1']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )

    lines += [
        "",
        "## Timing",
        "",
        "| metric | mean | std | min | max | n |",
        "|---|---|---|---|---|---|",
        f"| processing_time_ms | {proc['mean']:.2f} | {proc['std']:.2f} "
        f"| {proc['min']:.2f} | {proc['max']:.2f} | {proc['count']} |",
        f"| gesture_latency_ms | {lat['mean']:.2f} | {lat['std']:.2f} "
        f"| {lat['min']:.2f} | {lat['max']:.2f} | {lat['count']} |",
        f"| fps_rolling | {fps['mean']:.2f} | {fps['std']:.2f} "
        f"| {fps['min']:.2f} | {fps['max']:.2f} | {fps['count']} |",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+",
                        help="One or more metrics CSV files (globs allowed).")
    parser.add_argument("--markdown", metavar="FILE",
                        help="Also write a Markdown report to FILE.")
    args = parser.parse_args(argv)

    rows = _load_rows(args.paths)
    print(_format_report(rows))

    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(_format_markdown(rows))
        print(f"\nMarkdown report written to {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
