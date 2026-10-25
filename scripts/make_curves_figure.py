#!/usr/bin/env python3
"""Render train_error_rate / test_error_rate curves for all 10 conditions
into a single 2-panel matplotlib PDF, ready to drop into the LaTeX report.
Optionally also writes a CSV with the full curves and a final-value summary.

Usage (local Mac, .venv activated):
    python scripts/make_curves_figure.py \
        --logdir ~/sam_tb_logs \
        --out    report/curves.pdf

Or on CARC (event files live under /scratch1/$USER/sam_experiments/cN_*/...):
    python scripts/make_curves_figure.py \
        --logdir /scratch1/$USER/sam_experiments \
        --out    ~/curves.pdf \
        --csv    ~/curves_data.csv
then rsync both files back locally.

Notes:
    * Auto-discovers every events.out.tfevents.* file under the logdir.
    * Reads scalars from the 'tensors' tag category via tensorflow's
      EventAccumulator (TF2 actually stores scalars under 'tensors',
      not 'scalars').
    * Uses a consistent color palette across conditions (matches the
      slide deck and the TensorBoard default coloring).
    * --csv writes two files:
        <stem>_curves.csv  — full step-by-step data for all conditions
        <stem>_final.csv   — one row per condition: final train/test error
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from tensorflow.python.framework import tensor_util


# Fixed order so colors stay stable across regenerations of the figure.
COND_ORDER = [
    "c1_sgd_baseline",
    "c2_sam_only",
    "c3_mixup_only",
    "c4_cutmix_only",
    "c5_sam_mixup",
    "c6_sam_cutmix",
    "c7_sgd_cutout",
    "c8_sam_cutout",
    "c9_sgd_mixcut",
    "c10_sam_mixcut",
]
# tab10-ish, manually picked for readability
COLORS = {
    "c1_sgd_baseline": "#1f77b4",
    "c2_sam_only":     "#2ca02c",
    "c3_mixup_only":   "#1a1a1a",
    "c4_cutmix_only":  "#ff7f0e",
    "c5_sam_mixup":    "#e377c2",
    "c6_sam_cutmix":   "#ffbb33",
    "c7_sgd_cutout":   "#bcbd22",
    "c8_sam_cutout":   "#17becf",
    "c9_sgd_mixcut":   "#9467bd",
    "c10_sam_mixcut":  "#2ca02c",
}
LABELS = {
    "c1_sgd_baseline": r"C1 SGD",
    "c2_sam_only":     r"C2 SAM",
    "c3_mixup_only":   r"C3 SGD+Mixup",
    "c4_cutmix_only":  r"C4 SGD+CutMix",
    "c5_sam_mixup":    r"C5 SAM+Mixup",
    "c6_sam_cutmix":   r"C6 SAM+CutMix",
    "c7_sgd_cutout":   r"C7 SGD+Cutout",
    "c8_sam_cutout":   r"C8 SAM+Cutout",
    "c9_sgd_mixcut":   r"C9 SGD+mixcut",
    "c10_sam_mixcut":  r"C10 SAM+mixcut",
}


def find_event_dirs(root: Path) -> dict[str, Path]:
    """Find one event-file directory per condition.
    Returns {cond_name: path/to/dir/with/events.out.tfevents.*}
    """
    found = {}
    for ev in root.rglob("events.out.tfevents.*"):
        # walk up to first ancestor matching cN_*
        for parent in ev.parents:
            m = re.match(r"^(c\d+_[a-z_]+)", parent.name)
            if m:
                cond = m.group(1)
                # Pick the most recent event file (largest mtime) so multiple
                # re-runs of the same condition don't confuse the parser.
                cur = found.get(cond)
                if cur is None or ev.stat().st_mtime > cur[1]:
                    found[cond] = (ev.parent, ev.stat().st_mtime)
                break
    return {k: v[0] for k, v in found.items()}


def read_scalar(eventdir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for a given scalar tag.
    TF2 stores scalars under 'tensors' tag, not 'scalars'.
    """
    ea = EventAccumulator(
        str(eventdir),
        size_guidance={"tensors": 0, "scalars": 0},
    )
    ea.Reload()

    # try tensors first (TF2 default), then scalars (TF1 fallback)
    if tag in ea.Tags().get("tensors", []):
        events = ea.Tensors(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array(
            [tensor_util.MakeNdarray(e.tensor_proto).item() for e in events]
        )
        return steps, vals
    if tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events], dtype=np.int64)
        vals = np.array([e.value for e in events])
        return steps, vals
    return np.array([]), np.array([])


def smooth(y: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """TB-style EMA smoothing."""
    if len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * y[i]
    return out


def plot(panel_ax, eventdirs: dict[str, Path], tag: str, ylabel: str):
    panel_ax.set_xlabel("training step")
    panel_ax.set_ylabel(ylabel)
    panel_ax.set_ylim(0, 0.40)
    panel_ax.grid(True, ls=":", lw=0.5, alpha=0.6)

    plotted = 0
    for cond in COND_ORDER:
        if cond not in eventdirs:
            print(f"  [skip] {cond}: no event dir", file=sys.stderr)
            continue
        steps, vals = read_scalar(eventdirs[cond], tag)
        if len(steps) == 0:
            print(f"  [skip] {cond}: no '{tag}' tag", file=sys.stderr)
            continue
        # raw very-light, smoothed solid
        panel_ax.plot(steps, vals,
                      color=COLORS[cond], alpha=0.18, lw=0.6)
        panel_ax.plot(steps, smooth(vals),
                      color=COLORS[cond], lw=1.3, label=LABELS[cond])
        plotted += 1
    if plotted == 0:
        panel_ax.text(0.5, 0.5, f"no '{tag}' data found",
                      transform=panel_ax.transAxes,
                      ha="center", va="center", color="red")


def write_csv(eventdirs: dict[str, Path], csv_path: Path):
    """Write two CSV files:
      <stem>_curves.csv  — full step-by-step curves (condition, step, train_error, test_error)
      <stem>_final.csv   — one row per condition with final values
    """
    stem = csv_path.with_suffix("")
    curves_path = Path(str(stem) + "_curves.csv")
    final_path  = Path(str(stem) + "_final.csv")

    curves_path.parent.mkdir(parents=True, exist_ok=True)

    # collect all data
    all_data = {}  # cond -> {step -> {train_error, test_error}}
    for cond in COND_ORDER:
        if cond not in eventdirs:
            continue
        train_steps, train_vals = read_scalar(eventdirs[cond], "train_error_rate")
        test_steps,  test_vals  = read_scalar(eventdirs[cond], "test_error_rate")

        # merge by step
        rows = {}
        for s, v in zip(train_steps, train_vals):
            rows.setdefault(int(s), {})["train_error_rate"] = float(v)
        for s, v in zip(test_steps, test_vals):
            rows.setdefault(int(s), {})["test_error_rate"] = float(v)
        all_data[cond] = rows

    # write curves CSV
    with open(curves_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "step", "train_error_rate", "test_error_rate"])
        for cond in COND_ORDER:
            if cond not in all_data:
                continue
            for step in sorted(all_data[cond]):
                row = all_data[cond][step]
                writer.writerow([
                    cond,
                    step,
                    row.get("train_error_rate", ""),
                    row.get("test_error_rate", ""),
                ])
    print(f"Wrote {curves_path}")

    # write final CSV (last recorded value for each condition)
    with open(final_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "label",
                          "final_train_error_rate", "final_test_error_rate",
                          "final_train_accuracy",   "final_test_accuracy"])
        for cond in COND_ORDER:
            if cond not in all_data:
                continue
            rows = all_data[cond]
            steps_sorted = sorted(rows)
            last = rows[steps_sorted[-1]]
            train_err = last.get("train_error_rate", "")
            test_err  = last.get("test_error_rate",  "")
            train_acc = round(1.0 - train_err, 6) if train_err != "" else ""
            test_acc  = round(1.0 - test_err,  6) if test_err  != "" else ""
            writer.writerow([cond, LABELS[cond],
                              train_err, test_err,
                              train_acc, test_acc])
    print(f"Wrote {final_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", required=True,
                   help="root dir containing cN_* subdirs with TB events")
    p.add_argument("--out", required=True,
                   help="output PDF path (e.g. report/curves.pdf)")
    p.add_argument("--csv", default=None,
                   help="optional CSV output stem (e.g. ~/curves_data.csv); "
                        "produces <stem>_curves.csv and <stem>_final.csv")
    args = p.parse_args()

    root = Path(args.logdir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: logdir not found: {root}", file=sys.stderr)
        sys.exit(1)

    eventdirs = find_event_dirs(root)
    print(f"Found {len(eventdirs)} condition dirs under {root}:")
    for k in sorted(eventdirs):
        print(f"  {k:25s} -> {eventdirs[k]}")

    if not eventdirs:
        print("No event files found. Check --logdir.", file=sys.stderr)
        sys.exit(1)

    fig, (ax_test, ax_train) = plt.subplots(
        1, 2, figsize=(11.0, 3.6), sharey=True)
    plot(ax_test,  eventdirs, "test_error_rate",  "test error rate")
    plot(ax_train, eventdirs, "train_error_rate", "train error rate")

    # single legend for both panels, on the right
    handles, labels = ax_test.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center right",
               bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.78, bottom=0.14, top=0.93,
                        wspace=0.05)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"\nWrote {out}")

    if args.csv:
        write_csv(eventdirs, Path(args.csv).expanduser().resolve())


if __name__ == "__main__":
    main()