"""Compare predicted growth analytics with ground truth.

Predicted tracks are matched to ground-truth leaves with the IDF1 trajectory
mapping (:func:`leaftrack.track_metrics.trajectory_mapping`), then compared
at three levels:

- **plant, per frame**: leaf count, total projected leaf area and the 24 h
  relative growth rate (RGR);
- **leaf, per frame**: projected area of each matched leaf;
- **leaf, per life**: early (first 48 h) RGR and maximum area.

Errors are pooled over all sequences of a split. Percentage errors use the
ground truth as reference; RGR errors are absolute (per day), since a
relative error is unstable when the true rate is near zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from leaftrack.analytics import leaf_summary, plant_timeseries

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class SequenceGrowth:
    """Ground-truth and predicted analytics of one sequence."""

    key: str
    gt_records: pd.DataFrame
    pred_records: pd.DataFrame
    mapping: Mapping[int, int]  # gt leaf ID -> predicted track ID
    frame_hours: Sequence[float]
    gt_plant: pd.DataFrame = field(init=False)
    pred_plant: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.gt_plant = plant_timeseries(self.gt_records, self.frame_hours)
        self.pred_plant = plant_timeseries(self.pred_records, self.frame_hours)


def plant_pairs(seqs: Sequence[SequenceGrowth]) -> pd.DataFrame:
    """Per-frame plant traits, ground truth and prediction side by side."""
    parts = []
    for s in seqs:
        cols = ["frame", "hours", "leaf_count", "total_area_px", "rgr_per_day"]
        m = s.gt_plant[cols].merge(s.pred_plant[cols], on=["frame", "hours"], suffixes=("_gt", ""))
        parts.append(m.assign(sequence=s.key))
    return pd.concat(parts, ignore_index=True)


def leaf_frame_pairs(seqs: Sequence[SequenceGrowth]) -> pd.DataFrame:
    """Area of every ground-truth leaf per frame next to its mapped track's area.

    ``area_px`` is NaN where the mapped track is absent (missed or switched).
    """
    parts = []
    for s in seqs:
        gt = s.gt_records[["frame", "track_id", "area_px"]].rename(
            columns={"track_id": "gt_id", "area_px": "area_px_gt"}
        )
        gt["track_id"] = gt["gt_id"].map(s.mapping)
        pred = s.pred_records[["frame", "track_id", "area_px"]]
        parts.append(gt.merge(pred, on=["frame", "track_id"], how="left").assign(sequence=s.key))
    return pd.concat(parts, ignore_index=True)


def leaf_life_pairs(seqs: Sequence[SequenceGrowth]) -> pd.DataFrame:
    """Per-leaf summaries of each ground-truth leaf and its mapped track."""
    parts = []
    for s in seqs:
        gt = leaf_summary(s.gt_records).rename(columns={"track_id": "gt_id"})
        gt["track_id"] = gt["gt_id"].map(s.mapping)
        pred = leaf_summary(s.pred_records)
        m = gt.merge(pred, on="track_id", how="left", suffixes=("_gt", ""))
        parts.append(m.assign(sequence=s.key))
    return pd.concat(parts, ignore_index=True)


def _ape(pred: pd.Series, gt: pd.Series) -> pd.Series:
    return (pred - gt).abs() / gt.where(gt > 0)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def growth_metrics(seqs: Sequence[SequenceGrowth]) -> dict[str, Any]:
    """Pooled agreement between predicted and ground-truth growth analytics."""
    plant = plant_pairs(seqs)
    leaf = leaf_frame_pairs(seqs)
    life = leaf_life_pairs(seqs)
    covered = leaf["area_px"].notna()
    rgr = plant[["rgr_per_day_gt", "rgr_per_day"]].dropna()
    early = life[["early_rgr_per_day_gt", "early_rgr_per_day"]].dropna()
    r4 = lambda x: round(float(x), 4)  # noqa: E731
    return {
        "frames": len(plant),
        "leaves": len(life),
        "plant": {
            "leaf_count_mae": r4((plant["leaf_count"] - plant["leaf_count_gt"]).abs().mean()),
            "leaf_count_exact": r4((plant["leaf_count"] == plant["leaf_count_gt"]).mean()),
            "total_area_median_ape": r4(
                _ape(plant["total_area_px"], plant["total_area_px_gt"]).median()
            ),
            "total_area_r": r4(_pearson(plant["total_area_px"], plant["total_area_px_gt"])),
            "rgr_24h_mae_per_day": r4((rgr["rgr_per_day"] - rgr["rgr_per_day_gt"]).abs().mean()),
            "rgr_24h_r": r4(_pearson(rgr["rgr_per_day"], rgr["rgr_per_day_gt"])),
        },
        "leaf": {
            "area_coverage": r4(covered.mean()),
            "area_median_ape": r4(_ape(leaf["area_px"], leaf["area_px_gt"])[covered].median()),
            "max_area_median_ape": r4(_ape(life["max_area_px"], life["max_area_px_gt"]).median()),
            "early_rgr_leaves": len(early),
            "early_rgr_mae_per_day": r4(
                (early["early_rgr_per_day"] - early["early_rgr_per_day_gt"]).abs().mean()
            ),
            "early_rgr_r": r4(_pearson(early["early_rgr_per_day"], early["early_rgr_per_day_gt"])),
        },
    }


def growth_markdown(report: dict[str, Any]) -> str:
    """Render the growth-analytics report as Markdown."""
    p, lf = report["metrics"]["plant"], report["metrics"]["leaf"]
    return "\n".join(
        [
            f"## Growth analytics, {report['split']} split",
            "",
            f"{report['sequences']} sequences, {report['metrics']['frames']} frames, "
            f"{report['metrics']['leaves']} ground-truth leaves; tracker: {report['tracker']}.",
            "",
            "| Trait | Agreement with ground truth |",
            "|---|---|",
            f"| Leaf count per frame | exact in {p['leaf_count_exact']:.0%} of frames, "
            f"MAE {p['leaf_count_mae']:.2f} leaves |",
            f"| Total leaf area per frame | median error {p['total_area_median_ape']:.1%}, "
            f"r = {p['total_area_r']:.3f} |",
            f"| Plant RGR (24 h window) | MAE {p['rgr_24h_mae_per_day']:.3f} /day, "
            f"r = {p['rgr_24h_r']:.2f} |",
            f"| Area per leaf per frame | median error {lf['area_median_ape']:.1%} "
            f"({lf['area_coverage']:.0%} of leaf-frames followed by the leaf's track) |",
            f"| Maximum area per leaf | median error {lf['max_area_median_ape']:.1%} |",
            f"| Early RGR per leaf (first 48 h) | MAE {lf['early_rgr_mae_per_day']:.3f} /day, "
            f"r = {lf['early_rgr_r']:.2f} ({lf['early_rgr_leaves']} leaves) |",
            "",
        ]
    )


def comparison_figure(seq: SequenceGrowth, title: str | None = None) -> Figure:
    """Predicted (solid) vs ground-truth (dashed) leaf count, plant area and per-leaf area.

    Each ground-truth leaf and its mapped track share a color; unmatched
    predicted tracks are drawn thin and grey.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from leaftrack.viz import id_color

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    gd, pd_ = seq.gt_plant["hours"] / 24, seq.pred_plant["hours"] / 24
    dark, light = "#2f6b3a", "#8fbf8f"

    axes[0].step(gd, seq.gt_plant["leaf_count"], where="post", color=light, lw=2.5, ls="--",
                 label="ground truth")
    axes[0].step(pd_, seq.pred_plant["leaf_count"], where="post", color=dark, lw=1.6,
                 label="LeafTrack")
    axes[0].set(title="Leaf count", xlabel="day", ylabel="leaves")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(gd, seq.gt_plant["total_area_px"] / 1000, color=light, lw=2.5, ls="--",
                 label="ground truth")
    axes[1].plot(pd_, seq.pred_plant["total_area_px"] / 1000, color=dark, lw=1.6,
                 label="LeafTrack")
    axes[1].set(title="Plant leaf area", xlabel="day", ylabel="projected area (k px)")
    axes[1].legend(frameon=False, fontsize=8)

    ax = axes[2]
    mapped = set(seq.mapping.values())
    for gid, g in seq.gt_records.groupby("track_id"):
        color = np.array(id_color(int(gid))) / 255
        ax.plot(g["hours"] / 24, g["area_px"] / 1000, color=color, lw=2.2, ls="--", alpha=0.6)
        tid = seq.mapping.get(int(gid))
        if tid is not None:
            p = seq.pred_records[seq.pred_records["track_id"] == tid]
            ax.plot(p["hours"] / 24, p["area_px"] / 1000, color=color, lw=1.4)
    for tid, p in seq.pred_records.groupby("track_id"):
        if tid not in mapped:
            ax.plot(p["hours"] / 24, p["area_px"] / 1000, color="0.6", lw=0.8)
    ax.set(title="Area per leaf (dashed: ground truth)", xlabel="day",
           ylabel="projected area (k px)")

    for a in axes:
        a.grid(alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title)
    return fig
