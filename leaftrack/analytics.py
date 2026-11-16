"""Growth analytics from tracked leaves: tidy tables, growth rates and plots.

Inputs are the per-frame outputs of a tracker (or ground-truth masks) plus
the capture time of each frame. Outputs are pandas DataFrames:

- :func:`leaf_records`: one row per leaf per frame (long format).
- :func:`plant_timeseries`: leaf count, total leaf area and plant growth
  rates per frame.
- :func:`leaf_summary`: one row per leaf: lifespan, size and early growth.

Areas are *projected* leaf areas (what the camera sees), in pixels, or in
mm² when a scale is given. Two KOMATSUNA observations shape the growth-rate
definitions:

- Projected area changes by 15-30% between 4-hourly captures because
  leaves move over the day. Plant growth rates are therefore fitted over a
  trailing 24 h window, which spans one full day-night cycle.
- A leaf grows exponentially at first and then levels off (old seed leaves
  even shrink as new leaves cover them), so a single rate over its whole
  life mixes phases. The per-leaf rate is fitted over its first 48 h.

Relative growth rate (RGR) is the slope of ``ln(area)`` over time, reported
per day (0.69/day means the area doubles daily). Absolute growth rate (AGR)
is the slope of area over time, per hour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from leaftrack.track import TrackedLeaf

if TYPE_CHECKING:
    from matplotlib.figure import Figure

LEAF_COLUMNS = ["frame", "hours", "track_id", "area_px", "cx", "cy", "confidence"]


def records_from_masks(
    frames: Sequence[Mapping[int, np.ndarray]],
    hours: Sequence[float],
    confidences: Sequence[Mapping[int, float]] | None = None,
    mm2_per_px: float | None = None,
) -> pd.DataFrame:
    """Long table from ``{track_id: mask}`` per frame (works for ground truth too).

    Args:
        frames: one mapping per frame.
        hours: capture time of each frame, in hours from any origin.
        confidences: optional ``{track_id: score}`` per frame.
        mm2_per_px: area of one pixel in mm²; adds an ``area_mm2`` column.
    """
    if len(frames) != len(hours):
        raise ValueError("frames and hours must have the same length")
    rows = []
    for t, (frame, h) in enumerate(zip(frames, hours, strict=True)):
        for tid, mask in frame.items():
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            conf = confidences[t].get(tid, np.nan) if confidences else np.nan
            rows.append((t, float(h), int(tid), len(xs), xs.mean(), ys.mean(), conf))
    df = pd.DataFrame(rows, columns=LEAF_COLUMNS)
    if mm2_per_px is not None:
        df["area_mm2"] = df["area_px"] * mm2_per_px
    return df.sort_values(["frame", "track_id"], ignore_index=True)


def leaf_records(
    tracked: Sequence[Sequence[TrackedLeaf]],
    hours: Sequence[float],
    mm2_per_px: float | None = None,
) -> pd.DataFrame:
    """Long table (one row per leaf per frame) from tracker output."""
    frames = [{t.track_id: t.instance.mask for t in f} for f in tracked]
    confs = [{t.track_id: t.instance.confidence for t in f} for f in tracked]
    return records_from_masks(frames, hours, confs, mm2_per_px)


def _log_slope_per_day(hours: np.ndarray, area: np.ndarray) -> float:
    """Least-squares slope of ln(area) against time, per day (NaN if < 3 points)."""
    keep = area > 0
    if keep.sum() < 3 or np.ptp(hours[keep]) == 0:
        return float("nan")
    return float(np.polyfit(hours[keep] / 24.0, np.log(area[keep]), 1)[0])


def _slope_per_hour(hours: np.ndarray, area: np.ndarray) -> float:
    if len(area) < 3 or np.ptp(hours) == 0:
        return float("nan")
    return float(np.polyfit(hours, area, 1)[0])


def plant_timeseries(
    records: pd.DataFrame, frame_hours: Sequence[float] | None = None, window_h: float = 24.0
) -> pd.DataFrame:
    """Per-frame plant traits.

    Columns: ``frame, hours, leaf_count, total_area_px, rgr_per_day,
    agr_px_per_h``. Growth rates are fitted over the trailing ``window_h``
    hours (inclusive) and are NaN until the window holds 3 frames.

    Args:
        records: output of :func:`leaf_records` / :func:`records_from_masks`.
        frame_hours: capture time of *every* frame, so frames with no leaves
            appear with a count of 0. Defaults to the frames in ``records``.
        window_h: growth-rate window in hours.
    """
    per_frame = records.groupby("frame").agg(
        hours=("hours", "first"), leaf_count=("track_id", "size"), total_area_px=("area_px", "sum")
    )
    if frame_hours is not None:
        index = pd.RangeIndex(len(frame_hours), name="frame")
        per_frame = per_frame.reindex(index)
        per_frame["hours"] = np.asarray(frame_hours, float)
        per_frame[["leaf_count", "total_area_px"]] = (
            per_frame[["leaf_count", "total_area_px"]].fillna(0).astype(int)
        )
    per_frame = per_frame.reset_index()
    h = per_frame["hours"].to_numpy(float)
    a = per_frame["total_area_px"].to_numpy(float)
    rgr, agr = [], []
    for t in range(len(per_frame)):
        win = (h > h[t] - window_h - 1e-9) & (h <= h[t])
        rgr.append(_log_slope_per_day(h[win], a[win]))
        agr.append(_slope_per_hour(h[win], a[win]))
    per_frame["rgr_per_day"] = rgr
    per_frame["agr_px_per_h"] = agr
    return per_frame


def leaf_summary(records: pd.DataFrame, early_window_h: float = 48.0) -> pd.DataFrame:
    """One row per leaf.

    Columns: ``track_id, first_seen_h, last_seen_h, frames, max_area_px,
    final_area_px, early_rgr_per_day, early_agr_px_per_h``. Early rates are
    fitted over the leaf's first ``early_window_h`` hours (NaN with fewer
    than 3 frames in that window).
    """
    out = []
    for tid, g in records.groupby("track_id"):
        g = g.sort_values("hours")
        h = g["hours"].to_numpy(float)
        a = g["area_px"].to_numpy(float)
        early = h <= h[0] + early_window_h + 1e-9
        out.append(
            {
                "track_id": int(tid),
                "first_seen_h": h[0],
                "last_seen_h": h[-1],
                "frames": len(g),
                "max_area_px": int(a.max()),
                "final_area_px": int(a[-1]),
                "early_rgr_per_day": _log_slope_per_day(h[early], a[early]),
                "early_agr_px_per_h": _slope_per_hour(h[early], a[early]),
            }
        )
    columns = [
        "track_id", "first_seen_h", "last_seen_h", "frames", "max_area_px", "final_area_px",
        "early_rgr_per_day", "early_agr_px_per_h",
    ]  # fmt: skip
    return pd.DataFrame(out, columns=columns)


def growth_dashboard(
    records: pd.DataFrame, plant: pd.DataFrame, title: str | None = None
) -> Figure:
    """Three-panel figure: leaf count, plant area with RGR, and per-leaf area curves.

    Leaf curves use the same colors as the tracking overlays
    (:func:`leaftrack.viz.id_color`).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: scripts, tests, Spaces
    import matplotlib.pyplot as plt

    from leaftrack.viz import id_color

    days = plant["hours"] / 24.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.step(days, plant["leaf_count"], where="post", color="#2f6b3a", lw=2)
    ax.set(title="Leaf count", xlabel="day", ylabel="leaves")
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax = axes[1]
    ax.plot(days, plant["total_area_px"] / 1000, color="#2f6b3a", lw=2, label="total leaf area")
    ax.set(title="Plant leaf area", xlabel="day", ylabel="projected area (k px)")
    twin = ax.twinx()
    twin.plot(days, plant["rgr_per_day"], color="#c77d1a", lw=1.5, ls="--", label="RGR (24 h)")
    twin.set_ylabel("RGR per day", color="#c77d1a")
    twin.tick_params(axis="y", colors="#c77d1a")

    ax = axes[2]
    for tid, g in records.groupby("track_id"):
        color = np.array(id_color(int(tid))) / 255.0
        ax.plot(g["hours"] / 24.0, g["area_px"] / 1000, color=color, lw=1.6, label=str(tid))
    ax.set(title="Area per leaf (track ID)", xlabel="day", ylabel="projected area (k px)")
    if records["track_id"].nunique() <= 12:
        ax.legend(fontsize=7, ncol=2, frameon=False, title="leaf", title_fontsize=7)

    for a in axes:
        a.grid(alpha=0.3)
        a.spines[["top"]].set_visible(False)
    if title:
        fig.suptitle(title)
    return fig
