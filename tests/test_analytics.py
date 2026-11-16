"""Tests for growth analytics on synthetic leaves with known growth."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from leaftrack.analytics import (
    growth_dashboard,
    leaf_records,
    leaf_summary,
    plant_timeseries,
    records_from_masks,
)
from leaftrack.track import track_sequence
from tests.test_track import disk, leaf

HOURS = [4.0 * k for k in range(13)]  # two days, every 4 h


def square(area: int, x0: int = 0) -> np.ndarray:
    side = int(round(math.sqrt(area)))
    m = np.zeros((200, 200), bool)
    m[:side, x0 : x0 + side] = True
    return m


def exponential_frames(rgr_per_day: float, a0: float = 100.0):
    """One leaf whose area grows exactly exponentially (as far as pixels allow)."""
    sides = [math.sqrt(a0 * math.exp(rgr_per_day * h / 24)) for h in HOURS]
    return [{1: square(round(s) ** 2)} for s in sides]


def test_records_from_masks_columns_and_scale() -> None:
    frames = [{1: disk(30, 30, 5), 2: disk(70, 70, 5)}, {1: disk(30, 30, 6)}]
    df = records_from_masks(frames, [0, 4], mm2_per_px=0.25)
    assert list(df["track_id"]) == [1, 2, 1]
    assert df.loc[0, "area_mm2"] == pytest.approx(df.loc[0, "area_px"] * 0.25)
    assert df.loc[0, "cx"] == pytest.approx(30, abs=0.1)
    with pytest.raises(ValueError):
        records_from_masks(frames, [0])


def test_leaf_records_from_tracker_output() -> None:
    tracked = track_sequence([[leaf(30, 50, 8 + k), leaf(70, 50, 6)] for k in range(3)])
    df = leaf_records(tracked, [0, 4, 8])
    assert len(df) == 6 and set(df["track_id"]) == {1, 2}
    assert df["confidence"].eq(0.9).all()
    assert df[df.track_id == 1]["area_px"].is_monotonic_increasing


def test_plant_rgr_recovers_known_growth_rate() -> None:
    plant = plant_timeseries(records_from_masks(exponential_frames(0.5), HOURS))
    assert plant["rgr_per_day"].iloc[:2].isna().all()  # fewer than 3 frames in window
    assert plant["rgr_per_day"].iloc[-1] == pytest.approx(0.5, abs=0.05)
    assert (plant["agr_px_per_h"].dropna() > 0).all()


def test_plant_timeseries_fills_empty_frames() -> None:
    frames = [{1: disk(30, 30, 5)}, {}, {1: disk(30, 30, 6)}]
    plant = plant_timeseries(records_from_masks(frames, [0, 4, 8]), frame_hours=[0, 4, 8])
    assert list(plant["leaf_count"]) == [1, 0, 1]
    assert list(plant["hours"]) == [0, 4, 8]


def test_leaf_summary_early_window_and_lifespan() -> None:
    frames = exponential_frames(0.7)
    for k in range(6, 13):
        frames[k][2] = square(400, x0=100)  # a second leaf appears at 24 h, constant size
    summary = leaf_summary(records_from_masks(frames, HOURS)).set_index("track_id")
    assert summary.loc[1, "early_rgr_per_day"] == pytest.approx(0.7, abs=0.08)
    assert summary.loc[2, "first_seen_h"] == 24 and summary.loc[2, "frames"] == 7
    assert summary.loc[2, "early_rgr_per_day"] == pytest.approx(0.0, abs=1e-9)
    assert summary.loc[1, "max_area_px"] == summary.loc[1, "final_area_px"]


def test_short_leaf_has_nan_rates() -> None:
    summary = leaf_summary(records_from_masks([{1: disk(30, 30, 5)}, {1: disk(30, 30, 6)}], [0, 4]))
    assert np.isnan(summary.loc[0, "early_rgr_per_day"])


def test_growth_dashboard_renders() -> None:
    records = records_from_masks(exponential_frames(0.5), HOURS)
    fig = growth_dashboard(records, plant_timeseries(records), title="test")
    assert len(fig.axes) == 4  # three panels + the RGR twin axis
    assert isinstance(records, pd.DataFrame)


def test_analyze_sequence_with_leaftracker() -> None:
    from leaftrack.pipeline import analyze_sequence
    from tests.test_pipeline_tracking import FakeSegmenter

    dets = [[leaf(30, 50, 8 + k), leaf(70, 50, 6 + k)] for k in range(4)]
    images = [np.zeros((100, 100, 3), np.uint8)] * 4
    result = analyze_sequence(images, [0, 4, 8, 12], FakeSegmenter(dets), tracker="leaftrack")
    assert len(result.tracked) == 4
    assert list(result.plant["leaf_count"]) == [2, 2, 2, 2]
    assert set(result.leaves["track_id"]) == {1, 2}
    assert (result.leaves["early_rgr_per_day"] > 0).all()
    with pytest.raises(ValueError):
        analyze_sequence(images, [0, 4], FakeSegmenter(dets))


def test_cli_lists_images_and_reads_times(tmp_path) -> None:
    from analyze_growth import capture_hours, list_images

    for name in ["b.png", "a.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    files = list_images(tmp_path)
    assert [f.name for f in files] == ["a.png", "b.png"]
    assert capture_hours(files, 4.0, None) == [0.0, 4.0]
    times = tmp_path / "times.csv"
    times.write_text("filename,hours\na.png,0\nb.png,6.5\n", encoding="utf-8")
    assert capture_hours(files, 1.0, times) == [0.0, 6.5]
    times.write_text("filename,hours\na.png,0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        capture_hours(files, 1.0, times)


def test_growth_metrics_perfect_when_prediction_equals_truth() -> None:
    from leaftrack.growth_eval import SequenceGrowth, comparison_figure, growth_metrics

    gt = records_from_masks(exponential_frames(0.6), HOURS)
    pred = gt.assign(track_id=gt["track_id"] + 10)  # different IDs, same leaves
    seq = SequenceGrowth("s", gt, pred, {1: 11}, HOURS)
    m = growth_metrics([seq])
    assert m["plant"]["leaf_count_exact"] == 1.0
    assert m["plant"]["total_area_median_ape"] == 0.0
    assert m["leaf"]["area_coverage"] == 1.0 and m["leaf"]["early_rgr_mae_per_day"] == 0.0
    assert len(comparison_figure(seq).axes) == 3

    unmatched = growth_metrics([SequenceGrowth("s", gt, pred, {}, HOURS)])
    assert unmatched["leaf"]["area_coverage"] == 0.0
