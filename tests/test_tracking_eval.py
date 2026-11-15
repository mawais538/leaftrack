"""Tests for the tracking evaluation helpers (no model or dataset needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from leaftrack.track import track_sequence
from leaftrack.track_metrics import TrackingSummary
from leaftrack.tracking_eval import (
    SequenceRun,
    TrackerConfig,
    config_grid,
    group_sequences,
    gt_instances,
    per_frame_ids,
    render_frames,
    run_methods,
    save_gif,
    select_config,
    total,
    tracker_method,
)
from tests.test_track import disk, leaf


def row(seq: str, cam: str, frame: int) -> dict[str, str]:
    return {"sequence_id": seq, "camera": cam, "frame_index": str(frame), "day": "0", "hours": "0"}


def make_run(n: int = 4) -> SequenceRun:
    gt = [{1: disk(30, 50, 8 + k), 2: disk(70, 50, 6 + k)} for k in range(n)]
    dets = [[leaf(70, 50, 6 + k), leaf(30, 50, 8 + k)] for k in range(n)]
    images = [np.zeros((100, 100, 3), np.uint8) for _ in range(n)]
    return SequenceRun("s", [row("s", "", k) for k in range(n)], images, gt, dets)


def test_group_sequences_splits_cameras_and_sorts_by_time() -> None:
    rows = [row("multi_p00", "1", 1), row("multi_p00", "0", 1), row("multi_p00", "0", 0)]
    rows.append(row("rgbd_p01", "", 0))
    seqs = group_sequences(rows)
    assert list(seqs) == ["multi_p00_c0", "multi_p00_c1", "rgbd_p01"]
    assert [r["frame_index"] for r in seqs["multi_p00_c0"]] == ["0", "1"]


def test_config_grid_covers_all_combinations() -> None:
    grid = {"iou_threshold": (0.1, 0.3), "max_distance": (0.0,), "max_missed": (0, 5)}
    assert len(config_grid(grid)) == 4
    assert len(config_grid()) == 32


def test_select_config_prefers_idf1_then_simplicity() -> None:
    def s(idf1: float) -> TrackingSummary:
        return TrackingSummary(0.9, idf1, 0, 1.0, 1.0, 1.0, 10, 2)

    a, b, c = TrackerConfig(0.2, 1.0, 5), TrackerConfig(0.2, 0.0, 0), TrackerConfig(0.3, 1.0, 2)
    assert select_config({a: s(0.9), b: s(0.8)}) == a
    assert select_config({a: s(0.9), c: s(0.9)}) == c  # same score, less memory


def test_tracker_beats_per_frame_ids_on_reordered_detections() -> None:
    run = make_run()
    run.detections = [d if k % 2 else d[::-1] for k, d in enumerate(run.detections)]
    scores = run_methods(
        [run], {"none": per_frame_ids, "track": tracker_method(TrackerConfig(0.2, 1.0, 5))}
    )
    assert total(scores["track"]).idf1 == 1.0
    assert total(scores["none"]).idsw > 0


def test_oracle_method_tracks_ground_truth_perfectly() -> None:
    run = make_run()
    assert len(gt_instances(run.gt[0])) == 2
    scores = run_methods([run], {"oracle": tracker_method(TrackerConfig(0.2, 1.0, 5), True)})
    assert total(scores["oracle"]).mota == 1.0


def test_render_and_save_gif(tmp_path: Path) -> None:
    run = make_run(3)
    tracks = track_sequence(run.detections)
    frames = render_frames([run, run], [tracks, tracks], ["a", "b"], cell=80)
    assert len(frames) == 3 and frames[0].shape[1] > 160
    path = tmp_path / "t.gif"
    save_gif(frames, path, fps=4, hold_last=1)
    gif = Image.open(path)
    assert gif.n_frames == 3  # Pillow merges the held copy into a longer last frame
    gif.seek(2)
    assert gif.info["duration"] == 2 * 250
