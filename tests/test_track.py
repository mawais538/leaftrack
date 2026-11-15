"""Tests for the leaf tracker and the tracking metrics."""

from __future__ import annotations

import numpy as np
import pytest

from leaftrack.segment import LeafInstance
from leaftrack.track import (
    LeafTracker,
    leaf_radius_distance,
    mask_iou_matrix,
    mask_overlaps,
    track_sequence,
)
from leaftrack.track_metrics import TrackingSummary, evaluate_sequence, match_frame

H = W = 100


def disk(cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2


def leaf(cx: float, cy: float, r: float) -> LeafInstance:
    return LeafInstance(disk(cx, cy, r), 0.9, (cx - r, cy - r, cx + r, cy + r))


def ids(tracked) -> list[int]:
    return [t.track_id for t in tracked]


# --- geometry helpers ---------------------------------------------------------


def test_mask_iou_matrix_values() -> None:
    a = np.zeros((4, 4), bool)
    a[:2] = True  # 8 px
    b = np.zeros((4, 4), bool)
    b[1:3] = True  # 8 px, 4 shared
    iou = mask_iou_matrix([a, b], [b, np.zeros((4, 4), bool)])
    assert iou.shape == (2, 2)
    assert iou[0, 0] == pytest.approx(4 / 12)
    assert iou[1, 0] == pytest.approx(1.0)
    assert iou[0, 1] == 0.0
    assert mask_iou_matrix([], [a]).shape == (0, 1)


def test_mask_overlaps_containment() -> None:
    iou, iomin = mask_overlaps([disk(50, 50, 5)], [disk(50, 50, 10)])
    assert iou[0, 0] < 0.3 and iomin[0, 0] == pytest.approx(1.0)


def test_whole_leaf_beats_fragment() -> None:
    # Seen whole, then detected both as a fragment and as the whole leaf.
    left_half = disk(50, 50, 10) & (np.mgrid[:H, :W][1] < 50)
    tracker = LeafTracker()
    tracker.update([leaf(50, 50, 10)])
    fragment = LeafInstance(left_half, 0.9, (40, 40, 50, 60))
    assert ids(tracker.update([fragment, leaf(50, 50, 10)])) == [2, 1]


def test_leaf_radius_distance_scales_with_leaf_size() -> None:
    area = np.array([np.pi * 10**2])
    d = leaf_radius_distance(np.array([[0.0, 0.0]]), area, np.array([[10.0, 0.0]]), area)
    assert d[0, 0] == pytest.approx(1.0)


# --- tracker ------------------------------------------------------------------


def test_growing_leaves_keep_their_ids() -> None:
    frames = [[leaf(30, 50, 8 + k), leaf(70, 50, 6 + k)] for k in range(6)]
    out = track_sequence(frames)
    assert all(ids(f) == [1, 2] for f in out)
    assert out[-1][0].age == 6 and out[-1][0].hits == 6


def test_ids_follow_leaves_not_input_order() -> None:
    tracker = LeafTracker()
    tracker.update([leaf(30, 50, 8), leaf(70, 50, 8)])
    assert ids(tracker.update([leaf(71, 50, 9), leaf(31, 50, 9)])) == [2, 1]


def test_new_leaf_gets_a_new_id() -> None:
    tracker = LeafTracker()
    tracker.update([leaf(30, 50, 10)])
    assert ids(tracker.update([leaf(30, 50, 10), leaf(70, 50, 4)])) == [1, 2]


def test_short_occlusion_recovers_id_with_memory() -> None:
    frames = [[leaf(30, 50, 8), leaf(70, 50, 8)], [leaf(30, 50, 8)], [leaf(30, 50, 8)]]
    frames.append([leaf(30, 50, 8), leaf(70, 50, 8)])
    assert ids(track_sequence(frames, LeafTracker(max_missed=2))[-1]) == [1, 2]
    assert ids(track_sequence(frames, LeafTracker(max_missed=1))[-1]) == [1, 3]


def test_distance_gate_rescues_low_iou_match() -> None:
    # A small leaf shifts by more than its radius: IoU is zero, distance is 1.2 radii.
    frames = [[leaf(50, 50, 5)], [leaf(56, 50, 5)]]
    assert ids(track_sequence(frames, LeafTracker(max_distance=1.5))[-1]) == [1]
    assert ids(track_sequence(frames)[-1]) == [2]  # gate is off by default


def test_far_leaf_is_not_matched() -> None:
    frames = [[leaf(20, 20, 5)], [leaf(80, 80, 5)]]
    assert ids(track_sequence(frames)[-1]) == [2]


def test_reset_and_empty_frames() -> None:
    tracker = LeafTracker()
    tracker.update([leaf(30, 50, 8)])
    assert tracker.update([]) == []
    assert tracker.active_tracks == 1
    tracker.reset()
    assert ids(tracker.update([leaf(70, 50, 8)])) == [1]


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        LeafTracker(iou_threshold=1.5)
    with pytest.raises(ValueError):
        LeafTracker(max_missed=-1)


# --- metrics ------------------------------------------------------------------


def gt_sequence(n: int = 5) -> list[dict[int, np.ndarray]]:
    return [{1: disk(30, 50, 10), 2: disk(70, 50, 10)} for _ in range(n)]


def test_match_frame_requires_overlap() -> None:
    gt = {1: disk(30, 50, 10)}
    assert match_frame(gt, {7: disk(31, 50, 10)}) == [(1, 7)]
    assert match_frame(gt, {7: disk(70, 50, 10)}) == []
    assert match_frame({}, {7: disk(70, 50, 10)}) == []


def test_perfect_tracking_scores_one() -> None:
    gt = gt_sequence()
    pred = [{10: f[1], 20: f[2]} for f in gt]
    s = TrackingSummary.from_counts(evaluate_sequence(gt, pred))
    assert (s.mota, s.idf1, s.idsw, s.mostly_tracked) == (1.0, 1.0, 0, 1.0)


def test_id_swap_is_penalized() -> None:
    gt = gt_sequence(4)
    pred = [{10: f[1], 20: f[2]} if t < 2 else {20: f[1], 10: f[2]} for t, f in enumerate(gt)]
    c = evaluate_sequence(gt, pred)
    assert c.idsw == 2
    s = TrackingSummary.from_counts(c)
    assert s.mota == pytest.approx(1 - 2 / 8)
    assert s.idf1 == pytest.approx(0.5)


def test_misses_and_false_positives() -> None:
    gt = gt_sequence(2)
    pred = [{10: gt[0][1]}, {10: gt[1][1], 20: gt[1][2], 30: disk(50, 90, 5)}]
    c = evaluate_sequence(gt, pred)
    assert (c.tp, c.fn, c.fp, c.idsw) == (3, 1, 1, 0)
    assert (c + c).tp == 6


def test_tracker_on_synthetic_sequence_is_perfect() -> None:
    gt = [{1: disk(30, 50, 8 + k), 2: disk(70, 50, 6 + k)} for k in range(6)]
    tracked = track_sequence([[leaf(30, 50, 8 + k), leaf(70, 50, 6 + k)] for k in range(6)])
    pred = [{t.track_id: t.instance.mask for t in f} for f in tracked]
    s = TrackingSummary.from_counts(evaluate_sequence(gt, pred))
    assert s.idf1 == 1.0 and s.idsw == 0
