"""Tests for track_frames and the ByteTrack leaf tracker."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from leaftrack.pipeline import track_frames
from tests.test_track import leaf

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(
    os.environ.get("LEAFTRACK_TEST_WEIGHTS", ROOT / "runs/segment/yolo11n/weights/best.pt")
)
DATA = ROOT / "data/processed/komatsuna"


class FakeSegmenter:
    """Returns pre-made detections, one list per call."""

    def __init__(self, frames):
        self.frames = iter(frames)

    def predict(self, image):
        return next(self.frames)


def test_track_frames_with_leaftracker() -> None:
    dets = [[leaf(30, 50, 8 + k), leaf(70, 50, 6 + k)] for k in range(3)]
    images = [np.zeros((100, 100, 3), np.uint8)] * 3
    out = list(track_frames(images, FakeSegmenter(dets), tracker="leaftrack"))
    assert [[t.track_id for t in f] for f in out] == [[1, 2]] * 3


def test_track_frames_rejects_unknown_tracker() -> None:
    with pytest.raises(ValueError):
        list(track_frames([np.zeros((4, 4, 3), np.uint8)], FakeSegmenter([[]]), tracker="sort"))


@pytest.mark.skipif(
    not (WEIGHTS.exists() and (DATA / "test.txt").exists()),
    reason="needs trained weights and the processed KOMATSUNA data",
)
def test_bytetrack_ids_are_stable_and_reset() -> None:
    pytest.importorskip("ultralytics")
    from leaftrack.bytetrack import ByteTrackLeafTracker
    from leaftrack.evaluation import load_image, load_split
    from leaftrack.segment import LeafSegmenter

    rows = [r for r in load_split(DATA, "test") if r["sequence_id"] == "rgbd_p01"]
    image = load_image(sorted(rows, key=lambda r: int(r["frame_index"]))[-1], DATA)
    tracker = ByteTrackLeafTracker(LeafSegmenter(WEIGHTS, device="cpu", conf=0.30))

    first = [sorted(t.track_id for t in tracker.update(image)) for _ in range(3)]
    assert first[0] and first[0] == first[1] == first[2]  # same plant, same IDs
    assert min(first[0]) == 1
    tracker.reset()
    assert sorted(t.track_id for t in tracker.update(image)) == first[0]  # IDs restart

    leaves = tracker.update(image)
    assert all(t.age == 2 and t.hits == 2 for t in leaves)
    assert len(leaves) == len(LeafSegmenter(WEIGHTS, device="cpu", conf=0.30).predict(image))
