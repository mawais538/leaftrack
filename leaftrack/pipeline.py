"""One entry point for segmenting and tracking a plant time-lapse.

::

    from leaftrack.pipeline import track_frames
    from leaftrack.segment import LeafSegmenter

    segmenter = LeafSegmenter.from_hub("Awais05/leaftrack-yolo11n-seg", conf=0.30)
    for leaves in track_frames(frames, segmenter):  # ByteTrack by default
        print([leaf.track_id for leaf in leaves])
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Literal

import numpy as np

from leaftrack.bytetrack import ByteTrackLeafTracker
from leaftrack.segment import LeafSegmenter
from leaftrack.track import LeafTracker, TrackedLeaf

TrackerName = Literal["bytetrack", "leaftrack"]
TRACKERS: tuple[str, ...] = ("bytetrack", "leaftrack")


def track_frames(
    frames: Iterable[np.ndarray],
    segmenter: LeafSegmenter,
    tracker: TrackerName = "bytetrack",
    **tracker_kwargs: Any,
) -> Iterator[list[TrackedLeaf]]:
    """Segment and track time-ordered RGB frames of one plant, from a fresh state.

    Args:
        frames: ``(H, W, 3)`` uint8 RGB images in capture order.
        segmenter: the leaf segmenter (model and post-processing).
        tracker: ``"bytetrack"`` (default, selected on validation) or
            ``"leaftrack"`` (mask-overlap :class:`LeafTracker`).
        tracker_kwargs: passed to the tracker's constructor.

    Yields:
        The tracked leaves of each frame.
    """
    if tracker == "bytetrack":
        bt = ByteTrackLeafTracker(segmenter, **tracker_kwargs)
        for image in frames:
            yield bt.update(image)
    elif tracker == "leaftrack":
        lt = LeafTracker(**tracker_kwargs)
        for image in frames:
            yield lt.update(segmenter.predict(image))
    else:
        raise ValueError(f"tracker must be one of {TRACKERS}, got {tracker!r}")
