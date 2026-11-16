"""One entry point for segmenting and tracking a plant time-lapse.

::

    from leaftrack.pipeline import track_frames
    from leaftrack.segment import LeafSegmenter

    segmenter = LeafSegmenter.from_hub("Awais05/leaftrack-yolo11n-seg", conf=0.30)
    for leaves in track_frames(frames, segmenter):  # ByteTrack by default
        print([leaf.track_id for leaf in leaves])
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from leaftrack.bytetrack import ByteTrackLeafTracker
from leaftrack.segment import LeafSegmenter
from leaftrack.track import LeafTracker, TrackedLeaf

if TYPE_CHECKING:
    import pandas as pd

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


@dataclass
class GrowthAnalysis:
    """Everything produced for one plant time-lapse.

    Attributes:
        tracked: tracked leaves per frame.
        records: one row per leaf per frame (:func:`~leaftrack.analytics.leaf_records`).
        plant: per-frame plant traits (:func:`~leaftrack.analytics.plant_timeseries`).
        leaves: one row per leaf (:func:`~leaftrack.analytics.leaf_summary`).
    """

    tracked: list[list[TrackedLeaf]]
    records: pd.DataFrame
    plant: pd.DataFrame
    leaves: pd.DataFrame


def analyze_sequence(
    frames: Sequence[np.ndarray],
    hours: Sequence[float],
    segmenter: LeafSegmenter,
    tracker: TrackerName = "bytetrack",
    mm2_per_px: float | None = None,
) -> GrowthAnalysis:
    """Segment, track and measure a time-ordered sequence of RGB frames of one plant.

    Args:
        frames: ``(H, W, 3)`` uint8 RGB images in capture order.
        hours: capture time of each frame in hours (any origin).
        segmenter: the leaf segmenter.
        tracker: ``"bytetrack"`` (default) or ``"leaftrack"``.
        mm2_per_px: optional pixel area in mm² for an ``area_mm2`` column.
    """
    from leaftrack.analytics import leaf_records, leaf_summary, plant_timeseries

    if len(frames) != len(hours):
        raise ValueError("frames and hours must have the same length")
    tracked = list(track_frames(frames, segmenter, tracker))
    records = leaf_records(tracked, hours, mm2_per_px)
    return GrowthAnalysis(
        tracked, records, plant_timeseries(records, hours), leaf_summary(records)
    )
