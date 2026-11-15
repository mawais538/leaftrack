"""Leaf tracking: stable leaf IDs across the frames of a time-lapse.

Top-down plant time-lapses are captured hours apart. Between two captures a
leaf grows, shifts slightly and is sometimes hidden by a neighbouring leaf,
while new leaves appear near the plant centre. On KOMATSUNA ground truth the
median mask IoU of the same leaf in consecutive frames is 0.82 and a
*different* leaf overlaps more in only 0.13% of cases, so mask overlap is a
strong cue.

:class:`LeafTracker` matches detections to tracks with the Hungarian
algorithm. A pair may match when its mask IoU reaches a threshold; among
allowed pairs the cost ``(1 - IoU) + (1 - overlap of the smaller mask)``
prefers a whole leaf over a fragment of it when segmentation flickers.
Unmatched tracks stay alive for a few frames, so a leaf missed by the
segmenter gets its old ID back. An optional centroid-distance gate (in leaf
radii) can admit low-overlap pairs; it did not improve validation scores on
KOMATSUNA and is off by default. No GPU or appearance model is needed.

Typical use::

    tracker = LeafTracker()
    for image in frames:
        tracked = tracker.update(segmenter.predict(image))
        ids = [leaf.track_id for leaf in tracked]
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from leaftrack.segment import LeafInstance

_FORBIDDEN = 1e6  # assignment cost for pairs that fail both gates


@dataclass(eq=False)  # holds arrays: compare by identity
class TrackedLeaf:
    """A detection annotated with its persistent track ID.

    Attributes:
        track_id: stable ID, starting at 1, unique within one sequence.
        instance: the detected leaf in the current frame.
        age: frames since the track was created, counting this one.
        hits: frames in which the track was matched, counting this one.
    """

    track_id: int
    instance: LeafInstance
    age: int
    hits: int


@dataclass(eq=False)
class _Track:
    track_id: int
    mask: np.ndarray
    centroid: tuple[float, float]
    area: int
    start_frame: int
    hits: int = 1
    missed: int = 0


def mask_overlaps(
    a: Sequence[np.ndarray], b: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise IoU and intersection-over-smaller-mask, each ``(len(a), len(b))``.

    The second measure is 1 when one mask lies inside the other, which stays
    high while a leaf grows or is partly covered.
    """
    if len(a) == 0 or len(b) == 0:
        empty = np.zeros((len(a), len(b)), np.float32)
        return empty, empty.copy()
    fa = np.stack([np.asarray(m, bool).ravel() for m in a]).astype(np.float32)
    fb = np.stack([np.asarray(m, bool).ravel() for m in b]).astype(np.float32)
    inter = fa @ fb.T
    sa, sb = fa.sum(1)[:, None], fb.sum(1)[None, :]
    iou = inter / np.maximum(sa + sb - inter, 1)
    iomin = inter / np.maximum(np.minimum(sa, sb), 1)
    return iou.astype(np.float32), iomin.astype(np.float32)


def mask_iou_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise IoU between two lists of same-shape boolean masks, shape ``(len(a), len(b))``."""
    return mask_overlaps(a, b)[0]


def leaf_radius_distance(
    a_centroids: np.ndarray, a_areas: np.ndarray, b_centroids: np.ndarray, b_areas: np.ndarray
) -> np.ndarray:
    """Centroid distance divided by the radius of the larger leaf of each pair.

    A value of 1 means the centroids are one leaf radius apart, which keeps the
    gate meaningful for both tiny new leaves and large mature ones.
    """
    diff = a_centroids[:, None, :] - b_centroids[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    radius = np.sqrt(np.maximum(a_areas[:, None], b_areas[None, :]) / np.pi)
    return dist / np.maximum(radius, 1.0)


class LeafTracker:
    """Online multi-leaf tracker for one plant sequence.

    Args:
        iou_threshold: a pair can match if its mask IoU reaches this value...
        max_distance: ...or if its centroids are at most this many leaf radii
            apart (see :func:`leaf_radius_distance`). ``0`` (default)
            disables the distance gate.
        max_missed: frames an unmatched track is kept before it is retired.
            ``0`` forgets a leaf as soon as one frame misses it.

    The defaults were selected by IDF1 on the KOMATSUNA validation plants.
    """

    def __init__(
        self, iou_threshold: float = 0.3, max_distance: float = 0.0, max_missed: int = 5
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_distance < 0 or max_missed < 0:
            raise ValueError("max_distance and max_missed must be >= 0")
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.reset()

    def reset(self) -> None:
        """Forget all tracks; the next frame starts a new sequence with ID 1."""
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._frame = -1

    @property
    def active_tracks(self) -> int:
        """Tracks currently alive, including ones missed in recent frames."""
        return len(self._tracks)

    def _cost(self, instances: Sequence[LeafInstance]) -> np.ndarray:
        tracks = self._tracks
        iou, iomin = mask_overlaps([t.mask for t in tracks], [i.mask for i in instances])
        allowed = iou >= max(self.iou_threshold, 1e-6)
        if self.max_distance > 0:
            dist = leaf_radius_distance(
                np.array([t.centroid for t in tracks], float),
                np.array([t.area for t in tracks], float),
                np.array([i.centroid for i in instances], float),
                np.array([i.area for i in instances], float),
            )
            allowed |= dist <= self.max_distance
        return np.where(allowed, (1.0 - iou) + (1.0 - iomin), _FORBIDDEN)

    def update(self, instances: Sequence[LeafInstance]) -> list[TrackedLeaf]:
        """Associate one frame's detections with the existing tracks.

        Args:
            instances: the leaves detected in this frame, all with masks of
                the same shape as in previous frames.

        Returns:
            One :class:`TrackedLeaf` per input instance, in input order.
            Detections that match no track start a new one.
        """
        self._frame += 1
        instances = list(instances)
        assigned: dict[int, _Track] = {}  # detection index -> track

        if self._tracks and instances:
            cost = self._cost(instances)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] < _FORBIDDEN:
                    assigned[int(c)] = self._tracks[int(r)]

        matched = {id(t) for t in assigned.values()}
        survivors = []
        for track in self._tracks:
            if id(track) in matched:
                survivors.append(track)
                continue
            track.missed += 1
            if track.missed <= self.max_missed:
                survivors.append(track)

        out = []
        for j, inst in enumerate(instances):
            track = assigned.get(j)
            if track is None:
                track = _Track(self._next_id, inst.mask, inst.centroid, inst.area, self._frame, 0)
                self._next_id += 1
                survivors.append(track)
            track.mask, track.centroid, track.area = inst.mask, inst.centroid, inst.area
            track.hits += 1
            track.missed = 0
            age = self._frame - track.start_frame + 1
            out.append(TrackedLeaf(track.track_id, inst, age, track.hits))
        self._tracks = survivors
        return out


def track_sequence(
    frames: Iterable[Sequence[LeafInstance]], tracker: LeafTracker | None = None
) -> list[list[TrackedLeaf]]:
    """Track a whole sequence of per-frame detections from a fresh state."""
    tracker = tracker or LeafTracker()
    tracker.reset()
    return [tracker.update(frame) for frame in frames]
