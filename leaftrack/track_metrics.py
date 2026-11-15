"""Multi-object tracking metrics for leaf masks: CLEAR MOT (MOTA) and IDF1.

Each frame is a mapping ``{id: mask}`` for ground truth and for the tracker.
A ground-truth leaf and a predicted leaf correspond in a frame when their
mask IoU is at least ``iou_threshold`` (0.5 by default, as in MOTS). With
mask IoU above 0.5 such correspondences are essentially one-to-one; the
Hungarian algorithm resolves any remaining ties.

- **MOTA** = 1 - (FN + FP + IDSW) / GT, counted over all frames.
- **IDSW** counts a ground-truth leaf whose matched track ID differs from
  the one it was last matched to.
- **IDF1** (Ristani et al., 2016) is the F1 score of the best one-to-one
  mapping between whole ground-truth and predicted trajectories, so it
  rewards keeping the *same* ID for the life of a leaf.
- **Mostly tracked** is the share of ground-truth leaves that the single
  track mapped to them by IDF1 covers in at least 80% of their frames.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from leaftrack.track import mask_iou_matrix

Frame = Mapping[int, np.ndarray]


def match_frame(
    gt: Frame, pred: Frame, iou_threshold: float = 0.5
) -> list[tuple[int, int]]:
    """One-to-one ``(gt_id, pred_id)`` correspondences with IoU >= ``iou_threshold``."""
    gt_ids, pred_ids = list(gt), list(pred)
    iou = mask_iou_matrix([gt[i] for i in gt_ids], [pred[i] for i in pred_ids])
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(-iou)
    return [
        (gt_ids[r], pred_ids[c])
        for r, c in zip(rows, cols, strict=True)
        if iou[r, c] >= iou_threshold
    ]


@dataclass
class TrackingCounts:
    """Raw counts, additive across sequences."""

    frames: int = 0
    gt: int = 0
    pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    idtp: int = 0
    gt_tracks: int = 0
    mostly_tracked: int = 0

    def __add__(self, other: TrackingCounts) -> TrackingCounts:
        return TrackingCounts(**{k: v + getattr(other, k) for k, v in asdict(self).items()})


@dataclass(frozen=True)
class TrackingSummary:
    """Tracking scores derived from :class:`TrackingCounts`."""

    mota: float
    idf1: float
    idsw: int
    mostly_tracked: float
    precision: float
    recall: float
    frames: int
    gt_leaves: int

    @classmethod
    def from_counts(cls, c: TrackingCounts) -> TrackingSummary:
        return cls(
            mota=1.0 - (c.fn + c.fp + c.idsw) / c.gt if c.gt else 0.0,
            idf1=2 * c.idtp / (c.gt + c.pred) if c.gt + c.pred else 0.0,
            idsw=c.idsw,
            mostly_tracked=c.mostly_tracked / c.gt_tracks if c.gt_tracks else 0.0,
            precision=c.tp / c.pred if c.pred else 0.0,
            recall=c.tp / c.gt if c.gt else 0.0,
            frames=c.frames,
            gt_leaves=c.gt_tracks,
        )


def evaluate_sequence(
    gt_frames: Sequence[Frame], pred_frames: Sequence[Frame], iou_threshold: float = 0.5
) -> TrackingCounts:
    """Count tracking errors over one sequence (IDs are local to the sequence)."""
    if len(gt_frames) != len(pred_frames):
        raise ValueError("gt_frames and pred_frames must have the same length")
    c = TrackingCounts(frames=len(gt_frames))
    last_match: dict[int, int] = {}
    co_occurrence: dict[tuple[int, int], int] = defaultdict(int)
    gt_len: dict[int, int] = defaultdict(int)
    pred_len: dict[int, int] = defaultdict(int)

    for gt, pred in zip(gt_frames, pred_frames, strict=True):
        pairs = match_frame(gt, pred, iou_threshold)
        c.gt += len(gt)
        c.pred += len(pred)
        c.tp += len(pairs)
        c.fn += len(gt) - len(pairs)
        c.fp += len(pred) - len(pairs)
        for g, p in pairs:
            if g in last_match and last_match[g] != p:
                c.idsw += 1
            last_match[g] = p
            co_occurrence[g, p] += 1
        for g in gt:
            gt_len[g] += 1
        for p in pred:
            pred_len[p] += 1

    c.gt_tracks = len(gt_len)
    if co_occurrence:
        gts, preds = sorted(gt_len), sorted(pred_len)
        gi = {g: k for k, g in enumerate(gts)}
        pi = {p: k for k, p in enumerate(preds)}
        counts = np.zeros((len(gts), len(preds)))
        for (g, p), n in co_occurrence.items():
            counts[gi[g], pi[p]] = n
        rows, cols = linear_sum_assignment(-counts)
        c.idtp = int(counts[rows, cols].sum())
        c.mostly_tracked = sum(
            1 for r, col in zip(rows, cols, strict=True) if counts[r, col] >= 0.8 * gt_len[gts[r]]
        )
    return c
