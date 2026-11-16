"""ByteTrack leaf tracking: LeafTrack's default tracker.

ByteTrack (Zhang et al., ECCV 2022) associates boxes with a Kalman filter
and IoU in two stages. The second stage re-uses *low-confidence* detections
to continue existing tracks, which is what keeps a partly covered leaf's ID
while its segmentation flickers. On the KOMATSUNA validation plants it ties
with the mask-overlap :class:`~leaftrack.track.LeafTracker` (IDF1 0.886);
on test it makes fewer ID switches (IDF1 0.968, 6 switches, vs 0.949, 11),
and it is the default.

:class:`ByteTrackLeafTracker` runs the Ultralytics implementation on the
model of a :class:`~leaftrack.segment.LeafSegmenter` and applies the same
post-processing as :meth:`LeafSegmenter.predict` (confidence threshold and
neighbour-leaf filter), so tracked and untracked results agree.
"""

from __future__ import annotations

import numpy as np

from leaftrack.segment import LeafSegmenter, drop_neighbor_leaves, to_instances
from leaftrack.track import TrackedLeaf

#: Ultralytics' stock ByteTrack settings, as evaluated.
DEFAULT_TRACKER_CFG = "bytetrack.yaml"


class ByteTrackLeafTracker:
    """Segment and track the leaves of one plant, frame by frame.

    Args:
        segmenter: provides the model and the post-processing settings.
        tracker_cfg: Ultralytics tracker YAML.
        low_conf: detections down to this score are passed to ByteTrack for
            its second association stage. Only tracks whose current
            detection reaches ``segmenter.conf`` are returned.
    """

    def __init__(
        self,
        segmenter: LeafSegmenter,
        tracker_cfg: str = DEFAULT_TRACKER_CFG,
        low_conf: float = 0.1,
    ) -> None:
        self.segmenter = segmenter
        self.tracker_cfg = tracker_cfg
        self.low_conf = min(low_conf, segmenter.conf)
        self.reset()

    def reset(self) -> None:
        """Start a new sequence: the next frame gets fresh IDs starting at 1."""
        self._fresh = True
        self._frame = -1
        self._first_seen: dict[int, int] = {}
        self._hits: dict[int, int] = {}

    def update(self, image: np.ndarray) -> list[TrackedLeaf]:
        """Segment one ``(H, W, 3)`` uint8 RGB frame and return its tracked leaves."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) RGB image, got shape {image.shape}")
        seg = self.segmenter
        result = seg.model.track(
            np.ascontiguousarray(image[..., ::-1]),  # Ultralytics treats arrays as BGR
            persist=not self._fresh,  # persist=False rebuilds the tracker, IDs restart at 1
            tracker=self.tracker_cfg,
            conf=self.low_conf,
            iou=seg.iou,
            imgsz=seg.imgsz,
            max_det=seg.max_det,
            device=seg.device,
            retina_masks=True,
            verbose=False,
        )[0]
        self._fresh = False
        self._frame += 1
        if result.masks is None or result.boxes.id is None:
            return []

        ids = result.boxes.id.int().cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        polygons = list(result.masks.xy)
        track_of: dict[int, int] = {}
        leaves = []
        for k, track_id in enumerate(ids):
            sl = slice(k, k + 1)
            inst = to_instances(masks[sl], confs[sl], boxes[sl], polygons[sl])
            if inst and inst[0].confidence >= seg.conf:
                track_of[id(inst[0])] = int(track_id)
                leaves.append(inst[0])
        if seg.drop_neighbors:
            leaves = drop_neighbor_leaves(leaves)

        out = []
        for inst in leaves:
            tid = track_of[id(inst)]
            self._first_seen.setdefault(tid, self._frame)
            self._hits[tid] = self._hits.get(tid, 0) + 1
            age = self._frame - self._first_seen[tid] + 1
            out.append(TrackedLeaf(tid, inst, age, self._hits[tid]))
        return out
