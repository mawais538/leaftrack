"""LeafTrack: leaf instance segmentation, tracking and growth analytics.

Modules:

- :mod:`leaftrack.segment`    -- YOLO11-seg inference wrapper and neighbour-leaf filter.
- :mod:`leaftrack.metrics`    -- CVPPP leaf metrics (SBD, FBD, DiC).
- :mod:`leaftrack.evaluation` -- test-set evaluation, threshold selection, latency.
- :mod:`leaftrack.training`   -- training plans around Ultralytics.
- :mod:`leaftrack.hub`        -- publishing weights to the Hugging Face Hub.
- :mod:`leaftrack.datasets`   -- KOMATSUNA reader, YOLO-seg conversion, splitting.
- :mod:`leaftrack.viz`        -- drawing helpers.
- :mod:`leaftrack.pipeline`   -- ``track_frames``: segment + track a time-lapse.
- :mod:`leaftrack.bytetrack`  -- ByteTrack leaf tracker (default).
- :mod:`leaftrack.track`      -- mask-overlap ``LeafTracker`` and ``TrackedLeaf``.
- :mod:`leaftrack.track_metrics` -- tracking metrics (MOTA, IDF1, ID switches).
- :mod:`leaftrack.tracking_eval` -- tracking evaluation, baselines and rendering.
- :mod:`leaftrack.analytics`  -- per-leaf growth analytics (Phase 4).
"""

__version__ = "0.3.0"
