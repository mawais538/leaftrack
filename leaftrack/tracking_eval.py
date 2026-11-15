"""Tracking evaluation on KOMATSUNA: sequences, baselines, parameter sweep, reports.

A *sequence* is one camera view of one plant over time (60 frames). The
ground-truth track ID of a leaf is its stable label color.

To keep memory flat, every sequence is segmented once and then scored for all
tracker settings and baselines before moving on, so counts are accumulated
per setting rather than storing masks for the whole split.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from leaftrack.datasets.komatsuna import decode_label
from leaftrack.evaluation import load_image
from leaftrack.segment import LeafInstance, LeafSegmenter, drop_neighbor_leaves, to_instances
from leaftrack.track import LeafTracker, TrackedLeaf, track_sequence
from leaftrack.track_metrics import Frame, TrackingCounts, TrackingSummary, evaluate_sequence

Row = dict[str, str]


@dataclass(frozen=True)
class TrackerConfig:
    """Hyper-parameters of :class:`LeafTracker`."""

    iou_threshold: float
    max_distance: float
    max_missed: int

    def build(self) -> LeafTracker:
        return LeafTracker(self.iou_threshold, self.max_distance, self.max_missed)

    def label(self) -> str:
        return f"iou>={self.iou_threshold}, dist<={self.max_distance}, missed<={self.max_missed}"


DEFAULT_GRID: dict[str, Sequence[float]] = {
    "iou_threshold": (0.2, 0.3, 0.4, 0.5),
    "max_distance": (0.0, 1.0),
    "max_missed": (0, 2, 5, 10),
}


def config_grid(grid: dict[str, Sequence[float]] | None = None) -> list[TrackerConfig]:
    """All combinations of a parameter grid (defaults to :data:`DEFAULT_GRID`)."""
    g = grid or DEFAULT_GRID
    return [
        TrackerConfig(float(i), float(d), int(m))
        for i, d, m in product(g["iou_threshold"], g["max_distance"], g["max_missed"])
    ]


def group_sequences(rows: Sequence[Row]) -> dict[str, list[Row]]:
    """Group split rows into time-ordered sequences keyed ``<sequence_id>[_c<camera>]``."""
    seqs: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        key = r["sequence_id"] + (f"_c{r['camera']}" if r.get("camera") else "")
        seqs[key].append(r)
    return {k: sorted(v, key=lambda r: int(r["frame_index"])) for k, v in sorted(seqs.items())}


def load_gt_frame(row: Row) -> Frame:
    """``{leaf_id: mask}`` from the original color label (IDs are stable over time)."""
    return decode_label(np.asarray(Image.open(row["raw_label"]).convert("RGB")))


def to_frame(tracked: Sequence[TrackedLeaf]) -> Frame:
    return {t.track_id: t.instance.mask for t in tracked}


def gt_instances(frame: Frame) -> list[LeafInstance]:
    """Wrap ground-truth masks as detections (for the oracle-detection run)."""
    out = []
    for mask in frame.values():
        ys, xs = np.nonzero(mask)
        out.append(LeafInstance(mask, 1.0, (xs.min(), ys.min(), xs.max(), ys.max())))
    return out


def bytetrack_sequence(
    weights: str, images: Sequence[np.ndarray], conf: float, device: str, imgsz: int
) -> list[Frame]:
    """Ultralytics ByteTrack baseline (box IoU + Kalman filter), same post-processing.

    A fresh model per sequence guarantees the tracker state starts empty.
    """
    from ultralytics import YOLO

    model = YOLO(weights, task="segment")
    frames: list[Frame] = []
    for rgb in images:
        r = model.track(
            np.ascontiguousarray(rgb[..., ::-1]),
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.1,  # ByteTrack uses low-score boxes in its second association
            imgsz=imgsz,
            device=device,
            retina_masks=True,
            verbose=False,
        )[0]
        if r.masks is None or r.boxes.id is None:
            frames.append({})
            continue
        ids = r.boxes.id.int().cpu().numpy()
        masks, confs = r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        track_of: dict[int, int] = {}
        insts = []
        for k in range(len(ids)):
            inst = to_instances(masks[k : k + 1], confs[k : k + 1], boxes[k : k + 1])
            if inst and inst[0].confidence >= conf:
                track_of[id(inst[0])] = int(ids[k])
                insts.append(inst[0])
        frames.append({track_of[id(i)]: i.mask for i in drop_neighbor_leaves(insts)})
    return frames


@dataclass
class SequenceRun:
    """Everything computed for one sequence (kept only for the rendering step)."""

    key: str
    rows: list[Row]
    images: list[np.ndarray]
    gt: list[Frame]
    detections: list[list[LeafInstance]]


def segment_sequence(
    segmenter: LeafSegmenter, key: str, rows: Sequence[Row], data_dir: Path, conf: float
) -> SequenceRun:
    """Run the segmenter on every frame and load the matching ground truth.

    The segmenter should be built with its own confidence threshold at or
    below ``conf``; detections below ``conf`` are dropped here.
    """
    images = [load_image(r, data_dir) for r in rows]
    dets = [[i for i in segmenter.predict(img) if i.confidence >= conf] for img in images]
    gts = [load_gt_frame(r) for r in rows]
    return SequenceRun(key, list(rows), images, gts, dets)


Method = Callable[[SequenceRun], list[Frame]]


def run_methods(
    runs: Sequence[SequenceRun], methods: dict[str, Method]
) -> dict[str, dict[str, TrackingCounts]]:
    """Score each method on each sequence: ``{method: {sequence: counts}}``."""
    out: dict[str, dict[str, TrackingCounts]] = defaultdict(dict)
    for run in runs:
        for name, method in methods.items():
            out[name][run.key] = evaluate_sequence(run.gt, method(run))
    return dict(out)


def tracker_method(config: TrackerConfig, oracle: bool = False) -> Method:
    """A method that runs :class:`LeafTracker` on detections (or on ground truth)."""

    def method(run: SequenceRun) -> list[Frame]:
        frames = [gt_instances(g) for g in run.gt] if oracle else run.detections
        return [to_frame(f) for f in track_sequence(frames, config.build())]

    return method


def per_frame_ids(run: SequenceRun) -> list[Frame]:
    """No tracking: IDs are re-assigned 1..N in every frame (lower bound)."""
    return [{k + 1: d.mask for k, d in enumerate(f)} for f in run.detections]


def total(per_sequence: dict[str, TrackingCounts]) -> TrackingSummary:
    acc = TrackingCounts()
    for c in per_sequence.values():
        acc = acc + c
    return TrackingSummary.from_counts(acc)


def select_config(scores: dict[TrackerConfig, TrackingSummary]) -> TrackerConfig:
    """Best IDF1, then MOTA, then the simplest setting (lower thresholds, less memory)."""
    return max(
        scores,
        key=lambda c: (
            round(scores[c].idf1, 6),
            round(scores[c].mota, 6),
            -c.max_missed,
            -c.max_distance,
        ),
    )


def measure_tracker_latency(runs: Sequence[SequenceRun], config: TrackerConfig) -> float:
    """Median tracker-only time per frame in milliseconds (detections precomputed)."""
    times = []
    for run in runs:
        tracker = config.build()
        for frame in run.detections:
            t0 = time.perf_counter()
            tracker.update(frame)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)) if times else 0.0


def summary_dict(s: TrackingSummary) -> dict[str, Any]:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(s).items()}


def tracking_markdown(report: dict[str, Any]) -> str:
    """Render the tracking report as Markdown tables."""
    lines = [
        f"## Leaf tracking, {report['split']} split",
        "",
        f"{report['sequences']} sequences, {report['frames']} frames, "
        f"{report['gt_leaves']} ground-truth leaf tracks. Tracker settings selected on "
        f"{report['selected_on']}: {report['config_label']}.",
        "",
        "| Method | IDF1 ↑ | MOTA ↑ | ID switches ↓ | Mostly tracked ↑ | Val IDF1 |",
        "|---|---|---|---|---|---|",
    ]
    val = report.get("val_idf1", {})
    for name, s in report["methods"].items():
        v = f"{val[name]:.3f}" if name in val else "—"
        lines.append(
            f"| {name} | {s['idf1']:.3f} | {s['mota']:.3f} | {s['idsw']} | "
            f"{s['mostly_tracked']:.0%} | {v} |"
        )
    lines += [
        "",
        f"Tracker CPU time: {report['tracker_ms_per_frame']:.1f} ms per frame (median), "
        "on top of segmentation.",
        "",
        "### Per sequence (LeafTrack, predicted masks)",
        "",
        "| Sequence | Frames | Leaves | IDF1 | MOTA | ID switches |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in report["per_sequence"].items():
        lines.append(
            f"| {key} | {s['frames']} | {s['gt_leaves']} | {s['idf1']:.3f} | "
            f"{s['mota']:.3f} | {s['idsw']} |"
        )
    return "\n".join(lines) + "\n"


# --- rendering ------------------------------------------------------------------


def _tile(run: SequenceRun, tracked: Sequence[TrackedLeaf], t: int, title: str, cell: int):
    from leaftrack.evaluation import _letterbox
    from leaftrack.viz import draw_tracks, with_caption

    drawn = draw_tracks(
        run.images[t], [x.instance.mask for x in tracked], [x.track_id for x in tracked], cell
    )
    row = run.rows[t]
    parts = [title, f"day {int(row['day']) + 1}", f"{len(tracked)} leaves"]
    text = " | ".join(p for p in parts if p)
    return with_caption(_letterbox(drawn, cell), text)


def render_frames(
    runs: Sequence[SequenceRun],
    tracks: Sequence[Sequence[Sequence[TrackedLeaf]]],
    titles: Sequence[str],
    cell: int = 300,
) -> list[np.ndarray]:
    """Side-by-side animation frames, one tile per sequence, colors = track IDs."""
    n = min(len(r.rows) for r in runs)
    gap = None
    frames = []
    for t in range(n):
        tiles = [_tile(r, tr[t], t, title, cell) for r, tr, title in zip(runs, tracks, titles,
                                                                         strict=True)]
        gap = gap if gap is not None else np.full((tiles[0].shape[0], 8, 3), 255, np.uint8)
        frames.append(np.hstack([x for tile in tiles for x in (tile, gap)][:-1]))
    return frames


def filmstrip(
    run: SequenceRun,
    tracked: Sequence[Sequence[TrackedLeaf]],
    indices: Sequence[int],
    title: str,
    cell: int = 240,
) -> np.ndarray:
    """A row of time points from one sequence with consistent track colors."""
    tiles = [_tile(run, tracked[t], t, title, cell) for t in indices]
    gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    return np.hstack([x for tile in tiles for x in (tile, gap)][:-1])


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: float = 6, hold_last: int = 6) -> None:
    """Write RGB frames as a looping GIF with a shared adaptive palette."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = list(frames) + [frames[-1]] * hold_last
    # Build one palette from frames spread over the sequence, so colors that
    # only appear late (new leaves, grown leaves) are represented.
    sample = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
    montage = Image.fromarray(np.vstack([frames[i] for i in sample]))
    palette = montage.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = [
        Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in seq
    ]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0,
        optimize=True,
    )
