"""Evaluate leaf tracking on KOMATSUNA time-lapse sequences.

Tracker settings are selected by IDF1 on the validation plants, then the test
plants are scored once, alongside baselines:

- per-frame IDs (no tracking), a lower bound;
- Ultralytics ByteTrack, a box-based tracker with a Kalman filter;
- LeafTrack ablations (mask IoU only, + track memory, + distance fallback);
- LeafTrack on ground-truth masks, which isolates association errors from
  segmentation errors.

Usage::

    python evaluate_tracking.py --weights runs/segment/yolo11n/weights/best.pt --device 0

Writes ``metrics.json``, ``results.md``, ``per_sequence.csv``, a tracking GIF
and a filmstrip to ``outputs/track/<name>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import cv2

from leaftrack.evaluation import load_split
from leaftrack.segment import LeafSegmenter
from leaftrack.track import track_sequence
from leaftrack.track_metrics import TrackingCounts
from leaftrack.tracking_eval import (
    Method,
    SequenceRun,
    TrackerConfig,
    bytetrack_sequence,
    config_grid,
    filmstrip,
    group_sequences,
    measure_tracker_latency,
    per_frame_ids,
    render_frames,
    run_methods,
    save_gif,
    segment_sequence,
    select_config,
    summary_dict,
    total,
    tracker_method,
    tracking_markdown,
)

RENDER_SEQUENCES = {"multi_p00_c0": "multi-view", "rgbd_p01": "RGB-D"}
MAIN = "**LeafTrack**"
BYTETRACK = "ByteTrack (Ultralytics, box IoU + Kalman)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", default="runs/segment/yolo11n/weights/best.pt")
    p.add_argument("--data", type=Path, default=Path("data/processed/komatsuna"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--seg-report",
        type=Path,
        default=Path("outputs/eval/yolo11n/metrics.json"),
        help="evaluate.py report; its val-selected confidence threshold is reused",
    )
    p.add_argument("--conf", type=float, help="override the confidence threshold")
    p.add_argument("--val-split", default="val")
    p.add_argument("--split", default="test")
    p.add_argument("--no-bytetrack", action="store_true", help="skip the ByteTrack baseline")
    p.add_argument("--name", default="yolo11n")
    p.add_argument("--out", type=Path, default=Path("outputs/track"))
    return p.parse_args(argv)


def resolve_conf(args: argparse.Namespace) -> float:
    if args.conf is not None:
        return args.conf
    if args.seg_report.exists():
        return float(json.loads(args.seg_report.read_text())["postprocess"]["conf"])
    raise SystemExit("Pass --conf or a --seg-report written by evaluate.py")


def sweep(
    segmenter: LeafSegmenter,
    data: Path,
    split: str,
    conf: float,
    baselines: dict[str, Method],
) -> tuple[dict[TrackerConfig, TrackingCounts], dict[str, TrackingCounts]]:
    """Score every grid setting and each baseline over a split, one sequence at a time."""
    methods: dict = {c: tracker_method(c) for c in config_grid()} | baselines
    acc: dict = defaultdict(TrackingCounts)
    for key, rows in group_sequences(load_split(data, split)).items():
        run = segment_sequence(segmenter, key, rows, data, conf)
        for m, counts in run_methods([run], methods).items():
            acc[m] = acc[m] + counts[key]
        print(f"  {split} {key}: {len(rows)} frames")
    grid = {m: c for m, c in acc.items() if isinstance(m, TrackerConfig)}
    return grid, {m: c for m, c in acc.items() if not isinstance(m, TrackerConfig)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    conf = resolve_conf(args)
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    segmenter = LeafSegmenter(args.weights, device=args.device, conf=conf, imgsz=args.imgsz)

    baselines: dict[str, Method] = {}
    if not args.no_bytetrack:
        baselines[BYTETRACK] = lambda run: bytetrack_sequence(
            args.weights, run.images, conf, args.device, args.imgsz
        )

    print(f"Selecting tracker settings on '{args.val_split}' (conf {conf})")
    val, val_baselines = sweep(segmenter, args.data, args.val_split, conf, baselines)
    val_scores = {c: total({"all": n}) for c, n in val.items()}
    best = select_config(val_scores)
    print(f"  selected {best.label()}: val IDF1 {val_scores[best].idf1:.3f}")
    val_idf1 = {MAIN: val_scores[best].idf1} | {
        m: total({"all": c}).idf1 for m, c in val_baselines.items()
    }

    methods: dict[str, Method] = {"Per-frame IDs (no tracking)": per_frame_ids, **baselines}
    methods["LeafTrack without track memory"] = tracker_method(replace(best, max_missed=0))
    methods[MAIN] = tracker_method(best)
    methods["LeafTrack on ground-truth masks"] = tracker_method(best, oracle=True)

    print(f"Scoring '{args.split}'")
    per_method: dict[str, dict[str, TrackingCounts]] = defaultdict(dict)
    kept: dict[str, SequenceRun] = {}
    latencies = []
    for key, rows in group_sequences(load_split(args.data, args.split)).items():
        run = segment_sequence(segmenter, key, rows, args.data, conf)
        for name, counts in run_methods([run], methods).items():
            per_method[name].update(counts)
        latencies.append(measure_tracker_latency([run], best))
        if key in RENDER_SEQUENCES:
            kept[key] = run
        print(f"  {args.split} {key}: {len(rows)} frames")

    main_name = MAIN
    summary = {name: summary_dict(total(c)) for name, c in per_method.items()}
    per_seq = {k: summary_dict(total({k: c})) for k, c in per_method[main_name].items()}
    report = {
        "weights": str(args.weights),
        "split": args.split,
        "conf": conf,
        "selected_on": f"{args.val_split} split (best IDF1 over {len(val)} settings)",
        "config": best.__dict__,
        "config_label": best.label(),
        "val_idf1": {m: round(v, 4) for m, v in val_idf1.items()},
        "sequences": len(per_seq),
        "frames": summary[main_name]["frames"],
        "gt_leaves": summary[main_name]["gt_leaves"],
        "tracker_ms_per_frame": round(sorted(latencies)[len(latencies) // 2], 2),
        "methods": summary,
        "per_sequence": per_seq,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "results.md").write_text(tracking_markdown(report), encoding="utf-8")
    with open(out / "per_sequence.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "sequence", *next(iter(per_seq.values())).keys()])
        for name, seqs in per_method.items():
            for key, c in seqs.items():
                writer.writerow([name.strip("*"), key, *summary_dict(total({key: c})).values()])

    if kept:
        keys = [k for k in RENDER_SEQUENCES if k in kept]
        runs = [kept[k] for k in keys]
        tracks = [track_sequence(r.detections, best.build()) for r in runs]
        frames = render_frames(runs, tracks, [RENDER_SEQUENCES[k] for k in keys], cell=300)
        save_gif(frames, out / "tracking.gif", fps=6)
        n = len(runs[0].rows)
        strip = filmstrip(runs[0], tracks[0], [round(i * (n - 1) / 4) for i in range(5)],
                          "")
        cv2.imwrite(str(out / "filmstrip.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(tracking_markdown(report))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
