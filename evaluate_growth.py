"""Evaluate growth analytics (leaf count, leaf area, growth rates) against ground truth.

Runs segmentation + the default tracker (ByteTrack) on every test sequence,
computes the same analytics from the ground-truth label masks, and compares
them.

Usage::

    python evaluate_growth.py --weights runs/segment/yolo11n/weights/best.pt --device 0

Writes ``metrics.json``, ``results.md``, CSV tables and one comparison figure
plus one growth dashboard per sequence to ``outputs/growth/<name>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leaftrack.analytics import growth_dashboard, leaf_records, records_from_masks
from leaftrack.evaluation import load_image, load_split
from leaftrack.growth_eval import (
    SequenceGrowth,
    comparison_figure,
    growth_markdown,
    growth_metrics,
    leaf_frame_pairs,
    leaf_life_pairs,
    plant_pairs,
)
from leaftrack.segment import LeafSegmenter
from leaftrack.track_metrics import trajectory_mapping
from leaftrack.tracking_eval import bytetrack_tracks, group_sequences, load_gt_frame, to_frame

RIG_NAMES = {"multi": "multi-view", "rgbd": "RGB-D"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", default="runs/segment/yolo11n/weights/best.pt")
    p.add_argument("--data", type=Path, default=Path("data/processed/komatsuna"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seg-report", type=Path, default=Path("outputs/eval/yolo11n/metrics.json"))
    p.add_argument("--conf", type=float, help="override the val-selected confidence threshold")
    p.add_argument("--split", default="test")
    p.add_argument("--name", default="yolo11n")
    p.add_argument("--out", type=Path, default=Path("outputs/growth"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.conf is not None:
        conf = args.conf
    elif args.seg_report.exists():
        conf = float(json.loads(args.seg_report.read_text())["postprocess"]["conf"])
    else:
        raise SystemExit("Pass --conf or a --seg-report written by evaluate.py")
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    segmenter = LeafSegmenter(args.weights, device=args.device, conf=conf, imgsz=args.imgsz)

    seqs: list[SequenceGrowth] = []
    for key, rows in group_sequences(load_split(args.data, args.split)).items():
        hours = [float(r["hours"]) for r in rows]
        images = [load_image(r, args.data) for r in rows]
        gt = [load_gt_frame(r) for r in rows]
        tracked = bytetrack_tracks(segmenter, images)
        mapping = trajectory_mapping(gt, [to_frame(f) for f in tracked])
        seqs.append(
            SequenceGrowth(
                key, records_from_masks(gt, hours), leaf_records(tracked, hours), mapping, hours
            )
        )
        print(f"  {key}: {len(rows)} frames, {len(mapping)}/{len({i for f in gt for i in f})} "
              "leaves matched to a track")

    report = {
        "weights": str(args.weights),
        "split": args.split,
        "conf": conf,
        "tracker": "ByteTrack",
        "sequences": len(seqs),
        "metrics": growth_metrics(seqs),
        "per_sequence": {s.key: growth_metrics([s]) for s in seqs},
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "results.md").write_text(growth_markdown(report), encoding="utf-8")
    plant_pairs(seqs).to_csv(out / "plant_per_frame.csv", index=False)
    leaf_frame_pairs(seqs).to_csv(out / "leaf_area_per_frame.csv", index=False)
    leaf_life_pairs(seqs).to_csv(out / "leaf_summary.csv", index=False)

    import matplotlib.pyplot as plt

    for s in seqs:
        rig = RIG_NAMES.get(s.key.split("_")[0], s.key)
        fig = comparison_figure(s, f"Test plant {s.key} ({rig}): LeafTrack vs ground truth")
        fig.savefig(out / f"compare_{s.key}.png", dpi=110)
        plt.close(fig)
        fig = growth_dashboard(s.pred_records, s.pred_plant, f"Growth dashboard: {s.key}")
        fig.savefig(out / f"dashboard_{s.key}.png", dpi=110)
        plt.close(fig)

    print(growth_markdown(report))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
