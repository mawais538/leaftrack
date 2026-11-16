"""Measure leaf growth in a folder of top-down time-lapse images of one plant.

Images are processed in file-name order. Capture times come from a fixed
interval (``--interval-hours``) or, if file names are not evenly spaced in
time, from a CSV with ``filename,hours`` columns (``--times``).

Usage::

    python analyze_growth.py --images path/to/plant_frames --interval-hours 4

Writes to ``outputs/analyze/<folder name>/``:

- ``leaves.csv``: one row per leaf per frame (track ID, area, centroid, score),
- ``plant.csv``: leaf count, total leaf area and growth rates per frame,
- ``leaf_summary.csv``: one row per leaf (lifespan, size, early growth rate),
- ``dashboard.png``: growth charts, and ``tracking.gif`` (unless ``--no-gif``).

Weights default to the published YOLO11n model on the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_REPO = "Awais05/leaftrack-yolo11n-seg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="folder of frames of one plant")
    timing = p.add_mutually_exclusive_group()
    timing.add_argument("--interval-hours", type=float, default=1.0, help="time between frames")
    timing.add_argument("--times", type=Path, help="CSV with filename,hours columns")
    p.add_argument("--weights", help="local .pt file (default: download from the Hub)")
    p.add_argument("--hub-repo", default=DEFAULT_REPO)
    p.add_argument("--conf", type=float, default=0.30, help="val-selected for YOLO11n")
    p.add_argument("--tracker", choices=["bytetrack", "leaftrack"], default="bytetrack")
    p.add_argument("--mm-per-px", type=float, help="pixel size in mm, adds mm² areas")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, help="output folder (default: outputs/analyze/<name>)")
    p.add_argument("--no-gif", action="store_true")
    return p.parse_args(argv)


def list_images(folder: Path) -> list[Path]:
    """Image files in a folder, sorted by name."""
    files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"No images found in {folder}")
    return files


def capture_hours(files: list[Path], interval: float, times_csv: Path | None) -> list[float]:
    """Capture time per file, from a fixed interval or a ``filename,hours`` CSV."""
    if times_csv is None:
        return [k * interval for k in range(len(files))]
    with open(times_csv, newline="", encoding="utf-8") as f:
        table = {row["filename"]: float(row["hours"]) for row in csv.DictReader(f)}
    missing = [f.name for f in files if f.name not in table]
    if missing:
        raise SystemExit(f"{len(missing)} images have no time in {times_csv}, e.g. {missing[0]}")
    return [table[f.name] for f in files]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from leaftrack.analytics import growth_dashboard
    from leaftrack.hub import use_system_certificates
    from leaftrack.pipeline import analyze_sequence
    from leaftrack.segment import LeafSegmenter
    from leaftrack.viz import draw_tracks, save_gif, with_caption

    files = list_images(args.images)
    hours = capture_hours(files, args.interval_hours, args.times)
    order = np.argsort(hours, kind="stable")
    files, hours = [files[i] for i in order], [hours[i] for i in order]
    frames = [np.asarray(Image.open(f).convert("RGB")) for f in files]

    if args.weights:
        segmenter = LeafSegmenter(args.weights, device=args.device, conf=args.conf)
    else:
        use_system_certificates()
        segmenter = LeafSegmenter.from_hub(args.hub_repo, device=args.device, conf=args.conf)

    mm2 = args.mm_per_px**2 if args.mm_per_px else None
    result = analyze_sequence(frames, hours, segmenter, args.tracker, mm2)

    out = args.out or Path("outputs/analyze") / args.images.name
    out.mkdir(parents=True, exist_ok=True)
    result.records.insert(1, "file", [files[t].name for t in result.records["frame"]])
    result.records.to_csv(out / "leaves.csv", index=False)
    result.plant.to_csv(out / "plant.csv", index=False)
    result.leaves.to_csv(out / "leaf_summary.csv", index=False)
    fig = growth_dashboard(result.records, result.plant, args.images.name)
    fig.savefig(out / "dashboard.png", dpi=110)

    if not args.no_gif:
        tiles = []
        for image, leaves, h in zip(frames, result.tracked, hours, strict=True):
            drawn = draw_tracks(
                image, [x.instance.mask for x in leaves], [x.track_id for x in leaves], 400
            )
            tiles.append(with_caption(drawn, f"t = {h:g} h | {len(leaves)} leaves"))
        save_gif(tiles, out / "tracking.gif", fps=6)

    last = result.plant.iloc[-1]
    print(
        f"{len(files)} frames, {result.leaves.shape[0]} leaves tracked; last frame: "
        f"{int(last.leaf_count)} leaves, {int(last.total_area_px)} px total leaf area"
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
