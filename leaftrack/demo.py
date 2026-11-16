"""Demo logic shared by the Gradio app and the REST API (no UI code here).

Example sequences are **downloaded at startup from the official KOMATSUNA
server**, never stored in this repository or the Space: the dataset page
grants no redistribution licence. Only the 9 MB RGB-D archive is fetched,
and only plants the model never saw in training are offered.
"""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from leaftrack.datasets.download import KOMATSUNA_FILES, KOMATSUNA_URL, download
from leaftrack.pipeline import GrowthAnalysis, analyze_sequence
from leaftrack.segment import LeafSegmenter
from leaftrack.viz import draw_tracks, save_gif, with_caption

#: Example plants: label -> KOMATSUNA RGB-D plant number (held out from training).
EXAMPLES: dict[str, int] = {
    "KOMATSUNA plant 01, test split (unseen)": 1,
    "KOMATSUNA plant 04, validation split (unseen)": 4,
}
EXAMPLE_INTERVAL_HOURS = 4.0  # KOMATSUNA captures every 4 hours
MAX_FRAMES = 120
MAX_SIDE = 1024  # larger uploads are downscaled to bound CPU time and memory
RENDER_SIZE = 360


def fetch_example(plant: int, cache_dir: Path) -> list[Path]:
    """Download (once) the RGB-D archive and return one plant's frames in time order.

    Raises:
        RuntimeError: if the KOMATSUNA server cannot be reached.
    """
    cache_dir = Path(cache_dir)
    archive = download(
        KOMATSUNA_URL + "rgbd_plant.zip",
        cache_dir / "rgbd_plant.zip",
        KOMATSUNA_FILES["rgbd_plant.zip"],
    )
    out_dir = cache_dir / f"rgbd_plant_{plant:02d}"
    prefix = f"rgbd_plant/rgb_{plant:02d}_"
    if not out_dir.exists():
        tmp = cache_dir / f".tmp_{plant:02d}"
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = Path(name).name
                if name.startswith(prefix) and base.endswith(".png") and "/" not in base:
                    (tmp / base).write_bytes(z.read(name))
        tmp.rename(out_dir)
    return sorted(out_dir.glob("*.png"))


def load_frames(paths: Sequence[str | Path], max_side: int = MAX_SIDE) -> list[np.ndarray]:
    """Read images as RGB arrays, downscaling any whose longer side exceeds ``max_side``."""
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        frames.append(np.asarray(img))
    return frames


def check_sequence(paths: Sequence[str | Path]) -> list[Path]:
    """Validate an uploaded sequence and return it sorted by file name."""
    files = sorted((Path(p) for p in paths), key=lambda p: p.name)
    if not files:
        raise ValueError("Upload at least one image.")
    if len(files) > MAX_FRAMES:
        raise ValueError(f"At most {MAX_FRAMES} frames per run on the free CPU tier.")
    return files


@dataclass
class TimelapseResult:
    """Everything the UI shows for one time-lapse."""

    analysis: GrowthAnalysis
    gif_path: Path
    csv_paths: list[Path]
    summary: str


def render_gif(
    frames: Sequence[np.ndarray], analysis: GrowthAnalysis, hours: Sequence[float], path: Path
) -> Path:
    """Animated overlay: every leaf keeps its color (track ID) across frames."""
    tiles = []
    for image, leaves, h in zip(frames, analysis.tracked, hours, strict=True):
        drawn = draw_tracks(
            image, [x.instance.mask for x in leaves], [x.track_id for x in leaves], RENDER_SIZE
        )
        tiles.append(with_caption(drawn, f"day {h / 24:.1f} | {len(leaves)} leaves"))
    save_gif(tiles, path, fps=6)
    return path


def summarize(analysis: GrowthAnalysis) -> str:
    """Short Markdown summary of a growth analysis."""
    plant = analysis.plant
    if plant.empty or analysis.records.empty:
        return "No leaves were detected. Is this a top-down image of a small plant?"
    first, last = plant.iloc[0], plant.iloc[-1]
    span_days = (last.hours - first.hours) / 24
    lines = [
        f"**{len(plant)} frames over {span_days:.1f} days, "
        f"{analysis.leaves.shape[0]} leaves tracked.**",
        "",
        f"- Leaf count: {int(first.leaf_count)} → {int(last.leaf_count)}",
        f"- Total leaf area: {int(first.total_area_px):,} → {int(last.total_area_px):,} px",
    ]
    rgr = plant["rgr_per_day"].dropna()
    if len(rgr):
        lines.append(
            f"- Plant growth rate (24 h window): median {rgr.median():.2f} per day "
            f"(area ×{np.exp(rgr.median()):.2f} per day)"
        )
    return "\n".join(lines)


def finish_timelapse(
    frames: Sequence[np.ndarray],
    hours: Sequence[float],
    analysis: GrowthAnalysis,
    out_dir: Path | None = None,
) -> TimelapseResult:
    """Write the GIF and CSV downloads for a finished analysis (CPU only)."""
    out = Path(out_dir or tempfile.mkdtemp(prefix="leaftrack_"))
    out.mkdir(parents=True, exist_ok=True)
    gif = render_gif(frames, analysis, hours, out / "tracking.gif")
    csvs = []
    for name, df in [
        ("leaves.csv", analysis.records),
        ("plant.csv", analysis.plant),
        ("leaf_summary.csv", analysis.leaves),
    ]:
        df.to_csv(out / name, index=False)
        csvs.append(out / name)
    return TimelapseResult(analysis, gif, csvs, summarize(analysis))


def run_timelapse(
    frames: Sequence[np.ndarray],
    hours: Sequence[float],
    segmenter: LeafSegmenter,
    tracker: str = "bytetrack",
    out_dir: Path | None = None,
) -> TimelapseResult:
    """Analyze a sequence and write the GIF and CSV downloads to ``out_dir``."""
    analysis = analyze_sequence(frames, hours, segmenter, tracker)  # type: ignore[arg-type]
    return finish_timelapse(frames, hours, analysis, out_dir)


def leaf_table(analysis: GrowthAnalysis) -> pd.DataFrame:
    """Per-leaf summary rounded and labelled for display."""
    df = analysis.leaves.copy()
    if df.empty:
        return df
    df["first_seen_day"] = (df.pop("first_seen_h") / 24).round(1)
    df["last_seen_day"] = (df.pop("last_seen_h") / 24).round(1)
    df["early_rgr_per_day"] = df["early_rgr_per_day"].round(2)
    df = df.drop(columns=["early_agr_px_per_h"])
    return df.rename(columns={"track_id": "leaf"})[
        ["leaf", "first_seen_day", "last_seen_day", "frames", "max_area_px", "final_area_px",
         "early_rgr_per_day"]
    ]  # fmt: skip


def run_single(image: np.ndarray, segmenter: LeafSegmenter) -> tuple[np.ndarray, pd.DataFrame]:
    """Segment one image: overlay with numbered leaves and a per-leaf table."""
    leaves = segmenter.predict(image)
    ids = list(range(1, len(leaves) + 1))
    overlay = draw_tracks(image, [x.mask for x in leaves], ids, max(RENDER_SIZE, *image.shape[:2]))
    table = pd.DataFrame(
        {
            "leaf": ids,
            "area_px": [x.area for x in leaves],
            "confidence": [round(x.confidence, 3) for x in leaves],
        }
    )
    return overlay, table
