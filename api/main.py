"""LeafTrack REST API (FastAPI).

Endpoints:

- ``GET /health``: liveness and model info.
- ``POST /segment``: one image -> leaves (polygon, box, area, confidence).
- ``POST /analyze``: an ordered image sequence of one plant -> tracked leaves
  per frame, per-frame plant traits and per-leaf growth summary.

Run::

    pip install -e ".[demo]"
    uvicorn api.main:app --port 8000
    curl -F "image=@plant.png" http://localhost:8000/segment

Interactive docs are served at ``/docs``. The model comes from
``LEAFTRACK_WEIGHTS`` (local .pt) or the Hub repo in ``LEAFTRACK_HUB_REPO``.
"""

from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from leaftrack import __version__
from leaftrack.demo import MAX_FRAMES, MAX_SIDE
from leaftrack.segment import LeafSegmenter

HUB_REPO = os.environ.get("LEAFTRACK_HUB_REPO", "Awais05/leaftrack-yolo11n-seg")
CONF = float(os.environ.get("LEAFTRACK_CONF", "0.30"))

app = FastAPI(
    title="LeafTrack API",
    version=__version__,
    description="Leaf instance segmentation, tracking and growth analytics for top-down "
    "plant images.",
)


@lru_cache(maxsize=1)
def get_segmenter() -> LeafSegmenter:
    """Load the model once per process (overridden in tests)."""
    weights = os.environ.get("LEAFTRACK_WEIGHTS")
    if weights:
        return LeafSegmenter(weights, device="cpu", conf=CONF)
    from leaftrack.hub import use_system_certificates

    use_system_certificates()
    return LeafSegmenter.from_hub(HUB_REPO, device="cpu", conf=CONF)


Segmenter = Annotated[LeafSegmenter, Depends(get_segmenter)]


async def read_image(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image to RGB, downscaled to at most ``MAX_SIDE`` pixels."""
    try:
        img = Image.open(io.BytesIO(await upload.read())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(415, f"{upload.filename}: not a readable image") from exc
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _clean(value: Any) -> Any:
    """JSON-safe scalars: NaN -> None, numpy types -> Python types."""
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _rows(df) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "model": os.environ.get(
        "LEAFTRACK_WEIGHTS", HUB_REPO)}


@app.post("/segment")
async def segment(image: Annotated[UploadFile, File()], segmenter: Segmenter) -> dict[str, Any]:
    """Segment the leaves in one top-down plant image."""
    rgb = await read_image(image)
    leaves = segmenter.predict(rgb)
    return {
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "count": len(leaves),
        "leaves": [
            {
                "id": k + 1,
                "confidence": round(leaf.confidence, 4),
                "area_px": leaf.area,
                "box": [round(v, 1) for v in leaf.box],
                "polygon": np.round(leaf.polygon, 1).tolist(),
            }
            for k, leaf in enumerate(leaves)
        ],
    }


@app.post("/analyze")
async def analyze(
    images: Annotated[list[UploadFile], File(description="frames of one plant, in time order")],
    segmenter: Segmenter,
    interval_hours: Annotated[float, Form(gt=0)] = 1.0,
    tracker: Annotated[Literal["bytetrack", "leaftrack"], Form()] = "bytetrack",
) -> dict[str, Any]:
    """Track the leaves of one plant over time and compute growth analytics.

    Frames are used in the order they are sent, ``interval_hours`` apart.
    """
    from leaftrack.pipeline import analyze_sequence

    if not images:
        raise HTTPException(422, "send at least one image")
    if len(images) > MAX_FRAMES:
        raise HTTPException(413, f"at most {MAX_FRAMES} frames per request")
    frames = [await read_image(f) for f in images]
    hours = [k * interval_hours for k in range(len(frames))]
    result = analyze_sequence(frames, hours, segmenter, tracker)
    return {
        "frames": len(frames),
        "tracker": tracker,
        "leaves_per_frame": _rows(result.records),
        "plant": _rows(result.plant),
        "leaf_summary": _rows(result.leaves),
    }
