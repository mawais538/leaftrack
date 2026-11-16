"""Tests for the REST API and the demo helpers, with a fake segmenter (no model)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, get_segmenter
from leaftrack.demo import check_sequence, fetch_example, leaf_table, load_frames, run_timelapse
from tests.test_track import leaf


class CyclingSegmenter:
    """Returns two growing leaves; the k-th call gets slightly larger leaves."""

    conf = 0.3

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        k = self.calls
        return [leaf(30, 50, 6 + k), leaf(70, 50, 5 + k)]


def png_bytes(size: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((size, size, 3), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    app.dependency_overrides[get_segmenter] = CyclingSegmenter
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["version"]


def test_segment_returns_leaves(client) -> None:
    r = client.post("/segment", files={"image": ("p.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and body["width"] == 100
    assert {"id", "confidence", "area_px", "box", "polygon"} <= set(body["leaves"][0])


def test_segment_rejects_non_images(client) -> None:
    r = client.post("/segment", files={"image": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_analyze_tracks_and_measures(client) -> None:
    files = [("images", (f"{k}.png", png_bytes(), "image/png")) for k in range(4)]
    r = client.post("/analyze", files=files, data={"interval_hours": "6", "tracker": "leaftrack"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frames"] == 4
    assert [p["leaf_count"] for p in body["plant"]] == [2, 2, 2, 2]
    assert [p["hours"] for p in body["plant"]] == [0, 6, 12, 18]
    assert {s["track_id"] for s in body["leaf_summary"]} == {1, 2}
    assert body["plant"][0]["rgr_per_day"] is None  # NaN serialized as null


def test_analyze_validates_tracker(client) -> None:
    files = [("images", ("0.png", png_bytes(), "image/png"))]
    assert client.post("/analyze", files=files, data={"tracker": "sort"}).status_code == 422


def test_demo_helpers(tmp_path: Path) -> None:
    for name in ["b.png", "a.png"]:
        Image.fromarray(np.zeros((2000, 1000, 3), np.uint8)).save(tmp_path / name)
    files = check_sequence([tmp_path / "b.png", tmp_path / "a.png"])
    assert [f.name for f in files] == ["a.png", "b.png"]
    frames = load_frames(files)
    assert frames[0].shape == (1024, 512, 3)  # downscaled
    with pytest.raises(ValueError):
        check_sequence([])

    images = [np.zeros((100, 100, 3), np.uint8)] * 3
    result = run_timelapse(images, [0, 4, 8], CyclingSegmenter(), "leaftrack", tmp_path / "out")
    assert result.gif_path.exists() and len(result.csv_paths) == 3
    assert "3 frames" in result.summary
    assert list(leaf_table(result.analysis)["leaf"]) == [1, 2]


def test_fetch_example_extracts_one_plant(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "rgbd_plant.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for plant in (0, 1):
            for t in (1, 0):
                z.writestr(f"rgbd_plant/rgb_{plant:02d}_000_{t:02d}.png", png_bytes(10))
        z.writestr("rgbd_plant/../evil.png", b"x")  # must never be extracted
    monkeypatch.setattr("leaftrack.demo.download", lambda url, dest, size: archive)
    paths = fetch_example(1, tmp_path)
    assert [p.name for p in paths] == ["rgb_01_000_00.png", "rgb_01_000_01.png"]
    assert not (tmp_path / "evil.png").exists()
