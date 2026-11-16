# LeafTrack

**Leaf instance segmentation, tracking, and growth analytics for top-down plant time-lapse imagery.**

> Work in progress: segmentation and tracking are trained and evaluated. Growth analytics and the live Hugging Face demo come next.

![Leaf tracking over 10 days on two held-out test plants](docs/assets/tracking_demo.gif)

*Two test plants the model never saw in training, one image every 4 hours for 10 days. Each leaf keeps its color and number for as long as it is visible. Left: multi-view rig; right: RGB-D rig.*

## What it does

1. **Segment** every leaf in a top-down plant image with a YOLO11-seg model.
2. **Track** each leaf across a time-lapse sequence so it keeps a stable ID.
3. **Analyze** growth: leaf count, per-leaf area, and growth rate over time.

## Project structure

```
leaftrack/        Core package: segment, metrics, evaluation, training, hub, track, analytics
data/scripts/     Dataset download, conversion (to YOLO-seg), splitting, visual checks
configs/          Training configs and the committed plant-level split
train.py          Training entry point (used by Colab/Kaggle notebooks)
evaluate.py       Segmentation evaluation and results table
evaluate_tracking.py  Tracking evaluation, baselines and the demo GIF
api/              FastAPI service
app.py            Gradio demo (Hugging Face Spaces)
notebooks/        Thin GPU launcher notebooks
tests/            pytest suite
```

## Dataset

[KOMATSUNA](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) (Uchiyama et al., 2017): 1,200 labelled top-down frames of 10 plants over 10 days, with leaf identities that stay consistent over time. That makes it a fit for both segmentation and tracking evaluation.

The data is converted to YOLO-seg polygons and split **by plant, not by frame**, so no plant seen in training appears in validation or test. See [data/README.md](data/README.md) for the verified data format, conversion quality and split. Datasets and model weights are never committed.

## Results

Test split: 240 images of 2 held-out plants (one per capture rig).

| Model | Params | Mask mAP50 | Mask mAP50-95 | SBD ↑ | FBD ↑ | \|DiC\| ↓ | Exact leaf count | CPU latency* |
|---|---|---|---|---|---|---|---|---|
| **[YOLO11n-seg](https://huggingface.co/Awais05/leaftrack-yolo11n-seg)** (demo model) | 2.8 M | **0.987** | **0.825** | 0.930 | 0.970 | 0.15 | 85% | **241 ms** |
| [YOLO11s-seg](https://huggingface.co/Awais05/leaftrack-yolo11s-seg) | 10.1 M | 0.965 | 0.806 | **0.934** | **0.971** | **0.15** | **87%** | 523 ms |

\* Median per image at 640 px on 2 CPU threads, matching the free Hugging Face Spaces tier.

- **The nano model is the right choice for the demo.** It matches the small model within 0.004 SBD and is 2.2× faster on CPU.
- **SBD / FBD / DiC** are the [CVPPP leaf segmentation](https://link.springer.com/article/10.1007/s00138-015-0737-3) metrics. SBD measures how well each leaf is segmented, FBD how well the plant is separated from the background, and |DiC| the error in leaf count.
- **How the leaf metrics are computed:** with post-processing chosen on the **validation** plants (confidence threshold, and NMS IoU 0.5), so the test plants stay unseen, scored against the original color labels.

### What made the difference

COCO mAP looked excellent from the start, but leaf counting was off by 1.4 leaves per image. The cause was not the model: multi-view crops contain **leaves of neighbouring plants** at the image edge, which the labels correctly ignore. LeafTrack drops edge leaves that never reach toward the plant centre. On the ground truth this rule removes 0 of 5,820 true leaves.

| YOLO11n, test split | SBD ↑ | \|DiC\| ↓ | Exact leaf count |
|---|---|---|---|
| Ultralytics default (conf 0.25) | 0.746 | 1.38 | 26% |
| + threshold selected on validation (0.30) | 0.756 | 1.27 | 28% |
| + neighbour-leaf filter | **0.930** | **0.15** | **85%** |

Full reports, including per-rig breakdowns, are written by `evaluate.py` (`metrics.json`, `results.md`, `per_image.csv`).

![YOLO11n-seg predictions on held-out test plants](docs/assets/yolo11n_test_predictions.jpg)

*Segmentation on test plants, early to late growth: image, prediction, ground truth.*

## Leaf tracking

Every leaf gets a stable ID across the time-lapse. The default tracker is **ByteTrack** (Ultralytics' implementation): it matches boxes with a Kalman filter in two stages, and its second stage uses *low-confidence* detections to continue existing tracks, which keeps a partly covered leaf's ID while its segmentation flickers. On 2 CPU threads it adds about 3 ms per frame to segmentation (237 → 240 ms).

```python
from leaftrack.pipeline import track_frames
from leaftrack.segment import LeafSegmenter

segmenter = LeafSegmenter.from_hub("Awais05/leaftrack-yolo11n-seg", conf=0.30)
for leaves in track_frames(frames, segmenter):  # time-ordered RGB images of one plant
    for leaf in leaves:
        print(leaf.track_id, leaf.instance.area)
```

LeafTrack also includes its own tracker, `LeafTracker` (`track_frames(..., tracker="leaftrack")`). Frames are 4 hours apart, so it skips motion modelling and matches each detection to the existing tracks by **mask overlap** with the Hungarian algorithm. The cost combines IoU with overlap of the smaller mask, so a whole leaf wins over a fragment of it, and a track survives up to 10 missed frames.

Test split: 4 sequences (3 camera views of one plant, one RGB-D plant), 240 frames, 28 leaves. The ground-truth IDs are KOMATSUNA's per-leaf label colors. `LeafTracker` settings were selected on the validation plants.

| Method (YOLO11n masks) | IDF1 ↑ | MOTA ↑ | ID switches ↓ | Mostly tracked ↑ | Val IDF1 |
|---|---|---|---|---|---|
| Per-frame IDs (no tracking) | 0.323 | 0.678 | 309 | 4% | — |
| LeafTracker without track memory | 0.945 | 0.950 | 13 | 93% | — |
| LeafTracker (mask overlap) | 0.949 | **0.952** | 11 | 96% | 0.886 |
| **ByteTrack (default)** | **0.968** | **0.952** | **6** | 96% | 0.886 |
| LeafTracker on ground-truth masks | 0.973 | 0.988 | 13 | 96% | — |

- **IDF1** measures how consistently each leaf keeps one ID over its whole life; **MOTA** combines misses, false detections and ID switches; **mostly tracked** is the share of leaves followed by one ID for at least 80% of their frames.
- **Why ByteTrack is the default:** the two trackers tie on validation (IDF1 0.886). On test, ByteTrack makes about half as many ID switches (6 vs 11) and is easy to swap for any Ultralytics tracker.
- **Tracking quality is now limited by segmentation.** Given ground-truth masks, `LeafTracker` reaches 0.973 IDF1. The ID switches left there happen when a leaf changes shape faster than a 4-hour interval can follow, such as an old seed leaf (cotyledon) being covered by new growth.
- **Approaches that did not help on validation:** a centroid-distance fallback for low-overlap pairs, and a ByteTrack-style low-confidence pass inside `LeafTracker`.

![Tracked leaves at five time points of one test plant](docs/assets/tracking_filmstrip.jpg)

Reports (`metrics.json`, `results.md`, `per_sequence.csv`), the GIF and this filmstrip are written by:

```bash
python evaluate_tracking.py --weights runs/segment/yolo11n/weights/best.pt --device 0
```

## Training

Training runs on a free T4 via the launcher notebooks ([Colab](notebooks/train_colab.ipynb), [Kaggle](notebooks/train_kaggle.ipynb)) or on any local GPU:

```bash
pip install -e ".[train]"
python data/scripts/download_komatsuna.py     # ~125 MB, resumable
python data/scripts/convert_komatsuna.py
python data/scripts/split_dataset.py
python train.py --config configs/train_yolo11n.yaml --project runs/segment --name yolo11n
python evaluate.py --weights runs/segment/yolo11n/weights/best.pt --device 0
```

- **Recipe:** pretrained YOLO11-seg, 640 px, cosine LR, early stopping, plus full-rotation and vertical-flip augmentation (top-down plants have no canonical orientation).
- **Checkpoints:** written every 10 epochs, and `--resume` continues an interrupted run from `last.pt`.
- **These results** come from a laptop RTX 3050 (6 GB). YOLO11n trained with a 150-epoch budget, early-stopped at epoch 116 (best epoch 73), in 42 minutes. YOLO11s trained for 80 epochs at batch 8 in 53 minutes. The configs now default to 80 epochs with patience 20.

## Pretrained weights

Both models are on the Hugging Face Hub, each with its test metrics and a model card:
[`Awais05/leaftrack-yolo11n-seg`](https://huggingface.co/Awais05/leaftrack-yolo11n-seg) and
[`Awais05/leaftrack-yolo11s-seg`](https://huggingface.co/Awais05/leaftrack-yolo11s-seg).

```python
from leaftrack.segment import LeafSegmenter

segmenter = LeafSegmenter.from_hub("Awais05/leaftrack-yolo11n-seg", conf=0.30)
leaves = segmenter.predict(rgb_image)  # list of LeafInstance: mask, box, confidence, area
```

`conf=0.30` is the threshold selected on the validation plants. Weights are published with `python -m leaftrack.hub` and never committed to git.

## Development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
