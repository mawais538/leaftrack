"""LeafTrack Gradio demo (Hugging Face Spaces, free CPU tier).

- **Time-lapse**: pick an example plant or upload frames of one plant; get a
  tracking animation with stable leaf colors, growth charts, a per-leaf table
  and CSV downloads.
- **Single image**: segment and count the leaves in one top-down image.

Run locally::

    pip install -e ".[demo]"
    python app.py

Environment variables: ``LEAFTRACK_WEIGHTS`` (local .pt, otherwise the model is
downloaded from ``LEAFTRACK_HUB_REPO``), ``LEAFTRACK_CACHE`` (example cache).

On a Hugging Face ZeroGPU Space only segmentation + tracking runs on the GPU
(``@spaces.GPU``); rendering and tables stay on CPU to save visitors' GPU
quota. Elsewhere the decorator is a no-op and everything runs on CPU.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt

from leaftrack.analytics import growth_dashboard
from leaftrack.demo import (
    EXAMPLE_INTERVAL_HOURS,
    EXAMPLES,
    check_sequence,
    fetch_example,
    finish_timelapse,
    leaf_table,
    load_frames,
    run_single,
)
from leaftrack.hub import use_system_certificates
from leaftrack.pipeline import analyze_sequence
from leaftrack.segment import LeafSegmenter

try:  # Hugging Face ZeroGPU; a no-op decorator everywhere else
    import spaces

    gpu = spaces.GPU
except ImportError:

    def gpu(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)


ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true"}
DEVICE = "cuda" if ZERO_GPU else "cpu"

HUB_REPO = os.environ.get("LEAFTRACK_HUB_REPO", "Awais05/leaftrack-yolo11n-seg")
CONF = 0.30  # selected on the validation plants for YOLO11n
CACHE = Path(os.environ.get("LEAFTRACK_CACHE", Path(tempfile.gettempdir()) / "leaftrack_cache"))
TRACKERS = {"ByteTrack (default)": "bytetrack", "LeafTracker (mask overlap)": "leaftrack"}
GITHUB = "https://github.com/mawais538/leaftrack"


def load_segmenter() -> LeafSegmenter:
    weights = os.environ.get("LEAFTRACK_WEIGHTS")
    if weights:
        seg = LeafSegmenter(weights, device=DEVICE, conf=CONF)
    else:
        use_system_certificates()
        seg = LeafSegmenter.from_hub(HUB_REPO, device=DEVICE, conf=CONF)
    if ZERO_GPU:
        seg.model.to("cuda")  # ZeroGPU: place the model on CUDA at startup
    return seg


SEGMENTER = load_segmenter()


def _gpu_seconds(frames, hours, tracker) -> int:
    return min(120, 15 + len(frames) // 2)


@gpu(duration=_gpu_seconds)
def track_and_measure(frames, hours, tracker):
    """The only step that needs the model: segmentation + tracking + analytics."""
    return analyze_sequence(frames, hours, SEGMENTER, tracker)


@gpu(duration=20)
def segment_image(image):
    return run_single(image, SEGMENTER)


def analyze(example, uploads, interval_hours, tracker_label, progress=gr.Progress()):  # noqa: B008
    """Gradio callback for the time-lapse tab."""
    if uploads:
        files = check_sequence(uploads)
        interval = float(interval_hours)
    elif example:
        progress(0, desc="Fetching example frames from the KOMATSUNA server")
        try:
            files = fetch_example(EXAMPLES[example], CACHE)
        except Exception as exc:  # network: show a message instead of a stack trace
            raise gr.Error(
                f"Could not reach the KOMATSUNA server ({exc}). Upload frames instead."
            ) from exc
        interval = EXAMPLE_INTERVAL_HOURS
    else:
        raise gr.Error("Pick an example plant or upload frames of one plant.")

    frames = load_frames(files)
    hours = [k * interval for k in range(len(frames))]
    progress(0.1, desc=f"Segmenting and tracking {len(frames)} frames")
    analysis = track_and_measure(frames, hours, TRACKERS[tracker_label])
    progress(0.9, desc="Rendering")
    result = finish_timelapse(frames, hours, analysis)
    fig = growth_dashboard(result.analysis.records, result.analysis.plant)
    plt.close(fig)
    return str(result.gif_path), fig, result.summary, leaf_table(result.analysis), [
        str(p) for p in result.csv_paths
    ]


def segment_one(image):
    """Gradio callback for the single-image tab."""
    if image is None:
        raise gr.Error("Upload an image first.")
    overlay, table = segment_image(image)
    return overlay, f"**{len(table)} leaves detected.**", table


INTRO = f"""
# LeafTrack: leaf segmentation, tracking and growth analytics

Top-down plant images in; every leaf segmented, followed over time with a stable ID, and
measured. YOLO11n-seg ([weights](https://huggingface.co/{HUB_REPO})) + ByteTrack, light enough
for a CPU. [Code and results on GitHub]({GITHUB}).
"""

ABOUT = f"""
### How it was evaluated

All numbers are on test plants the model never saw in training (KOMATSUNA dataset,
split by plant; settings chosen on separate validation plants).

| | Result |
|---|---|
| Leaf segmentation (YOLO11n-seg) | mask mAP50 0.987, leaf Dice (SBD) 0.930, exact leaf count 85% |
| Leaf tracking (ByteTrack) | IDF1 0.968, 6 ID switches in 240 frames |
| Total leaf area vs ground truth | median error 1.7% |
| Area per leaf vs ground truth | median error 2.6% |

Areas are projected (visible) areas in pixels. Plant growth rate (RGR, per day) is fitted over a
24 h window; each leaf's early growth rate over its first 48 h.

**Data.** The examples are downloaded on demand from the
[KOMATSUNA dataset](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) (Uchiyama et al., ICCV
Workshops 2017); no dataset images are stored in this Space. Details and code: [GitHub]({GITHUB}).
"""

with gr.Blocks(title="LeafTrack") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Time-lapse"):
        with gr.Row():
            with gr.Column(scale=1):
                example = gr.Dropdown(
                    list(EXAMPLES),
                    value=next(iter(EXAMPLES)),
                    label="Example plant (60 frames, 10 days)",
                )
                uploads = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="…or upload frames of one plant (sorted by file name, up to 120)",
                )
                interval = gr.Number(4, label="Hours between uploaded frames", minimum=0.01)
                tracker = gr.Radio(list(TRACKERS), value=next(iter(TRACKERS)), label="Tracker")
                run = gr.Button("Analyze growth", variant="primary")
                summary = gr.Markdown()
            with gr.Column(scale=1):
                gif = gr.Image(
                    label="Tracked leaves (color = leaf ID)", type="filepath", height=420
                )
        plot = gr.Plot(label="Growth")
        table = gr.Dataframe(label="Leaves", interactive=False, wrap=True)
        csvs = gr.File(label="Download CSV tables", file_count="multiple")
        run.click(
            analyze, [example, uploads, interval, tracker], [gif, plot, summary, table, csvs]
        )
    with gr.Tab("Single image"):
        with gr.Row():
            single_in = gr.Image(label="Top-down plant image", type="numpy")
            single_out = gr.Image(label="Segmented leaves", type="numpy")
        single_count = gr.Markdown()
        single_table = gr.Dataframe(label="Leaves", interactive=False)
        gr.Button("Segment leaves", variant="primary").click(
            segment_one, single_in, [single_out, single_count, single_table]
        )
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue(max_size=8).launch(theme=gr.themes.Soft(primary_hue="green"))
