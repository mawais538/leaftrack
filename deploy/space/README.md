---
title: LeafTrack
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Leaf segmentation, tracking and growth analytics
models:
  - Awais05/leaftrack-yolo11n-seg
tags:
  - plant-phenotyping
  - instance-segmentation
  - tracking
  - yolo11
---

# LeafTrack demo

Segments every leaf in top-down plant images, follows each leaf over a time-lapse with a stable
ID, and measures growth (leaf count, leaf area, growth rates). YOLO11n-seg + ByteTrack, on
ZeroGPU (the GPU is used only while your request is being analyzed).

Code, evaluation and documentation: [github.com/mawais538/leaftrack](https://github.com/mawais538/leaftrack).

Example plants are downloaded on demand from the
[KOMATSUNA dataset](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/) (Uchiyama et al., ICCV
Workshops 2017); no dataset images are stored in this Space.
