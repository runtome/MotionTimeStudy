# MotionTimeStudy

A video time-study analyser that automatically detects a person's activities in a video, extracts timestamps, and generates a Gantt chart and activity log.

## How it works

1. **Person detection** — RF-DETR locates the primary person in each frame.
2. **Action recognition** — Crops are fed into X3D-S (Kinetics-400, 13-frame clips at ~6 fps) to classify the current activity.
3. **Segment tracking** — Activity segments are recorded whenever the top-1 prediction changes (3-frame debounce to reduce noise).
4. **Output** — Annotated video, Gantt chart, and a downloadable activity log (CSV / Excel).

## Live demo

Deployed on Streamlit Community Cloud:  
👉 **https://motion-time-study.streamlit.app**

## Project structure

```
MotionTimeStudy/
├── app.py              # Streamlit UI
├── inference.py        # Detection & action-recognition pipeline
├── requirements.txt    # Python dependencies
└── packages.txt        # System packages (apt-get) for Streamlit Cloud
```

## Local setup

### Prerequisites

- Python 3.10+
- ffmpeg installed and on `PATH`
- A CUDA-capable GPU is recommended (falls back to CPU automatically)

### Install

```bash
git clone https://github.com/runtome/MotionTimeStudy.git
cd MotionTimeStudy
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Usage

1. Upload a video file (MP4, AVI, MOV, MKV, or WebM).
2. Adjust **Detection Threshold** — lower values detect more people but may include false positives.
3. Adjust **Top-K Actions** — number of action predictions shown per clip.
4. Click **Analyse Video** and wait for processing to complete.
5. Review the annotated video, Gantt chart, and activity log.
6. Download results as CSV or Excel.

## Models

| Model | Purpose | Source |
|---|---|---|
| RF-DETR Base | Person detection | [`rfdetr`](https://github.com/roboflow/rf-detr) |
| X3D-S | Action recognition (Kinetics-400) | [facebookresearch/pytorchvideo](https://github.com/facebookresearch/pytorchvideo) |

## Deploy on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repository, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**.

No secrets or API tokens are required.
