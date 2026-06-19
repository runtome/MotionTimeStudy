"""
Video time-study inference pipeline.
Models are loaded once at module import; call process_video() per request.
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rfdetr import RFDETRBase
import supervision as sv

# ── Constants (match action_recognition.py exactly) ───────────────────────────
CLIP_FRAMES     = 13
FRAME_SIZE      = 182
SAMPLE_EVERY    = 5
CROP_PAD        = 0.15
MIN_CROP_PX     = 32
PERSON_CLASS_ID = 1
KINETICS_MEAN   = [0.45, 0.45, 0.45]
KINETICS_STD    = [0.225, 0.225, 0.225]

COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

_DIR = Path(__file__).parent
KINETICS_LABELS_PATH = _DIR / "kinetics400_classnames.json"
KINETICS_LABELS_URL  = (
    "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
)

TRANSITION_CONFIRM = 3  # consecutive same-label inferences required to switch segment

device = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level model references (populated by _load_models)
object_model: RFDETRBase | None = None
action_model: torch.nn.Module | None = None
box_annotator: sv.BoxAnnotator | None = None
label_annotator: sv.LabelAnnotator | None = None
kinetics_classes: dict[int, str] = {}


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_models() -> None:
    global object_model, action_model, box_annotator, label_annotator, kinetics_classes

    # Kinetics-400 class names
    if not KINETICS_LABELS_PATH.exists():
        print("Downloading Kinetics-400 class names...")
        try:
            urllib.request.urlretrieve(KINETICS_LABELS_URL, str(KINETICS_LABELS_PATH))
        except Exception as e:
            print(f"  Warning: could not download class names ({e})")

    if KINETICS_LABELS_PATH.exists():
        with open(KINETICS_LABELS_PATH) as f:
            raw = json.load(f)
        kinetics_classes = {int(v): k for k, v in raw.items()}

    print(f"Loading RF-DETR (device={device})...")
    object_model = RFDETRBase(device=device)
    object_model.optimize_for_inference()
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    print("Loading X3D-S (~200 MB download on first run)...")
    action_model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_s", pretrained=True)
    action_model.eval()
    action_model = action_model.to(device)
    print("Models ready.")


_load_models()


# ── Low-level helpers (ported from action_recognition.py) ─────────────────────
def crop_person(frame: np.ndarray, xyxy) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    pw = int((x2 - x1) * CROP_PAD)
    ph = int((y2 - y1) * CROP_PAD)
    x1 = max(0, x1 - pw);  y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw);  y2 = min(h, y2 + ph)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.shape[0] >= MIN_CROP_PX and crop.shape[1] >= MIN_CROP_PX else None


def preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE))
    t = torch.from_numpy(resized).float() / 255.0
    t = t.permute(2, 0, 1)
    for c in range(3):
        t[c] = (t[c] - KINETICS_MEAN[c]) / KINETICS_STD[c]
    return t


def run_action_model(buffer: collections.deque, top_k: int) -> list[tuple[str, float]]:
    clip = torch.stack(list(buffer))
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
    with torch.no_grad():
        logits = action_model(clip.to(device))
    probs = torch.softmax(logits[0], dim=0)
    topk  = torch.topk(probs, top_k)
    return [
        (kinetics_classes.get(idx.item(), f"action_{idx.item()}"), prob.item())
        for idx, prob in zip(topk.indices, topk.values)
    ]


def draw_action_panel(frame: np.ndarray, results: list, buffer_len: int) -> None:
    y = 100
    status = f"Buffer: {buffer_len}/{CLIP_FRAMES}" if buffer_len < CLIP_FRAMES else "Action:"
    cv2.putText(frame, status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += 22
    for label, conf in results:
        bar_w = int(conf * 150)
        cv2.rectangle(frame, (10, y - 14), (10 + bar_w, y + 2), (0, 180, 255), -1)
        cv2.putText(frame, f"{label}  {conf:.2f}", (14, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def draw_time_accumulator(frame: np.ndarray, time_accum: dict) -> None:
    if not time_accum:
        return
    h, w = frame.shape[:2]
    top5 = sorted(time_accum.items(), key=lambda x: x[1], reverse=True)[:5]
    row_h   = 20
    panel_h = row_h * (len(top5) + 1) + 10
    x_off   = w - 280
    y_start = h - panel_h - 5
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_off - 5, y_start - 5), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    y = y_start + row_h
    cv2.putText(frame, "Accumulated Time", (x_off, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    y += row_h
    for label, secs in top5:
        short = label[:24]
        cv2.putText(frame, f"{short}: {secs:.1f}s", (x_off, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
        y += row_h


# ── Segment tracking ─────────────────────────────────────────────────────────
def update_segments(
    segments: list[dict],
    current_segment: dict | None,
    new_top1: str,
    video_ts: float,
) -> tuple[list[dict], dict | None]:
    if current_segment is None:
        return segments, {"activity": new_top1, "start": video_ts}

    if new_top1 == current_segment["activity"]:
        return segments, current_segment

    # Label changed — close old segment, open new
    closed = {
        "activity": current_segment["activity"],
        "start":    current_segment["start"],
        "end":      video_ts,
        "duration": round(video_ts - current_segment["start"], 3),
    }
    segments = segments + [closed]
    return segments, {"activity": new_top1, "start": video_ts}


def close_last_segment(
    segments: list[dict],
    current_segment: dict | None,
    video_ts: float,
) -> list[dict]:
    if current_segment is None:
        return segments
    closed = {
        "activity": current_segment["activity"],
        "start":    current_segment["start"],
        "end":      video_ts,
        "duration": round(video_ts - current_segment["start"], 3),
    }
    return segments + [closed]


# ── Main processing function ──────────────────────────────────────────────────
def process_video(
    input_path: str,
    threshold: float = 0.5,
    top_k: int = 3,
    progress_callback=None,
) -> tuple[str, list[dict]]:
    """
    Process a video file through RF-DETR → X3D-S pipeline.

    Returns (annotated_video_path, segments) where segments is a list of dicts
    with keys: activity, start, end, duration (all in seconds of the source video).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    writer = cv2.VideoWriter(
        tmp_out.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (frame_w, frame_h),
    )

    frame_buffer    = collections.deque(maxlen=CLIP_FRAMES)
    action_labels: list[tuple[str, float]] = []
    time_accum: dict[str, float]           = collections.defaultdict(float)
    segments: list[dict]                   = []
    current_segment: dict | None           = None
    top1_history                           = collections.deque(maxlen=TRANSITION_CONFIRM)
    confirmed_top1: str                    = ""
    frame_count                            = 0
    video_ts                               = 0.0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            video_ts = frame_count / video_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ── RF-DETR detection ──────────────────────────────────────────────
            detections = object_model.predict(Image.fromarray(rgb), threshold=threshold)

            primary_xyxy = None
            primary_conf = -1.0

            if len(detections) > 0:
                coco_labels = [
                    f"{COCO_CLASSES.get(int(cid), f'cls{cid}')} {conf:.2f}"
                    for cid, conf in zip(detections.class_id, detections.confidence)
                ]
                frame = box_annotator.annotate(frame, detections)
                frame = label_annotator.annotate(frame, detections, labels=coco_labels)

                for cid, conf, xyxy in zip(
                    detections.class_id, detections.confidence, detections.xyxy
                ):
                    if int(cid) == PERSON_CLASS_ID and float(conf) > primary_conf:
                        primary_conf = float(conf)
                        primary_xyxy = xyxy

            if primary_xyxy is not None:
                x1, y1, x2, y2 = (int(v) for v in primary_xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            # ── Build clip buffer ──────────────────────────────────────────────
            new_sample = False
            if frame_count % SAMPLE_EVERY == 0 and primary_xyxy is not None:
                crop = crop_person(frame, primary_xyxy)
                if crop is not None:
                    frame_buffer.append(preprocess_crop(crop))
                    new_sample = True

            # ── Run X3D-S when buffer full ─────────────────────────────────────
            if new_sample and len(frame_buffer) == CLIP_FRAMES:
                action_labels = run_action_model(frame_buffer, top_k)
                new_top1 = action_labels[0][0] if action_labels else ""

                # Accumulate wall-video time for the HUD panel
                if confirmed_top1:
                    time_accum[confirmed_top1] += SAMPLE_EVERY / video_fps

                # Debounce: only act on label when N consecutive agree
                top1_history.append(new_top1)
                if (
                    len(top1_history) == TRANSITION_CONFIRM
                    and len(set(top1_history)) == 1
                    and new_top1 != confirmed_top1
                ):
                    confirmed_top1 = new_top1
                    segments, current_segment = update_segments(
                        segments, current_segment, confirmed_top1, video_ts
                    )

            # ── HUD ───────────────────────────────────────────────────────────
            cv2.putText(frame, f"FPS: {video_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Device: {device.upper()}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            draw_action_panel(frame, action_labels, len(frame_buffer))
            draw_time_accumulator(frame, time_accum)

            writer.write(frame)

            if progress_callback and frame_count % 30 == 0:
                progress_callback(frame_count / total_frames, f"Frame {frame_count}/{total_frames}")

    finally:
        writer.release()
        cap.release()

    segments = close_last_segment(segments, current_segment, video_ts)
    return _remux_h264(tmp_out.name), segments


def _remux_h264(src: str) -> str:
    """Re-encode src video to H.264 MP4 for browser compatibility. Falls back to src on failure."""
    dst = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    dst.close()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src,
                "-vcodec", "libx264", "-crf", "23", "-preset", "fast",
                "-movflags", "+faststart",
                "-an",   # no audio track (source has none)
                dst.name,
            ],
            check=True,
            capture_output=True,
        )
        os.unlink(src)
        return dst.name
    except Exception as e:
        print(f"ffmpeg re-encode skipped ({e}); using original mp4v file")
        try:
            os.unlink(dst.name)
        except OSError:
            pass
        return src


# ── Output helpers ────────────────────────────────────────────────────────────
def build_dataframe(segments: list[dict]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame(columns=["Activity", "Start (s)", "End (s)", "Duration (s)"])
    rows = [
        {
            "Activity":    s["activity"],
            "Start (s)":   round(s["start"], 2),
            "End (s)":     round(s["end"], 2),
            "Duration (s)": round(s["duration"], 2),
        }
        for s in segments
    ]
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def save_excel(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    df.to_excel(tmp.name, index=False, engine="openpyxl")
    return tmp.name


def build_gantt_chart(segments: list[dict]):
    if not segments:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No segments detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")
        return fig

    unique_activities = list(dict.fromkeys(s["activity"] for s in segments))
    n = len(unique_activities)
    cmap = matplotlib.colormaps["tab20"]
    color_map = {act: cmap((i % 20) / 20) for i, act in enumerate(unique_activities)}

    fig, ax = plt.subplots(figsize=(12, max(3, n * 0.65 + 1.5)))

    for seg in segments:
        y_pos = unique_activities.index(seg["activity"])
        ax.barh(
            y=y_pos,
            width=seg["duration"],
            left=seg["start"],
            height=0.5,
            color=color_map[seg["activity"]],
            edgecolor="white",
            linewidth=0.5,
        )
        # Label the bar with its duration if wide enough
        if seg["duration"] > 0.3:
            ax.text(
                seg["start"] + seg["duration"] / 2,
                y_pos,
                f"{seg['duration']:.1f}s",
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

    ax.set_yticks(range(n))
    ax.set_yticklabels(unique_activities, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_title("Activity Timeline", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    patches = [
        mpatches.Patch(color=color_map[act], label=act)
        for act in unique_activities
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, framealpha=0.8, title="Activities")

    plt.tight_layout()
    return fig


def upload_to_hf(
    paths: list[str],
    filenames: list[str],
    repo_id: str = "suphot/motion-study",
) -> list[str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping HF upload.")
        return []

    from huggingface_hub import HfApi
    api = HfApi()
    ts  = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    urls = []

    for local_path, remote_name in zip(paths, filenames):
        path_in_repo = f"uploads/{ts}/{remote_name}"
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            urls.append(
                f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"
            )
            print(f"Uploaded → {path_in_repo}")
        except Exception as e:
            print(f"Upload failed for {remote_name}: {e}")

    return urls
