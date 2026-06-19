# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
streamlit run app.py
```

Requires `ffmpeg` on `PATH` and the Python deps from `requirements.txt`. GPU is auto-detected; CPU fallback is automatic.

## Architecture

The project has two files:

**`inference.py`** — pure pipeline, no UI dependency. Models are loaded **once at module import** via `_load_models()` into module-level globals (`object_model`, `action_model`, etc.). This means the first `import inference` is slow (~minutes on CPU); subsequent calls to `process_video()` are fast.

**`app.py`** — Streamlit UI that wraps `inference.py`. Key rendering constraint: **results must be rendered *after* the processing block**, not before it. Session state is populated during the processing block, so any `st.*` call that reads `st.session_state` must come after `st.session_state.update(...)` in script execution order. Placing display code above the processing block causes it to render from an empty session state on the first run.

### Pipeline inside `inference.py`

```
process_video()
  └─ per frame: RF-DETR → detect persons → crop primary person
  └─ every SAMPLE_EVERY (5) frames: append crop to deque[CLIP_FRAMES=13]
  └─ when deque full: X3D-S → top-K actions
  └─ debounce (TRANSITION_CONFIRM=3 consecutive same labels) → update segment
  └─ write annotated frame to temp mp4v file
→ _remux_h264() re-encodes to H.264 via ffmpeg (browser-compatible)
→ returns (annotated_path, segments)
```

Segments are plain dicts: `{activity, start, end, duration}` (all in seconds).

### Streamlit Cloud deployment

- `packages.txt` at the repo root installs apt packages before pip. Currently provides `libgl1` (OpenCV), `libglib2.0-0t64` (Debian trixie name), and `ffmpeg` (required for H.264 remux — without it the video saves as mp4v which browsers cannot play).
- No secrets or environment variables are required.
- The Streamlit `width="stretch"` API (not the deprecated `use_container_width=True`) is used throughout.
