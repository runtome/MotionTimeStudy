import os
import tempfile

import streamlit as st

import inference

st.set_page_config(page_title="MotionStudy — Video Time Study", layout="wide")

st.title("🏃 MotionStudy — Video Time Study Analyser")
st.markdown(
    "Upload a video to automatically detect activities, "
    "extract timestamps, and build a Gantt chart."
)

# ── Controls ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    video_file = st.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"]
    )
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05,
        help="Lower = detect more people (may include false positives)",
    )
    top_k = st.slider(
        "Top-K Actions",
        min_value=1, max_value=5, value=3, step=1,
        help="Number of top action predictions shown per clip",
    )
    run_btn = st.button(
        "Analyse Video",
        type="primary",
        use_container_width=True,
        disabled=video_file is None,
    )

with col_right:
    if video_file is not None:
        st.caption("Uploaded video")
        st.video(video_file)

# ── Processing ────────────────────────────────────────────────────────────────
if run_btn and video_file is not None:
    video_file.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    try:
        prog = st.progress(0.0, text="Analysing video...")

        annotated_path, segments = inference.process_video(
            tmp_path,
            threshold=float(threshold),
            top_k=int(top_k),
            progress_callback=lambda f, d: prog.progress(min(f * 0.85, 1.0), text=d),
        )

        if not segments:
            prog.empty()
            st.error(
                "No activity segments detected. Try lowering the Detection Threshold "
                "or use a video where a person is clearly visible."
            )
        else:
            prog.progress(0.87, text="Building outputs...")
            df         = inference.build_dataframe(segments)
            csv_path   = inference.save_csv(df)
            excel_path = inference.save_excel(df)
            gantt_fig  = inference.build_gantt_chart(segments)

            prog.progress(1.0, text="Done.")

            with open(annotated_path, "rb") as f:
                annotated_bytes = f.read()
            with open(csv_path, "rb") as f:
                csv_bytes = f.read()
            with open(excel_path, "rb") as f:
                excel_bytes = f.read()

            st.session_state.update(
                {
                    "annotated_video": annotated_bytes,
                    "gantt_fig":       gantt_fig,
                    "df":              df,
                    "csv_bytes":       csv_bytes,
                    "excel_bytes":     excel_bytes,
                }
            )

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# ── Results (rendered after processing so session_state is current) ───────────
if "annotated_video" in st.session_state:
    st.subheader("Annotated Video")
    st.video(st.session_state["annotated_video"])

if "df" in st.session_state:
    st.subheader("Activity Gantt Chart")
    st.pyplot(st.session_state["gantt_fig"])

    st.subheader("Activity Log")
    st.dataframe(st.session_state["df"], use_container_width=True)

    col_csv, col_excel = st.columns(2)
    with col_csv:
        st.download_button(
            "Download CSV",
            data=st.session_state["csv_bytes"],
            file_name="activity_log.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_excel:
        st.download_button(
            "Download Excel",
            data=st.session_state["excel_bytes"],
            file_name="activity_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

st.markdown("---")
st.markdown(
    "**How it works:** RF-DETR detects the primary person each frame. "
    "Crops are fed into X3D-S (Kinetics-400, 13-frame clips at ~6 fps) to classify the action. "
    "Activity segments are recorded whenever the top-1 prediction changes (3-frame debounce)."
)
