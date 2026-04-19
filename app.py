"""
PPE Multi-Model Detection System
Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import sys
import os

import pandas as pd
import streamlit as st
from PIL import Image

# ── make sure the project root is on the path ───────────────────────── #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODEL_REGISTRY, get_model

# ══════════════════════════════════════════════════════════════════════ #
#  Page config — must be first Streamlit call                           #
# ══════════════════════════════════════════════════════════════════════ #
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════ #
#  Custom CSS — industrial / dark-panel aesthetic                        #
# ══════════════════════════════════════════════════════════════════════ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── App background ── */
    .stApp { background-color: #0d0f14; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #13161d;
        border-right: 1px solid #2a2d36;
    }
    [data-testid="stSidebar"] * { color: #c8cdd8 !important; }

    /* ── Main heading ── */
    .ppe-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.05rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #e8eaf0;
        border-bottom: 2px solid #f5a623;
        padding-bottom: 0.45rem;
        margin-bottom: 1.4rem;
    }

    /* ── Model card (sidebar) ── */
    .model-card {
        background: #1a1d26;
        border: 1px solid #2e3140;
        border-left: 3px solid #f5a623;
        border-radius: 4px;
        padding: 0.9rem 1rem;
        margin-top: 0.6rem;
        font-size: 0.82rem;
        line-height: 1.6;
    }
    .model-card .badge {
        display: inline-block;
        background: #f5a623;
        color: #0d0f14;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding: 2px 7px;
        border-radius: 2px;
        margin-bottom: 0.4rem;
    }
    .model-card .task-pill {
        display: inline-block;
        background: transparent;
        border: 1px solid #3e4255;
        color: #8a8fa8;
        font-size: 0.7rem;
        font-family: 'IBM Plex Mono', monospace;
        padding: 1px 6px;
        border-radius: 2px;
        margin-left: 6px;
    }

    /* ── Result panel ── */
    .result-panel {
        background: #13161d;
        border: 1px solid #2a2d36;
        border-radius: 6px;
        padding: 1.1rem 1.3rem;
        margin-top: 1rem;
    }
    .summary-text {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88rem;
        color: #f5a623;
        margin-bottom: 0.8rem;
    }

    /* ── Metric strip ── */
    .metric-strip {
        display: flex;
        gap: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #1a1d26;
        border: 1px solid #2e3140;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        text-align: center;
        min-width: 90px;
    }
    .metric-box .val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.45rem;
        font-weight: 600;
        color: #e8eaf0;
    }
    .metric-box .lbl {
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7080;
    }

    /* ── Confidence bar (classification) ── */
    .conf-row { display: flex; align-items: center; gap: 0.6rem; margin: 3px 0; }
    .conf-label { font-size: 0.75rem; color: #c8cdd8; min-width: 130px; }
    .conf-bar-bg { flex:1; background:#1e2130; border-radius:2px; height:7px; }
    .conf-bar-fill { height:7px; border-radius:2px; }
    .conf-pct { font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#8a8fa8; min-width:38px; text-align:right; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #2a2d36; }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px 4px 0 0;
        color: #6b7080;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        padding: 8px 18px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #1a1d26 !important;
        color: #f5a623 !important;
        border-color: #2a2d36 !important;
        border-bottom-color: #1a1d26 !important;
    }

    /* ── Dataframe text ── */
    [data-testid="stDataFrame"] { font-size: 0.78rem; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #2e3140;
        border-radius: 6px;
        background: #13161d;
        padding: 0.5rem;
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #f5a623 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════ #
#  Model loading (cached — runs once per model per session)             #
# ══════════════════════════════════════════════════════════════════════ #

@st.cache_resource(show_spinner=False)
def load_model_cached(model_name: str):
    model = get_model(model_name)
    model.load_model()
    return model


# ══════════════════════════════════════════════════════════════════════ #
#  Result rendering helpers                                             #
# ══════════════════════════════════════════════════════════════════════ #

def render_detection_results(result: dict) -> None:
    """Render annotated image + detection table for detection models."""
    detections = result["detections"]
    annotated = result["annotated_image"]

    # Summary metrics
    avg_conf = (
        sum(d["confidence"] for d in detections) / len(detections)
        if detections
        else 0.0
    )

    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="metric-box">
            <div class="val">{len(detections)}</div>
            <div class="lbl">Objects</div>
          </div>
          <div class="metric-box">
            <div class="val">{avg_conf:.0%}</div>
            <div class="lbl">Avg Conf</div>
          </div>
          <div class="metric-box">
            <div class="val">{len(set(d['label'] for d in detections))}</div>
            <div class="lbl">Unique Classes</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_img, col_table = st.columns([1.2, 1])

    with col_img:
        st.image(annotated, caption="Annotated Image", use_container_width=True)

    with col_table:
        if detections:
            df = pd.DataFrame(
                [
                    {
                        "Label": d["label"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "BBox (x1,y1,x2,y2)": str(d["bbox"]),
                    }
                    for d in sorted(detections, key=lambda x: -x["confidence"])
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download annotated image
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            st.download_button(
                "⬇ Download Annotated Image",
                data=buf.getvalue(),
                file_name="detection_result.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_{model.name}_{id(result)}",
            )
        else:
            st.info("No objects detected above the confidence threshold.")


def _conf_bar(label: str, conf: float, detected: bool) -> str:
    fill_color = "#f5a623" if detected else "#2e3140"
    width_pct = int(conf * 100)
    dot = "●" if detected else "○"
    dot_color = "#f5a623" if detected else "#3e4255"
    return (
        f'<div class="conf-row">'
        f'<span style="color:{dot_color};font-size:0.7rem;">{dot}</span>'
        f'<span class="conf-label">{label}</span>'
        f'<div class="conf-bar-bg"><div class="conf-bar-fill" '
        f'style="width:{width_pct}%;background:{fill_color};"></div></div>'
        f'<span class="conf-pct">{conf:.0%}</span>'
        f"</div>"
    )


def render_classification_results(result: dict) -> None:
    """Render confidence bars for classification models."""
    detections = result["detections"]

    detected = [d for d in detections if d["detected"]]
    not_detected = [d for d in detections if not d["detected"]]

    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="metric-box">
            <div class="val">{len(detected)}</div>
            <div class="lbl">Detected</div>
          </div>
          <div class="metric-box">
            <div class="val">{len(not_detected)}</div>
            <div class="lbl">Not Detected</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_img, col_bars = st.columns([1, 1.1])

    with col_img:
        st.image(result["annotated_image"], caption="Input Image", use_container_width=True)

    with col_bars:
        bars_html = "".join(
            _conf_bar(d["label"], d["confidence"], d["detected"])
            for d in sorted(detections, key=lambda x: -x["confidence"])
        )
        st.markdown(bars_html, unsafe_allow_html=True)


def run_inference(image: Image.Image, model) -> None:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    with st.spinner("Running inference…"):
        result = model.predict(image)

    st.markdown(
        f'<div class="summary-text">▶ {result["summary"]}</div>',
        unsafe_allow_html=True,
    )

    if model.task_type == "detection":
        render_detection_results(result)
    else:
        render_classification_results(result)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════ #
#  Sidebar                                                              #
# ══════════════════════════════════════════════════════════════════════ #

with st.sidebar:
    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
        'letter-spacing:0.18em;text-transform:uppercase;color:#f5a623;">'
        "PPE DETECTION SYSTEM</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        '<p style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;'
        'color:#6b7080;margin-bottom:0.3rem;">SELECT MODEL</p>',
        unsafe_allow_html=True,
    )

    model_options = list(MODEL_REGISTRY.keys())
    selected_name = st.selectbox(
        "model",
        options=model_options,
        format_func=lambda k: f"{MODEL_REGISTRY[k]['icon']}  {k}",
        label_visibility="collapsed",
    )

    info = MODEL_REGISTRY[selected_name]
    st.markdown(
        f"""
        <div class="model-card">
          <span class="badge">{info['badge']}</span>
          <span class="task-pill">{info['task']}</span>
          <br/>
          <span style="color:#e8eaf0;">{info['description']}</span>
          <br/><br/>
          <span style="color:#6b7080;font-size:0.7rem;">CLASSES</span>
          &nbsp;
          <span style="font-family:'IBM Plex Mono',monospace;color:#f5a623;">
            {info['num_classes']}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Load model
    with st.spinner(f"Loading {selected_name}…"):
        try:
            model = load_model_cached(selected_name)
            st.markdown(
                '<p style="color:#4caf82;font-size:0.76rem;font-family:\'IBM Plex Mono\','
                f'monospace;">✓ {selected_name} ready</p>',
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            st.error(f"Weights file not found:\n`{info['weights']}`")
            model = None
        except Exception as exc:
            st.error(f"Load error:\n{exc}")
            model = None

    st.markdown(
        '<p style="font-size:0.68rem;color:#3e4255;margin-top:1rem;">'
        "Models are cached after first load.</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════ #
#  Main content                                                         #
# ══════════════════════════════════════════════════════════════════════ #

st.markdown(
    '<p class="ppe-header">Personal Protective Equipment — Detection System</p>',
    unsafe_allow_html=True,
)

tab_upload, tab_camera = st.tabs(["📁  UPLOAD IMAGE", "📷  LIVE CAMERA"])

# ── Upload tab ──────────────────────────────────────────────────────── #
with tab_upload:
    st.markdown(
        '<p style="font-size:0.78rem;color:#6b7080;margin-bottom:0.6rem;">'
        "Supported formats: JPG, JPEG, PNG, BMP, WEBP</p>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        if model is not None:
            run_inference(image, model)
        else:
            st.warning("⚠️ Model failed to load — check the sidebar for details.")
    else:
        st.markdown(
            """
            <div style="border:1px dashed #2a2d36;border-radius:6px;padding:2.5rem;
                        text-align:center;margin-top:1rem;">
              <p style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                         color:#3e4255;letter-spacing:0.08em;">
                NO IMAGE LOADED
              </p>
              <p style="font-size:0.75rem;color:#2e3140;">
                Upload an image to begin inference
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Camera tab ──────────────────────────────────────────────────────── #
with tab_camera:
    st.markdown(
        '<p style="font-size:0.78rem;color:#6b7080;margin-bottom:0.4rem;">'
        "Click <strong style='color:#c8cdd8;'>Take Photo</strong> to capture a frame "
        "and run inference on it.</p>",
        unsafe_allow_html=True,
    )
    camera_frame = st.camera_input(
        "Capture",
        label_visibility="collapsed",
    )

    if camera_frame is not None:
        image = Image.open(camera_frame).convert("RGB")
        if model is not None:
            run_inference(image, model)
        else:
            st.warning("⚠️ Model failed to load — check the sidebar for details.")