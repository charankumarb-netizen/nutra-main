"""
NutriCheck — AI-Powered Nutrient Deficiency Detection Dashboard.
Unified, modern, single-page clinical screening dashboard with integrated AI Chatbot.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Setup python path to include src and root
ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nutricheck_nlp import NutriCheckChatbot, NutriCheckNLP  # noqa: E402
from nutricheck_nlp.context_model import load_default_classifier  # noqa: E402
from nutricheck_nlp.crosscheck import cross_check  # noqa: E402
from nutricheck_nlp.document_extractor import DocumentError  # noqa: E402
from nutricheck_nlp.explain import explain as explain_report  # noqa: E402
from nutricheck_nlp.lexicon import ANALYTES, TARGETS, reference_table  # noqa: E402
from nutricheck_nlp.timeline import build_timeline  # noqa: E402
from nutricheck_nlp.visit_prep import build_visit_prep  # noqa: E402

# -----------------------------------------------------------------------------
# Page Configuration & Healthcare Theme Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NutriCheck — AI Nutrient Deficiency Screening",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern Healthcare + AI styling
st.markdown(
    """
    <style>
    /* Global Clean Healthcare Theme */
    .stApp {
        background-color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
    }
    
    /* Header & Branding */
    .nutricheck-header {
        background: linear-gradient(135deg, #0f2b48 0%, #1e4976 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(15, 43, 72, 0.12);
    }
    .nutricheck-header h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .nutricheck-header p {
        color: #e0f2fe !important;
        font-size: 1.05rem;
        margin: 0;
        max-width: 900px;
        line-height: 1.5;
    }
    .header-pills {
        margin-top: 1rem;
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .header-pill {
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: #ffffff;
    }

    /* Cards */
    .medical-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .nutrient-card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .nutrient-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-deficient, .badge-low, .badge-high {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fca5a5;
    }
    .badge-borderline {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
    }
    .badge-normal, .badge-not-deficient {
        background-color: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .badge-not-detected, .badge-insufficient {
        background-color: #f1f5f9;
        color: #64748b;
        border: 1px solid #cbd5e1;
    }

    /* Floating Chatbot Button */
    .floating-chat-btn {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.85rem 1.6rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.4);
        cursor: pointer;
        z-index: 9999;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-decoration: none;
    }
    .floating-chat-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(2, 132, 199, 0.5);
    }

    /* Chatbot Box */
    .chat-container {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        height: 100%;
    }

    /* Hide Streamlit default technical branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Core Engine & Automatic Processing Mechanism
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_local_engine() -> NutriCheckNLP:
    """Load local NLP pipeline and context classifier once."""
    return NutriCheckNLP(classifier=load_default_classifier())


def check_remote_health(url: str, timeout: float = 0.8) -> bool:
    """Quickly probe if a remote or local FastAPI service is healthy."""
    try:
        import requests
        r = requests.get(url.rstrip("/") + "/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def get_active_backend() -> tuple[str, str, str]:
    """
    Automatic processing connection:
    1. Checks configured environment variables (NUTRICHECK_BACKEND_URL or NUTRICHECK_API_URL).
    2. Probes local FastAPI service (http://localhost:8000).
    3. Safely falls back to local in-memory engine.
    """
    env_url = os.getenv("NUTRICHECK_BACKEND_URL") or os.getenv("NUTRICHECK_API_URL")
    api_key = os.getenv("NUTRICHECK_API_KEY", "")

    if env_url and check_remote_health(env_url, timeout=1.5):
        return "remote", env_url, api_key

    if check_remote_health("http://localhost:8000", timeout=0.6):
        return "remote", "http://localhost:8000", api_key

    return "local", "", ""


def process_report_text(text: str) -> dict[str, Any]:
    """Unified text processing with automatic remote/local fallback."""
    backend_mode, api_url, api_key = get_active_backend()
    if backend_mode == "remote":
        try:
            import requests
            headers = {"X-API-Key": api_key} if api_key else {}
            r = requests.post(
                api_url.rstrip("/") + "/v1/analyze/text",
                headers=headers,
                timeout=120,
                json={"text": text},
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            # Safe internal fallback
            pass

    return get_local_engine().analyze_text(text)


def process_report_file(name: str, data: bytes, mime_type: str | None = None) -> list[dict[str, Any]]:
    """Unified file processing (PDF, DOCX, TXT, CSV, images) with automatic remote/local fallback."""
    backend_mode, api_url, api_key = get_active_backend()
    if backend_mode == "remote":
        try:
            import requests
            headers = {"X-API-Key": api_key} if api_key else {}
            r = requests.post(
                api_url.rstrip("/") + "/v1/analyze/file",
                headers=headers,
                timeout=300,
                files={"file": (name, data, mime_type or "application/octet-stream")},
            )
            if r.status_code < 400:
                return r.json()["results"]
        except Exception:
            # Safe internal fallback
            pass

    return get_local_engine().analyze_file(name, data, mime_type)


def export_csv_summary(result: dict[str, Any]) -> str:
    """Generate clean downloadable CSV summary."""
    rows = []
    for n in result.get("nutrients", []) + result.get("findings", []):
        rows.append({
            "Nutrient/Finding": n.get("name"),
            "Status": n.get("status_label", n.get("status")),
            "Confidence": f"{n.get('confidence', 1.0):.0%}" if n.get("confidence") else "N/A",
            "Reliability": n.get("reliability", "ok"),
            "Needs Review": "Yes" if n.get("needs_review") else "No",
            "Evidence": " | ".join(n.get("evidence", [])),
            "Recommendation": n.get("recommendation", ""),
        })
    return pd.DataFrame(rows).to_csv(index=False)


# -----------------------------------------------------------------------------
# Status Helpers & Mappings
# -----------------------------------------------------------------------------

def get_status_info(status: str) -> tuple[str, str]:
    """Map internal NLP status codes to clean English labels and badge CSS classes."""
    s = (status or "").lower()
    if s in ("deficient", "low"):
        return "Low", "badge-deficient"
    if s == "borderline":
        return "Borderline", "badge-borderline"
    if s in ("normal", "not_deficient"):
        return "Normal", "badge-normal"
    if s == "high":
        return "High", "badge-high"
    if s == "suspected":
        return "Insufficient information", "badge-insufficient"
    return "Not detected", "badge-not-detected"


STANDARD_REFERENCES = {
    "vitamin_b12": "Normal: 200 – 900 pg/mL (Deficient: < 200 pg/mL, Borderline: 200 – 300 pg/mL)",
    "vitamin_d": "Normal: 30 – 100 ng/mL (Deficient: < 20 ng/mL, Borderline: 20 – 30 ng/mL)",
    "iron": "Ferritin Normal: 30 – 400 ng/mL (Deficient: < 30 ng/mL, Borderline: 30 – 50 ng/mL)",
    "calcium": "Total Calcium Normal: 8.8 – 10.5 mg/dL (Deficient: < 8.5 mg/dL, Borderline: 8.5 – 8.8 mg/dL)",
    "folate": "Serum Folate Normal: 5.0 – 20.0 ng/mL (Deficient: < 4.0 ng/mL, Borderline: 4.0 – 5.0 ng/mL)",
    "anemia": "Hemoglobin Normal: 12.0 – 16.0 g/dL (Females) / 13.0 – 17.5 g/dL (Males)",
}


# -----------------------------------------------------------------------------
# SECTION 1: HEADER & INTRODUCTION
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="nutricheck-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1>🥗 NutriCheck</h1>
                <p>AI-Powered Nutrient Deficiency Detection from Medical Reports using NLP</p>
            </div>
            <div style="text-align: right;">
                <span class="header-pill" style="background: rgba(16, 185, 129, 0.25); border-color: rgba(16, 185, 129, 0.4);">
                    ● Analysis Engine Ready
                </span>
            </div>
        </div>
        <div class="header-pills">
            <span class="header-pill">Vitamin B12</span>
            <span class="header-pill">Vitamin D</span>
            <span class="header-pill">Iron & Ferritin</span>
            <span class="header-pill">Calcium</span>
            <span class="header-pill">Folate</span>
            <span class="header-pill">Hemoglobin</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Initialize Session States
# -----------------------------------------------------------------------------

if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **NutriCheck AI Assistant**. "
                "Once your medical report is analysed, I can answer questions about your exact biomarker levels, "
                "explain your results in simple English, suggest dietary changes, and prepare questions for your doctor."
            ),
        }
    ]

if "multi_reports_queue" not in st.session_state:
    st.session_state["multi_reports_queue"] = []


# -----------------------------------------------------------------------------
# SECTIONS 2, 3, 4, 5: REPORT INPUT (UPLOAD, CAMERA, MANUAL TEXT, MODE)
# -----------------------------------------------------------------------------

st.markdown('<div class="medical-card">', unsafe_allow_html=True)

# Section 5: Single / Multiple Reports Selection
col_mode_title, col_mode_radio = st.columns([1, 2])
with col_mode_title:
    st.markdown("### 📋 Medical Report Input")
with col_mode_radio:
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Single Report Analysis", "Multiple Reports Analysis"],
        horizontal=True,
        label_visibility="collapsed",
        key="analysis_mode_radio",
    )

is_multi_mode = "Multiple" in analysis_mode

# Side-by-side Upload Reports and Camera Capture
col_upload, col_camera = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Option A: Upload Reports")
    uploaded_files = st.file_uploader(
        "Select medical report files",
        type=["pdf", "docx", "txt", "csv", "jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=is_multi_mode,
        key="file_uploader",
        help="Upload PDF, Word (DOCX), Text (TXT), Lab CSV, or Scanned Images (JPG/PNG).",
    )
    if is_multi_mode:
        st.caption("ℹ️ In Multiple Reports mode, you can select several files at once from your device.")

with col_camera:
    st.markdown("#### Option B: Camera Capture")
    show_camera = st.checkbox("📷 Open Camera to Capture Report", key="show_camera_checkbox")
    camera_file = None
    if show_camera:
        try:
            camera_file = st.camera_input(
                "Position your medical report clearly in the camera frame",
                key="camera_capture_input",
                help="Point your mobile rear camera or webcam directly at the report page.",
            )
            if camera_file is not None:
                st.image(camera_file, caption="Captured Report Image", use_container_width=True)
                st.success("✅ Photo captured! It will be analysed when you click 'Analyse Report'.")
        except Exception as cam_err:
            st.warning(f"Camera access is not available on this device: {cam_err}. Please use file upload.")

# Section 4: Manual Medical Text Input
st.markdown("---")
st.markdown("#### Or paste your medical report text here")
manual_text = st.text_area(
    "Report or clinical note text",
    placeholder="Example:\nPatient: John Doe\nDate: 2025-03-12\nFerritin: 14 ng/mL\nVitamin B12: 160 pg/mL\nVitamin D (25-OH): 18 ng/mL\nHemoglobin: 11.2 g/dL\nCalcium: 9.4 mg/dL",
    height=120,
    label_visibility="collapsed",
    key="manual_text_area",
)

# Manage Multi-Reports Queue if in Multi-Mode
if is_multi_mode:
    st.markdown("##### 📚 Selected Reports for Comparison")
    st.info("⚠️ **Patient Safeguard**: Please only compare reports belonging to the **same individual**. NutriCheck automatically checks patient consistency.")
    
    # Add manual text as a report to queue if provided
    c_add1, c_add2 = st.columns([3, 1])
    with c_add1:
        new_report_label = st.text_input("Optional Report Label / Date", placeholder="e.g. Visit 1 (Jan 2025)", key="new_rep_label")
    with c_add2:
        st.write("")
        st.write("")
        if st.button("➕ Add Current Text to Queue", use_container_width=True):
            if manual_text.strip():
                st.session_state["multi_reports_queue"].append({
                    "label": new_report_label or f"Report {len(st.session_state['multi_reports_queue']) + 1}",
                    "text": manual_text.strip(),
                    "source": "Manual Text",
                })
                st.success("Added report to queue!")
            else:
                st.warning("Please paste report text first to add it.")

    # Show queued reports list
    if st.session_state["multi_reports_queue"]:
        st.write("**Queued Reports for Analysis:**")
        to_delete = None
        for idx, rep in enumerate(st.session_state["multi_reports_queue"]):
            q_col1, q_col2, q_col3 = st.columns([3, 2, 1])
            q_col1.write(f"📄 **{rep['label']}** ({rep['source']})")
            q_col2.caption(rep["text"][:80] + ("..." if len(rep["text"]) > 80 else ""))
            if q_col3.button("❌ Remove", key=f"del_rep_{idx}"):
                to_delete = idx
        if to_delete is not None:
            st.session_state["multi_reports_queue"].pop(to_delete)
            st.rerun()

# Section 6: Analyse Report Button
st.markdown("---")
col_btn, col_status = st.columns([1, 2])
with col_btn:
    analyse_clicked = st.button("🔬 Analyse Report", type="primary", use_container_width=True)

with col_status:
    if st.session_state["analysis_results"]:
        st.caption("✅ Analysis results are ready below. You can also chat with the AI assistant.")

st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# EXECUTION LOGIC ON BUTTON CLICK
# -----------------------------------------------------------------------------

if analyse_clicked:
    # Determine input source
    has_uploaded = uploaded_files is not None and (
        (isinstance(uploaded_files, list) and len(uploaded_files) > 0) or
        (not isinstance(uploaded_files, list) and uploaded_files is not None)
    )
    has_camera = camera_file is not None
    has_text = bool(manual_text.strip())
    has_queue = is_multi_mode and len(st.session_state["multi_reports_queue"]) > 0

    if not (has_uploaded or has_camera or has_text or has_queue):
        st.error("Please upload a medical report file, capture a photo using the camera, or paste your report text.")
    else:
        with st.spinner("Analysing medical report with AI and clinical NLP pipeline..."):
            try:
                processed_list = []
                report_source_texts = []

                if not is_multi_mode:
                    # Single Report Analysis
                    if has_camera:
                        # Process camera photo
                        res = process_report_file("camera_capture.jpg", camera_file.getvalue(), "image/jpeg")
                        processed_list.extend(res)
                        report_source_texts.append("Camera Capture")
                    elif has_uploaded:
                        # Process single uploaded file
                        f = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
                        res = process_report_file(f.name, f.getvalue(), f.type)
                        processed_list.extend(res)
                        report_source_texts.append(f.name)
                    elif has_text:
                        # Process manual text
                        res = process_report_text(manual_text.strip())
                        processed_list.append(res)
                        report_source_texts.append(manual_text.strip())

                else:
                    # Multiple Reports Analysis
                    # 1. From files
                    if has_uploaded:
                        files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
                        for f in files:
                            res = process_report_file(f.name, f.getvalue(), f.type)
                            for r in res:
                                r["report_id"] = f.name
                                processed_list.append(r)
                            report_source_texts.append(f.name)

                    # 2. From queue
                    if has_queue:
                        for rep in st.session_state["multi_reports_queue"]:
                            r = process_report_text(rep["text"])
                            r["report_id"] = rep["label"]
                            processed_list.append(r)
                            report_source_texts.append(rep["text"])

                    # 3. From camera if also taken
                    if has_camera:
                        res = process_report_file("camera_capture.jpg", camera_file.getvalue(), "image/jpeg")
                        for r in res:
                            r["report_id"] = "Camera Capture"
                            processed_list.append(r)

                if processed_list:
                    st.session_state["analysis_results"] = {
                        "is_multi": is_multi_mode or len(processed_list) > 1,
                        "results": processed_list,
                        "primary_result": processed_list[0],
                        "source_text": "\n---\n".join(report_source_texts),
                    }
                    # Update Chatbot context
                    bot = NutriCheckChatbot(result=processed_list[0], report_text=report_source_texts[0] if report_source_texts else "")
                    st.session_state["chatbot"] = bot
                    st.success("✅ Analysis successfully completed!")
                else:
                    st.warning("No clinical text could be extracted from the provided input.")

            except DocumentError as doc_err:
                st.error(f"Document error: {doc_err.message}")
            except Exception as exc:
                st.error(f"Analysis error: {exc}")


# -----------------------------------------------------------------------------
# SECTIONS 7 & 8: SCREENING RESULTS DASHBOARD + INTEGRATED AI CHATBOT
# -----------------------------------------------------------------------------

if st.session_state.get("analysis_results"):
    data = st.session_state["analysis_results"]
    results = data["results"]
    primary = data["primary_result"]
    is_multi = data["is_multi"]

    st.markdown("---")

    # Side-by-side layout: Screening Results on the Left, AI Chatbot on the Right
    col_results, col_chatbot = st.columns([1.15, 0.85], gap="large")

    # =========================================================================
    # LEFT COLUMN: SCREENING RESULTS DASHBOARD
    # =========================================================================
    with col_results:
        st.markdown("### 📊 Nutrient Deficiency Screening Results")

        # Summary Header Card
        nutrients = primary.get("nutrients", [])
        findings = primary.get("findings", [])
        all_screened = nutrients + findings

        def_count = sum(1 for n in all_screened if n.get("status") == "deficient")
        bor_count = sum(1 for n in all_screened if n.get("status") == "borderline")
        norm_count = sum(1 for n in all_screened if n.get("status") in ("normal", "not_deficient"))

        st.markdown(
            f"""
            <div style="background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 1.2rem; margin-bottom: 1.5rem; display: flex; justify-content: space-around; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.03);">
                <div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #dc2626;">{def_count}</div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">DEFICIENT (LOW)</div>
                </div>
                <div style="border-left: 1px solid #e2e8f0;"></div>
                <div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #d97706;">{bor_count}</div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">BORDERLINE</div>
                </div>
                <div style="border-left: 1px solid #e2e8f0;"></div>
                <div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #059669;">{norm_count}</div>
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">NORMAL</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quality indicator
        quality = primary.get("quality", {})
        if quality.get("verdict") == "ok":
            st.caption("✅ **Quality Check**: All data quality checks passed.")
        else:
            st.caption(f"🔎 **Quality Note**: {quality.get('message', 'Some values need verification against original report')}")

        # Multi-record selector if batch CSV was processed
        if len(results) > 1 and not is_multi:
            rec_options = [f"Record {r.get('record_id', i+1)}" for i, r in enumerate(results)]
            selected_rec_idx = st.selectbox("Select Record to Inspect", range(len(results)), format_func=lambda i: rec_options[i])
            active_res = results[selected_rec_idx]
        else:
            active_res = primary

        # Render Individual Nutrient Cards
        active_nutrients = active_res.get("nutrients", []) + active_res.get("findings", [])

        # Match labs to nutrients
        labs_by_target: dict[str, list[dict]] = {}
        for lab in active_res.get("labs", []):
            t = lab.get("target")
            if t:
                labs_by_target.setdefault(t, []).append(lab)

        for item in active_nutrients:
            key = item.get("key")
            name = item.get("name")
            status = item.get("status", "not_assessed")
            conf = item.get("confidence")
            label_text, badge_class = get_status_info(status)

            # Get measured laboratory value and reference range
            my_labs = labs_by_target.get(key, [])
            measured_str = "Not detected in laboratory panel"
            is_measured = False
            ref_range_str = STANDARD_REFERENCES.get(key, "Standard Clinical Adult Range")

            if my_labs:
                usable_labs = [l for l in my_labs if l.get("usable")]
                if usable_labs:
                    l = usable_labs[0]
                    measured_str = f"{l.get('value')} {l.get('unit')}"
                    is_measured = True
                    if l.get("lab_reference"):
                        ref_range_str = f"Report Range: {l.get('lab_reference')}"

            # Format Card
            conf_str = f" • Confidence: {conf:.0%}" if conf else ""
            measured_badge = "🔬 Measured Lab Value" if is_measured else "📝 Clinical Text Finding"

            st.markdown(
                f"""
                <div class="nutrient-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #0f2b48; font-size: 1.15rem;">{name}</h4>
                        <span class="badge {badge_class}">{label_text}</span>
                    </div>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap; font-size: 0.9rem; margin-bottom: 0.6rem;">
                        <div><strong>Measured:</strong> <span style="color: #0369a1; font-weight: 600;">{measured_str}</span></div>
                        <div style="color: #64748b;">|</div>
                        <div style="color: #475569;"><strong>Reference:</strong> {ref_range_str}</div>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.5rem;">
                        <span>{measured_badge}</span>{conf_str}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Evidence and Explanation in clean expander
            evidence_list = item.get("evidence", [])
            recommendation = item.get("recommendation", "")
            if evidence_list or recommendation:
                with st.expander(f"Evidence & Clinical Notes for {name}"):
                    if evidence_list:
                        st.markdown("**Supporting Evidence from Report:**")
                        for ev in evidence_list:
                            st.text("• " + ev)
                    if recommendation:
                        st.markdown(f"**Recommendation:** {recommendation}")

        # Additional CBC / Hematology Laboratory Measurements (e.g. MCV / VCM, Hematocrit, MCH, MCHC)
        cbc_labs = [l for l in active_res.get("labs", []) if l.get("analyte") in ("mcv", "hematocrit", "mch", "mchc")]
        if cbc_labs:
            st.markdown("#### 🔬 Additional Complete Blood Count (CBC) Measurements")
            for cbc in cbc_labs:
                v = cbc.get("value")
                u = cbc.get("unit")
                an = cbc.get("analyte")
                st_code = cbc.get("status")

                if st_code == "high":
                    tag_badge = '<span class="badge badge-high">Elevated (High)</span>'
                elif st_code == "deficient":
                    tag_badge = '<span class="badge badge-deficient">Low</span>'
                else:
                    tag_badge = '<span class="badge badge-normal">Within Normal Limits</span>'

                ref_text = "Standard Adult Reference: 80 – 100 fL" if an == "mcv" else f"Reference: {cbc.get('canonical_unit', '')}"
                note_str = " • <i>Elevated MCV (Macrocytosis) is clinically consistent with Vitamin B12 or Folate deficiency</i>" if an == "mcv" and st_code == "high" else ""

                st.markdown(
                    f"""
                    <div class="nutrient-card" style="border-left: 4px solid #6366f1;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <h5 style="margin: 0; color: #1e1b4b; font-size: 1.05rem;">{cbc.get('label')}</h5>
                            {tag_badge}
                        </div>
                        <div style="font-size: 0.9rem; margin-bottom: 0.4rem;">
                            <strong>Measured:</strong> <span style="color: #4338ca; font-weight: 700;">{v:g} {u}</span>
                            <span style="color: #64748b; margin: 0 0.5rem;">|</span>
                            <span style="color: #475569;">{ref_text}</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #475569;">
                            <span>🔬 Additional CBC Laboratory Measurement</span>{note_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Section 7 & 8: Longitudinal Comparison for Multiple Reports
        if is_multi and len(results) > 1:
            st.markdown("---")
            st.markdown("### 📈 Longitudinal Comparison Across Reports")

            # Check if patient names are consistent
            names_detected = set()
            for r in results:
                p_text = r.get("source_text") or ""
                m = re.search(r"(?i)patient(?:\s*name)?\s*[:=]\s*([a-zA-Z\s]+)", p_text)
                if m:
                    names_detected.add(m.group(1).strip().lower())

            if len(names_detected) > 1:
                st.warning("⚠️ **Different Patients Detected**: These reports appear to belong to different individuals. Results are presented separately without merged longitudinal timeline trends.")
            else:
                tl = build_timeline(results)
                cc = cross_check(results)

                # Timeline Series
                if tl.get("series"):
                    st.markdown("#### Biomarker Trends Over Time")
                    for s in tl["series"]:
                        flag_txt = f" — {s['flag_text']}" if s.get("flag_text") else ""
                        st.write(f"**{s['label']}** ({s.get('trend', 'tracked')}){flag_txt}")
                        points_df = pd.DataFrame([
                            {"Date": p["date"] or "Report", "Value": p["canonical_value"], "Status": p["status"], "Report": p["report_id"]}
                            for p in s["points"]
                        ])
                        st.dataframe(points_df, use_container_width=True)
                        if len(points_df) >= 2:
                            st.line_chart(points_df.set_index("Date")["Value"])

                # Cross-check Contradictions
                if cc.get("issues"):
                    st.markdown("#### ⚠️ Noted Contradictions / Missing Information Across Reports")
                    for iss in cc["issues"]:
                        st.warning(f"**{iss['title']}**: {iss['detail']}")

        # Batch CSV Summary if CSV contained multiple rows
        if len(results) > 1 and not is_multi:
            st.markdown("---")
            st.markdown("### 📦 Batch Summary Table")
            batch_rows = []
            for r in results:
                row_data = {"Record ID": r.get("record_id", "1")}
                for n in r.get("nutrients", []) + r.get("findings", []):
                    row_data[n["name"]] = n.get("status_label", n.get("status"))
                batch_rows.append(row_data)
            batch_df = pd.DataFrame(batch_rows)
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "📥 Download Batch Results (CSV)",
                batch_df.to_csv(index=False),
                "nutricheck_batch_results.csv",
                "text/csv",
            )

        # Download Report Buttons
        st.markdown("---")
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button(
                "📥 Download Report Summary (CSV)",
                export_csv_summary(primary),
                "nutricheck_summary.csv",
                "text/csv",
                use_container_width=True,
            )
        with c_dl2:
            st.download_button(
                "📥 Download Complete JSON",
                json.dumps(primary, indent=2),
                "nutricheck_result.json",
                "application/json",
                use_container_width=True,
            )

        st.caption("⚠️ Screening-support tool only. This is not a medical diagnosis; results must be confirmed by a healthcare professional.")

    # =========================================================================
    # RIGHT COLUMN: INTEGRATED AI CHATBOT
    # =========================================================================
    with col_chatbot:
        st.markdown(
            """
            <div id="ai-chatbot-panel" style="background: white; border-radius: 14px; border: 1px solid #e2e8f0; padding: 1.25rem; box-shadow: 0 2px 10px rgba(0,0,0,0.04);">
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">🤖</span>
                    <div>
                        <h4 style="margin: 0; color: #0f2b48;">NutriCheck AI Assistant</h4>
                        <div style="font-size: 0.8rem; color: #64748b;">Context-Aware Medical Knowledge Assistant</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Question Buttons
        st.markdown("##### Quick Questions:")
        quick_cols1 = st.columns(2)
        quick_cols2 = st.columns(2)

        prompt_to_send = None

        if quick_cols1[0].button("Why is my B12 low?", use_container_width=True):
            prompt_to_send = "Why is my Vitamin B12 low?"
        if quick_cols1[1].button("What does Vitamin D mean?", use_container_width=True):
            prompt_to_send = "What does my Vitamin D result mean?"
        if quick_cols2[0].button("Which nutrients need attention?", use_container_width=True):
            prompt_to_send = "Which nutrients need attention?"
        if quick_cols2[1].button("Explain in simple English", use_container_width=True):
            prompt_to_send = "Explain my report in simple English"

        st.caption("Or ask about foods to eat, borderline vs deficient, or questions for your doctor.")

        # Chat Message History Container
        chat_box = st.container(height=420)
        with chat_box:
            for msg in st.session_state["chat_messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat Input
        user_input = st.chat_input("Ask a question about your report results...")

        if prompt_to_send:
            user_input = prompt_to_send

        if user_input:
            # 1. Append user message
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(user_input)

            # 2. Get answer from chatbot
            bot = st.session_state.get("chatbot")
            if not bot:
                bot = NutriCheckChatbot(result=primary, report_text=data.get("source_text", ""))
                st.session_state["chatbot"] = bot

            answer = bot.answer(user_input)

            # 3. Append assistant message
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            with chat_box:
                with st.chat_message("assistant"):
                    st.markdown(answer)

            st.rerun()

    # Floating AI Chatbot Link Button
    st.markdown(
        """
        <a href="#ai-chatbot-panel" class="floating-chat-btn">
            <span>💬</span> <span>Ask AI Chatbot</span>
        </a>
        """,
        unsafe_allow_html=True,
    )
