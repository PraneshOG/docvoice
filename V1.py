import streamlit as st
import whisper
import tempfile
import os
import re
from datetime import datetime
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Doctor Voice → Text", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
    .big-title { font-size: 38px; font-weight: 800; margin-bottom: 0px; }
    .sub-title { font-size: 16px; opacity: 0.75; margin-top: 0px; }
    .card {
        padding: 16px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
        box-shadow: 0 6px 16px rgba(0,0,0,0.10);
    }
    .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 12px;
        margin-right: 6px;
    }
    mark {
        padding: 0.1em 0.3em;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🩺 Doctor Voice → Text Converter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Mic Recording • Deep Learning Transcription • Translation • Highlighting</div>', unsafe_allow_html=True)
st.write("")

if "history" not in st.session_state:
    st.session_state.history = []
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None

st.sidebar.title("⚙️ Settings")

model_size = st.sidebar.selectbox(
    "Whisper Model",
    ["tiny", "base", "small"],
    index=2,
    help="For your laptop (i5 + 8GB), 'small' is best."
)

output_mode = st.sidebar.selectbox(
    "Output Mode",
    ["Transcribe (Same Language)", "Translate to English"]
)

language_choice = st.sidebar.selectbox("Input Language", ["Auto Detect", "English", "Tamil"])
enable_autocorrect = st.sidebar.checkbox("Medical autocorrect", value=True)

st.sidebar.divider()
st.sidebar.subheader("🔎 Keywords")

DEFAULT_KEYWORDS = [
    "fever", "cough", "cold", "headache", "pain", "vomiting", "diarrhea",
    "bp", "blood pressure", "diabetes", "sugar", "hypertension",
    "asthma", "infection", "allergy", "gastric", "chest pain", "dizziness",
    "paracetamol", "dolo", "tablet", "mg", "metformin", "insulin",
    "ecg", "copd", "gerd", "hba1c"
]

custom_keywords = st.sidebar.text_input(
    "Add extra keywords (comma separated)",
    placeholder="Eg: nephrolithiasis, thrombocytopenia"
)

KEYWORDS = DEFAULT_KEYWORDS.copy()
if custom_keywords.strip():
    KEYWORDS.extend([k.strip() for k in custom_keywords.split(",") if k.strip()])

@st.cache_resource
def load_model(size: str):
    return whisper.load_model(size)

model = load_model(model_size)

def medical_autocorrect(text: str) -> str:
    if not text:
        return text

    fixes = {
        "thar sitamon": "paracetamol",
        "sitamon": "paracetamol",
        "parasitamol": "paracetamol",
        "paracitamol": "paracetamol",
        "dolo six fifty": "dolo 650",
        "six fifty mg": "650 mg",
        "six fifty milligram": "650 mg",
        "blood pressure is": "blood pressure:",
        "bp is": "bp:",
        "sugar is": "sugar:",
    }

    out = text
    for wrong, right in fixes.items():
        out = re.sub(wrong, right, out, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", out).strip()

def clean_transcript(text: str) -> str:
    fillers = ["uh", "um", "hmm", "aaa", "ah", "like", "you know"]
    for f in fillers:
        text = re.sub(rf"\b{re.escape(f)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

def highlight_keywords(text: str) -> str:
    for kw in sorted(set(KEYWORDS), key=len, reverse=True):
        text = re.sub(
            rf"(?i)\b{re.escape(kw)}\b",
            r"<mark>\g<0></mark>",
            text
        )
    return text

def count_keywords_found(text: str) -> int:
    return sum(
        1 for kw in set(KEYWORDS)
        if re.search(rf"(?i)\b{re.escape(kw)}\b", text)
    )

def build_export_text(patient_name, patient_age, transcript, detected_lang, model_size, mode):
    ts = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    return f"""
DOCTOR VOICE → TEXT TRANSCRIPT
Generated At: {ts}
Model: {model_size}
Mode: {mode}
Detected Language: {detected_lang}

PATIENT DETAILS:
Name: {patient_name if patient_name else "-"}
Age: {patient_age if patient_age else "-"}

TRANSCRIPT:
{transcript}
""".strip()

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Patient Details")

    c1, c2 = st.columns(2)
    with c1:
        patient_name = st.text_input("Patient Name")
    with c2:
        patient_age = st.text_input("Age")

    st.divider()

    st.subheader("🎙️ Live Recording")
    audio = mic_recorder(
        start_prompt="▶️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        use_container_width=True
    )

    if audio:
        st.audio(audio["bytes"], format="audio/wav")

    transcribe_btn = st.button("🧠 Process Audio", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📚 Session History")

    if not st.session_state.history:
        st.caption("No records yet")
    else:
        for h in reversed(st.session_state.history):
            st.caption(f"{h['time']} | {h['mode']}")
            st.caption(h["text"][:150])
            st.divider()

    st.markdown("</div>", unsafe_allow_html=True)

if transcribe_btn and audio:
    with st.spinner("Processing with Whisper..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            path = tmp.name

        lang = None
        if language_choice == "English":
            lang = "en"
        elif language_choice == "Tamil":
            lang = "ta"

        task = "translate" if output_mode == "Translate to English" else "transcribe"

        result = model.transcribe(path, language=lang, task=task)
        transcript = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        os.remove(path)

        if enable_autocorrect:
            transcript = medical_autocorrect(transcript)

    kw_count = count_keywords_found(transcript)

    st.markdown(
        f"""
        <span class="pill">Model: {model_size}</span>
        <span class="pill">Mode: {output_mode}</span>
        <span class="pill">Keywords Found: {kw_count}</span>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📄 Output")
    st.text_area("Transcript", transcript, height=150)

    st.subheader("✨ Highlighted")
    st.markdown(
        f"<div style='font-size:16px;line-height:1.7'>{highlight_keywords(transcript)}</div>",
        unsafe_allow_html=True
    )

    if st.button("✨ Clean Transcript"):
        transcript = clean_transcript(transcript)
        st.text_area("Cleaned", transcript, height=150)

    export_text = build_export_text(
        patient_name,
        patient_age,
        transcript,
        detected_lang,
        model_size,
        output_mode
    )

    st.download_button(
        "⬇️ Download Transcript",
        data=export_text,
        file_name="doctor_transcript.txt",
        mime="text/plain"
    )

    st.session_state.history.append({
        "time": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        "mode": output_mode,
        "text": transcript
    })
