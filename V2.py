import streamlit as st
import whisper
import tempfile
import os
import re
from datetime import datetime
from streamlit_mic_recorder import mic_recorder

import torch
from transformers import MarianTokenizer, MarianMTModel, logging
logging.set_verbosity_error()

st.set_page_config(page_title="Doctor Voice → Text", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 36px; font-weight: 800; }
.sub-title { font-size: 15px; opacity: 0.75; }
.card {
    padding: 16px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.04);
}
.pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    font-size: 12px;
    margin-right: 6px;
}
mark { padding: 0.1em 0.3em; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🩺 Doctor Voice → Text Converter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Whisper ASR • MarianMT Translation • Fully Offline • Open Source</div>', unsafe_allow_html=True)
st.write("")

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("⚙️ Settings")

model_size = st.sidebar.selectbox(
    "Whisper Model",
    ["tiny", "base", "small"],
    index=2
)

output_mode = st.sidebar.selectbox(
    "Output Mode",
    [
        "Transcribe (Same Language)",
        "Translate to English (Enhanced)"
    ]
)

language_choice = st.sidebar.selectbox(
    "Input Language",
    ["Auto Detect", "English", "Tamil"]
)

@st.cache_resource
def load_whisper(size):
    return whisper.load_model(size)

@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-ta-en"
    tokenizer = MarianTokenizer.from_pretrained(
        model_name,
        local_files_only=False
    )
    model = MarianMTModel.from_pretrained(
        model_name,
        local_files_only=False
    )
    return tokenizer, model

whisper_model = load_whisper(model_size)

translator_tokenizer = None
translator_model = None
if output_mode.startswith("Translate"):
    translator_tokenizer, translator_model = load_translator()

def translate_tamil_to_english(text: str) -> str:
    if not text.strip():
        return text
    inputs = translator_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        translated = translator_model.generate(**inputs, max_length=512)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

def highlight_keywords(text: str) -> str:
    KEYWORDS = [
        "fever","cough","pain","bp","blood pressure","diabetes",
        "hypertension","asthma","infection","paracetamol",
        "tablet","mg","metformin","insulin"
    ]
    for kw in sorted(set(KEYWORDS), key=len, reverse=True):
        text = re.sub(
            rf"(?i)\b{re.escape(kw)}\b",
            r"<mark>\g<0></mark>",
            text
        )
    return text

left, right = st.columns([1.4, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎙️ Record Doctor Voice")

    audio = mic_recorder(
        start_prompt="▶️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        use_container_width=True
    )

    if audio:
        st.audio(audio["bytes"], format="audio/wav")

    process_btn = st.button("🧠 Process Audio", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📚 Session History")

    if not st.session_state.history:
        st.caption("No records yet")
    else:
        for h in reversed(st.session_state.history):
            st.caption(f"{h['time']} | {h['mode']}")
            st.caption(h["text"][:140])
            st.divider()

    st.markdown('</div>', unsafe_allow_html=True)

if process_btn and audio:
    with st.spinner("Processing audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio["bytes"])
            path = tmp.name

        lang = None
        if language_choice == "English":
            lang = "en"
        elif language_choice == "Tamil":
            lang = "ta"

        result = whisper_model.transcribe(path, language=lang)
        tamil_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        os.remove(path)

        if output_mode.startswith("Translate"):
            final_text = translate_tamil_to_english(tamil_text)
        else:
            final_text = tamil_text

    st.success("✅ Processing Complete")

    st.markdown(
        f"""
        <span class="pill">Model: {model_size}</span>
        <span class="pill">Mode: {output_mode}</span>
        <span class="pill">Detected Language: {detected_lang}</span>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📄 Output")
    st.text_area("Transcript", final_text, height=160)

    st.subheader("✨ Highlighted Medical Terms")
    st.markdown(
        f"<div style='font-size:16px;line-height:1.7'>{highlight_keywords(final_text)}</div>",
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download Transcript",
        data=final_text,
        file_name="doctor_transcript.txt",
        mime="text/plain"
    )

    st.session_state.history.append({
        "time": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        "mode": output_mode,
        "text": final_text
    })
