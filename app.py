import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import tempfile
import os

# -------------------------------------------
# 🔐 Load Gemini API key
# -------------------------------------------
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("⚠️ GOOGLE_API_KEY not found in secrets. Please add it in Streamlit Cloud settings.")
    st.stop()

# -------------------------------------------
# 🎨 Streamlit UI Setup
# -------------------------------------------
st.set_page_config(page_title="AI Medical Voice Agent", page_icon="🩺", layout="centered")
st.title("🩺 AI Medical Voice Agent (Gemini)")
st.caption("Speak or upload your question. The AI provides safe, factual, and general medical guidance — not diagnosis or prescription.")

# -------------------------------------------
# 🧠 Initialize Conversation Memory
# -------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# -------------------------------------------
# 🩺 Medical Mode Selector
# -------------------------------------------
mode = st.selectbox(
    "🩹 Select Consultation Mode:",
    ["General Health", "Medicine Info", "Nutrition & Diet", "Mental Health Support"]
)

# -------------------------------------------
# 🌐 Language Selector
# -------------------------------------------
lang = st.selectbox(
    "🌍 Response Language:",
    ["en", "hi", "es", "fr"]
)

# -------------------------------------------
# 🎙️ Audio Input (Streamlit native)
# -------------------------------------------
audio_file = st.audio_input("🎤 Record your question")

if audio_file is not None:
    st.success("✅ Audio received!")

    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_file.read())
        audio_path = tmpfile.name

    # 🎧 Convert Speech → Text
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            user_text = recognizer.recognize_google(audio_data)
            st.session_state.conversation.append({"role": "user", "content": user_text})
    except sr.UnknownValueError:
        st.error("⚠ Sorry, I couldn't understand your voice. Please try again.")
        os.unlink(audio_path)
        st.stop()
    except Exception as e:
        st.error(f"⚠ Error processing audio: {e}")
        os.unlink(audio_path)
        st.stop()

    os.unlink(audio_path)

    # -------------------------------------------
    # 🧠 Generate Gemini Response
    # -------------------------------------------
    with st.spinner("💬 Thinking..."):
        model = genai.GenerativeModel("gemini-1.5-pro")
        chat_history = "\n".join(
            [f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.conversation]
        )
        prompt = (
            f"You are a medical information assistant in {mode} mode.\n"
            f"Provide factual, safe, and general health guidance. "
            f"Do not diagnose or prescribe.\n\nConversation:\n{chat_history}"
        )
        try:
            response = model.generate_content(prompt)
            ai_text = response.text
            st.session_state.conversation.append({"role": "assistant", "content": ai_text})
        except Exception as e:
            st.error(f"⚠ Gemini error: {e}")
            st.stop()

    # -------------------------------------------
    # 💬 Chat Bubbles (UI)
    # -------------------------------------------
    st.subheader("💬 Chat History")
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='text-align:right;background-color:#DCF8C6;padding:10px;border-radius:10px;margin:5px;'>🧍‍♀️ {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='text-align:left;background-color:#E9E9EB;padding:10px;border-radius:10px;margin:5px;'>🤖 {msg['content']}</div>",
                unsafe_allow_html=True,
            )

    # -------------------------------------------
    # 🔊 Text-to-Speech Response
    # -------------------------------------------
    try:
        tts = gTTS(ai_text, lang=lang)
        audio_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_out.name)
        st.audio(audio_out.name, format="audio/mp3")
        os.unlink(audio_out.name)
    except Exception as e:
        st.warning(f"Speech synthesis failed: {e}")

# -------------------------------------------
# ⚠️ Disclaimer
# -------------------------------------------
st.divider()
st.caption("⚠️ This AI provides general medical information only. It is **not a substitute** for professional diagnosis or treatment.")
