# app.py
import streamlit as st
import sqlite3
import hashlib
import tempfile
import os
import time
import numpy as np
import wave
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import google.generativeai as genai

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="AI Medical Dashboard", page_icon="🏥", layout="wide")

# Configure Gemini (use Streamlit secrets or fallback)
API_KEY = st.secrets.get("GOOGLE_API_KEY", None) or "YOUR_API_KEY_HERE"
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    # We'll show a warning but do not hard-stop (some parts can still run)
    st.warning("Warning: Gemini API configuration failed. Add valid GOOGLE_API_KEY in Streamlit secrets.")
    # Optionally: st.stop() if you want to force key presence

# -----------------------
# DATABASE (users + appointments + reports)
# -----------------------
DB_PATH = "app_data.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

# Users table
c.execute(
    """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
"""
)
# Appointments table
c.execute(
    """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    patient_name TEXT,
    age INTEGER,
    gender TEXT,
    phone TEXT,
    email TEXT,
    department TEXT,
    doctor TEXT,
    date TEXT,
    time TEXT,
    type TEXT,
    symptoms TEXT,
    emergency INTEGER,
    followup INTEGER,
    created_at TEXT
)
"""
)
# Reports metadata table
c.execute(
    """
CREATE TABLE IF NOT EXISTS medical_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    name TEXT,
    file_name TEXT,
    type TEXT,
    date TEXT,
    notes TEXT,
    uploaded_at TEXT
)
"""
)
conn.commit()

# -----------------------
# UTILS - hashing & session
# -----------------------
def make_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_hash(password: str, hashed: str) -> bool:
    return make_hash(password) == hashed

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# -----------------------
# STYLE (make it visually closer to screenshot)
# -----------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg,#1e3c72,#2a5298);
        color: #fff;
    }
    .big-title {
        font-size:32px;
        font-weight:700;
        margin-bottom:0;
    }
    .section-title {
        font-size:24px;
        font-weight:700;
        color:#fff;
        margin-top:30px;
    }
    .card {
        background: rgba(255,255,255,0.06);
        padding:16px;
        border-radius:10px;
        margin-bottom:16px;
    }
    .form-label {
        color:#dfe7ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# AUTH: signup & login
# -----------------------
def signup_ui():
    st.header("Create an account")
    with st.form("signup_form", clear_on_submit=True):
        username = st.text_input("Choose username")
        password = st.text_input("Choose password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Sign up")
        if submitted:
            if not username or not password:
                st.error("Please fill all fields.")
                return
            if password != password2:
                st.error("Passwords do not match.")
                return
            c.execute("SELECT username FROM users WHERE username=?", (username,))
            if c.fetchone():
                st.error("Username already exists.")
                return
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, make_hash(password)))
            conn.commit()
            st.success("Account created! You can now login.")

def login_ui():
    st.header("Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if not username or not password:
                st.error("Please fill both fields.")
                return
            c.execute("SELECT password FROM users WHERE username=?", (username,))
            row = c.fetchone()
            if row and verify_hash(password, row[0]):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome, {username}!")
                time.sleep(0.8)
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

# -----------------------
# Helper: save uploaded file to local folder and DB metadata
# -----------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_medical_report(username, file_uploader_obj, name, report_type, date_val, notes):
    if not file_uploader_obj:
        return False
    file_bytes = file_uploader_obj.read()
    file_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{file_uploader_obj.name}")
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    c.execute(
        "INSERT INTO medical_reports (username, name, file_name, type, date, notes, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, name or file_uploader_obj.name, os.path.basename(file_path), report_type, date_val, notes, time.strftime("%Y-%m-%d %H:%M")),
    )
    conn.commit()
    return True

# -----------------------
# AUDIO: record using sounddevice OR file upload fallback
# -----------------------
try:
    import sounddevice as sd  # may fail in some environments
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False

def record_sound_device(duration_sec=5, fs=44100):
    """Record via sounddevice; returns path to wav file"""
    if not SD_AVAILABLE:
        raise RuntimeError("sounddevice not available")
    recording = sd.rec(int(duration_sec * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    tempf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tempf.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    return tempf.name

def transcribe_audio_file(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    except Exception:
        return None

def tts_play(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)
    try:
        os.unlink(tmp.name)
    except:
        pass

# -----------------------
# Gemini helper (wrap to avoid hard crash)
# -----------------------
def gemini_medical_answer(user_prompt, mode="General Health", lang="en"):
    """Call Gemini and return text (handles exceptions)"""
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        prompt = (
            f"You are a medical information assistant (mode: {mode}).\n"
            "Provide safe, factual, and general health guidance. DO NOT diagnose or prescribe medications.\n"
            f"User question: {user_prompt}\n\nPlease respond clearly and concisely."
        )
        resp = model.generate_content(prompt)
        text = resp.text if hasattr(resp, "text") else str(resp)
        return text
    except Exception as e:
        # Return a helpful fallback message
        return f"(Gemini error: {e})\nI couldn't fetch an AI response — check API key/network."

# -----------------------
# UI: The big dashboard (shown after login)
# -----------------------
def show_dashboard():
    # Top header area (like screenshot)
    st.markdown("<div style='padding:18px 6px 6px 6px'>", unsafe_allow_html=True)
    st.markdown("<h1 class='big-title'>MedMind AI Agent</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Tabs across the top
    tabs = st.tabs(
        ["Voice Consultation", "Image Analysis", "Medical Reports", "Appointments", "Prescriptions & Pharmacy", "Health Dashboard"]
    )

    # -----------------------
    # Tab 1: Voice Consultation
    # -----------------------
    with tabs[0]:
        st.header("🎙 Voice Consultation")
        st.write("Speak or upload your question. The AI provides general medical information only.")

        col1, col2 = st.columns([2, 1])
        with col1:
            mode = st.selectbox("Consultation Mode:", ["General Health", "Medicine Info", "Nutrition & Diet", "Mental Health Support"])
            lang = st.selectbox("Response Language:", ["en", "hi", "es", "fr"])
            st.write("Choose to record (if supported) or upload an audio file.")

            user_text = None

            if SD_AVAILABLE:
                rec_dur = st.slider("Recording duration (seconds)", 3, 12, 5)
                if st.button("Start Recording (mic)"):
                    try:
                        audio_path = record_sound_device(rec_dur)
                        st.success("Recording saved.")
                        # show audio player
                        st.audio(audio_path)
                        user_text = transcribe_audio_file(audio_path)
                        if user_text:
                            st.info(f"You said: {user_text}")
                        else:
                            st.error("Could not transcribe audio.")
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
                    except Exception as e:
                        st.error(f"Recording error: {e}")
            else:
                st.info("Recording via microphone not available in this environment. Please upload audio file.")

            uploaded_audio = st.file_uploader("Or upload audio (wav/mp3)", type=["wav", "mp3"])
            if uploaded_audio and not user_text:
                # save to temp and transcribe
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1])
                tmp.write(uploaded_audio.read())
                tmp.flush()
                tmp.close()
                st.audio(tmp.name)
                user_text = transcribe_audio_file(tmp.name)
                try:
                    os.unlink(tmp.name)
                except:
                    pass
                if user_text:
                    st.info(f"You said: {user_text}")
                else:
                    st.error("Could not transcribe uploaded audio.")

            # Text fallback
            if st.text_area("Or type your question:", "", key="voice_text_input"):
                typed = st.session_state["voice_text_input"]
            else:
                typed = None
            if not user_text and typed:
                user_text = typed

            if user_text:
                if st.button("Ask AI"):
                    with st.spinner("Asking AI..."):
                        ai_text = gemini_medical_answer(user_text, mode=mode, lang=lang)
                        st.markdown("### 🤖 AI Response")
                        st.write(ai_text)
                        # TTS
                        try:
                            tts_play(ai_text, lang=lang)
                        except Exception:
                            st.warning("TTS failed in this environment.")

        with col2:
            st.subheader("💬 Chat History")
            if "history" not in st.session_state:
                st.session_state.history = []
            # display last 6 messages
            for entry in st.session_state.history[-6:]:
                st.markdown(f"You: {entry['q']}")
                st.markdown(f"AI: {entry['a']}")
            if st.button("Clear History"):
                st.session_state.history = []

            st.divider()
            st.subheader("📹 Video Consultation (Jitsi)")
            doc_name = st.text_input("Doctor name to start video call", key="video_doc")
            if doc_name:
                room = doc_name.strip().replace(" ", "_").lower()
                st.markdown(
                    f"""
                    <iframe src="https://meet.jit.si/{room}" allow="camera; microphone; fullscreen; display-capture"
                    style="height:400px; width:100%; border: none; border-radius:8px;"></iframe>
                    """,
                    unsafe_allow_html=True,
                )

    # -----------------------
    # Tab 2: Image Analysis (placeholder)
    # -----------------------
    with tabs[1]:
        st.header("🩻 Medical Image Analysis")
        st.write("Upload an image (X-ray, MRI, skin lesion). This is a demo placeholder.")
        uploaded_image = st.file_uploader("Upload medical image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded image", use_column_width=True)
            st.info("Image analysis with Gemini/vision would go here (requires image->model integration).")

    # -----------------------
    # Tab 3: Medical Reports
    # -----------------------
    with tabs[2]:
        st.header("📄 Medical Reports")
        st.write("Upload your reports (PDF/image) and store them securely.")
        col1, col2 = st.columns([2, 1])
        with col1:
            report_file = st.file_uploader("Upload report (pdf/jpg/png)", type=["pdf", "jpg", "jpeg", "png"])
            report_name = st.text_input("Report name", value="")
            report_type = st.selectbox("Report type", ["Blood Test", "X-Ray", "MRI", "CT Scan", "Ultrasound", "Prescription", "Other"])
            report_date = st.date_input("Report date")
            notes = st.text_area("Notes (optional)")
            if st.button("Save Report"):
                if not report_file:
                    st.error("Pick a file first.")
                else:
                    ok = save_medical_report(st.session_state.user, report_file, report_name, report_type, str(report_date), notes)
                    if ok:
                        st.success("Report saved.")
        with col2:
            st.subheader("Your Reports")
            c.execute("SELECT id, name, file_name, type, date, uploaded_at FROM medical_reports WHERE username=? ORDER BY uploaded_at DESC", (st.session_state.user,))
            rows = c.fetchall()
            if rows:
                for r in rows:
                    st.markdown(f"{r[1]}** ({r[3]}) — {r[4]} — uploaded {r[5]}")
                    if st.button(f"Download {r[2]}", key=f"down_{r[0]}"):
                        file_path = os.path.join(UPLOAD_DIR, r[2])
                        if os.path.exists(file_path):
                            with open(file_path, "rb") as f:
                                st.download_button(label="Download file", data=f, file_name=r[2])
                        else:
                            st.error("File not found on server.")
            else:
                st.info("No reports uploaded yet.")

                

    # -----------------------
    # Tab 4: Appointments (form saved to DB)
    # -----------------------
    with tabs[3]:
        st.header("📅 Book Doctor Appointment")
        st.write("Schedule appointments with healthcare professionals.")
        with st.form("appointment_form"):
            patient_name = st.text_input("Patient Name", value=st.session_state.user or "")
            age = st.number_input("Age", min_value=1, max_value=120, value=25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
            phone = st.text_input("Phone Number", placeholder="+91 XXXXX XXXXX")
            email = st.text_input("Email")
            department = st.selectbox("Department/Specialization", ["General Physician", "Cardiologist", "Dermatologist", "Dentist", "Psychiatrist", "Orthopedic", "Pediatrician", "Gynecologist", "ENT Specialist"])
            doctor = st.selectbox("Select Doctor", [f"Dr. {d.split()[0]} Sharma" for d in ["Available", "Available", "Available"]])
            appointment_date = st.date_input("Preferred Date")
            appointment_time = st.time_input("Preferred Time")
            consultation_type = st.radio("Consultation Type", ["In-Person", "Video Call", "Phone Call"])
            symptoms = st.text_area("Describe your symptoms or reason for visit", height=100)
            emergency = st.checkbox("Mark as Emergency")
            followup = st.checkbox("Follow-up Appointment")
            submit_apt = st.form_submit_button("Confirm Appointment")
            if submit_apt:
                # save to DB
                c.execute(
                    """
                    INSERT INTO appointments (username, patient_name, age, gender, phone, email, department, doctor, date, time, type, symptoms, emergency, followup, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        st.session_state.user,
                        patient_name,
                        int(age),
                        gender,
                        phone,
                        email,
                        department,
                        doctor,
                        str(appointment_date),
                        appointment_time.strftime("%H:%M"),
                        consultation_type,
                        symptoms,
                        1 if emergency else 0,
                        1 if followup else 0,
                        time.strftime("%Y-%m-%d %H:%M"),
                    ),
                )
                conn.commit()
                st.success("✅ Appointment confirmed!")

        st.divider()
        st.subheader("Your Appointments")
        c.execute("SELECT id, doctor, date, time, status FROM appointments WHERE username=? ORDER BY created_at DESC LIMIT 10", (st.session_state.user,))
        appts = c.fetchall()
        if appts:
            for a in appts:
                st.markdown(f"{a[0]}** — {a[1]} — {a[2]} at {a[3]} — status: {a[4] if a[4] else 'Confirmed'}")
        else:
            st.info("No appointments found.")

    # -----------------------
    # Tab 5: Prescriptions & Pharmacy (basic)
    # -----------------------
    with tabs[4]:
        st.header("💊 Prescriptions & Pharmacy")
        st.write("AI can suggest OTC meds for simple symptoms (educational only).")
        symptom_input = st.text_area("Describe symptoms e.g. fever, sore throat, cough")
        if st.button("Generate Suggestion"):
            if symptom_input.strip():
                ai_text = gemini_medical_answer(f"Suggest general over-the-counter medicines and home remedies for: {symptom_input}")
                st.markdown("### Suggested (Educational Only)")
                st.write(ai_text)
            else:
                st.warning("Please describe symptoms.")

        st.divider()
        st.subheader("Pharmacy (Demo)")
        meds = [
            {"name": "Paracetamol", "dosage": "500mg", "price": 50},
            {"name": "Ibuprofen", "dosage": "400mg", "price": 80},
            {"name": "Vitamin D3", "dosage": "1000 IU", "price": 200},
        ]
        cols = st.columns(3)
        for idx, med in enumerate(meds):
            with cols[idx]:
                st.markdown(f"{med['name']}")
                st.write(med['dosage'])
                st.write(f"₹{med['price']}")
                qty = st.number_input(f"Qty {med['name']}", min_value=0, max_value=10, value=0, key=f"qty_{idx}")
                if st.button(f"Add {med['name']}", key=f"add_{idx}") and qty > 0:
                    if "cart" not in st.session_state:
                        st.session_state.cart = {}
                    st.session_state.cart[med['name']] = {"qty": qty, "total": med['price'] * qty}
                    st.success("Added to cart")

        if "cart" in st.session_state and st.session_state.cart:
            st.divider()
            st.subheader("Cart")
            total = 0
            for name, info in st.session_state.cart.items():
                st.write(f"{name} — {info['qty']} pcs — ₹{info['total']}")
                total += info['total']
            st.markdown(f"Total: ₹{total}")
            if st.button("Checkout"):
                st.success("Order placed (demo).")
                st.session_state.cart = {}

    # -----------------------
    # Tab 6: Health Dashboard (summary)
    # -----------------------
    with tabs[5]:
        st.header("📊 Health Dashboard")
        st.write("Summary of your activity")
        # quick metrics (from DB)
        c.execute("SELECT COUNT(*) FROM appointments WHERE username=?", (st.session_state.user,))
        total_appointments = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM medical_reports WHERE username=?", (st.session_state.user,))
        total_reports = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM appointments WHERE username=? AND emergency=1", (st.session_state.user,))
        emergency_count = c.fetchone()[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Appointments", total_appointments)
        col2.metric("Reports", total_reports)
        col3.metric("Emergencies", emergency_count)

        st.markdown("### Recent activity")
        c.execute("SELECT date, doctor, type FROM appointments WHERE username=? ORDER BY created_at DESC LIMIT 5", (st.session_state.user,))
        for row in c.fetchall():
            st.write(f"{row[0]} — {row[1]} — {row[2]}")

    # footer disclaimer
    st.markdown("---")
    st.markdown(
        "<div style='padding:10px; font-size:12px; color:#e8f0ff'>⚠ This AI provides general medical information only. It is NOT a substitute for professional medical advice.</div>",
        unsafe_allow_html=True,
    )

# -----------------------
# Main controller
# -----------------------
def main():
    st.sidebar.title("AI Health")
    if st.session_state.logged_in:
        st.sidebar.markdown(f"Logged in: {st.session_state.user}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.sidebar.success("Logged out")
            st.experimental_rerun()
        show_dashboard()
    else:
        choice = st.sidebar.radio("Go to", ["Login", "Sign Up"])
        if choice == "Login":
            login_ui()
        else:
            signup_ui()

if __name__ == "__main__":
    main()