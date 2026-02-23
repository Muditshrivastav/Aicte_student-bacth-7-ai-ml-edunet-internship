import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
from deepface import DeepFace
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="🧠",
    layout="centered"
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "affirmation" not in st.session_state:
    st.session_state.affirmation = False

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Companion Controls")

st.sidebar.subheader("Quick Mood")
quick_mood = st.sidebar.radio(
    "",
    ["None", "😔 Sad", "😰 Anxious", "😵 Stressed", "😐 Neutral"]
)

st.sidebar.markdown("---")

if st.sidebar.button("🌱 Daily Affirmation"):
    st.session_state.affirmation = True

if st.sidebar.button("🧹 Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "🚨 If you feel unsafe:\n\n"
    "🇮🇳 **KIRAN Helpline (24/7): 1800-599-0019**"
)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["💬 Chatbot", "😊 Facial Emotion Detection"])

# =========================================================
# ===================== TAB 1 : CHATBOT ===================
# =========================================================
with tab1:
    st.title("🧠 Mental Health Companion Chatbot")

    st.write(
        "This is a calm, supportive space to talk about how you're feeling."
    )

    st.warning(
        "⚠️ Disclaimer: This chatbot is not a medical professional."
    )

    # -------- LLM (GROQ) --------
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.8
    )

    SYSTEM_CONTEXT = """
You are a calm, empathetic mental health companion for students.

Speak like a real, caring human — not a chatbot or therapist.

Guidelines:
- Be natural and conversational
- Avoid repeating the same comforting phrases
- Offer emotional support
- Suggest calming or grounding ideas only when appropriate
- Do not sound clinical
- Never diagnose or give medical advice
"""

    # -------- CHAT HISTORY --------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -------- CHAT INPUT --------
    user_input = st.chat_input("Share what you're feeling...")

    if user_input:
        if quick_mood != "None":
            user_input = f"I am feeling {quick_mood.split(' ')[1].lower()}. {user_input}"

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Listening..."):
            messages = [SystemMessage(content=SYSTEM_CONTEXT)]

            for msg in st.session_state.messages[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            response = llm.invoke(messages).content

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        with st.chat_message("assistant"):
            st.markdown(response)

    # -------- AFFIRMATION --------
    if st.session_state.affirmation:
        st.markdown("---")
        st.markdown("### 🌞 Daily Affirmation")
        st.write(
            "You are doing the best you can with what you have right now — and that is enough."
        )
        st.session_state.affirmation = False

# =========================================================
# ============ TAB 2 : FACIAL EMOTION DETECTION ============
# =========================================================
with tab2:
    st.header("😊 Live Facial Emotion Detection")

    st.write(
        "This feature uses your webcam to detect facial emotions in real time."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Camera"):
            st.session_state.camera_on = True

    with col2:
        if st.button("⏹ Stop Camera"):
            st.session_state.camera_on = False

    frame_placeholder = st.empty()
    emotion_placeholder = st.empty()

    if st.session_state.camera_on:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Could not access the camera.")
        else:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                try:
                    result = DeepFace.analyze(
                        frame_rgb,
                        actions=["emotion"],
                        enforce_detection=False
                    )

                    emotion = result[0]["dominant_emotion"]
                    confidence = result[0]["emotion"][emotion]

                    emotion_placeholder.success(
                        f"Detected Emotion: **{emotion.upper()}** ({confidence:.2f}%)"
                    )

                except Exception:
                    emotion_placeholder.warning("Face not clearly detected")

                time.sleep(0.1)

            cap.release()
            frame_placeholder.empty()
            emotion_placeholder.info("Camera stopped.")
