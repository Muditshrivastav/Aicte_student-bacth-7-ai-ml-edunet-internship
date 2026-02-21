import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

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

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Companion Controls")
st.sidebar.markdown("---")

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
    "India Helpline: **1800-599-0019 (KIRAN)**"
)

# ---------------- MAIN UI ----------------
st.title("🧠 Mental Health Companion Chatbot")

st.write(
    "This is a calm, supportive space to talk about how you're feeling. "
    "You can chat freely and optionally share a mood image."
)

st.warning(
    "⚠️ Disclaimer: This chatbot is not a medical professional."
)

# ---------------- IMAGE INPUT ----------------
st.markdown("🖼️ Optional: Share a mood image")

chat_image = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    key="chat_image"
)

if chat_image:
    image = Image.open(chat_image)
    st.image(image, caption="Shared mood image", width=220)

# ---------------- CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# ---------------- SYSTEM CONTEXT ----------------
SYSTEM_CONTEXT = """
You are a calm, empathetic mental health companion for students.

You speak like a real, caring human — not a chatbot — and gently support the user emotionally.

Guidelines:
- Respond naturally and conversationally
- Avoid repeating the same comforting phrases
- Offer gentle emotional support
- Suggest calming or grounding ideas only when appropriate
- Do not sound scripted or clinical
- Never diagnose or give medical advice
"""

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Share what you're feeling...")

if user_input:
    # Apply quick mood
    if quick_mood != "None":
        user_input = f"I am feeling {quick_mood.split(' ')[1].lower()}. {user_input}"

    # Add image context gently
    if chat_image:
        user_input = (
            "The user has shared a mood-related image. "
            "Respond gently using both the image context and the message.\n\n"
            f"{user_input}"
        )

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Listening..."):
        # ✅ STRICT LangChain message format
        messages = [SystemMessage(content=SYSTEM_CONTEXT)]

        for msg in st.session_state.messages[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        response = llm.invoke(messages)
        assistant_reply = response.content

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

# ---------------- AFFIRMATION ----------------
if st.session_state.affirmation:
    st.markdown("---")
    st.markdown("### 🌞 Daily Affirmation")
    st.write(
        "You are doing the best you can with what you have right now — and that is enough."
    )
    st.session_state.affirmation = False
