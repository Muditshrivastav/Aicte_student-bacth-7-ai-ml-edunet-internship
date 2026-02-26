import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["groq_api_key"] = os.getenv("groq_api_key")

st.set_page_config(page_title="Mental Health Companion", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    /* Styled Disclaimer Box */
    .disclaimer {
        background-color: #2b2c11;
        border-left: 5px solid #eab308;
        padding: 15px;
        border-radius: 8px;
        color: #fef08a;
        margin-bottom: 25px;
        font-size: 0.9rem;
    }
    /* Sidebar Helpline Box */
    .helpline-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3b82f6;
        margin-top: 20px;
    }
    .helpline-text { color: #93c5fd; font-size: 0.85rem; }
    .helpline-phone { color: #3b82f6; font-weight: bold; font-size: 1rem; }
    </style>
""", unsafe_allow_html=True)

# --- 2. INITIALIZE GROQ ---

def get_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR (Companion Controls) ---
with st.sidebar:
    st.header("🧠 Companion Controls")
    
    st.subheader("Quick Mood")
    mood = st.radio(
        "How are you feeling?",
        ["None", "😔 Sad", "😨 Anxious", "🤬 Stressed", "😐 Neutral"],
        index=0
    )
    
    st.divider()
    
    if st.button("🌱 Daily Affirmation", use_container_width=True):
        st.toast("You are resilient, and your feelings are valid.")
        
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Emergency Helpline Box
    st.markdown("""
    <div class="helpline-box">
        <p class="helpline-text">🚨 If you feel unsafe:</p>
        <p style='font-weight: bold; color: white; margin-bottom: 5px;'>IN KIRAN Mental Health Helpline</p>
        <p class="helpline-phone">(24/7): 1800-599-0019</p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. MAIN INTERFACE ---
st.title("🧠 Mental Health Companion Chatbot")
st.write("This is a calm, supportive space to talk about how you're feeling.")

# The Yellow Disclaimer Box
st.markdown("""
<div class="disclaimer">
    ⚠️ Disclaimer: This chatbot is not a medical professional. If you are in a crisis, please reach out to the helplines in the sidebar.
</div>
""", unsafe_allow_html=True)

# Display Message History
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

# --- 5. CHAT INPUT (Root level to prevent jumping) ---
if prompt := st.chat_input("Share what you're feeling..."):
    
    # 1. Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # 2. Setup Persona
    system_prompt = SystemMessage(content=(
        f"You are a kind mental health companion. The user has selected a mood of {mood}. "
        "Be empathetic, non-judgmental, and supportive. Focus on active listening."
    ))

    # 3. Generate response with streaming
    with st.chat_message("assistant"):
        llm = get_llm()
        # Combine system prompt + history + current user input
        full_context = [system_prompt] + st.session_state.messages + [HumanMessage(content=prompt)]
        
        response = st.write_stream(llm.stream(full_context))
    
    # 4. Save to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=response))
