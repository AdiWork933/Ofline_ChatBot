import streamlit as st
import os
import json
from datetime import datetime
from model import ModelHandler
from streamlit_extras.add_vertical_space import add_vertical_space
import nltk
from base64 import b64encode

nltk.download("punkt")


# Set favicon and page title
st.set_page_config(page_title="EDU GEN", page_icon="img/edugen.png", layout="wide")

# --- Load and encode EDUGEN logo
from base64 import b64encode

# Load both logos
def load_logo_base64(path):
    with open(path, "rb") as f:
        return b64encode(f.read()).decode()

edugen_logo = load_logo_base64("img/bot_img.png")  # Chatbot logo
user_logo = load_logo_base64("img/images.png")     # User logo

# --- Custom chat bubble rendering
def chat_bubble(role, content):
    alignment = "row-reverse" if role == "user" else "row"
    bubble_color = "#d1eaff" if role == "user" else "#f1f1f1"
    logo = user_logo if role == "user" else edugen_logo

    st.markdown(f"""
        <div style='display: flex; flex-direction: {alignment}; align-items: flex-start; margin-bottom: 1rem;'>
            <img src="data:image/png;base64,{logo}" style='width: 40px; height: 40px; margin: 0 12px; border-radius: 50%;' />
            <div style='background-color: {bubble_color}; padding: 12px 18px; border-radius: 16px; max-width: 75%; font-size: 15px; line-height: 1.5;'>{content}</div>
        </div>
    """, unsafe_allow_html=True)


# --- CSS Styling
st.markdown("""
    <style>
        input[type="text"] {
            color: black !important;
            background-color: white !important;
        }
        .feedback-button {
            font-size: 0.8rem;
            padding: 6px 12px;
            margin-top: 6px;
            background-color: #f1f1f1;
            border: 1px solid #ccc;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model handler
@st.cache_resource
def get_model_handler():
    return ModelHandler()

model_handler = get_model_handler()

# --- Load and process documents from uploads_data/
@st.cache_resource
def initialize_vector_store():
    folder_path = "uploads_data"
    if not os.path.exists(folder_path):
        return None
    all_chunks = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            chunks = model_handler.chunk_pdf_text(filepath)
        elif filename.endswith(".docx"):
            chunks = model_handler.chunk_docx_text(filepath)
        elif filename.endswith(".xlsx"):
            chunks = model_handler.chunk_excel_text(filepath)
        else:
            continue
        if chunks:
            all_chunks.extend(chunks)

    if all_chunks:
        vector_store, _ = model_handler.get_vector_store_from_chunks(all_chunks)
        return vector_store
    return None

# --- Init vector store
vector_store = initialize_vector_store()

# --- Session state setup
for key in ['chat_history', 'show_feedback']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'chat_history' else False

# --- Title
st.title("🤖 Chat with Me")

# --- Show chat history with logo bubbles
for entry in st.session_state.chat_history:
    chat_bubble("user", entry["question"])
    chat_bubble("assistant", entry["answer"])
    if entry.get("feedback"):
        st.markdown(f"🗣 Feedback: {entry['feedback']}")

# --- Feedback form
if st.session_state.show_feedback:
    with st.expander("📝 Provide Feedback for the Last Answer"):
        app_id = st.text_input("Application ID")
        feedback_text = st.text_input("Your feedback:")

        if st.button("Submit Feedback"):
            if not app_id.strip() or not feedback_text.strip():
                st.warning("⚠ Both Application ID and Feedback are required.")
            else:
                if st.session_state.chat_history:
                    last_qna = st.session_state.chat_history[-1]
                    last_qna["feedback"] = feedback_text
                    last_qna["app_id"] = app_id

                    # Save to TXT
                    with open("feedback_log.txt", "a", encoding="utf-8") as file:
                        file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        file.write(f"Application ID: {app_id}\n")
                        file.write(f"Question: {last_qna['question']}\n")
                        file.write(f"Answer: {last_qna['answer']}\n")
                        file.write(f"Feedback: {feedback_text}\n")
                        file.write("-" * 40 + "\n")

                    # Save to JSON
                    feedback_data = {
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "application_id": app_id,
                        "question": last_qna["question"],
                        "answer": last_qna["answer"],
                        "feedback": feedback_text
                    }

                    if os.path.exists("feedback_log.json"):
                        with open("feedback_log.json", "r+", encoding="utf-8") as jf:
                            data = json.load(jf)
                            data.append(feedback_data)
                            jf.seek(0)
                            json.dump(data, jf, indent=4)
                    else:
                        with open("feedback_log.json", "w", encoding="utf-8") as jf:
                            json.dump([feedback_data], jf, indent=4)

                    st.success("✅ Feedback and App ID submitted successfully!")
                    st.session_state.show_feedback = False

# --- Chat input
question = st.chat_input("Type your question here...")
if st.button("💬 Give Feedback"):
    st.session_state.show_feedback = True

if question:
    with st.spinner("🤔 Generating answer..."):
        if not vector_store:
            dummy_answer = "⚠ The model is under maintenance. No documents are loaded."
        else:
            small_talk = model_handler.handle_small_talk(question)
            if small_talk:
                dummy_answer = small_talk
            else:
                docs = vector_store.similarity_search(question, k=5)
                context = "\n".join(doc.page_content for doc in docs)
                dummy_answer = model_handler.generate_answer(question, context)

        # Show new messages with logo
        chat_bubble("user", question)
        chat_bubble("assistant", dummy_answer)

        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": dummy_answer,
            "feedback": None
        })
        st.session_state.show_feedback = False

