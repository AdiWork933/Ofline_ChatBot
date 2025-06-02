import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

# Set favicon and page title
st.set_page_config(page_title="EDU GEN", page_icon="img/edugen.png", layout="wide")

# Load environment variables from .env file
load_dotenv()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Ensure upload directory exists
UPLOAD_DIR = Path("./uploads_data")
UPLOAD_DIR.mkdir(exist_ok=True)

# Session state for login
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Login Form
def login():
    st.title("🔐 Admin Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login 🔓")
        if submitted:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials")

# Upload Page
def admin_dashboard():
    st.title("📁 Admin File Upload Dashboard")
    st.markdown(f"Welcome, *{ADMIN_USERNAME}*")

    uploaded_file = st.file_uploader(
        "Upload a document (.pdf, .docx, .xlsx):", 
        type=["pdf", "docx", "xlsx"]
    )

    if uploaded_file is not None:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ File {uploaded_file.name} uploaded and saved!")

    # Display uploaded files with delete buttons
    st.subheader("Uploaded Files:")
    files = list(UPLOAD_DIR.glob("*"))
    if not files:
        st.info("No files uploaded yet.")
    else:
        for file in files:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                if st.button("🗑 Delete", key=f"delete_{file.name}"):
                    try:
                        file.unlink()  # Delete the file
                        st.success(f"Deleted {file.name} successfully.")
                        st.experimental_rerun()  # Refresh the UI
                    except Exception as e:
                        st.error(f"Error deleting {file.name}: {e}")

# App Flow
if not st.session_state.authenticated:
    login()
else:
    admin_dashboard()

#streamlit run admin_app.py 