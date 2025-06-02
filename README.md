# 🧠 Offline_ChatBoat

An **offline AI-powered chatbot application** built using **Django**. This chatbot allows admins to upload knowledge base documents (PDF, DOCX, XLSX), and users to chat in a ChatGPT-style interface with file-based knowledge using local LLMs like `flan-t5-large` or quantized `Mistral/TinyLLaMA`.

---

## 📌 Project Overview

This offline chatbot platform features:

-   **Admin panel** : Upload and manage document files.
-   **User panel** : Chat with AI on uploaded knowledge.
-   **Local Model Support**: All inference is performed using local models, no external API.
-   **Feedback System**: User feedback is stored in both `.json` and `.txt` formats.
-   **Vector Store**: Efficient document retrieval via chunking , FAISS and chromadb==1.0.8.
-   **Secure Login** for admin.

---

## 🔧 Features

### 🔒 Admin Panel

-   Secure login for Admin
-   Upload `.pdf`, `.docx`, `.xlsx` files
-   Files processed and stored for vector-based retrieval
-   Manage users (optional)
    -login page of admin pannel

![image](https://github.com/user-attachments/assets/3e614cda-1253-4361-ae0e-e76e6f4ed9f0)

-Admin File Upload Dashboard

![image](https://github.com/user-attachments/assets/f4ed7137-88fc-4b7e-8828-2e96b9945bd2)

### 🧑‍💻 User Panel

-   Ask questions and receive intelligent answers from chatbot
-   Powered by offline LLM (transformers)
-   Feedback input (helpful / not helpful)

![image](https://github.com/user-attachments/assets/f81b2fed-f030-4142-afb5-fff51e1bf353)

### ⚙️ Backend Logic

-   `model.py`: Handles model loading, chunking, embedding, and answering
-   Local LLM (no HuggingFace API dependency)
-   SentenceTransformer + FAISS vector store for fast retrieval
-   Feedback with Application id stored in:
    -   `feedback.json`
    -   `feedback.txt`

---

## 🔧 Setup For Window OS

-   `Creating virtual environment`: python -m venv venv
-   `set Exicute Access` : Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
-   `Activate virtual environment` : .\venv\Scripts\Activate
-   `Installing Required Librady` : pip install -r requirements.txt
-   `Starting streamlit Server(Terminal 1)` : streamlit run admin_app.py
-   `Starting streamlit Server(Terminal 2)` : streamlit run user_panel.py
-   `To stope streamlit Server` : ctrl + c (for both terminal)
-   `To Deactivate virtual environment` : deactivate ( into terminal)

---

## 🔧 Setup For ubuntu OS

-   `Creating virtual environment`: python3 -m venv .venv
-   `Activate virtual environment` : source .venv /bin/activate
-   `Installing Required Librady` : pip install -r requirements.txt
-   `Starting streamlit Server(Terminal 1)` : streamlit run admin_app.py
-   `Starting streamlit Server(Terminal 2)` : streamlit run user_panel.py
-   `To stope streamlit Server` : ctrl + c (for both terminal)
-   `To Deactivate virtual environment` : deactivate ( into terminal)

Creating virtual environment: python3 -m venv venv

Set Execute Access: source venv/bin/activate

Activate virtual environment: source venv/bin/activate

Installing Required Library: pip install -r requirements.txt

Starting streamlit Server(Terminal 1): streamlit run admin_app.py

Starting streamlit Server(Terminal 2): streamlit run user_panel.py

To stop streamlit Server: ctrl + c (for both terminals)

To Deactivate virtual environment: deactivate (into terminal)
