# 🤖 Offline AI Chatbot using Fine-Tuned Qwen 2.5 3B (Django) - under update

An **offline AI-powered chatbot application** built using **Django** and powered by a **fine-tuned Qwen 2.5 3B large language model**.

This system allows administrators to upload knowledge base documents (PDF, DOCX, XLSX), which are automatically processed and indexed. Users can chat with the bot in a **ChatGPT-style interface** and receive intelligent, context-aware answers — all running completely **offline with zero cloud dependency**.

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


### 🧑‍💻 User Panel

-   Ask questions and receive intelligent answers from chatbot
-   Powered by offline LLM (transformers)
-   Feedback input (helpful / not helpful)

![image](https://github.com/user-attachments/assets/59968a0d-ab83-407f-997a-b7b9acb52754)

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
