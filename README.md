# 🧠 Gemini-Powered Chatbot with Django, JWT Auth, MongoDB, and Document-Based Q&A
This project is a Gemini-integrated intelligent chatbot built using Django REST Framework, offering contextual document question answering and secure JWT-based authentication. Users can upload documents (PDF, DOCX, XLSX, JSON) and ask context-based questions. All interactions and chat history are stored in MongoDB.

## 📁 Features
🔒 JWT Authentication (User Sign Up, Login)

📄 File Upload: PDF, DOCX, XLSX, JSON

🤖 Contextual Chat with Gemini (Google Generative AI)

🧠 Document-Based Question Answering (Text extraction + Prompting)

🗃️ MongoDB for storing chats and documents

📤 RESTful API design

### 🚀 Getting Started
#### 1. Clone the Repository
```
  git clone https://github.com/AdiWork933/Ofline_ChatBot.git
  cd gemini-chatbot-backend
```
#### 2. Create Virtual Environment
```
  python -m venv .venv
  source .venv/bin/activate (For Linux)
  .venv/Scripts/activate (For window)
```
if activation error occur(window) then 
```
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .venv/Scripts/activate
```

#### 3. Install Dependencies
```
  pip install -r requirements.txt
```
#### 4. Environment Variables
Create a .env file in the root directory with the following:
```
GOOGLE_API_KEY = your_gemini_api_key_here<br>
SUPER_ADMIN_USERNAME = admin123
SUPER_ADMIN_PASSWORD = sceret@123
MONGODB_URI = your_mongodb_connection_uri
DJANGO_SECRET_KEY = your_django_secret_key
MONGO_DB_NAME = mongo db name
```
#### 5. Run Server
```
  python manage.py runserver
```
for any specific port 
```
  python manage.py runserver (port number like 8505)
```
#### API will be live(normal) at: http://127.0.0.1:8000/
### If google.api_core error appears, install:

pip install google-generativeai<br>
## 🔑 Environment Variables
Create a .env file in the root directory with the following:
```
GOOGLE_API_KEY = your_gemini_api_key_here<br>
SUPER_ADMIN_USERNAME = admin123
SUPER_ADMIN_PASSWORD = sceret@123
MONGODB_URI = your_mongodb_connection_uri
DJANGO_SECRET_KEY = your_django_secret_key
MONGO_DB_NAME = mongo db name
```


## 🧪 API Endpoints
Updat...........very soon.<br>
Authentication is via JWT. Pass token in the header:

Authorization: Bearer <your_token>
## 🗂️ Folder Structure
project/<br>
├── chatbot/<br>
│   ├── views.py<br>
│   ├── models.py<br>
│   ├── urls.py<br>
│   └── ...<br>
├── documents/<br>
│   ├── handlers/        # PDF, DOCX, XLSX parsing<br>
│   └── ...<br>
├── users/<br>
│   ├── views.py<br>
│   └── ...<br>
├── templates/<br>
├── media/              # Uploaded documents<br>
├── manage.py<br>
├── requirements.txt<br>
└── .env

## 📚 Dependencies
The project relies on the following Python libraries:

### 🛠 Core Framework
Django – Web framework
djangorestframework – For building RESTful APIs

### 🔐 Authentication
PyJWT – JWT encoding/decoding for secure auth

### 🤖 AI Integration
google-generativeai – Gemini-powered conversational AI

### 🗃 Database
pymongo – MongoDB access for storing chat history and uploaded files

### 📄 Document Parsing
python-docx – Word (.docx) file processing
openpyxl – Excel (.xlsx) file parsing
PyMuPDF – PDF file reading and text extraction
Built-in json – For handling .json documents

### 📦 Installation
#### Install all dependencies at once using:

pip install -r requirements.txt

