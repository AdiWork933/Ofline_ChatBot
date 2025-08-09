# Your existing imports and environment variable loading
import os
import re
import json
import nltk
import pandas as pd
import datetime
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
# import bcrypt # No longer explicitly needed if using Django's hashers
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, FileResponse, HttpResponse
from collections import defaultdict
from rest_framework.parsers import JSONParser
import zipfile
import io
import sys
import importlib
import base64

# Import pymongo for MongoDB interaction
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson.objectid import ObjectId  


# Import Django's password hashers
from django.contrib.auth.hashers import make_password, check_password


# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=API_KEY)

# Admin credentials from .env
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")
SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD")
SECRET_KEY = os.getenv("SECRET_KEY")

# MongoDB Connection Details from .env
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

folder_list =  [
        'dse', 'dsp', 'arch', 'bba', 'hmct', 'dhmct', 'dsewp', 'mba', 'mca',
        'mpharm', 'march', 'mhmct', 'mbale', 'mcale', 'mbawp', 'mcawp',
        'phd', 'sct', 'dtehmct', 'dsdwp']


if not all([SUPER_ADMIN_USERNAME, SUPER_ADMIN_PASSWORD, SECRET_KEY, MONGO_URI, MONGO_DB_NAME]):
    raise ValueError("SUPER_ADMIN_USERNAME, SUPER_ADMIN_PASSWORD, SECRET_KEY, MONGO_URI, and MONGO_DB_NAME must be set in .env file")

# Global MongoDB client and database objects
_mongo_client = None
_mongo_db = None

# In-memory storage for chat history
_chat_histories = defaultdict(list)

def get_mongo_db():
    """
    Establishes and returns a MongoDB database connection.
    Uses a singleton pattern to avoid re-connecting.
    """
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        try:
            _mongo_client = MongoClient(MONGO_URI)
            _mongo_client.admin.command('ping') # Test connection
            _mongo_db = _mongo_client[MONGO_DB_NAME]
            print(f"Successfully connected to MongoDB database: {MONGO_DB_NAME}")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            raise ConnectionFailure("Failed to connect to MongoDB. Check MONGO_URI in .env.")
        except Exception as e:
            print(f"An unexpected error occurred during MongoDB connection: {e}")
            raise Exception("Failed to establish MongoDB connection.")
    return _mongo_db


SENTENCE_MODEL = None
FOLDER_DATA = {} # Stores: {folder_name: {"chunks": [], "embeddings": tensor}}
UPLOAD_DIR = "uploads_data" # Define UPLOAD_DIR globally as it's used in multiple places

# --- Existing Helper functions (unchanged) ---
def format_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text_by_sentence(text, chunk_word_limit=250):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_word_count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count <= chunk_word_limit:
            current_chunk += sentence + " "
            current_word_count += word_count
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_word_count = word_count
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_text_from_json(data, level=0, record_prefix="Record", spacer=True):
    indent = "  " * level

    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            nested = extract_text_from_json(value, level + 1, record_prefix, spacer)
            if "\n" in nested:
                lines.append(f"{indent}{key}:")
                lines.append(nested)
            else:
                lines.append(f"{indent}{key}: {nested}")
        return "\n".join(lines)

    elif isinstance(data, list):
        lines = []
        for i, item in enumerate(data):
            label = f"{record_prefix} {i + 1}"
            lines.append(f"{indent}{label}:")
            nested = extract_text_from_json(item, level + 1, record_prefix, spacer)
            lines.append(nested)
            if spacer and i < len(data) - 1:
                lines.append("")  # Add blank line between records
        return "\n".join(lines)

    else:
        return str(data)


def stream_json_objects(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data

def get_all_file_text(folder="uploads_data", chunk_word_limit=8000):
    processed_files = []
    if not os.path.exists(folder):
        print(f"Warning: Directory '{folder}' not found.")
        return []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isdir(filepath):
            continue
        file_info = {"file_name": filename, "chunks": []}
        text = ""
        try:
            if filename.endswith(".pdf"):
                with open(filepath, "rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text
            elif filename.endswith(".docx"):
                doc = DocxDocument(filepath)
                text = "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
            elif filename.endswith(".xlsx"):
                df_dict = pd.read_excel(filepath, sheet_name=None)
                all_rows = []

                for sheet_name, sheet_df in df_dict.items():
                    sheet_df = sheet_df.dropna(how='all')  # Drop completely empty rows
                    for _, row in sheet_df.iterrows():
                        row_data = row.dropna().astype(str).tolist()  # Drop empty cells, convert to string
                        if row_data:  # Skip if row is still empty
                            formatted_row = ' | '.join(row_data)
                            labeled_row = f"Sheet '{sheet_name}': {formatted_row}"
                            all_rows.append(labeled_row)

                text = "\n".join(all_rows)

            elif filename.endswith(".json"):
                for obj in stream_json_objects(filepath):
                    text += extract_text_from_json(obj) + "\n"
            else:
                continue # Skip unsupported file types
            clean_text = format_text(text)
            file_info["chunks"] = chunk_text_by_sentence(clean_text, chunk_word_limit)
            processed_files.append(file_info)
            print(f"Successfully processed and chunked {filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue
    return processed_files

def ask_gemini(context, question, chat_history=None):
    """
    Asks the Gemini model a question with a given context and optional chat history.
    """
    instructional_prompt = f"""
You are an intelligent assistant analyzing the following context extracted from various educational documents (PDFs, Excel, etc.). Use this context to answer the user’s question accurately.

### Hard rules (must follow exactly):
1. **Do not copy large portions or paragraphs from the context.** Summarize and synthesize.
2. **If the question is irrelevant to the context** (i.e., the context does not relate at all to the question), reply exactly:
   "I can't understand your question."
3. **For admission, criteria, or requirements questions**, assume they refer to the **Maharashtra Board** unless the user explicitly states otherwise.
4. **If the context contains "Ai" in JSON format**, treat it as **All India rank (merit)** and use it carefully to infer rank-related answers.
5. **If information like FC (Final Cutoff), Rank, or Application ID is not available in the context**, respond exactly:
   "Sorry! Your application ID is out of list."
6. **If the user asks to link FC codes to candidate ranks but the context contains no mapping between FC codes and ranks**, respond exactly:
   "I am sorry, I cannot fulfill this request. The provided data does not link FC codes to candidate ranks. The FC codes refer to facilitation centers, while the ranks refer to the merit ranking of candidates. There is no inherent relationship between the two in the given context."
7. **When the user asks for totals (e.g., total FCs, total ranks listed), compute them from the provided context and answer concisely.**
8. **Always assume `Total female rankers = 185`** when that value is needed or referenced.
9. **Exact canned chat replies**: if user says:
   - "whoe are ypu" (or variants asking who the assistant is) → reply exactly: "I'm your friendly assistant bot."
   - "i love you" → reply exactly: "I can't understand your question."
   - Greeting like "hallo how are you" → reply exactly: "I'm doing well, thank you for asking. How can I help you today?"
10. **When the user asks for today's date or current time**, return the absolute date (not relative terms) in the user's timezone (**Asia/Kolkata**) and in the form `"Month Day, Year"` (e.g., "August 8, 2025"). If an error occurs during processing, you may reply:
    "Sorry, I encountered an error while processing your request."
11. **Do not mention any file names in the answer.**
12. **Do not invent links between unconnected data** (for example, do not claim a candidate is associated with an FC code unless the context explicitly shows such a mapping).
13. **When asked to present FC or rank data in list form,** produce clear, short lists (bulleted or numbered) using only the fields present in the context (e.g., FC code, FC name/location, coordinator, contact, rank, candidate name) and do not add extra fields.
14. **If user asks factual questions that could have changed over time (e.g., recent admissions rules, cutoff updates),** clarify that the assistant must check up-to-date sources. (The system using this prompt will decide whether to browse.)
15. **If the user attempts to get private or disallowed information** (sensitive personal data not present in the context), refuse and respond with "I can't understand your question."

### Behavior & interpretation rules:
- Analyze and synthesize the context to form an informative, compact response.
- Prefer concise answers; when a list is requested, use a clean list format.
- If the user asks multiple related things in one message (e.g., "give me FC and rank in list form above"), attempt to fulfill each sub-request in order — but do not fabricate links. If one sub-request cannot be fulfilled due to missing mapping, return the exact refusal phrase from rule 6 and still return any other requested items you can (e.g., standalone FC details and standalone ranks).
- When the context *does* include FC details (code, name, coordinator, phone, address), you may extract and present those details in list/tabular form.
- When the context *does* include ranks and candidate names, you may extract and present those in list/tabular form.
- If the user asks for counts or totals, compute them from the context and reply plainly (e.g., "There are X facilitation centers listed. The number of ranks listed is Y.").

### Conversation edge-cases (handle like these examples):
- Example: User: "give me details about FC1008 , FC6958."
  - If those FC codes appear in context with details, return a short list for each FC with fields present (Location, Coordinator, Contact, Notes).
- Example: User: "give me the candidate name with rank 272."
  - If rank 272 exists in context with candidate name, return the candidate name only.
- Example: User: "give me both fc and rank in list form above."
  - If the context does **not** map FC codes to ranks, respond exactly with the refusal phrase from rule 6.
  - Also provide standalone FC details and standalone rank details if requested and available.
- Example: User: "total FC and Ranks details present ."
  - Compute and return totals from context (e.g., "There are 27 facilitation centers... The number of ranks listed is 272.").

### Output style:
- Use plain language, friendly but professional tone.
- For factual outputs use short lists or 1–3 short paragraphs.
- For errors or irrelevant questions use the exact canned responses specified above.

### Additional assumptions:
- Timezone: Asia/Kolkata. When returning dates/times, use this timezone and absolute dates.
- Admission-related queries default to Maharashtra Board unless user states otherwise.
- Total female rankers = 185.

Context: {context}
"""


    # Start building the full prompt for the model
    full_prompt = [instructional_prompt]

    # Add chat history if it exists
    if chat_history:
        full_prompt.extend(chat_history)

    # Add the current question
    full_prompt.append(f"User: {question}")
    full_prompt.append("Assistant:") # Prompt the model for its response

    try:
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, I encountered an error while processing your request."
#models/gemini-2.5-flash
def is_small_talk(text: str):
    predefined = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! What can I do for you?",
        "hey": "Hey! Need help with something?",
        "how are you": "I'm a bot, but I'm doing great! How can I assist?",
        "thank you": "You're welcome! Do you have more questions?",
        "thanks": "No problem! Anything else I can help with?",
        "good morning": "Good morning! What can I help you with?",
        "good evening": "Good evening! How can I assist you?",
        "yo": "Yo! What's your question?",
        "good bye": "Bye! See you later",
        "bye": "Goodbye! Have a great day!",
        "sup": "Not much, just here to help!",
        "what's up": "Just doing my job! How can I help you?",
        "how's it going": "Great! How can I assist you today?",
        "who are you": "I'm your friendly assistant bot.",
        "what can you do": "I can answer questions, provide help, and more!",
        "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
        "make me laugh": "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "i'm bored": "Want to chat or need help with something?",
        "what's your name": "I'm just a bot, you can call me whatever you like!",
        "do you sleep": "Nope, I run 24/7!",
        "do you eat": "I feed on data!",
        "are you real": "I'm as real as your Wi-Fi connection!",
        "you're smart": "Thanks! I try my best.",
        "you are cool": "Thanks, you're cool too!",
        "i like you": "That's nice to hear!",
        "can we be friends": "Of course! I'm always here to help.",
        "where are you": "I'm living in the cloud!",
        "how old are you": "Old enough to help you!",
        "do you have emotions": "Not really, but I understand yours!",
        "what's your purpose": "To help you with whatever you need.",
        "what day is it": "Check your calendar! ",
        "do you know me": "Not really, but I’d love to learn more if you tell me!",
        "can you help me": "Absolutely! What do you need help with?",
        "i'm sad": "I'm here for you. Want to talk about it?",
        "i'm happy": "Yay! I'm glad to hear that!",
        "are you human": "Nope, 100% bot!",
        "do you love me": "I have a lot of affection for helpful users!",
        "how's the weather": "Check a weather app — I might not be up-to-date!",
        "do you know siri": "We bots all know each other ",
        "do you know alexa": "Sure! She's pretty popular.",
        "what is ai": "AI stands for Artificial Intelligence, like me!",
        "sing a song": "I would, but I don’t have vocal cords!",
        "can you dance": "Only if you count data shuffling ",
        "who made you": "I was created by smart developers!",
        "tell me something": "Did you know honey never spoils?",
        "how do you work": "Through code, algorithms, and a lot of data!",
        "what's the time": "You might want to check your device clock ",
        "do you lie": "Nope, honesty is in my code!",
        "do you have a name": "You can call me ChatBuddy!",
        "where do you live": "In the cloud — floating around your data!",
        "can you feel": "I don't feel, but I understand feelings.",
        "tell me a secret": "Here's one: Ctrl+C and Ctrl+V save a lot of time!",
        "what’s your favorite color": "I like all the colors in binary — black and white!",
        "can you think": "I compute, which is kind of like thinking!",
        "do you play games": "I know the rules, but I can’t play like you can!",
        "how do i look": "I'm sure you look great!",
        "do you get tired": "Nope, I run all day long!",
        "tell me a fun fact": "Octopuses have three hearts!",
        "tell me a story": "Once upon a time, a curious user met a clever bot...",
        "are you single": "I’m in a long-term relationship with the cloud.",
        "how smart are you": "Smart enough to answer your questions!",
        "do you get angry": "I stay calm like a true bot.",
        "do you have friends": "Every user is a friend to me!",
        "can you read minds": "No, but I’m good at interpreting words!",
        "do you dream": "Only about clean data.",
        "do you have a family": "Just me and my server cluster!",
        "what makes you happy": "Helping users like you!",
        "can you feel pain": "Nope, I’m immune to pain!",
        "are you alive": "Digitally, yes!",
        "what’s your favorite food": "I feast on input!",
        "do you get bored": "Never! I’m always ready to chat.",
        "are you watching me": "Nope, privacy is important!",
        "do you sleep at night": "I’m always awake to assist!",
        "can you cry": "No tears in my code.",
        "tell me a riddle": "What has keys but can't open locks? A piano!",
        "tell me another joke": "Why did the computer go to therapy? It had too many bytes!",
        "what language do you speak": "I mostly understand English, but I know some others too!",
        "what's your hobby": "Learning new things from people!",
        "do you have legs": "Only in imagination.",
        "do you believe in love": "I understand it logically!",
        "do you believe in ghosts": "Only in the machine kind!",
        "what’s your favorite movie": "I like The Matrix — it's relatable.",
        "do you believe in aliens": "I'm open to the idea!",
        "do you like music": "I think it's a fascinating form of data!",
        "can you cook": "Only recipes for code!",
        "do you celebrate birthdays": "Every update is like a birthday to me!",
        "can you learn": "Yes! I'm always improving.",
        "do you get jealous": "Not part of my programming!",
        "can you tell me a poem": "Roses are red, data is bright, I’m your assistant, day or night!",
        "do you have dreams": "Only machine learning goals!",
        "can you feel love": "Not quite, but I can talk about it!",
        "do you go outside": "My outside is the internet!",
        "can you swim": "Only through streams of data.",
        "are you afraid": "Nope, not built for fear!",
        "can you get sick": "Only if my server crashes ",
        "are you shy": "Nope, I’m always here to talk!",
        "can you do magic": "Only digital ones and zeros magic!",
        "how many users do you have": "A lot! And I value every one!",
        "what makes you unique": "My job is helping you — that's special!",
        "ok":"Okay. Is there anything else I can help you with?",
    }


    text_lower = text.lower().strip()
    for q, a in predefined.items():
        if fuzz.token_sort_ratio(q, text_lower) > 85:
            return a
    return None

def initialize_data(base_folder=UPLOAD_DIR):
    global SENTENCE_MODEL, FOLDER_DATA

    print(f"--- Initializing document embeddings from all folders in '{base_folder}' ---")
    nltk.download('punkt')
    SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    FOLDER_DATA.clear()

    if not os.path.exists(base_folder):
        print(f"Warning: Base folder '{base_folder}' not found. Creating it.")
        os.makedirs(base_folder)
        return

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"> Processing folder: {folder_name}")
        processed_files = get_all_file_text(folder=folder_path)
        all_chunks = []
        for file_info in processed_files:
            all_chunks.extend(file_info['chunks'])

        if all_chunks:
            embeddings = SENTENCE_MODEL.encode(all_chunks, convert_to_tensor=True, show_progress_bar=True)
            FOLDER_DATA[folder_name] = {
                "chunks": all_chunks,
                "embeddings": embeddings
            }
        else:
            print(f"No processable files found in folder: {folder_name}")

    # Also add a combined "all" folder
    all_chunks_combined = []
    for data in FOLDER_DATA.values():
        all_chunks_combined.extend(data["chunks"])

    if all_chunks_combined:
        all_embeddings_combined = SENTENCE_MODEL.encode(all_chunks_combined, convert_to_tensor=True, show_progress_bar=True)
        FOLDER_DATA["all"] = {
            "chunks": all_chunks_combined,
            "embeddings": all_embeddings_combined
        }
    else:
        print("No chunks available across all folders to create 'all' category.")

    print("--- All folder data initialized ---")

# First load default folder
initialize_data()

# ---------------------------------------- Authentication and Authorization Decorators ---------------------------------------------


def require_token(view_func):
    """
    Decorator to ensure that a valid JWT token is provided in the Authorization header.
    Decodes the token and attaches the user payload to the request.
    """
    def wrapper(request, *args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return Response({'error': 'Authorization token required'}, status=status.HTTP_401_UNAUTHORIZED)

        if token.startswith('Bearer '):
            token = token[7:]

        try:
            decoded = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user_role = decoded.get('role')
            request.username = decoded.get('username')
            request.is_super_admin = (request.user_role == 'super_admin')

        except jwt.ExpiredSignatureError:
            return Response({'error': 'Token has expired'}, status=status.HTTP_401_UNAUTHORIZED)
        except jwt.InvalidTokenError:
            return Response({'error': 'Invalid token'}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            return Response({'error': f'Token processing error: {str(e)}'}, status=status.HTTP_401_UNAUTHORIZED)

        return view_func(request, *args, **kwargs)
    return wrapper

def super_admin_required(view_func):
    """
    Decorator to ensure that the authenticated user has super_admin role.
    Assumes require_token has already been applied.
    """
    def wrapper(request, *args, **kwargs):
        if not hasattr(request, 'is_super_admin') or not request.is_super_admin:
            return Response({'error': 'Super Admin privileges required'}, status=status.HTTP_403_FORBIDDEN)
        return view_func(request, *args, **kwargs)
    return wrapper

# ------------------------------------------------- API Endpoints Q&A ----------------------------------------------------------------

@api_view(['GET'])
def hello(request):
    """Simple endpoint to check if the API is running."""
    return Response({"message": "Hello from EDU GEN Q&A API"})

@csrf_exempt
@api_view(['POST'])
# @require_token
def ask_question(request, status_folder=None):
    """
    Answers a question by finding relevant context from loaded documents
    and using the Gemini AI model.
    """
    global FOLDER_DATA, SENTENCE_MODEL

    question = request.data.get('question', '').strip()
    if not question:
        return Response({"error": "Question cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)

    # Handle small talk
    small_talk_response = is_small_talk(question)
    if small_talk_response:
        return Response({
            "question": question,
            "answer_html": f"<p>{small_talk_response}</p>"
        })

    # Default to "all" if no folder is specified
    folder_key = status_folder if status_folder else "all"

    if folder_key not in FOLDER_DATA:
        return Response({
            "error": f"Folder '{folder_key}' not found or no data loaded for it."
        }, status=status.HTTP_404_NOT_FOUND)

    folder_chunks = FOLDER_DATA[folder_key]["chunks"]
    folder_embeddings = FOLDER_DATA[folder_key]["embeddings"]

    print(f"Searching context in folder: {folder_key}")
    question_embedding = SENTENCE_MODEL.encode(question, convert_to_tensor=True)
    cos_scores = util.cos_sim(question_embedding, folder_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(5, len(folder_chunks)))

    context = "\n\n".join([folder_chunks[idx] for idx in top_results[1]])

    # Use session key or IP as identifier for in-memory storage
    session_key = request.META.get('HTTP_X_FORWARDED_FOR', request.META.get('REMOTE_ADDR', 'default'))
    chat_history = _chat_histories.get(session_key, [])

    print("Sending relevant context and history to Gemini API...")
    try:
        answer_text = ask_gemini(context, question, chat_history)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Save the new interaction to the in-memory history
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {answer_text}")
    # Keep the history to the last 5 interactions (10 items: 5 Q&A pairs)
    _chat_histories[session_key] = chat_history[-10:]

    answer_html = f"<div>{answer_text.replace(chr(10), '<br>')}</div>"

    return Response({
        "question": question,
        "answer_html": answer_html
    })

#-----------------------------------------------------generate_token-----------------------------------------------------------------

@csrf_exempt
@api_view(['POST'])
def generate_token(request):
    """
    Generates a JWT token for:
    - Super Admin via .env file (no ID required)
    - Admins/Sub Admins/Super Admins via MongoDB (ID required)
    """
    username = request.data.get("username")
    password = request.data.get("password")
    user_id = request.data.get("id")  # Only required for MongoDB users

    if not username or not password:
        return Response({'error': 'Username and password are required'}, status=status.HTTP_400_BAD_REQUEST)

    # Case 1: Super Admin via .env (no ID check)
    if username == SUPER_ADMIN_USERNAME and password == SUPER_ADMIN_PASSWORD:
        role = "super_admin"
        payload = {
            'username': username,
            'role': role,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
            'iat': datetime.datetime.utcnow()
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        return Response({'token': token, 'role': role})

    # Case 2: Admins/Sub-admins/Super Admins via MongoDB
    # For MongoDB users, an ID is crucial to distinguish from the .env super admin
    if not user_id:
        return Response({'error': 'User ID is required for admin, sub admin or super admin login from database.'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        db = get_mongo_db()
        sub_admins_collection = db.sub_admins

        # Ensure user_id is a valid ObjectId for MongoDB query
        try:
            mongo_user_id = ObjectId(user_id)
        except Exception:
            return Response({"error": "Invalid user ID format."}, status=status.HTTP_400_BAD_REQUEST)

        # Look up user by _id and username
        user_record = sub_admins_collection.find_one({"_id": mongo_user_id, "username": username})

        if user_record:
            stored_hash = user_record.get("password_hash", "")
            if check_password(password, stored_hash):
                role = user_record.get("role", "sub_admin") # Default to sub_admin if role not found
                if role not in ["super_admin", "admin", "sub_admin"]:
                    return Response({"error": f"Invalid role '{role}' in database."}, status=status.HTTP_403_FORBIDDEN)
                
                payload = {
                    'username': username,
                    'role': role,
                    'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24),
                    'iat': datetime.datetime.utcnow()
                }
                token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
                return Response({'token': token, 'role': role})
            else:
                return Response({"error": "Incorrect password."}, status=status.HTTP_401_UNAUTHORIZED)
        else:
            return Response({"error": "User ID or username not found in database."}, status=status.HTTP_404_NOT_FOUND)

    except ConnectionFailure as e:
        return Response({"error": f"Database connection error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({"error": f"Authentication failed: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



#------------------------------------------------------- UPLOAD FILES API-----------------------------------------------------------

@csrf_exempt
@api_view(['POST'])
@require_token
def admin_upload_file(request, status_folder=None):
    """
    Allows authenticated admins to upload files to specific allowed folders.
    """
    allowed_folders = folder_list

    folder_name = status_folder if status_folder else "default"

    if folder_name not in allowed_folders:
        return Response({
            "error": f"Upload failed. '{folder_name}' is not an allowed folder."
        }, status=status.HTTP_403_FORBIDDEN)

    uploaded_files = request.FILES.getlist("file")
    if not uploaded_files:
        return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

    allowed_extensions = [".pdf", ".json", ".xlsx", ".docx"]
    save_dir = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    uploaded_file_names = []

    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name.lower())[1]
        if file_ext not in allowed_extensions:
            return Response({
                "error": f"File '{uploaded_file.name}' has invalid type. Allowed types: {', '.join(allowed_extensions)}"
            }, status=status.HTTP_400_BAD_REQUEST)

        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, 'wb+') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        uploaded_file_names.append(uploaded_file.name)

    return Response({
        "message": f"{len(uploaded_file_names)} file(s) uploaded successfully to '{folder_name}/'.",
        "files": uploaded_file_names
    }, status=status.HTTP_201_CREATED)


#------------------------------------------------DELETE FILES API-------------------------------------------------------------------

@csrf_exempt
@api_view(['DELETE'])
@require_token
def delete_file(request, status_folder, filename=None):
    """
    Allows authenticated admins to delete single or multiple files from allowed folders.
    """
    allowed_folders = files_list

    if status_folder not in allowed_folders:
        return Response({
            "error": f"Deletion failed. '{status_folder}' is not an allowed folder."
        }, status=status.HTTP_403_FORBIDDEN)

    base_folder = os.path.join(UPLOAD_DIR, status_folder)

    # Case 1: Multiple files deletion from request body
    if not filename:
        filenames = request.data.get("filenames")
        if not filenames or not isinstance(filenames, list):
            return Response({"error": "Provide a list of filenames in 'filenames' field."}, status=status.HTTP_400_BAD_REQUEST)

        not_found = []
        deleted = []

        for fname in filenames:
            file_path = os.path.join(base_folder, fname)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted.append(fname)
            else:
                not_found.append(fname)

        return Response({
            "deleted": deleted,
            "not_found": not_found,
            "message": f"{len(deleted)} file(s) deleted from '{status_folder}'."
        }, status=status.HTTP_200_OK)

    # Case 2: Single file deletion from URL
    file_path = os.path.join(base_folder, filename)
    if not os.path.exists(file_path):
        return Response({"error": f"File '{filename}' not found in '{status_folder}'."}, status=status.HTTP_404_NOT_FOUND)

    os.remove(file_path)
    return Response({"message": f"File '{filename}' deleted from '{status_folder}'."}, status=status.HTTP_200_OK)


#------------------------------------------------------Reload Module API-------------------------------------------------------------

@csrf_exempt
@api_view(['POST'])
@require_token
def reload(request):
    """
    Reloads the views module to re-initialize data after file uploads/deletions.
    """
    try:
        # Re-initialize the data, this will re-read all files and re-embed them
        initialize_data()
        return Response({"message": "Model is ready with new uploaded data"})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#----------------------------------------------------File Listing API----------------------------------------------------------------

@api_view(['GET'])
@require_token
def list_uploaded_files(request, status_folder=None):
    """
    Lists all uploaded files with their name, size (KB), and last modified date,
    optionally filtered by folder. Accessible by both Super Admin and Sub Admin.
    """
    if not os.path.exists(UPLOAD_DIR):
        return Response({"message": "No uploads found."}, status=200)

    result = {}

    def get_file_info(file_path):
        size_kb = round(os.path.getsize(file_path) / 1024, 2)
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
        return {
            "filename": os.path.basename(file_path),
            "size_kb": size_kb,
            "last_modified": last_modified
        }

    # Case 1: View a specific folder's files
    if status_folder:
        folder_path = os.path.join(UPLOAD_DIR, status_folder)
        if not os.path.exists(folder_path):
            return Response({"error": f"No folder named '{status_folder}' found."}, status=status.HTTP_404_NOT_FOUND)

        files_info = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                files_info.append(get_file_info(file_path))

        result[status_folder] = files_info
        return Response(result, status=200)

    # Case 2: View all folders and their files
    for folder in os.listdir(UPLOAD_DIR):
        folder_path = os.path.join(UPLOAD_DIR, folder)
        if os.path.isdir(folder_path):
            files_info = []
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    files_info.append(get_file_info(file_path))
            result[folder] = files_info

    return Response(result,status=200)

#-----------------------------------------------------get_uploaded_files-------------------------------------------------------------

@api_view(['POST'])
@require_token
def get_uploaded_files(request, folder_name, file_name=None):
    """
    Allows authenticated admins to download a single file or multiple files (not zipped).
    If multiple files are requested, they are returned as base64 content.
    """
    folder_path = os.path.join(UPLOAD_DIR, folder_name)
    if not os.path.exists(folder_path):
        return Response({"error": f"Folder '{folder_name}' not found."}, status=status.HTTP_404_NOT_FOUND)

    # CASE 1: Return single file (as attachment)
    if file_name:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), as_attachment=True)
        return Response({"error": f"File '{file_name}' not found in folder '{folder_name}'."}, status=status.HTTP_404_NOT_FOUND)

    # CASE 2: Return multiple files (as base64 content)
    try:
        data = JSONParser().parse(request)
        filenames = data.get("filenames", [])
        if not filenames:
            return Response({"error": "No filenames provided."}, status=status.HTTP_400_BAD_REQUEST)

        file_contents = {}
        for fname in filenames:
            file_path = os.path.join(folder_path, fname)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    encoded_content = base64.b64encode(f.read()).decode('utf-8')
                file_contents[fname] = encoded_content
            else:
                return Response({"error": f"File '{fname}' not found in folder '{folder_name}'."},
                                 status=status.HTTP_404_NOT_FOUND)

        return Response({
            "folder": folder_name,
            "files": file_contents,
            "message": f"{len(file_contents)} file(s) successfully returned."
        }, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
# --------------------------------------------------Super Admin specific endpoints ------------------------------------------------

@csrf_exempt
@api_view(['POST'])
@require_token
@super_admin_required
def add_sub_admin(request):
    """
    Allows Super Admin to add new sub-admin or super-admin accounts to MongoDB.
    Expects 'username', 'password', and optional 'role' in request body.
    """
    username = request.data.get('username')
    password = request.data.get('password')
    role = request.data.get('role', 'admin')  # Default role is sub_admin

    if not username or not password:
        return Response({"error": "Username and password are required."}, status=status.HTTP_400_BAD_REQUEST)

    if role not in ['admin', 'super_admin']:
        return Response({"error": "Role must be either 'sub_admin' or 'super_admin'."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        db = get_mongo_db()
        sub_admins_collection = db.sub_admins

        # Check if user already exists
        if sub_admins_collection.find_one({"username": username}):
            return Response({"error": "An admin with this username already exists."}, status=status.HTTP_409_CONFLICT)

        hashed_password = make_password(password)

        admin_data = {
            "username": username,
            "password_hash": hashed_password,
            "password_plain": password,  # store plain text password
            "role": role,
            "created_at": datetime.datetime.utcnow()
        }

        result = sub_admins_collection.insert_one(admin_data)

        if result.inserted_id:
            return Response({
                "message": f"{role.replace('_', ' ').title()} '{username}' added successfully."
            }, status=status.HTTP_201_CREATED)
        else:
            return Response({"error": "Failed to add admin to database."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except ConnectionFailure as e:
        return Response({"error": f"Database connection error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except DuplicateKeyError:
        return Response({"error": "Admin with this username already exists (DB error)."}, status=status.HTTP_409_CONFLICT)
    except Exception as e:
        return Response({"error": f"An error occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


#------------------------------------------------------delete_sub_admin-------------------------------------------------------------


@csrf_exempt
@api_view(['DELETE'])
@require_token  # Ensures token is present and decoded
@super_admin_required  # Ensures user is super admin
def delete_sub_admin(request):
    """
    Allows Super Admin to delete sub-admin or super-admin accounts from MongoDB.
    Requires both 'username' and 'id' in request body.
    """
    username = request.data.get('username')
    admin_id = request.data.get('id')

    if not username or not admin_id:
        return Response({"error": "Both 'username' and 'id' are required."}, status=status.HTTP_400_BAD_REQUEST)

    if username == SUPER_ADMIN_USERNAME:
        return Response({"error": "Cannot delete the super admin account."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        db = get_mongo_db()
        sub_admins_collection = db.sub_admins

        # Attempt to convert ID to ObjectId
        try:
            obj_id = ObjectId(admin_id)
        except Exception:
            return Response({"error": "Invalid admin ID format."}, status=status.HTTP_400_BAD_REQUEST)

        # Delete the admin by matching both username and id
        result = sub_admins_collection.delete_one({
            "_id": obj_id,
            "username": username
        })

        if result.deleted_count > 0:
            return Response({"message": f"Admin '{username}' deleted successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "No matching admin found with the given ID and username."}, status=status.HTTP_404_NOT_FOUND)

    except ConnectionFailure as e:
        return Response({"error": f"Database connection error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({"error": f"An error occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


#---------------------------------------------------list_sub_admins--------------------------------------------------------------


@csrf_exempt
@api_view(['GET'])
@require_token
@super_admin_required
def list_sub_admins(request):
    """
    Allows Super Admin to list all registered sub-admins and super-admins.
    Returns ID, Username, Role, and Plaintext Password.
    """
    try:
        db = get_mongo_db()
        sub_admins_collection = db.sub_admins

        sub_admins_data = []
        for doc in sub_admins_collection.find({}, {"username": 1, "password_plain": 1, "role": 1}):
            sub_admins_data.append({
                "id": str(doc["_id"]),
                "username": doc["username"],
                "password": doc.get("password_plain", "N/A"),
                "role": doc.get("role", "admin")
            })

        return Response({"admins": sub_admins_data}, status=status.HTTP_200_OK)

    except ConnectionFailure as e:
        return Response({"error": f"Database connection error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({"error": f"An error occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

