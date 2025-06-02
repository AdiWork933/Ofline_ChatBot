import re
import nltk
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import os
import pandas as pd
from docx import Document as DocxDocument
import torch

# Only needed once
nltk.download('punkt')

class ModelHandler:
    def __init__(self):
        # You could replace the below with a quantized model if using one that supports INT8, like bitsandbytes
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        
        # Embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_model_chroma = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        self.chroma_persist_dir = "chroma_store"
        os.makedirs(self.chroma_persist_dir, exist_ok=True)

    def format_text(self, text):
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        lines = text.splitlines()
        formatted = [
            line.title() if line.isupper() and len(line.split()) < 10 else line.strip()
            for line in lines if line.strip()
        ]
        return "\n\n".join(formatted)

    def chunk_docx_text(self, file, tokenizer_name="gpt2", chunk_size=500):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        doc = DocxDocument(file)
        text = "\n".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
        text = self.format_text(text)
        sentences = sent_tokenize(text)

        chunks, current_chunk, current_tokens = [], "", 0
        for sentence in sentences:
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_tokens + len(tokens) <= chunk_size:
                current_chunk += " " + sentence
                current_tokens += len(tokens)
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = len(tokens)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return list(dict.fromkeys(chunks))

    def chunk_excel_text(self, file, tokenizer_name="gpt2", chunk_size=500):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = self.embedding_model
        try:
            df = pd.read_excel(file, sheet_name=None)
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return []
        all_sentences = []
        for sheet_name, sheet_df in df.items():
            try:
                rows = sheet_df.astype(str).apply(lambda x: ' | '.join(x), axis=1).tolist()
                labeled_rows = [f"Sheet: {sheet_name}\n{row}" for row in rows]
                all_sentences.extend(labeled_rows)
            except Exception as e:
                print(f"Error in sheet {sheet_name}: {e}")
                continue
        all_sentences = list(dict.fromkeys(all_sentences))

        embeddings = model.encode(all_sentences, convert_to_tensor=True)
        mean_embedding = torch.mean(embeddings, dim=0, keepdim=True)
        similarities = util.cos_sim(mean_embedding, embeddings)[0]
        ranked_indices = torch.argsort(similarities, descending=True)
        sorted_sentences = [all_sentences[i] for i in ranked_indices]

        chunks, current_chunk, current_tokens = [], "", 0
        for sentence in sorted_sentences:
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if len(tokens) > chunk_size:
                truncated = tokenizer.decode(tokens[:chunk_size])
                chunks.append(truncated.strip())
                continue
            if current_tokens + len(tokens) <= chunk_size:
                current_chunk += " " + sentence
                current_tokens += len(tokens)
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = len(tokens)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return list(dict.fromkeys(chunks))

    def chunk_pdf_text(self, file, tokenizer_name="gpt2", chunk_size=500):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
        text = re.sub(r'(\+?\d{1,3}[-.\s]?|\()?\d{3}[-.\s)]*\d{3}[-.\s]?\d{4}', '', text)
        text = self.format_text(text)
        sentences = sent_tokenize(text)

        chunks, current_chunk, current_tokens = [], "", 0
        for sentence in sentences:
            tokens = tokenizer.encode(sentence, add_special_tokens=False)
            if current_tokens + len(tokens) <= chunk_size:
                current_chunk += " " + sentence
                current_tokens += len(tokens)
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = len(tokens)
        if current_chunk:
            chunks.append(current_chunk.strip())
        return list(dict.fromkeys(chunks))

    def get_vector_store_from_chunks(self, chunks: List[str]):
        if not chunks or not isinstance(chunks, list):
            raise ValueError("Chunks must be a non-empty list of strings.")
        texts = chunks
        metadatas = [{"source": f"chunk_{i}"} for i in range(len(texts))]
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding_model_chroma,
            metadatas=metadatas,
            persist_directory=self.chroma_persist_dir
        )
        vector_store.persist()
        documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
        return vector_store, documents

    def generate_answer(self, question, context):
        is_true_false = any(question.strip().lower().startswith(prefix)
                            for prefix in ["is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "has", "have", "had"])
        if is_true_false:
            prompt = (
                f"You are a fact-checking assistant. Based on the context below, answer ONLY with 'True' or 'False'. "
                f"Do not explain your answer.\n\nContext:\n{context[:1800]}\n\n"
                f"Question: {question}\n\nAnswer (True/False only):"
            )
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            output_ids = self.t5_model.generate(**inputs, max_new_tokens=10)
            answer = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
            return "True" if "true" in answer else "False" if "false" in answer else "Unable to determine."
        else:
            prompt = (
                f"You are a helpful assistant. Based on the following academic context, answer the question in a descriptive and detailed way.\n"
                f"Your answer should include at least 4 complete sentences.\n\n"
                f"Context:\n{context[:1800]}\n\nQuestion: {question}\n\nAnswer:"
            )
            inputs = self.t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            output_ids = self.t5_model.generate(**inputs, max_new_tokens=300)
            answer = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if len(sent_tokenize(answer)) < 4:
                prompt += "\n\nPlease elaborate further."
                inputs = self.t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                output_ids = self.t5_model.generate(**inputs, max_new_tokens=300)
                answer = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return answer.strip()

    def handle_small_talk(self, query):
        greetings = ["hi", "hello", "hey"]
        thanks = ["thanks", "thank you"]
        bye = ["bye", "goodbye", "good night"]

        q = query.lower().strip()
        if q in greetings:
            return "Hello! I'm ready to help you with your documents."
        if q in thanks:
            return "You're welcome! Let me know if you have more questions."
        if q in bye:
            return "Goodbye! Have a great day."
        return None
