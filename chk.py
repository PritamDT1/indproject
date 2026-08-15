from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# client = genai.Client()
# models = client.models.list()

# for m in models:
#     print(m.name, m.supported_actions)


import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- File reader (your function) ---
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    file_text = ""

    if ext == ".pdf":
        reader = PdfReader(file_path)
        file_text = "".join(page.extract_text() + "\n" for page in reader.pages)

    elif ext == ".docx":
        doc = Document(file_path)
        file_text = "\n".join([para.text for para in doc.paragraphs])

    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            file_text = f.read()

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        file_text = df.to_string()

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        file_text = df.to_string()

    elif ext == ".json":
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_text = str(data)

    else:
        file_text = "Unsupported file type."

    return file_text

# --- Chunking ---
def chunk_text(raw_text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(raw_text)

# --- Build vector store from multiple files ---
def create_vector_store(file_paths):
    all_chunks = []
    for path in file_paths:
        text = read_file(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = FAISS.from_texts(all_chunks, embeddings)
    return vector_store

# --- Semantic search ---
def semantic_search(vector_store, query, k=3):
    docs = vector_store.similarity_search(query, k=k)
    return docs

