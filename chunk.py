from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from reader import read_file

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

