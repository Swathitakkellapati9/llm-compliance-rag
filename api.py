from fastapi import FastAPI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

app = FastAPI()

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load PDF text
def load_text():
    text = ""
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):
            path = os.path.join("docs", filename)
            reader = PdfReader(path)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
    return text

# Build chunks + index once at startup
text = load_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/ask")
def ask(question: str):
    query_vec = model.encode([question]).astype("float32")
    distances, indices = index.search(query_vec, k=3)

    retrieved = [chunks[i] for i in indices[0]]
    answer = "\n\n".join(retrieved)

    return {
        "question": question,
        "answer": answer
    }