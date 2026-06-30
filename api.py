from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
import faiss
import numpy as np
import os
from dotenv import load_dotenv

from src.rag import build_index, retrieve, generate_answer

load_dotenv()

app = FastAPI()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load PDF
text = ""

for filename in os.listdir("docs"):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join("docs", filename))
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

# Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_text(text)

# Index
model, index = build_index(chunks)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    retrieved = retrieve(q.question, model, index, chunks)
    context = "\n\n".join(retrieved)

    answer = generate_answer(q.question, context, client)

    return {
        "question": q.question,
        "answer": answer,
        "sources": retrieved
    }