from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
import os

def build_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def retrieve(query, model, index, chunks, k=3):
    query_vec = model.encode([query]).astype("float32")
    _, indices = index.search(query_vec, k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks


def generate_answer(query, context, client):
    prompt = f"""
You are a compliance assistant.

Your job is to explain concepts using policy statements.

Even if a formal definition is not present, infer the meaning from the given rules.

Context:
{context}

Question:
{query}

Instructions:
- Combine all relevant policy statements
- Explain what the concept means in simple terms
- Do NOT say information is missing unless completely unrelated

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text