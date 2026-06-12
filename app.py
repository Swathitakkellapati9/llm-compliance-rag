from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from google import genai
import faiss
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)
# Load PDF

text = ""

for filename in os.listdir("docs"):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join("docs", filename)
        
        print("Loading:", filename)

        reader = PdfReader(pdf_path)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"
# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query
query = input("Ask a question: ")

query_vec = model.encode([query]).astype("float32")

distances, indices = index.search(query_vec, k=3)

retrieved_chunks = []

for idx in indices[0]:
    retrieved_chunks.append(chunks[idx])

answer_chunk = "\n\n".join(retrieved_chunks)
print("\nRETRIEVED CONTEXT:\n")
print(answer_chunk)

prompt = f"""
You are a compliance assistant.

Answer ONLY from the provided context.

If the answer is not in the context, say:
'I could not find that information in the provided document.'

Context:
{answer_chunk}

Question:
{query}
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print("\nANSWER:\n")
print(response.text)