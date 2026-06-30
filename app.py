from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import os
from dotenv import load_dotenv

from src.rag import build_index, retrieve, generate_answer

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# ---------------- LOAD PDF ----------------
text = ""

for filename in os.listdir("docs"):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join("docs", filename))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

# ---------------- CHUNKING ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# ---------------- BUILD INDEX ----------------
model, index = build_index(chunks)

# ---------------- QUERY LOOP ----------------
while True:
    query = input("\nAsk a question (type 'exit' to stop): ")

    if query.lower() == "exit":
        break

    retrieved_chunks = retrieve(query, model, index, chunks)

    context = "\n\n".join(retrieved_chunks)

    print("\nSOURCES USED:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\nSource {i}: {chunk[:200]}")

    answer = generate_answer(query, context, client)

    print("\nANSWER:\n")
    print(answer)