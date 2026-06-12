from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load PDF
reader = PdfReader("docs/bank_policy.pdf")

text = ""
for page in reader.pages:
    if page.extract_text():
        text += page.extract_text()

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

# QUERY (test)
query = "How long should customer records be stored?"

query_vec = model.encode([query]).astype("float32")

# Search top 2 similar chunks
distances, indices = index.search(query_vec, k=2)

print("\nQUERY:", query)
print("\nMOST RELEVANT CHUNK:\n")
print(chunks[indices[0][0]])