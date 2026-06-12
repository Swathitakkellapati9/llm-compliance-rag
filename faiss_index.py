from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load PDF
reader = PdfReader("docs/bank_policy.pdf")

text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text

# 2. Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# 3. Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert chunks → vectors
embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

# 5. FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index built")
print("Total chunks:", len(chunks))