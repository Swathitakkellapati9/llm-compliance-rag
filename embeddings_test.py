from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Customer records must be retained for 7 years."

embedding = model.encode(text)

print(type(embedding))
print(len(embedding))
print(embedding[:10])