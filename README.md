# 📄 RAG-Based Compliance Assistant

## 🚀 Overview

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions from PDF documents and get accurate, context-based answers using LLMs.

---

## 🧠 Problem Statement

Manual reading of long policy/compliance documents is slow and inefficient.
LLMs alone may hallucinate without proper context.

---

## 💡 Solution

This system solves it using a RAG pipeline:

* PDF documents are loaded and split into chunks
* Text is converted into embeddings using SentenceTransformer
* FAISS is used for semantic similarity search
* Gemini LLM generates final answers using retrieved context

---

## 🏗️ Architecture

User Query
→ Embedding Model
→ FAISS Vector Search
→ Top Relevant Chunk
→ Gemini LLM
→ Final Answer

---

## 🛠️ Tech Stack

* Python
* FAISS
* SentenceTransformers
* Google Gemini API
* PyPDF

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
```

Create `.env` file:

```
GOOGLE_API_KEY=your_api_key
```

Run:

```bash
python app.py
```

---

## 📌 Example Queries

* What is password policy?
* How long should customer records be retained?
* When must security incidents be reported?

---

## 🔗 GitHub

(https://github.com/Swathitakkellapati9/llm-compliance-assistant)
