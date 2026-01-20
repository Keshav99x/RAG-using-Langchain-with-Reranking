# 🔍 Local RAG with Hybrid Search & Reranking

A high-precision, **100% offline** RAG (Retrieval-Augmented Generation) system for chatting with PDF documents. 

Unlike standard RAG implementations that rely solely on vector similarity, this project implements a **Two-Stage Retrieval Pipeline** using a Cross-Encoder (FlashRank) to drastically improve answer quality and reduce hallucinations.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Privacy](https://img.shields.io/badge/privacy-100%25%20Offline-green)

## 🚀 Key Features

* **🔒 Privacy-First:** Runs entirely locally. No data leaves your machine. Telemetry is explicitly disabled for ChromaDB.
* **🧠 Advanced Reranking:** Uses **FlashRank** (ms-marco-MiniLM-L-12-v2) to re-score and filter documents, ensuring the LLM only sees the most relevant context.
* **⚡ Hybrid Pipeline:**
    1.  **Broad Search:** Retrieves top 20 candidates via Vector Search (ChromaDB).
    2.  **Precision Filter:** Reranks and selects top 5 "Gold" chunks.
* **📝 Smart Citations:** Answers include exact source page numbers (e.g., `[Source: document.pdf, page 4]`).
* **🤖 State-of-the-Art LLM:** Powered by **Llama 3.1** via Ollama.

## 🛠️ Architecture

1.  **Ingestion:** PDF → Text Chunks (1000 chars) → Embeddings (nomic-embed-text) → ChromaDB.
2.  **Retrieval:** User Query → Vector Search (Top 20) → **Cross-Encoder Reranking** (Top 5).
3.  **Generation:** Top 5 Chunks + System Prompt → Llama 3.1 → Answer.

## 📦 Installation

### 1. Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running.

### 2. Clone the Repository
```bash
git clone [https://github.com/Keshav99x/RAG-using-Langchain-with-Reranking.git](https://github.com/Keshav99x/RAG-using-Langchain-with-Reranking.git)
cd RAG-using-Langchain-with-Reranking
