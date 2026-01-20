# config.py
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
# Database Settings
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "pdf_rag"

# Text Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# AI Model Settings
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"