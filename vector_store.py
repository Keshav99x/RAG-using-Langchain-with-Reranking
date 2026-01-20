from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config

def build_or_load_vectorstore(chunks):
    """
    Creates (or loads) a Chroma vector database.
    """
    embedding = OllamaEmbeddings(model=config.EMBED_MODEL)

    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=config.PERSIST_DIR
    )

    if vectordb._collection.count() == 0 and chunks:
        print("Indexing new document...")
        vectordb.add_documents(chunks)
    
    return vectordb