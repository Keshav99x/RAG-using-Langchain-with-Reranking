from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_chroma import Chroma

def get_advanced_retriever(vectordb: Chroma):
    """
    Creates a retrieval pipeline:
    1. Vector Search (Broad): Fetches 20 docs.
    2. Reranking (Precise): Picks the top 5 most relevant.
    """
    
   
    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": 20} 
    )

   
    compressor = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2", 
        top_n=5  
    )

 
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )

    return compression_retriever