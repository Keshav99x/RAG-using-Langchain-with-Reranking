import os
import sys
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "pdf_rag"


def load_and_chunk_pdf(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_or_load_vectorstore(chunks, embed_model: str = "nomic-embed-text"):
   
    embedding = OllamaEmbeddings(model=embed_model)

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )

    if vectordb._collection.count() == 0:
        vectordb.add_documents(chunks)

    return vectordb


def format_sources(docs) -> str:

    seen = set()
    lines = []
    for d in docs:
        source = os.path.basename(d.metadata.get("source", "document"))
        page = d.metadata.get("page", None)
        key = (source, page)

        if key in seen:
            continue
        seen.add(key)

        if page is not None:
            lines.append(f"- {source} (page {page + 1})")
        else:
            lines.append(f"- {source}")
    return "\n".join(lines) if lines else "- No sources"


def answer_question(
    question: str,
    retrieved_docs,
    llm_model: str = "llama3.1"
) -> str:
    
    context = "\n\n".join(
        [f"[PAGE {d.metadata.get('page', '?') + 1}] {d.page_content}" for d in retrieved_docs]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a highly accurate assistant.\n"
         "Answer ONLY using the provided context.\n"
         "If the answer is not in the context, say: "
         "\"I couldn't find that in the document.\" "
         "Do NOT guess or add outside knowledge.\n"
         "Be concise and clear."),
        ("human",
         "QUESTION:\n{question}\n\n"
         "CONTEXT:\n{context}\n\n"
         "Write the answer in 4-10 bullet points (max).")
    ])

    llm = ChatOllama(model=llm_model, temperature=0.2)
    chain = prompt | llm

    response = chain.invoke({"question": question, "context": context})
    return response.content


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_rag_cli.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    print("\n[1/4] Loading & chunking PDF...")
    chunks = load_and_chunk_pdf(pdf_path)
    print(f"Total chunks created: {len(chunks)}")

    print("\n[2/4] Building / Loading vector database (Chroma)...")
    vectordb = build_or_load_vectorstore(chunks)
    print("Vector DB ready!")

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    print("\n[3/4] Ready for questions!")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit", "q"]:
            print("Bye")
            break

        print("\n[4/4] Retrieving relevant context...")
        retrieved_docs = retriever.invoke(q)


        total_len = sum(len(d.page_content) for d in retrieved_docs)
        if total_len < 800:
            print("\nAssistant:")
            print("I couldn't find that in the document.\n")
            continue

        answer = answer_question(q, retrieved_docs)

        print("\nAssistant:")
        print(answer)

        print("\nSources:")
        print(format_sources(retrieved_docs))
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
