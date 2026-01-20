import sys
import os
import ingestion
import vector_store
import rag
import reranker

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    chunks = ingestion.load_and_chunk_pdf(pdf_path)
    print(f"Total chunks created: {len(chunks)}")

    vectordb = vector_store.build_or_load_vectorstore(chunks)
    print("Vector DB ready!")

    smart_retriever = reranker.get_advanced_retriever(vectordb)
    print("\n----------------------------------")
    print("AI PDF Assistant Ready! (Type 'exit' to quit)")
    print("----------------------------------\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit", "q"]:
            print("Bye 👋")
            break

        retrieved_docs = smart_retriever.invoke(q)

        total_len = sum(len(d.page_content) for d in retrieved_docs)
        if total_len < 500:  
            print("\nAssistant:\nI couldn't find relevant information in the document.\n")
            continue

        answer = rag.answer_question(q, retrieved_docs)

        # Output
        print("\nAssistant:")
        print(answer)
        print("\nSources:")
        print(rag.format_sources(retrieved_docs))
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()