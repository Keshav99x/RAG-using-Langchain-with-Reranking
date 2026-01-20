import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import config

def format_sources(docs) -> str:
    """
    Builds a clean citations text block from retrieved docs.
    """
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

def answer_question(question: str, retrieved_docs) -> str:
    """
    Generates an answer using ONLY the retrieved context.
    """
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

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.2)
    chain = prompt | llm
    
    response = chain.invoke({"question": question, "context": context})
    return response.content