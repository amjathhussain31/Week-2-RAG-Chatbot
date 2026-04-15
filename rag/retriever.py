# rag/retriever.py

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(
    vectorstore: FAISS,
    search_type: str = "similarity",
    k: int = 4
) -> VectorStoreRetriever:
    """
    Build a retriever from the vectorstore.

    search_type options:
      'similarity' → pure cosine distance (default, recommended)
      'mmr'        → maximal marginal relevance (diverse results)

    k: number of chunks to retrieve per query
    """
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type   = "mmr",
            search_kwargs = {"k": k, "fetch_k": k * 4},
        )
    return vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": k},
    )

def format_docs(docs) -> str:
    """
    Convert retrieved Document objects into a single
    context string for the prompt.
    Includes page citation so the LLM can reference it.
    """
    formatted = []
    for doc in docs:
        page   = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"[Source: {source}, Page: {int(page)+1}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)