# rag/vectorstore.py — with both options

import os
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from rag.embeddings import get_embeddings
from typing import List, Literal

FAISS_PATH  = "data/faiss_index"
CHROMA_PATH = "data/chroma_db"

def build_vectorstore(
    chunks: List[Document],
    store_type: Literal["faiss", "chroma"] = "faiss"
):
    """
    Build a vector store from document chunks.
    
    store_type: "faiss"  → FAISS (default, fast, file-based)
                "chroma" → ChromaDB (auto-persist, metadata filtering)
    """
    embeddings = get_embeddings()
    print(f"  Building {store_type.upper()} vectorstore "
          f"from {len(chunks)} chunks...")

    if store_type == "chroma":
        vectorstore = Chroma.from_documents(
            documents         = chunks,
            embedding         = embeddings,
            persist_directory = CHROMA_PATH
        )
        print(f"  ChromaDB saved to: {CHROMA_PATH}")
        return vectorstore

    # Default: FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"  FAISS index saved to: {FAISS_PATH}")
    return vectorstore


def load_vectorstore(
    store_type: Literal["faiss", "chroma"] = "faiss"
):
    """Load a previously saved vector store."""
    embeddings = get_embeddings()

    if store_type == "chroma":
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(
                f"No ChromaDB found at '{CHROMA_PATH}'. "
                f"Run build_vectorstore() first."
            )
        vectorstore = Chroma(
            persist_directory  = CHROMA_PATH,
            embedding_function = embeddings
        )
        print(f"  ChromaDB loaded from: {CHROMA_PATH}")
        return vectorstore

    # Default: FAISS
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(
            f"No FAISS index found at '{FAISS_PATH}'. "
            f"Run build_vectorstore() first."
        )
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"  FAISS index loaded from: {FAISS_PATH}")
    return vectorstore


def get_or_build_vectorstore(
    chunks: List[Document] = None,
    store_type: Literal["faiss", "chroma"] = "faiss"
):
    """
    Smart loader:
      - If index exists → load it (fast path)
      - If not → build from chunks (one-time cost)
    """
    path_exists = (
        os.path.exists(FAISS_PATH)  if store_type == "faiss"
        else os.path.exists(CHROMA_PATH)
    )

    if path_exists:
        return load_vectorstore(store_type)

    if chunks is None:
        raise ValueError(
            "No existing vectorstore found and no chunks provided."
        )
    return build_vectorstore(chunks, store_type)
'''
```

Now in `app.py` sidebar you can add a store selector and it just works — nothing else in the chain needs to change. That's the whole point.

---

## One Line Summary
```
FAISS  = fastest pure vector search, explicit control, industry standard
Chroma = easier persistence, metadata filtering, better for multi-tenant

For this app: FAISS
For a production multi-user app with filters: Chroma
Both teach you the same LangChain retrieval pattern
'''