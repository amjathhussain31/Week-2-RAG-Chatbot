# rag/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model.
    Downloads ~90MB on first call, cached locally after that.
    Runs on CPU — no GPU needed.
    """
    return HuggingFaceEmbeddings(
        model_name      = EMBEDDING_MODEL,
        model_kwargs    = {"device": "cpu"},
        encode_kwargs   = {"normalize_embeddings": True},
    )
    
'''
```

### Why `normalize_embeddings=True`?
```
Without normalisation:
  Similarity score depends on vector length AND direction.
  Long documents get artificially high scores.

With normalisation:
  All vectors scaled to length 1.
  Similarity score depends on direction ONLY.
  Fairer comparison across chunks of different sizes.
'''