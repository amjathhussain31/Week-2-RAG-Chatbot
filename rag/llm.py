# rag/llm.py

from langchain_ollama import ChatOllama

# All available local models — add more as you pull them
AVAILABLE_MODELS = {
    "Mistral 7B":  "mistral",
    "LLaMA 2 7B":  "llama2",
}

def get_llm(model_name: str, streaming: bool = True) -> ChatOllama:
    """
    Return a ChatOllama instance for the selected model.
    
    model_name : key from AVAILABLE_MODELS dict
                 e.g. "Mistral 7B"
    streaming  : True = tokens stream live in UI
                 False = wait for full response (testing only)
    """
    model_id = AVAILABLE_MODELS.get(model_name)
    
    if model_id is None:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Choose from: {list(AVAILABLE_MODELS.keys())}"
        )
    
    return ChatOllama(
        model       = model_id,
        temperature = 0,         # 0 = deterministic, best for RAG
        streaming   = streaming,
    )