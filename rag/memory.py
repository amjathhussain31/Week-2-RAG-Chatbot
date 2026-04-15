# rag/memory.py

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable

# In-memory store: { session_id: InMemoryChatMessageHistory }
# For production, replace with SQLite or Redis backend
_store: dict = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Return the message history for a given session.
    Creates a new empty history if session doesn't exist yet.
    """
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

def clear_session(session_id: str) -> None:
    """Clear conversation history for a session."""
    if session_id in _store:
        _store[session_id].clear()
        print(f"  Session '{session_id}' cleared.")

def wrap_with_memory(chain: Runnable) -> RunnableWithMessageHistory:
    """
    Wrap any LCEL chain with automatic memory management.

    Before each invoke:
      - Fetches history for session_id
      - Injects it into the 'history' placeholder in the prompt

    After each invoke:
      - Appends (human message, ai response) to history

    The chain's prompt MUST have MessagesPlaceholder(variable_name='history')
    """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key   = "question",
        history_messages_key = "history",
    )