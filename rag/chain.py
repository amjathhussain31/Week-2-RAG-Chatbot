# rag/chain.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from operator import itemgetter
from typing import List

from rag.llm       import get_llm
from rag.retriever import get_retriever, format_docs
from rag.memory    import wrap_with_memory

# ── Stronger system prompt ────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a strict document assistant. You answer ONLY \
from the sources listed below. You have NO other knowledge.

RULE 1 — DOCUMENT CONTEXT
  If the answer is in the document chunks below → answer from it.
  Always cite the page: (Page 3)

RULE 2 — CONVERSATION HISTORY
  If the answer is not in the document but the user mentioned it
  earlier (e.g. their name) → answer from history.

RULE 3 — OUT OF SCOPE
  If the answer is in NEITHER the document NOR the conversation
  → say EXACTLY this, nothing more:
  "I cannot find this information in the document or our conversation."
  and never assume page numbers.

RULE 4 — FORBIDDEN BEHAVIOURS (never do these)
  ✗ Never say "based on common knowledge"
  ✗ Never say "it can be inferred"
  ✗ Never say "generally speaking"
  ✗ Never say "typically" to fill gaps
  ✗ Never use training data to answer
  ✗ Never guess page numbers
  ✗ Never add information not in the context

RULE 5 — PRONOUNS
  If the user says "it", "they", "those", "these" → resolve from
  conversation history before answering.

Document context:
{context}"""


# ── Query rewriter prompt ─────────────────────────────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter for a RAG retrieval system.

Rewrite the user's latest question into a clear, self-contained question
that can be understood without reading the conversation history.

Rules:
- Replace all pronouns (it, they, those, these, them, its) with the
  actual subject from the conversation history.
- If the question is already clear and standalone, return it unchanged.
- Return ONLY the rewritten question. Nothing else.
- If you cannot determine the subject, return the question unchanged."""),
    ("human", """Conversation so far:
{history_text}

User's question: {question}

Rewritten question:"""),
])


def history_to_text(messages: List[BaseMessage]) -> str:
    """Convert message objects to plain text for the rewriter prompt."""
    if not messages:
        return "No previous conversation."
    lines = []
    for m in messages:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def build_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",  RAG_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human",  "{question}"),
    ])


def build_rag_chain(
    vectorstore: FAISS,
    model_name:  str = "Mistral 7B",
    k:           int = 4,
):
    retriever    = get_retriever(vectorstore, k=k)
    llm          = get_llm(model_name, streaming=True)
    rewrite_llm  = get_llm(model_name, streaming=False)  # non-streaming for rewriter
    prompt       = build_rag_prompt()
    rewriter     = REWRITE_PROMPT | rewrite_llm | StrOutputParser()

    def rewrite_question(inputs: dict) -> str:
        """
        Rewrite the question to be standalone before retrieval.
        Uses conversation history to resolve pronouns and references.
        """
        question     = inputs["question"]
        history      = inputs.get("history", [])
        history_text = history_to_text(history)

        # Only rewrite if there IS history to refer back to
        if not history or len(history) == 0:
            return question

        # Check if question has ambiguous references worth rewriting
        ambiguous_words = [
            "it", "they", "those", "these", "them", "its",
            "that", "this", "such", "above", "previous",
            "mentioned", "those points", "these points"
        ]
        question_lower = question.lower()
        needs_rewrite  = any(w in question_lower for w in ambiguous_words)

        # Also rewrite very short questions that depend on context
        needs_rewrite = needs_rewrite or len(question.split()) <= 5

        if not needs_rewrite:
            return question

        try:
            rewritten = rewriter.invoke({
                "history_text": history_text,
                "question":     question,
            })
            # Clean up the rewritten question
            rewritten = rewritten.strip().strip('"').strip("'")
            print(f"\n  [Query Rewriter]")
            print(f"  Original : {question}")
            print(f"  Rewritten: {rewritten}\n")
            return rewritten
        except Exception:
            return question  # fallback to original if rewriter fails

    # ── Core chain ────────────────────────────────────────────────────────────
    core_chain = (
        RunnableMap({
            # Rewrite → retrieve → format
            "context":  RunnableLambda(rewrite_question)
                        | retriever
                        | format_docs,
            "question": itemgetter("question"),
            "history":  itemgetter("history"),
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return wrap_with_memory(core_chain)