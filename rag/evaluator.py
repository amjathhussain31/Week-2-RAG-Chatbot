# rag/evaluator.py

from __future__ import annotations
import uuid
from typing import List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from rag.retriever import get_retriever
from rag.chain import build_rag_chain
from rag.memory import clear_session
from rag.llm import AVAILABLE_MODELS

# ── Default 10 Q&A pairs ───────────────────────────────────────────────────────
# Edit these to match your actual PDF content.
# Questions should be answerable from the document.
# ground_truth should be the correct expected answer.
DEFAULT_EVAL_QA: List[dict] = [
    {
        "question": "What is MapReduce?",
        "ground_truth": "MapReduce is a programming model and data processing technique used to process large datasets in a distributed environment using two main phases: Map and Reduce.",
    },
    {
        "question": "What are the main phases of MapReduce?",
        "ground_truth": "The main phases of MapReduce are Map phase, Shuffle and Sort phase, and Reduce phase.",
    },
    {
        "question": "What happens in the Map phase?",
        "ground_truth": "In the Map phase, input data is split into smaller chunks and processed by mapper functions that generate intermediate key-value pairs.",
    },
]


def _build_ragas_llm(model_name: str) -> LangchainLLMWrapper:
    """
    Wrap the selected Ollama model as a RAGAS-compatible LLM judge.
    Uses non-streaming mode — RAGAS needs complete responses.
    """
    model_id = AVAILABLE_MODELS.get(model_name, "mistral")
    ollama_llm = ChatOllama(
        model       = model_id,
        temperature = 0,
        streaming   = False,   # RAGAS requires full response, not streamed tokens
        timeout         = 600,      # ← add this — 2 min max per judge call
        num_predict     = 256,      # ← cap output length, speeds up judging
    )
    return LangchainLLMWrapper(ollama_llm)


def _build_ragas_embeddings(model_name: str) -> LangchainEmbeddingsWrapper:
    """
    Wrap Ollama embeddings as a RAGAS-compatible embeddings model.
    Uses the same model for consistency with the retriever.
    """
    model_id = AVAILABLE_MODELS.get(model_name, "mistral")
    ollama_embeddings = OllamaEmbeddings(model=model_id)
    return LangchainEmbeddingsWrapper(ollama_embeddings)


def _invoke_chain_for_eval(
    chain,
    question: str,
    session_id: str,
) -> str:
    """Run the RAG chain for one question, return the full answer string."""
    cfg = {"configurable": {"session_id": session_id}}
    chunks = []
    for chunk in chain.stream({"question": question}, config=cfg):
        chunks.append(chunk)
    return "".join(chunks)


def _retrieve_contexts(
    vectorstore: FAISS,
    question: str,
    k: int = 4,
) -> List[str]:
    """
    Retrieve raw chunk texts for a question.
    RAGAS expects a list of strings, not Document objects.
    """
    retriever = get_retriever(vectorstore, k=k)
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def run_ragas_eval(
    vectorstore: FAISS,
    model_name: str = "Mistral 7B",
    k: int = 4,
    qa_pairs: List[dict] | None = None,
    progress_callback=None,
) -> dict:
    if qa_pairs is None:
        qa_pairs = DEFAULT_EVAL_QA

    questions, answers, contexts, ground_truths = [], [], [], []
    total = len(qa_pairs)

    eval_session = f"eval-{uuid.uuid4()}"
    chain = build_rag_chain(vectorstore, model_name=model_name, k=k)

    # ── Step 1: generate answers + retrieve contexts ──────────────────────────
    for i, qa in enumerate(qa_pairs):
        q  = qa["question"]
        gt = qa["ground_truth"]

        if progress_callback:
            progress_callback(i, total, f"Generating answer {i+1}/{total}: {q[:50]}…")

        answer = _invoke_chain_for_eval(chain, q, eval_session)
        ctx    = _retrieve_contexts(vectorstore, q, k=k)

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(gt)

    clear_session(eval_session)

    # ── Step 2: score each metric manually, row by row ────────────────────────
    # Bypasses evaluate() entirely — avoids the asyncio deadlock with Ollama.
    if progress_callback:
        progress_callback(total, total, "Running RAGAS scoring locally via Ollama…")

    ragas_llm        = _build_ragas_llm(model_name)
    ragas_embeddings = _build_ragas_embeddings(model_name)

    for metric in [faithfulness, answer_relevancy, context_recall]:
        metric.llm        = ragas_llm
        metric.embeddings = ragas_embeddings

    import asyncio

    def _get_loop():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _score_metric(metric, row: dict) -> float:
        """Score a single metric on a single row, with timeout protection."""
        loop = _get_loop()
        single = Dataset.from_dict({
            "question":     [row["question"]],
            "answer":       [row["answer"]],
            "contexts":     [row["contexts"]],
            "ground_truth": [row["ground_truth"]],
        })
        try:
            future = asyncio.ensure_future(
                metric.ascore(single.to_pandas().iloc[0]),
                loop=loop,
            )
            return loop.run_until_complete(
                asyncio.wait_for(future, timeout=600)   # 3 min hard cap per cell
            )
        except asyncio.TimeoutError:
            print(f"  ⚠ Timeout scoring {metric.name} — returning 0.0")
            return 0.0
        except Exception as e:
            print(f"  ⚠ Error scoring {metric.name}: {e} — returning 0.0")
            return 0.0

    # Build per-row results
    per_question = []
    faith_scores, relevancy_scores, recall_scores = [], [], []

    for i, (q, a, ctx, gt) in enumerate(
        zip(questions, answers, contexts, ground_truths)
    ):
        if progress_callback:
            progress_callback(
                i, total,
                f"Scoring row {i+1}/{total}: {q[:40]}…"
            )

        row = {"question": q, "answer": a, "contexts": ctx, "ground_truth": gt}

        f_score  = _score_metric(faithfulness,      row)
        r_score  = _score_metric(answer_relevancy,  row)
        rc_score = _score_metric(context_recall,    row)

        faith_scores.append(f_score)
        relevancy_scores.append(r_score)
        recall_scores.append(rc_score)

        per_question.append({
            "question":         q,
            "answer":           a,
            "ground_truth":     gt,
            "faithfulness":     round(f_score,  3),
            "answer_relevancy": round(r_score,  3),
            "context_recall":   round(rc_score, 3),
        })

        print(
            f"  Row {i+1}: faith={f_score:.2f}  "
            f"relevancy={r_score:.2f}  recall={rc_score:.2f}"
        )

    scores = {
        "faithfulness":     round(sum(faith_scores)     / max(len(faith_scores),     1), 3),
        "answer_relevancy": round(sum(relevancy_scores) / max(len(relevancy_scores), 1), 3),
        "context_recall":   round(sum(recall_scores)    / max(len(recall_scores),    1), 3),
    }

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    return {
        "scores":       scores,
        "per_question": per_question,
        "dataset":      dataset,
    }