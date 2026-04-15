# app.py

import streamlit as st
import uuid
from rag.loader      import load_pdf
from rag.splitter    import split_documents
from rag.vectorstore import get_or_build_vectorstore, build_vectorstore
from rag.chain       import build_rag_chain
from rag.memory      import clear_session
from rag.llm         import AVAILABLE_MODELS
from rag.evaluator import run_ragas_eval, DEFAULT_EVAL_QA
import tempfile, os

import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")


# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "RAG Chatbot",
    page_icon  = "🤖",
    layout     = "wide",
)

st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────
# These persist across Streamlit reruns
if "messages"     not in st.session_state:
    st.session_state.messages     = []          # chat history for display
if "rag_chain"    not in st.session_state:
    st.session_state.rag_chain    = None        # built chain
if "vectorstore"  not in st.session_state:
    st.session_state.vectorstore  = None        # FAISS index
if "session_id"   not in st.session_state:
    st.session_state.session_id   = str(uuid.uuid4())  # unique per browser tab
if "active_model" not in st.session_state:
    st.session_state.active_model = "Mistral 7B"
if "doc_loaded"   not in st.session_state:
    st.session_state.doc_loaded   = False

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    # File upload
    st.subheader("📄 Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload the PDF you want to chat with."
    )

    # Chunking settings
    st.subheader("✂️ Chunking")
    chunk_size    = st.slider("Chunk size",    200,  2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap",   0,   500,  200,  50)

    # Model selection
    st.subheader("🤖 Model")
    selected_model = st.selectbox(
        "Choose model",
        options = list(AVAILABLE_MODELS.keys()),
        index   = 0,
    )

    # Retrieval settings
    st.subheader("🔍 Retrieval")
    k_chunks = st.slider("Chunks to retrieve (k)", 1, 8, 4)

    st.divider()

    # Process button
    process_btn = st.button(
        "⚡ Process Document",
        type      = "primary",
        use_container_width = True,
        disabled  = uploaded_file is None,
    )

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # Model switch — rebuild chain if model changed
    if (st.session_state.vectorstore is not None
            and selected_model != st.session_state.active_model):
        with st.spinner(f"Switching to {selected_model}..."):
            st.session_state.rag_chain    = build_rag_chain(
                st.session_state.vectorstore,
                model_name = selected_model,
                k          = k_chunks,
            )
            st.session_state.active_model = selected_model
            # Clear chat — new model, fresh start
            st.session_state.messages     = []
            clear_session(st.session_state.session_id)
            st.session_state.session_id   = str(uuid.uuid4())
        st.success(f"Switched to {selected_model}")

# ── Process uploaded document ─────────────────────────────────
if process_btn and uploaded_file is not None:
    with st.spinner("Reading and indexing document..."):
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Full pipeline: load → split → embed → index
            pages      = load_pdf(tmp_path)
            chunks     = split_documents(
                pages,
                chunk_size    = chunk_size,
                chunk_overlap = chunk_overlap,
            )
            vectorstore = build_vectorstore(chunks)

            # Build chain with selected model
            rag_chain   = build_rag_chain(
                vectorstore,
                model_name = selected_model,
                k          = k_chunks,
            )

            # Store in session state
            st.session_state.vectorstore  = vectorstore
            st.session_state.rag_chain    = rag_chain
            st.session_state.active_model = selected_model
            st.session_state.doc_loaded   = True
            st.session_state.messages     = []
            clear_session(st.session_state.session_id)
            st.session_state.session_id   = str(uuid.uuid4())

        finally:
            os.unlink(tmp_path)  # clean up temp file

    st.sidebar.success(
        f"✅ Loaded {len(pages)} pages → {len(chunks)} chunks"
    )

# ── Main area — tabs ──────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluate (RAGAS)"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — chat (identical to your original, just indented)
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.title("🤖 RAG Chatbot")
    st.caption(
        f"Model: **{st.session_state.active_model}** | "
        f"Session: `{st.session_state.session_id[:8]}...`"
    )

    if not st.session_state.doc_loaded:
        st.info(
            "👈 Upload a PDF in the sidebar and click "
            "**Process Document** to start chatting."
        )
        with st.expander("ℹ️ How this app works"):
            st.markdown("""
            1. **Upload** a PDF in the sidebar
            2. **Process** — the app reads, chunks, and indexes it
            3. **Chat** — ask questions about the document
            4. **Memory** — the bot remembers earlier turns
            5. **Switch models** — change the model anytime
            """)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input(
            "Ask anything about your document...",
            disabled=st.session_state.rag_chain is None,
        ):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                try:
                    cfg = {"configurable": {"session_id": st.session_state.session_id}}
                    for chunk in st.session_state.rag_chain.stream(
                        {"question": question}, config=cfg
                    ):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"⚠️ Error: {str(e)}"
                    response_placeholder.error(full_response)

            st.session_state.messages.append({
                "role": "assistant", "content": full_response
            })

# ══════════════════════════════════════════════════════════════
# TAB 2 — RAGAS evaluation (fully local, no OpenAI)
# ══════════════════════════════════════════════════════════════
with tab_eval:
    st.title("📊 RAGAS Evaluation")
    st.caption("Fully local — runs via Ollama, no API keys needed.")

    if not st.session_state.doc_loaded:
        st.info("👈 Process a document first, then return here to evaluate.")
    else:
        st.markdown(
            "Runs your Q&A pairs through the RAG chain and scores with RAGAS. "
            "The **same Ollama model** acts as both the RAG chain and the RAGAS judge."
        )

        # ── Edit Q&A pairs ────────────────────────────────────
        with st.expander("✏️ Edit evaluation Q&A pairs", expanded=False):
            st.caption(
                "Replace these with questions and correct answers from your PDF."
            )
            if "eval_qa" not in st.session_state:
                st.session_state.eval_qa = list(DEFAULT_EVAL_QA)

            edited_qa = []
            for i, qa in enumerate(st.session_state.eval_qa):
                cols = st.columns([2, 2, 0.3])
                q  = cols[0].text_input(
                    f"Question {i+1}", value=qa["question"], key=f"q_{i}"
                )
                gt = cols[1].text_input(
                    f"Ground truth {i+1}", value=qa["ground_truth"], key=f"gt_{i}"
                )
                if cols[2].button("🗑", key=f"del_{i}"):
                    st.session_state.eval_qa.pop(i)
                    st.rerun()
                edited_qa.append({"question": q, "ground_truth": gt})

            col_add, col_save = st.columns(2)
            if col_add.button("➕ Add pair"):
                st.session_state.eval_qa.append({"question": "", "ground_truth": ""})
                st.rerun()
            if col_save.button("💾 Save pairs"):
                st.session_state.eval_qa = edited_qa
                st.success("Saved!")

        # ── Warn user about duration ───────────────────────────
        n_pairs = len(st.session_state.get("eval_qa", DEFAULT_EVAL_QA))
        st.info(
            f"⏱ {n_pairs} pairs × ~30–60s per pair on Ollama = "
            f"**~{n_pairs * 1} – {n_pairs * 2} mins** total. "
            "Keep Ollama running in the background."
        )

        # ── Run button ─────────────────────────────────────────
        if st.button("▶ Run RAGAS Evaluation", type="primary"):
            progress_bar = st.progress(0)
            status_text  = st.empty()
            qa_pairs     = st.session_state.get("eval_qa", DEFAULT_EVAL_QA)
            total        = len(qa_pairs)

            def update_progress(current, total, label):
                progress_bar.progress(
                    min(current / max(total, 1), 1.0)
                )
                status_text.caption(label)

            try:
                eval_result = run_ragas_eval(
                    vectorstore       = st.session_state.vectorstore,
                    model_name        = st.session_state.active_model,
                    k                 = k_chunks,
                    qa_pairs          = qa_pairs,
                    progress_callback = update_progress,
                )

                progress_bar.progress(1.0)
                status_text.empty()

                # ── Score cards ───────────────────────────────
                scores = eval_result["scores"]
                st.subheader("Overall scores")

                def badge(v):
                    if v >= 0.8: return "🟢"
                    if v >= 0.5: return "🟡"
                    return "🔴"

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    f"{badge(scores['faithfulness'])} Faithfulness",
                    f"{scores['faithfulness']:.3f}",
                    help="Does the answer stick to retrieved context? 1.0 = perfect.",
                )
                c2.metric(
                    f"{badge(scores['answer_relevancy'])} Answer relevancy",
                    f"{scores['answer_relevancy']:.3f}",
                    help="Is the answer on-topic for the question? 1.0 = perfect.",
                )
                c3.metric(
                    f"{badge(scores['context_recall'])} Context recall",
                    f"{scores['context_recall']:.3f}",
                    help="Did retrieval fetch the right chunks? 1.0 = perfect.",
                )

                # ── Interpretation ────────────────────────────
                st.subheader("What the scores reveal")
                issues = []
                if scores["faithfulness"] < 0.7:
                    issues.append(
                        "**Low faithfulness** — the LLM is adding info not in the "
                        "retrieved chunks. Tighten the system prompt or increase `k`."
                    )
                if scores["answer_relevancy"] < 0.7:
                    issues.append(
                        "**Low answer relevancy** — answers are drifting off-topic. "
                        "Retrieved context may be noisy; try MMR search or smaller chunks."
                    )
                if scores["context_recall"] < 0.7:
                    issues.append(
                        "**Low context recall** — retrieval is missing key chunks. "
                        "Try smaller chunk size, larger `k`, or check your embeddings model."
                    )
                if not issues:
                    st.success("All scores ≥ 0.7 — pipeline looks healthy! 🎉")
                else:
                    for issue in issues:
                        st.warning(issue)

                # ── Per-question table ────────────────────────
                with st.expander("🔍 Per-question breakdown", expanded=True):
                    import pandas as pd
                    df = pd.DataFrame(eval_result["per_question"])
                    show_cols = [
                        c for c in [
                            "question", "answer",
                            "faithfulness", "answer_relevancy", "context_recall"
                        ] if c in df.columns
                    ]
                    st.dataframe(df[show_cols], use_container_width=True)

                # ── Download results ──────────────────────────
                import pandas as pd
                csv = pd.DataFrame(eval_result["per_question"]).to_csv(index=False)
                st.download_button(
                    "⬇ Download results as CSV",
                    data      = csv,
                    file_name = "ragas_results.csv",
                    mime      = "text/csv",
                )

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Evaluation failed: {e}")
                st.info(
                    "Common causes: Ollama not running, model not pulled, "
                    "or ragas version mismatch. Check your terminal for details."
                )