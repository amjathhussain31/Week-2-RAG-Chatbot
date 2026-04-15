# 🚀 RAG MapReduce Chatbot

A **Retrieval-Augmented Generation (RAG) based chatbot** that answers questions from custom documents using **LLMs + FAISS + RAGAS evaluation**.

---

## 📌 Overview

This project implements an end-to-end **RAG pipeline** that enables users to:

* 📄 Upload and query PDF documents
* 🤖 Get context-aware answers using LLMs
* 🧠 Maintain conversation memory
* 📊 Evaluate performance using RAGAS metrics

Unlike traditional LLMs that rely only on training data, this system retrieves relevant document chunks and generates **grounded, accurate responses**. ([GitHub][1])

---

## 🧠 Architecture

```
User Query
   ↓
Query Rewriting (optional)
   ↓
Retriever (FAISS)
   ↓
Relevant Chunks (Top-K)
   ↓
LLM (Mistral / LLaMA)
   ↓
Final Answer + Citations
```

---

## ⚙️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Mistral 7B / LLaMA 2
* **Vector Store:** FAISS
* **Embeddings:** Sentence Transformers
* **Framework:** Custom modular RAG pipeline
* **Evaluation:** RAGAS (Faithfulness, Relevancy, Recall)

---

## 📁 Project Structure

```
rag/
│
├── loader.py          # Load PDF documents
├── splitter.py        # Chunking logic
├── embeddings.py      # Embedding model
├── vectorstore.py     # FAISS setup
├── retriever.py       # Retrieval logic
├── chain.py           # RAG pipeline
├── memory.py          # Chat memory
├── llm.py             # LLM configuration
├── evaluator.py       # RAGAS evaluation
│
app.py                 # Streamlit UI
requirements.txt
```

---

## 🚀 Features

### ✅ Core RAG Pipeline

* Document ingestion and chunking
* Semantic search using FAISS
* Context-aware answer generation

### ✅ Multi-Model Support

* Switch between:

  * Mistral 7B
  * LLaMA 2

### ✅ Conversation Memory

* Maintains session-based chat history
* Improves contextual understanding

### ✅ Evaluation with RAGAS

* Faithfulness (hallucination detection)
* Answer Relevancy
* Context Recall

---

## 📊 Example RAGAS Results

| Metric       | Score |
| ------------ | ----- |
| Faithfulness | 1.00  |
| Relevancy    | 1.00  |
| Recall       | 0.60  |

👉 Indicates strong grounding and response quality, with scope for improving retrieval.

---

## 🛠️ Installation

```bash
git clone https://github.com/amjathhussain31/Week-2-RAG-Chatbot.git
cd Week-2-RAG-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧪 Run Evaluation

```bash
python test_evaluator.py
```

---

## ⚠️ Limitations

* Retrieval recall can be improved (currently ~0.6)
* No reranking or hybrid search implemented
* Works best with structured PDF content

---

## 🔮 Future Improvements

* 🔼 Hybrid search (BM25 + vector search)
* 🔼 Reranking using cross-encoders
* 🔼 Better chunking strategies
* 🔼 Deployment (Streamlit Cloud / Docker)
* 🔼 Agentic RAG (multi-step reasoning)

---

## 🧠 Key Learnings

* Understanding of **end-to-end RAG systems**
* Importance of **retrieval quality over LLM power**
* Hands-on experience with **RAGAS evaluation metrics**
* Debugging real-world issues like:

  * hallucination
  * poor recall
  * context mismatch

---

[1]: https://github.com/amajji/llm-rag-chatbot-with-langchain?utm_source=chatgpt.com "GitHub - amajji/llm-rag-chatbot-with-langchain: Development and deployment on AWS of a question-answer LLM model using Llama2 with 7B parameters and RAG with LangChain"
