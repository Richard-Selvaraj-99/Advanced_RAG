#  Advanced RAG: Multi-Query Document Intelligence

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline designed for high-precision document intelligence.  
This system leverages **Multi-Query Generation** to overcome the vocabulary gap between user queries and technical documents, achieving significantly higher retrieval accuracy than standard RAG setups and uses Recursive text splitter(context aware text splitter)to maintain sematic meanings.

---

## Project Architecture

This project follows a modular **Service-Oriented Architecture (SOA)**.  
Instead of relying on a single query, the system expands each user question into multiple semantic variations before searching the vector database—maximizing recall and contextual coverage.

#Stages of RAG

    A[PDF Document] --> B[Embedding Model]
    B --> C[(ChromaDB)]
    C --> D[Retrieval]
    D --> E[LLM]
    E --> F[Final Output] 

#project structure
```text
my_project/
├── pyproject.toml
├── src/
│   └── my_app/
│       ├── __init__.py      # Hoists your main functions
│       ├── main.py          # FastAPI entry point
│       ├── core/
│       │   ├── hugging_face_hub.py
│       │   └── .env         # Config, constants, secrets
│       ├── ingestion/
│       │   ├── pdf_files/
│       │   ├── __init__.py
│       │   └── pdf_ingestion.py
│       ├── services/        # The "Heavy Lifters"
│       │   ├── __init__.py
│       │   ├── chunking.py
│       │   ├── embedder.py  # Embedding logic
│       │   ├── vector_db.py # ChromaDB logic
│       │   ├── retrieval.py
│       │   └── llm.py       # LLM / Inference logic
│       └── utils/
│           ├── __init__.py
│           └── logger.py
└── tests/
    └── test.py
```
-------------------------------------------------------------------------------

# Advanced Features
## Multi-Query Retrieval

Uses an LLM to generate multiple versions of a user’s question.
This ensures accurate retrieval even when user terminology differs from the source documents.
-----------------------------------------------------------------------------
# Recursive Contextual Chunking

Implements a recursive splitting strategy that respects paragraph and sentence boundaries, preventing semantic fragmentation.
---------------------------------------------------------------------------------
# Metadata-Aware Retrieval

Each retrieved chunk includes source PDF name and page number, enabling transparent and verifiable citations.
---------------------------------------------------------------------------------
# FastAPI Backend

High-performance asynchronous API endpoints for querying and document status monitoring.
------------------------------------------------------------------------------------
##Tech Stack
| Component       | Technology                           |
| --------------- | ------------------------------------ |
| Orchestration   | LangChain / Python                   |
| Vector Database | ChromaDB                             |
| Embeddings      | Hugging Face (sentence-transformers) |
| LLM Gateway     | Groq / OpenAI                        |
| API Framework   | FastAPI                              |
--------------------------------------------------------------------------------------------------
 Getting Started
1️ Prerequisites
Python 3.11+
--------------------------------------------------------------------------------------------------
2.
git clone https://github.com/Richard-Selvaraj-99/Advanced_RAG.git
cd Advanced_RAG
--------------------------------------------------------------------------------------------------
3. uv init
uv venv
uv add -r requriement.txt  or uv sync if used to handling TOML file
uvicorn src.my_app.main:app --reload
--------------------------------------------------------------------------------------------------
📉 Roadmap (Future Improvements)
⬆️ Upstream & Downstream
Upstream: Cloud storage support (AWS S3) and real-time web scraping
--------------------------------------------------------------------------------------------------
🚀 Planned Enhancements
Hybrid Search (Vector + BM25 keyword matching)
cross-Encoder reranking for top-k results
emantic chunking based on thematic shifts

