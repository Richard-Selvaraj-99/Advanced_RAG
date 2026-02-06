graph LR
    A[PDF Document] --> B[Embedding Model]
    B --> C[(ChromaDB)]
    C --> D[Retrieval]
    D --> E[LLM]
    E --> F[Final Output] 

#project structure
my_project/
├── pyproject.toml
├── src/
│   └── my_app/
│       ├── __init__.py      <-- Hoists main functions
│       ├── main.py          <-- FastAPI entry point
│       ├── core/
│       │   ├── hf_hub.py    
│       │   └── .env         <-- Config & Secrets
│       ├── ingestion/ 
│       │   ├── pdf_files/
│       │   ├── __init__.py 
│       │   └── ingestion.py
│       ├── services/        <-- The "Heavy Lifters"
│       │   ├── __init__.py
│       │   ├── chunking.py 
│       │   ├── embedder.py  <-- Embedding logic
│       │   ├── vector_db.py <-- ChromaDB logic
│       │   ├── retrieval.py    
│       │   └── llm.py       <-- LLM logic
│       └── utils/
│           ├── __init__.py
│           └── logger.py
└── tests/
    └── test.py
