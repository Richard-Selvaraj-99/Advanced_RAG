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
#
