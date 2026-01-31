"this is the first initialization" 
my_project/
├── pyproject.toml
├── src/
│   └── my_app/
│       ├── __init__.py      <-- (Hoists your main functions)
│       ├── main.py          <-- (FastAPI entry point)
│       ├── core/            <-- (Config, constants, secrets)
│       ├── services/        <-- (The "Heavy Lifters")
│       │   ├── __init__.py
│       │   ├── embedder.py  <-- (Embedding logic)
│       │   ├── vector_db.py <-- (ChromaDB logic)
│       │   └── llm.py       <-- (LLM/Inference logic)
│       └── utils/
│           ├── __init__.py
│           └── logger.py
└── tests/
