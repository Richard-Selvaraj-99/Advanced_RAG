import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# NOW your existing imports will work
from my_app.orchestration.rag_pipeline import RAGPipeline
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from my_app.orchestration.rag_pipeline import RAGPipeline

# Global instance to be used across the app
rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGPipeline()
    rag.bootstrap()
    yield

app = FastAPI(title="RAG Service", lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def handle_query(request: QueryRequest):
    answer, sources = rag.ask(request.question)
    return {"answer": answer, "sources": sources}
if __name__ == "__main__":
    import uvicorn
    # This tells uvicorn to look at 'app' inside this file (main.py)
    uvicorn.run(app, host="0.0.0.0", port=8000)