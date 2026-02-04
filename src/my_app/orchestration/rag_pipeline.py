from my_app.ingestion.pdf_ingestion import get_pdf_documents
from my_app.services.chunking import DocumentChunker
from my_app.services.embedder import EmbeddingManager
from my_app.services.vector_db import VectorStore
from my_app.services.retrieval import RAGRetriever
from my_app.services.llm import RAGGenerator
from my_app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        # Tools initialization
        self.embedder = EmbeddingManager()
        self.vector_db = VectorStore()
        self.chunker = DocumentChunker()
        self.generator = RAGGenerator()
        self.retriever = RAGRetriever(self.vector_db, self.embedder)

    def bootstrap(self):
        """One-time setup to fill the database."""
        if self.vector_db.count() == 0:
            logger.info("Database empty. Starting ingestion...")
            docs = get_pdf_documents()
            chunks = self.chunker.split(docs)
            embeddings = self.embedder.embed_documents(chunks)
            self.vector_db.add_documents(chunks, embeddings)
        else:
            logger.info("Database already contains data. Skipping ingestion.")

    def ask(self, question: str):
        """The main RAG loop."""
        # 1. Fetch relevant context
        context_chunks = self.retriever.retrieve(query=question, top_k=3)
        
        if not context_chunks:
            return "No matching information found in your documents.", []

        # 2. Format and Generate
        context_text = "\n\n".join([c["content"] for c in context_chunks])
        sources = list(set([c["metadata"].get("source", "Unknown") for c in context_chunks]))
        
        answer = self.generator.generate_answer(query=question, context=context_text)
        return answer, sources