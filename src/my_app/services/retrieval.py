from typing import List, Dict, Any, Optional
from my_app.services.vector_db import VectorStore
from my_app.services.embedder import EmbeddingManager
from my_app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0, filters: Optional[dict] = None) -> List[Dict[str, Any]]:
        logger.info(f"Retrieving: {query}")

        # FIXED: Calling the model encode directly with proper parameters
        query_embedding = self.embedding_manager.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        retrieved_chunks = []
        # Check if results exist to prevent index errors
        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return []

        # Chroma returns lists of lists
        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]),
            start=1
        ):
            similarity_score = 1.0 - distance
            if similarity_score >= score_threshold:
                retrieved_chunks.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": round(similarity_score, 4),
                    "rank": rank,
                })
        return retrieved_chunks