from typing import List, Dict, Any, Optional
import numpy as np

from my_app.services.vector_db import VectorStore
from my_app.services.embedder import EmbeddingManager
from my_app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """
    Handles similarity-based retrieval for RAG pipelines.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        """

        logger.info(
            "Retrieval started | query='%s' | top_k=%s | threshold=%s",
            query,
            top_k,
            score_threshold,
        )

        #  Embed query
        query_embedding = self.embedding_manager.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        #  Vector search
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        retrieved_chunks: List[Dict[str, Any]] = []

        if not results or not results.get("documents"):
            logger.info("No retrieval results")
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            similarity_score = 1.0 - distance

            if similarity_score < score_threshold:
                continue

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": round(similarity_score, 4),
                    "rank": rank,
                }
            )

        logger.info(
            "Retrieval completed | returned_chunks=%s",
            len(retrieved_chunks),
        )

        return retrieved_chunks
