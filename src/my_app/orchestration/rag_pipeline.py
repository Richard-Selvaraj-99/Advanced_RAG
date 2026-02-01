from typing import List, Dict, Any

from my_app.services.retrieval import RAGRetriever
from my_app.services.llm import RAGGenerator
from my_app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation workflow.
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        llm: RAGGenerator,
        top_k: int = 5,
        score_threshold: float = 0.0,
        max_context_chunks: int = 8,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_context_chunks = max_context_chunks

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a user query.
        """

        logger.info("RAG pipeline started | query='%s'", query)

        # Rewrite query (optional but powerful)
        queries = self.llm.rewrite_query(query)

        # Always include original query
        if query not in queries:
            queries.insert(0, query)

        # Retrieve chunks
        retrieved_chunks: List[Dict[str, Any]] = []

        for q in queries:
            results = self.retriever.retrieve(
                query=q,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
            )
            retrieved_chunks.extend(results)

        if not retrieved_chunks:
            logger.warning("No chunks retrieved for query")
            return {
                "answer": "I could not find relevant information in the documents.",
                "sources": [],
            }

        #  Deduplicate by chunk_id
        seen = set()
        unique_chunks = []

        for chunk in retrieved_chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk)

        # Limit context size
        unique_chunks = unique_chunks[: self.max_context_chunks]

        #  Build context text
        context_text = "\n\n".join(
            f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['content']}"
            for c in unique_chunks
        )

        #  Generate answer
        answer = self.llm.generate_answer(
            query=query,
            context=context_text,
        )

        logger.info(
            "RAG pipeline completed | chunks_used=%s",
            len(unique_chunks),
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": c["chunk_id"],
                    "metadata": c["metadata"],
                    "score": c["similarity_score"],
                }
                for c in unique_chunks
            ],
        }
