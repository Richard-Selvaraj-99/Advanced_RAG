from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from my_app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """
    Generates vector embeddings for document chunks using SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model into memory."""
        try:
            logger.info("Loading embedding model | model=%s", self.model_name)

            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )

            logger.info(
                "Embedding model loaded | dimension=%s",
                self.model.get_sentence_embedding_dimension(),
            )

        except Exception as exc:
            logger.exception("Failed to load embedding model")
            raise RuntimeError("Embedding model initialization failed") from exc

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for LangChain Documents.

        Returns:
            np.ndarray of shape (num_documents, embedding_dim)
        """

        if not documents:
            logger.warning("No documents provided for embedding")
            return np.array([])

        if not self.model:
            raise RuntimeError("Embedding model is not loaded")

        texts = [doc.page_content for doc in documents]

        logger.info(
            "Generating embeddings | documents=%s | batch_size=%s",
            len(texts),
            self.batch_size,
        )

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(
            "Embeddings generated | shape=%s",
            embeddings.shape,
        )

        return embeddings
