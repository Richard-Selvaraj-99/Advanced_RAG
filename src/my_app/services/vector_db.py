from typing import List, Optional
import os

import chromadb
from langchain_core.documents import Document
import numpy as np

from my_app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Manages storage and retrieval of document embeddings using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store",
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client: Optional[chromadb.ClientAPI] = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize persistent ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"},
            )

            logger.info(
                "Vector store initialized | collection=%s | count=%s",
                self.collection_name,
                self.collection.count(),
            )

        except Exception as exc:
            logger.exception("Failed to initialize vector store")
            raise RuntimeError("Vector store initialization failed") from exc

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add document chunks and their embeddings to the vector store.
        """

        if not documents:
            logger.warning("No documents provided to vector store")
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                "Documents count does not match embeddings count"
            )

        ids = []
        metadatas = []
        contents = []

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Each document must contain a 'chunk_id' in metadata"
                )

            ids.append(chunk_id)
            metadatas.append(doc.metadata)
            contents.append(doc.page_content)

        logger.info(
            "Adding documents to vector store | chunks=%s",
            len(ids),
        )

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=contents,
            )

            logger.info(
                "Documents added | total_chunks=%s",
                self.collection.count(),
            )

        except Exception as exc:
            logger.exception("Failed to add documents to vector store")
            raise RuntimeError("Vector store add operation failed") from exc

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        Perform similarity search using a query embedding.
        """

        logger.info(
            "Running similarity search | top_k=%s | filters=%s",
            top_k,
            filters,
        )

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
        )

    def count(self) -> int:
        """Return number of stored chunks."""
        return self.collection.count()
