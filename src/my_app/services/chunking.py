from typing import List
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from my_app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Splits documents into smaller, overlapping chunks suitable for
    embedding and retrieval in a RAG pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        logger.info(
            "Initialized DocumentChunker | chunk_size=%s | chunk_overlap=%s",
            self.chunk_size,
            self.chunk_overlap,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving and enriching metadata.

        Each chunk receives:
        - chunk_id (UUID)
        - source (original file if provided)
        - page (if available)
        """

        if not documents:
            logger.warning("No documents provided for chunking")
            return []

        logger.info("Starting document chunking | documents=%s", len(documents))

        chunks = self._text_splitter.split_documents(documents)

        enriched_chunks: List[Document] = []

        for idx, chunk in enumerate(chunks):
            metadata = chunk.metadata.copy() if chunk.metadata else {}

            metadata.update(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": idx,
                }
            )

            enriched_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )

        logger.info(
            "Chunking complete | input_docs=%s | output_chunks=%s",
            len(documents),
            len(enriched_chunks),
        )

        return enriched_chunks
