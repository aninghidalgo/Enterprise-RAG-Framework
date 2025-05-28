"""
Base vector store abstract class that defines the interface for all vector stores.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    All vector store implementations should inherit from this class.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store with configuration.

        Args:
            config: Vector store configuration
        """
        self.config = config
        self.embedding_model = self._initialize_embedding_model()

    def _initialize_embedding_model(self) -> Any:
        """
        Initialize the embedding model for vectorizing text.

        Returns:
            Embedding model instance
        """
        embedding_model = self.config.get(
            "embedding_model", "sentence-transformers/all-mpnet-base-v2"
        )

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embedding_model)
            logger.info("Initialized embedding model: %s", embedding_model)
            return model
        except ImportError:
            logger.error(
                "Failed to initialize SentenceTransformer. Please install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error("Error initializing embedding model: %s", str(e))
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error("Error generating embeddings: %s", str(e))
            raise

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dictionaries with text and metadata

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def search(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Query text
            top_k: Number of documents to return
            filters: Optional metadata filters

        Returns:
            List of document dictionaries with text, metadata, and similarity scores
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.

        Returns:
            Number of documents
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the vector store is ready for queries.

        Returns:
            True if ready, False otherwise
        """
        pass
