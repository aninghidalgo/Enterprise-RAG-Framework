"""
Milvus vector store implementation for document retrieval.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus vector store implementation for efficient similarity search.
    Supports high-dimensional vector indexing and fast retrieval.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Milvus vector store.

        Args:
            config: Configuration dictionary containing:
                - host: Milvus server host
                - port: Milvus server port
                - collection_name: Name of the collection
                - embedding_dimension: Dimension of embedding vectors
                - index_type: Type of index to use (e.g., "IVF_FLAT", "HNSW")
                - index_params: Parameters for the index
        """
        super().__init__(config)
        
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 19530)
        self.collection_name = config.get("collection_name", "document_vectors")
        self.embedding_dimension = config.get("embedding_dimension", 768)
        self.index_type = config.get("index_type", "IVF_FLAT")
        self.index_params = config.get("index_params", {
            "nlist": 1024,
            "metric_type": "L2"
        })
        
        # Connect to Milvus server
        self._connect()
        
        # Create or get collection
        self._initialize_collection()

    def _connect(self) -> None:
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info("Connected to Milvus server at %s:%d", self.host, self.port)
        except Exception as e:
            logger.error("Failed to connect to Milvus server: %s", str(e))
            raise

    def _initialize_collection(self) -> None:
        """Initialize or get existing collection."""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info("Using existing collection: %s", self.collection_name)
            else:
                # Define collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension)
                ]
                schema = CollectionSchema(fields=fields, description="Document vectors collection")
                
                # Create collection
                self.collection = Collection(name=self.collection_name, schema=schema)
                
                # Create index
                self.collection.create_index(
                    field_name="vector",
                    index_params={
                        "index_type": self.index_type,
                        "metric_type": self.index_params["metric_type"],
                        "params": {"nlist": self.index_params["nlist"]}
                    }
                )
                logger.info("Created new collection: %s", self.collection_name)
        except Exception as e:
            logger.error("Failed to initialize collection: %s", str(e))
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dictionaries containing text, metadata, and embeddings
        """
        try:
            # Prepare data for insertion
            ids = []
            texts = []
            metadata_list = []
            vectors = []
            
            for doc in documents:
                ids.append(doc["id"])
                texts.append(doc["text"])
                metadata_list.append(doc["metadata"])
                vectors.append(doc["embedding"])
            
            # Insert data
            data = [
                ids,
                texts,
                metadata_list,
                vectors
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info("Added %d documents to Milvus", len(documents))
        except Exception as e:
            logger.error("Failed to add documents: %s", str(e))
            raise

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of similar documents with scores
        """
        try:
            # Load collection into memory
            self.collection.load()
            
            # Prepare search parameters
            search_params = {
                "metric_type": self.index_params["metric_type"],
                "params": {"nprobe": 10}
            }
            
            # Execute search
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text", "metadata"]
            )
            
            # Format results
            documents = []
            for hits in results:
                for hit in hits:
                    doc = {
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "metadata": hit.entity.get("metadata"),
                        "score": hit.score
                    }
                    documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error("Failed to search documents: %s", str(e))
            raise
        finally:
            # Release collection from memory
            self.collection.release()

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete
        """
        try:
            expr = f'id in {document_ids}'
            self.collection.delete(expr)
            logger.info("Deleted %d documents from Milvus", len(document_ids))
        except Exception as e:
            logger.error("Failed to delete documents: %s", str(e))
            raise

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.collection.drop()
            self._initialize_collection()
            logger.info("Cleared all documents from Milvus")
        except Exception as e:
            logger.error("Failed to clear documents: %s", str(e))
            raise

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.

        Returns:
            Number of documents
        """
        try:
            return self.collection.num_entities
        except Exception as e:
            logger.error("Failed to get document count: %s", str(e))
            return 0 