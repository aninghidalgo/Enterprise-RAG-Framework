"""
Retrieval engine module for finding relevant documents.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """
    Core retrieval engine that handles document indexing and retrieval.
    Supports hybrid search combining dense and sparse retrieval methods.
    """

    def __init__(
        self, 
        retrieval_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the retrieval engine with configuration options.
        
        Args:
            retrieval_config: Configuration for retrieval strategies
            vector_store_config: Configuration for vector store
        """
        self.retrieval_config = retrieval_config or {
            "type": "hybrid",  # Options: dense, sparse, hybrid
            "top_k": 5,
            "similarity_threshold": 0.7,
            "reranker_threshold": 0.4,
            "use_reranker": True,
        }
        
        self.vector_store_config = vector_store_config or {
            "type": "faiss",  # Options: faiss, pinecone, qdrant, weaviate, chroma
            "index_name": "document_index",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "embedding_dimension": 768,
        }
        
        logger.info("Initializing retrieval engine with config: %s", 
                   {"retrieval": self.retrieval_config, "vector_store": self.vector_store_config})
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize sparse retriever if using hybrid or sparse
        if self.retrieval_config["type"] in ["sparse", "hybrid"]:
            self.sparse_retriever = self._initialize_sparse_retriever()
        else:
            self.sparse_retriever = None
        
        # Initialize reranker if enabled
        if self.retrieval_config.get("use_reranker", True):
            self.reranker = self._initialize_reranker()
        else:
            self.reranker = None
    
    def _initialize_vector_store(self) -> Any:
        """
        Initialize vector store based on configuration.
        
        Returns:
            Initialized vector store instance
        """
        vector_store_type = self.vector_store_config.get("type", "faiss")
        
        try:
            if vector_store_type == "faiss":
                from .vector_stores.faiss_store import FAISSVectorStore
                return FAISSVectorStore(self.vector_store_config)
            
            elif vector_store_type == "pinecone":
                from .vector_stores.pinecone_store import PineconeVectorStore
                return PineconeVectorStore(self.vector_store_config)
            
            elif vector_store_type == "qdrant":
                from .vector_stores.qdrant_store import QdrantVectorStore
                return QdrantVectorStore(self.vector_store_config)
            
            elif vector_store_type == "weaviate":
                from .vector_stores.weaviate_store import WeaviateVectorStore
                return WeaviateVectorStore(self.vector_store_config)
            
            elif vector_store_type == "chroma":
                from .vector_stores.chroma_store import ChromaVectorStore
                return ChromaVectorStore(self.vector_store_config)
            
            else:
                logger.warning(
                    "Unknown vector store type: %s, falling back to FAISS", 
                    vector_store_type
                )
                from .vector_stores.faiss_store import FAISSVectorStore
                return FAISSVectorStore(self.vector_store_config)
                
        except ImportError as e:
            logger.error(
                "Failed to initialize %s vector store: %s. Falling back to in-memory store.", 
                vector_store_type, str(e)
            )
            from .vector_stores.memory_store import InMemoryVectorStore
            return InMemoryVectorStore(self.vector_store_config)
    
    def _initialize_sparse_retriever(self) -> Any:
        """
        Initialize sparse retriever (BM25).
        
        Returns:
            Initialized sparse retriever instance
        """
        try:
            from .sparse.bm25_retriever import BM25Retriever
            return BM25Retriever(self.retrieval_config)
        except ImportError as e:
            logger.error("Failed to initialize sparse retriever: %s", str(e))
            return None
    
    def _initialize_reranker(self) -> Any:
        """
        Initialize reranker model.
        
        Returns:
            Initialized reranker instance
        """
        reranker_model = self.retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        try:
            from .reranker.cross_encoder import CrossEncoderReranker
            return CrossEncoderReranker(model_name=reranker_model)
        except ImportError as e:
            logger.error("Failed to initialize reranker: %s", str(e))
            return None
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a list of documents for retrieval.
        
        Args:
            documents: List of document dictionaries with text and metadata
        """
        start_time = time.time()
        
        # Index in vector store
        self.vector_store.add_documents(documents)
        
        # Index in sparse retriever if available
        if self.sparse_retriever is not None:
            self.sparse_retriever.add_documents(documents)
        
        elapsed_time = time.time() - start_time
        logger.info("Indexed %d documents in %.2f seconds", len(documents), elapsed_time)
    
    def retrieve(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            filters: Optional metadata filters for retrieval
            options: Optional parameters to customize retrieval
                - top_k: Number of documents to retrieve
                - similarity_threshold: Minimum similarity score threshold
                - retrieval_type: Override configured retrieval type
                - rerank: Whether to apply reranking
        
        Returns:
            List of retrieved documents with text, metadata, and relevance scores
        """
        start_time = time.time()
        
        # Extract options
        options = options or {}
        top_k = options.get("top_k", self.retrieval_config.get("top_k", 5))
        similarity_threshold = options.get(
            "similarity_threshold", 
            self.retrieval_config.get("similarity_threshold", 0.7)
        )
        retrieval_type = options.get("retrieval_type", self.retrieval_config.get("type", "hybrid"))
        use_reranker = options.get("rerank", self.retrieval_config.get("use_reranker", True))
        
        # Select retrieval strategy
        if retrieval_type == "dense":
            retrieved_docs = self._dense_retrieval(query, top_k, similarity_threshold, filters)
        elif retrieval_type == "sparse":
            retrieved_docs = self._sparse_retrieval(query, top_k, filters)
        elif retrieval_type == "hybrid":
            retrieved_docs = self._hybrid_retrieval(query, top_k, similarity_threshold, filters)
        else:
            logger.warning("Unknown retrieval type: %s, using hybrid retrieval", retrieval_type)
            retrieved_docs = self._hybrid_retrieval(query, top_k, similarity_threshold, filters)
        
        # Apply reranking if enabled and available
        if use_reranker and self.reranker is not None and len(retrieved_docs) > 1:
            reranker_threshold = self.retrieval_config.get("reranker_threshold", 0.4)
            retrieved_docs = self._rerank_documents(query, retrieved_docs, reranker_threshold)
        
        # Log retrieval metrics
        elapsed_time = time.time() - start_time
        doc_count = len(retrieved_docs)
        
        logger.info(
            "Retrieved %d documents in %.2f seconds using %s retrieval%s", 
            doc_count, elapsed_time, retrieval_type,
            " with reranking" if use_reranker and self.reranker is not None else ""
        )
        
        if doc_count > 0:
            max_score = retrieved_docs[0].get("score", 0)
            min_score = retrieved_docs[-1].get("score", 0) if doc_count > 1 else max_score
            logger.debug("Retrieval score range: %.4f to %.4f", max_score, min_score)
        
        return retrieved_docs
    
    def _dense_retrieval(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform dense retrieval using vector embeddings.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            filters: Optional metadata filters
        
        Returns:
            List of retrieved documents with scores
        """
        # Get results from vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Retrieve more initially to allow for filtering
            filters=filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            doc for doc in results 
            if doc.get("score", 0) >= similarity_threshold
        ]
        
        # Limit to top_k
        return filtered_results[:top_k]
    
    def _sparse_retrieval(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform sparse retrieval using BM25 or similar algorithm.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
        
        Returns:
            List of retrieved documents with scores
        """
        if self.sparse_retriever is None:
            logger.warning("Sparse retriever not initialized, falling back to dense retrieval")
            return self._dense_retrieval(query, top_k, 0.0, filters)
        
        # Get results from sparse retriever
        results = self.sparse_retriever.search(
            query=query,
            top_k=top_k * 2,  # Retrieve more initially to allow for filtering
            filters=filters
        )
        
        # Apply any post-filtering
        if filters and not self.sparse_retriever.supports_filters():
            results = self._apply_filters(results, filters)
        
        # Limit to top_k
        return results[:top_k]
    
    def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining dense and sparse methods.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            filters: Optional metadata filters
        
        Returns:
            List of retrieved documents with scores
        """
        # If sparse retriever is not available, fall back to dense retrieval
        if self.sparse_retriever is None:
            logger.warning("Sparse retriever not initialized, falling back to dense retrieval")
            return self._dense_retrieval(query, top_k, similarity_threshold, filters)
        
        # Get results from both retrievers
        dense_results = self._dense_retrieval(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters
        )
        
        sparse_results = self._sparse_retrieval(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        # Combine results with a reciprocal rank fusion
        combined_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, top_k
        )
        
        return combined_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int,
        k: int = 60  # Constant for RRF formula
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return
            k: Constant in RRF formula
        
        Returns:
            Combined list of documents with fused scores
        """
        # Create a dictionary to track document scores by ID
        doc_scores = {}
        
        # Process dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc.get("id")
            if not doc_id:
                continue
                
            rrf_score = 1 / (rank + k)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "dense_score": doc.get("score", 0),
                    "sparse_score": 0,
                    "rrf_score": rrf_score
                }
            else:
                doc_scores[doc_id]["dense_score"] = doc.get("score", 0)
                doc_scores[doc_id]["rrf_score"] += rrf_score
        
        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.get("id")
            if not doc_id:
                continue
                
            rrf_score = 1 / (rank + k)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "dense_score": 0,
                    "sparse_score": doc.get("score", 0),
                    "rrf_score": rrf_score
                }
            else:
                doc_scores[doc_id]["sparse_score"] = doc.get("score", 0)
                doc_scores[doc_id]["rrf_score"] += rrf_score
        
        # Sort by RRF score and convert back to document list
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # Convert to final document format
        results = []
        for item in sorted_docs[:top_k]:
            doc = item["doc"].copy()
            doc["score"] = item["rrf_score"]
            doc["dense_score"] = item["dense_score"]
            doc["sparse_score"] = item["sparse_score"]
            results.append(doc)
        
        return results
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        score_threshold: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using a cross-encoder model.
        
        Args:
            query: Query text
            documents: List of retrieved documents
            score_threshold: Minimum score threshold for reranker
        
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
            
        if self.reranker is None:
            logger.warning("Reranker not initialized, skipping reranking")
            return documents
        
        try:
            # Extract text content from documents
            doc_texts = [doc.get("text", "") for doc in documents]
            
            # Get reranker scores
            reranker_scores = self.reranker.compute_scores(query, doc_texts)
            
            # Add reranker scores to documents
            for i, score in enumerate(reranker_scores):
                documents[i]["reranker_score"] = float(score)
            
            # Filter by threshold
            filtered_docs = [
                doc for doc in documents 
                if doc.get("reranker_score", 0) >= score_threshold
            ]
            
            # If all documents were filtered out, return top document
            if not filtered_docs and documents:
                return [documents[0]]
            
            # Sort by reranker score
            reranked_docs = sorted(
                filtered_docs,
                key=lambda x: x.get("reranker_score", 0),
                reverse=True
            )
            
            return reranked_docs
            
        except Exception as e:
            logger.error("Error during reranking: %s", str(e))
            return documents
    
    def _apply_filters(
        self,
        documents: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to a list of documents.
        
        Args:
            documents: List of documents
            filters: Dictionary of metadata filters
        
        Returns:
            Filtered list of documents
        """
        if not filters:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            match = True
            
            for key, value in filters.items():
                # Handle nested keys with dot notation
                if "." in key:
                    parts = key.split(".")
                    current = metadata
                    for part in parts[:-1]:
                        if part not in current:
                            match = False
                            break
                        current = current[part]
                    
                    if match and (parts[-1] not in current or current[parts[-1]] != value):
                        match = False
                
                # Handle list membership
                elif isinstance(value, list):
                    if key not in metadata or metadata[key] not in value:
                        match = False
                
                # Handle range queries
                elif isinstance(value, dict) and ("$gt" in value or "$lt" in value or "$gte" in value or "$lte" in value):
                    if key not in metadata:
                        match = False
                    else:
                        field_value = metadata[key]
                        
                        if "$gt" in value and not field_value > value["$gt"]:
                            match = False
                        if "$lt" in value and not field_value < value["$lt"]:
                            match = False
                        if "$gte" in value and not field_value >= value["$gte"]:
                            match = False
                        if "$lte" in value and not field_value <= value["$lte"]:
                            match = False
                
                # Handle exact match
                elif key not in metadata or metadata[key] != value:
                    match = False
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the index.
        
        Returns:
            Number of indexed documents
        """
        return self.vector_store.get_document_count()
    
    def is_ready(self) -> bool:
        """
        Check if the retrieval engine is ready for queries.
        
        Returns:
            True if ready, False otherwise
        """
        return self.vector_store.is_ready()
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the index.
        
        Args:
            document_ids: List of document IDs to delete
        """
        # Delete from vector store
        self.vector_store.delete_documents(document_ids)
        
        # Delete from sparse retriever if available
        if self.sparse_retriever is not None:
            self.sparse_retriever.delete_documents(document_ids)
        
        logger.info("Deleted %d documents from index", len(document_ids))
    
    def clear_index(self) -> None:
        """
        Clear all documents from the index.
        """
        # Clear vector store
        self.vector_store.clear()
        
        # Clear sparse retriever if available
        if self.sparse_retriever is not None:
            self.sparse_retriever.clear()
        
        logger.info("Cleared all documents from index")
