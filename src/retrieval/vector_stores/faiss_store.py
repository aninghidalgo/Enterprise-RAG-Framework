"""
FAISS vector store implementation for efficient similarity search.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Any, Tuple
import uuid
import numpy as np

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Vector store implementation using Facebook AI Similarity Search (FAISS).
    Provides efficient similarity search for high-dimensional vectors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FAISS vector store.
        
        Args:
            config: Vector store configuration
        """
        from .base import BaseVectorStore
        
        self.config = config
        self.index_name = config.get("index_name", "document_index")
        self.embedding_dimension = config.get("embedding_dimension", 768)
        
        # Parent class initialization
        self.embedding_model = self._initialize_embedding_model()
        
        # Initialize FAISS
        self.index = self._initialize_index()
        
        # Document mapping for storing and retrieving documents
        self.docstore = {}
        
        # Document ID to index mapping
        self.id_to_index = {}
        self.index_to_id = {}
        
        # Next index to use
        self.next_index = 0
        
        logger.info(f"Initialized FAISS vector store with dimension {self.embedding_dimension}")
    
    def _initialize_embedding_model(self) -> Any:
        """
        Initialize the embedding model for vectorizing text.
        
        Returns:
            Embedding model instance
        """
        embedding_model = self.config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2")
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embedding_model)
            logger.info("Initialized embedding model: %s", embedding_model)
            return model
        except ImportError:
            logger.error("Failed to initialize SentenceTransformer. Please install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error("Error initializing embedding model: %s", str(e))
            raise
    
    def _initialize_index(self) -> Any:
        """
        Initialize FAISS index.
        
        Returns:
            FAISS index
        """
        try:
            import faiss
            
            # Check if we should load an existing index
            index_path = self.config.get("index_path")
            if index_path and os.path.exists(index_path):
                logger.info(f"Loading existing FAISS index from {index_path}")
                return self._load_index(index_path)
            
            # Create a new index
            dimension = self.embedding_dimension
            
            # Select index type based on configuration
            index_type = self.config.get("index_type", "flat")
            
            if index_type == "flat":
                # Exact nearest neighbor search
                index = faiss.IndexFlatIP(dimension)  # Inner product similarity
                logger.info("Created new FAISS flat index")
            
            elif index_type == "ivf":
                # Inverted file index for faster search
                nlist = self.config.get("nlist", 100)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # IVF indices need to be trained before use
                if self.config.get("random_training", False):
                    # Train with random vectors if no data available yet
                    import numpy as np
                    train_size = max(nlist * 10, 1000)
                    train_vectors = np.random.random((train_size, dimension)).astype(np.float32)
                    faiss.normalize_L2(train_vectors)  # Normalize for inner product
                    index.train(train_vectors)
                
                logger.info(f"Created new FAISS IVF index with {nlist} lists")
            
            elif index_type == "hnsw":
                # Hierarchical Navigable Small World graph index
                M = self.config.get("M", 16)  # Number of connections per layer
                ef_construction = self.config.get("ef_construction", 200)
                index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = ef_construction
                logger.info(f"Created new FAISS HNSW index with M={M}")
            
            else:
                # Default to flat index
                logger.warning(f"Unknown index type: {index_type}, using flat index")
                index = faiss.IndexFlatIP(dimension)
            
            return index
            
        except ImportError:
            logger.error("Failed to initialize FAISS. Please install with: pip install faiss-cpu or faiss-gpu")
            raise
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            raise
    
    def _load_index(self, index_path: str) -> Tuple[Any, Dict, Dict, Dict, int]:
        """
        Load FAISS index and associated data from disk.
        
        Args:
            index_path: Path to the index file
            
        Returns:
            Tuple of (index, docstore, id_to_index, index_to_id, next_index)
        """
        try:
            import faiss
            
            # Load index
            index = faiss.read_index(f"{index_path}.faiss")
            
            # Load document store and mappings
            with open(f"{index_path}.pkl", "rb") as f:
                data = pickle.load(f)
                self.docstore = data["docstore"]
                self.id_to_index = data["id_to_index"]
                self.index_to_id = data["index_to_id"]
                self.next_index = data["next_index"]
            
            logger.info(f"Loaded FAISS index with {len(self.docstore)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            # Create a new index instead
            self.docstore = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0
            
            # Initialize a new index
            dimension = self.embedding_dimension
            index = faiss.IndexFlatIP(dimension)
            logger.info("Created new FAISS index after failed load")
            return index
    
    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Save FAISS index and associated data to disk.
        
        Args:
            index_path: Path to save the index, defaults to config path
        """
        if index_path is None:
            index_path = self.config.get("index_path")
        
        if not index_path:
            logger.warning("No index path specified, cannot save index")
            return
        
        try:
            import faiss
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, f"{index_path}.faiss")
            
            # Save document store and mappings
            with open(f"{index_path}.pkl", "wb") as f:
                data = {
                    "docstore": self.docstore,
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                    "next_index": self.next_index,
                }
                pickle.dump(data, f)
            
            logger.info(f"Saved FAISS index with {len(self.docstore)} documents to {index_path}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
            
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings for inner product similarity
            import faiss
            faiss.normalize_L2(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with text and metadata
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Extract text from documents
        texts = [doc.get("text", "") for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Assign IDs to documents if they don't have them
        doc_ids = []
        for i, doc in enumerate(documents):
            if "id" not in doc:
                doc_id = str(uuid.uuid4())
                doc["id"] = doc_id
            else:
                doc_id = doc["id"]
            
            doc_ids.append(doc_id)
            
            # Store document
            self.docstore[doc_id] = doc
            
            # Map document ID to index
            index = self.next_index + i
            self.id_to_index[doc_id] = index
            self.index_to_id[index] = doc_id
        
        # Update next index
        self.next_index += len(documents)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Save index if auto_save is enabled
        if self.config.get("auto_save", False) and self.config.get("index_path"):
            self.save_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
        return doc_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
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
        if self.index.ntotal == 0:
            logger.warning("Empty index, no documents to search")
            return []
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])
        
        # Determine how many documents to fetch (more if filtering)
        fetch_k = top_k
        if filters:
            # Fetch more documents if we're going to filter
            fetch_k = min(top_k * 5, self.index.ntotal)
        
        # Search index
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        # Flatten results
        scores = scores[0]
        indices = indices[0]
        
        # Convert to document dictionaries
        results = []
        for i, (score, idx) in enumerate(zip(scores, indices)):
            # Skip invalid indices
            if idx == -1 or idx not in self.index_to_id:
                continue
            
            # Get document ID and document
            doc_id = self.index_to_id[idx]
            document = self.docstore.get(doc_id)
            
            if document:
                # Create result with score
                result = document.copy()
                result["score"] = float(score)
                results.append(result)
        
        # Apply filters if specified
        if filters:
            results = self._apply_filters(results, filters)
        
        # Limit to top_k
        return results[:top_k]
    
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
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
        """
        if not self.config.get("support_remove", False):
            logger.warning("Document deletion not supported in current FAISS index, rebuilding index")
            self._rebuild_index_without(document_ids)
            return
            
        try:
            import faiss
            
            # Get indices to remove
            indices_to_remove = []
            for doc_id in document_ids:
                if doc_id in self.id_to_index:
                    indices_to_remove.append(self.id_to_index[doc_id])
                    
                    # Remove from mappings
                    index = self.id_to_index.pop(doc_id)
                    self.index_to_id.pop(index)
                    
                    # Remove from docstore
                    self.docstore.pop(doc_id, None)
            
            # Remove from FAISS index if supported
            if indices_to_remove and hasattr(self.index, "remove_ids"):
                # Convert to FAISS-compatible format
                remove_array = np.array(indices_to_remove, dtype=np.int64)
                self.index.remove_ids(remove_array)
                
                logger.info(f"Removed {len(indices_to_remove)} documents from FAISS index")
            else:
                # Fallback to rebuilding the index
                self._rebuild_index_without(document_ids)
            
            # Save index if auto_save is enabled
            if self.config.get("auto_save", False) and self.config.get("index_path"):
                self.save_index()
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
    
    def _rebuild_index_without(self, document_ids_to_remove: List[str]) -> None:
        """
        Rebuild the index excluding specified documents.
        
        Args:
            document_ids_to_remove: List of document IDs to exclude
        """
        try:
            # Keep documents that are not in the removal list
            docs_to_keep = []
            for doc_id, doc in self.docstore.items():
                if doc_id not in document_ids_to_remove:
                    docs_to_keep.append(doc)
            
            # Reset state
            self.docstore = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0
            
            # Re-initialize the index
            self.index = self._initialize_index()
            
            # Add documents back
            if docs_to_keep:
                self.add_documents(docs_to_keep)
                
            logger.info(f"Rebuilt FAISS index with {len(docs_to_keep)} documents, removed {len(document_ids_to_remove)}")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        try:
            # Reset state
            self.docstore = {}
            self.id_to_index = {}
            self.index_to_id = {}
            self.next_index = 0
            
            # Re-initialize the index
            self.index = self._initialize_index()
            
            logger.info("Cleared FAISS index")
            
            # Save empty index if auto_save is enabled
            if self.config.get("auto_save", False) and self.config.get("index_path"):
                self.save_index()
                
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        return len(self.docstore)
    
    def is_ready(self) -> bool:
        """
        Check if the vector store is ready for queries.
        
        Returns:
            True if ready, False otherwise
        """
        # FAISS index is ready if it has been initialized
        return self.index is not None
