"""
ColBERT (Contextualized Late Interaction) retriever implementation.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

from ..base import BaseRetriever

logger = logging.getLogger(__name__)


class ColBERTRetriever(BaseRetriever):
    """
    ColBERT retriever implementation for efficient and effective document retrieval.
    Combines contextualized token embeddings with late interaction for better matching.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ColBERT retriever.

        Args:
            config: Configuration dictionary containing:
                - model_name: Name of the ColBERT model to use
                - max_length: Maximum sequence length
                - batch_size: Batch size for processing
                - use_gpu: Whether to use GPU for computation
        """
        super().__init__(config)
        
        self.model_name = config.get("model_name", "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco")
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 32)
        self.use_gpu = config.get("use_gpu", torch.cuda.is_available())
        
        # Initialize model and tokenizer
        self._initialize_model()
        
        # Document store
        self.documents = []
        self.doc_embeddings = None
        self.doc_tokens = None

    def _initialize_model(self) -> None:
        """Initialize ColBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info("Initialized ColBERT model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize ColBERT model: %s", str(e))
            raise

    def _tokenize_and_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode texts using the ColBERT model.

        Args:
            texts: List of texts to encode

        Returns:
            Dictionary containing tokenized inputs
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if self.use_gpu:
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        return encoded

    def _compute_embeddings(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute token embeddings using the ColBERT model.

        Args:
            encoded: Tokenized inputs

        Returns:
            Token embeddings tensor
        """
        with torch.no_grad():
            outputs = self.model(**encoded)
            token_embeddings = outputs.last_hidden_state
        
        return token_embeddings

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever.

        Args:
            documents: List of document dictionaries containing text and metadata
        """
        try:
            # Store documents
            self.documents.extend(documents)
            
            # Process documents in batches
            texts = [doc["text"] for doc in documents]
            all_embeddings = []
            all_tokens = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize and encode
                encoded = self._tokenize_and_encode(batch_texts)
                
                # Compute embeddings
                embeddings = self._compute_embeddings(encoded)
                
                # Store embeddings and tokens
                all_embeddings.append(embeddings.cpu().numpy())
                all_tokens.append(encoded["input_ids"].cpu().numpy())
            
            # Concatenate results
            if self.doc_embeddings is None:
                self.doc_embeddings = np.concatenate(all_embeddings, axis=0)
                self.doc_tokens = np.concatenate(all_tokens, axis=0)
            else:
                self.doc_embeddings = np.concatenate([self.doc_embeddings] + all_embeddings, axis=0)
                self.doc_tokens = np.concatenate([self.doc_tokens] + all_tokens, axis=0)
            
            logger.info("Added %d documents to ColBERT retriever", len(documents))
        except Exception as e:
            logger.error("Failed to add documents: %s", str(e))
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of relevant documents with scores
        """
        try:
            # Tokenize and encode query
            encoded = self._tokenize_and_encode([query])
            query_embeddings = self._compute_embeddings(encoded)
            query_embeddings = query_embeddings.cpu().numpy()
            
            # Compute late interaction scores
            scores = self._compute_late_interaction(query_embeddings[0])
            
            # Apply filters if provided
            if filters:
                scores = self._apply_filters(scores, filters)
            
            # Get top-k results
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Format results
            results = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                results.append(doc)
            
            return results
        except Exception as e:
            logger.error("Failed to search documents: %s", str(e))
            raise

    def _compute_late_interaction(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute late interaction scores between query and documents.

        Args:
            query_embeddings: Query token embeddings

        Returns:
            Array of similarity scores
        """
        # Reshape query embeddings for broadcasting
        query_embeddings = query_embeddings.reshape(1, -1, 1)
        
        # Compute maximum similarity for each query token
        similarities = np.matmul(self.doc_embeddings, query_embeddings).squeeze(-1)
        max_similarities = np.max(similarities, axis=1)
        
        # Sum the maximum similarities
        scores = np.sum(max_similarities, axis=1)
        
        return scores

    def _apply_filters(self, scores: np.ndarray, filters: Dict[str, Any]) -> np.ndarray:
        """
        Apply metadata filters to scores.

        Args:
            scores: Array of similarity scores
            filters: Metadata filters to apply

        Returns:
            Filtered scores array
        """
        filtered_scores = scores.copy()
        
        for i, doc in enumerate(self.documents):
            for key, value in filters.items():
                if doc["metadata"].get(key) != value:
                    filtered_scores[i] = float("-inf")
        
        return filtered_scores

    def clear(self) -> None:
        """Clear all documents from the retriever."""
        self.documents = []
        self.doc_embeddings = None
        self.doc_tokens = None
        logger.info("Cleared all documents from ColBERT retriever")

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the retriever.

        Returns:
            Number of documents
        """
        return len(self.documents) 