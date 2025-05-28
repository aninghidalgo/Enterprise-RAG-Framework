"""
Cross-encoder reranker implementation for improving retrieval precision.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranker implementation using cross-encoder models to score query-document pairs.
    Cross-encoders typically provide more accurate relevance scores than bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name or path of the cross-encoder model
            device: Device to run the model on (cpu, cuda, cuda:0, etc.)
            batch_size: Batch size for scoring
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = self._load_model()
        
        logger.info(f"Initialized cross-encoder reranker with model {model_name}")
    
    def _load_model(self) -> Any:
        """
        Load the cross-encoder model.
        
        Returns:
            Loaded cross-encoder model
        """
        try:
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
            
            return model
            
        except ImportError:
            logger.error("Failed to load cross-encoder. Please install with: pip install -U sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {str(e)}")
            raise
    
    def compute_scores(
        self,
        query: str,
        passages: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """
        Compute relevance scores for query-passage pairs.
        
        Args:
            query: Query text
            passages: List of passage texts
            batch_size: Optional batch size override
            
        Returns:
            List of relevance scores for each passage
        """
        if not passages:
            return []
        
        batch_size = batch_size or self.batch_size
        
        try:
            # Create query-passage pairs
            pairs = [(query, passage) for passage in passages]
            
            # Compute scores
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # Convert to float list
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            return scores
            
        except Exception as e:
            logger.error(f"Error computing cross-encoder scores: {str(e)}")
            # Return neutral scores in case of error
            return [0.5] * len(passages)
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        score_key: str = "reranker_score",
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on cross-encoder scores.
        
        Args:
            query: Query text
            documents: List of document dictionaries
            score_key: Key to store reranker scores in documents
            threshold: Optional score threshold to filter documents
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Extract passages from documents
        passages = [doc.get("text", "") for doc in documents]
        
        # Compute scores
        scores = self.compute_scores(query, passages)
        
        # Add scores to documents
        for i, score in enumerate(scores):
            documents[i][score_key] = float(score)
        
        # Apply threshold if specified
        if threshold is not None:
            documents = [doc for doc in documents if doc[score_key] >= threshold]
        
        # Sort by reranker score
        documents = sorted(documents, key=lambda x: x[score_key], reverse=True)
        
        return documents
