"""
MonoT5 reranker implementation for document reranking.
"""

import logging
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from ..base import BaseReranker

logger = logging.getLogger(__name__)


class MonoT5Reranker(BaseReranker):
    """
    MonoT5 reranker implementation for document reranking.
    Uses T5 model fine-tuned for passage reranking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MonoT5 reranker.

        Args:
            config: Configuration dictionary containing:
                - model_name: Name of the MonoT5 model to use
                - max_length: Maximum sequence length
                - batch_size: Batch size for processing
                - use_gpu: Whether to use GPU for computation
        """
        super().__init__(config)
        
        self.model_name = config.get("model_name", "castorini/monot5-base-msmarco")
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 8)
        self.use_gpu = config.get("use_gpu", torch.cuda.is_available())
        
        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize MonoT5 model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.cuda()
            
            self.model.eval()
            logger.info("Initialized MonoT5 model: %s", self.model_name)
        except Exception as e:
            logger.error("Failed to initialize MonoT5 model: %s", str(e))
            raise

    def _prepare_inputs(self, query: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare inputs for the MonoT5 model.

        Args:
            query: Query text
            documents: List of documents to rerank

        Returns:
            List of formatted input texts
        """
        inputs = []
        for doc in documents:
            # Format: "Query: {query} Document: {document} Relevant:"
            input_text = f"Query: {query} Document: {doc['text']} Relevant:"
            inputs.append(input_text)
        return inputs

    def _process_batch(self, batch_inputs: List[str]) -> torch.Tensor:
        """
        Process a batch of inputs through the MonoT5 model.

        Args:
            batch_inputs: List of input texts

        Returns:
            Tensor of relevance scores
        """
        # Tokenize inputs
        encoded = self.tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if self.use_gpu:
            encoded = {k: v.cuda() for k, v in encoded.items()}
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_length=1,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get scores for "true" token
        true_token_id = self.tokenizer.encode("true")[0]
        scores = outputs.sequences_scores
        
        return scores

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Optional number of top documents to return

        Returns:
            Reranked list of documents with scores
        """
        try:
            # Prepare inputs
            inputs = self._prepare_inputs(query, documents)
            
            # Process in batches
            all_scores = []
            for i in range(0, len(inputs), self.batch_size):
                batch_inputs = inputs[i:i + self.batch_size]
                batch_scores = self._process_batch(batch_inputs)
                all_scores.extend(batch_scores.cpu().numpy())
            
            # Add scores to documents
            for doc, score in zip(documents, all_scores):
                doc["score"] = float(score)
            
            # Sort by score
            reranked_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
            
            # Return top-k if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
            
            return reranked_docs
        except Exception as e:
            logger.error("Failed to rerank documents: %s", str(e))
            raise

    def clear(self) -> None:
        """Clear any cached data."""
        pass  # No cached data to clear 