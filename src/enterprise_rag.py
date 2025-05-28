"""
Enterprise-Ready RAG System with Evaluation Suite
Main module containing the core RAG system implementation.
"""

import logging
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Main class for the Enterprise-Ready RAG System.
    Coordinates document processing, retrieval, augmentation, and generation.
    """

    def __init__(
        self,
        retrieval_config: Optional[Dict[str, Any]] = None,
        augmentation_config: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the RAG system with configurable components.

        Args:
            retrieval_config: Configuration for the retrieval engine
            augmentation_config: Configuration for the augmentation layer
            generation_config: Configuration for the generation module
            vector_store_config: Configuration for the vector store
        """
        from .document_processing.processor import DocumentProcessor
        from .retrieval.engine import RetrievalEngine
        from .augmentation.augmenter import ContextAugmenter
        from .generation.generator import ResponseGenerator

        # Default configurations
        self.retrieval_config = retrieval_config or {"type": "hybrid", "top_k": 5}
        self.augmentation_config = augmentation_config or {"max_context_length": 4000}
        self.generation_config = generation_config or {"model": "gpt-3.5-turbo"}
        self.vector_store_config = vector_store_config or {"type": "faiss"}

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.retrieval_engine = RetrievalEngine(self.retrieval_config, self.vector_store_config)
        self.context_augmenter = ContextAugmenter(self.augmentation_config)
        self.response_generator = ResponseGenerator(self.generation_config)

        logger.info("RAG system initialized with configs: %s", 
                   {"retrieval": self.retrieval_config, 
                    "augmentation": self.augmentation_config,
                    "generation": self.generation_config,
                    "vector_store": self.vector_store_config})

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process and index a list of documents.

        Args:
            documents: List of document dictionaries with text and metadata
        """
        processed_docs = self.document_processor.process_documents(documents)
        self.retrieval_engine.index_documents(processed_docs)
        logger.info("Indexed %d documents", len(documents))

    def query(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        retrieval_options: Optional[Dict[str, Any]] = None,
        generation_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.

        Args:
            query: User query text
            filters: Optional metadata filters for retrieval
            retrieval_options: Optional parameters to customize retrieval
            generation_options: Optional parameters to customize generation

        Returns:
            Dict with answer, sources, and additional metadata
        """
        logger.info("Processing query: %s", query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieval_engine.retrieve(
            query, 
            filters=filters, 
            options=retrieval_options
        )
        
        # Augment context based on retrieved documents
        augmented_context = self.context_augmenter.augment(query, retrieved_docs)
        
        # Generate response
        response = self.response_generator.generate(
            query, 
            augmented_context, 
            options=generation_options
        )
        
        return {
            "query": query,
            "answer": response["text"],
            "sources": response["sources"],
            "metadata": {
                "retrieved_doc_count": len(retrieved_docs),
                "retrieved_docs": [doc["metadata"] for doc in retrieved_docs],
                "confidence": response["confidence"],
                "latency": {
                    "retrieval_ms": response["latency"]["retrieval_ms"],
                    "augmentation_ms": response["latency"]["augmentation_ms"],
                    "generation_ms": response["latency"]["generation_ms"],
                    "total_ms": response["latency"]["total_ms"],
                }
            }
        }
    
    def evaluate(
        self, 
        test_dataset: Union[str, List[Dict[str, Any]]], 
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system using the evaluation suite.

        Args:
            test_dataset: Path to test dataset file or list of test examples
            metrics: List of metrics to evaluate (default: all available metrics)

        Returns:
            Dictionary of evaluation results by metric
        """
        from .evaluation.evaluator import EvaluationSuite
        
        evaluator = EvaluationSuite()
        return evaluator.evaluate(self, test_dataset, metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG system.

        Returns:
            Dictionary with system status information
        """
        return {
            "document_count": self.retrieval_engine.get_document_count(),
            "vector_store_type": self.vector_store_config["type"],
            "retrieval_type": self.retrieval_config["type"],
            "generation_model": self.generation_config["model"],
            "is_ready": self.retrieval_engine.is_ready(),
        }
