"""
Test configuration and fixtures for the Enterprise-Ready RAG System.
"""

import os
import sys
import pytest
import json
from typing import Dict, Any, List
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enterprise_rag import RAGSystem
from src.document_processing.processor import DocumentProcessor
from src.retrieval.engine import RetrievalEngine
from src.retrieval.vector_stores.faiss_store import FAISSVectorStore
from src.retrieval.sparse.bm25_retriever import BM25Retriever
from src.augmentation.augmenter import ContextAugmenter
from src.generation.generator import ResponseGenerator
from src.evaluation.evaluator import EvaluationSuite


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """
    Sample documents for testing.
    """
    return [
        {
            "id": "doc1",
            "text": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge sources before generating responses.",
            "metadata": {
                "source": "test",
                "title": "RAG Overview"
            }
        },
        {
            "id": "doc2",
            "text": "Dense retrieval uses embeddings to find semantically similar documents, while sparse retrieval like BM25 focuses on keyword matching. Hybrid retrieval combines both approaches.",
            "metadata": {
                "source": "test",
                "title": "Retrieval Methods"
            }
        },
        {
            "id": "doc3",
            "text": "Vector databases like FAISS (Facebook AI Similarity Search) enable efficient similarity search in high-dimensional spaces, making them ideal for dense retrieval applications.",
            "metadata": {
                "source": "test",
                "title": "Vector Databases"
            }
        },
        {
            "id": "doc4",
            "text": "Chunking strategies affect retrieval performance. Smaller chunks may preserve local context but lose broader meaning, while larger chunks may contain more information but include irrelevant content.",
            "metadata": {
                "source": "test",
                "title": "Document Chunking"
            }
        },
        {
            "id": "doc5",
            "text": "Rerankers improve retrieval quality by taking an initial set of documents and reordering them based on relevance to the query, often using more compute-intensive models than the initial retriever.",
            "metadata": {
                "source": "test",
                "title": "Reranking"
            }
        }
    ]


@pytest.fixture
def temp_index_dir():
    """
    Create a temporary directory for vector indices.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_store(temp_index_dir):
    """
    FAISS vector store fixture.
    """
    store = FAISSVectorStore(
        index_name="test_index",
        embedding_model="all-MiniLM-L6-v2",
        dimension=384,
        index_path=temp_index_dir
    )
    return store


@pytest.fixture
def sparse_retriever():
    """
    BM25 retriever fixture.
    """
    return BM25Retriever()


@pytest.fixture
def retrieval_engine(vector_store, sparse_retriever):
    """
    Retrieval engine fixture.
    """
    return RetrievalEngine(
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        retrieval_type="hybrid",
        top_k=3
    )


@pytest.fixture
def document_processor():
    """
    Document processor fixture.
    """
    return DocumentProcessor(
        chunk_size=500,
        chunk_overlap=50
    )


@pytest.fixture
def context_augmenter():
    """
    Context augmenter fixture.
    """
    return ContextAugmenter(
        max_context_length=2000,
        citation_format="inline"
    )


@pytest.fixture
def response_generator():
    """
    Mock response generator that doesn't require API keys.
    """
    class MockResponseGenerator(ResponseGenerator):
        def generate(self, query, context, options=None):
            # Simple mock implementation
            sources = [doc["id"] for doc in context.get("documents", [])]
            return {
                "query": query,
                "answer": f"This is a mock answer for: {query}",
                "sources": sources,
                "metadata": {
                    "model": "mock-model",
                    "latency_ms": 100
                }
            }
    
    return MockResponseGenerator(
        model="mock-model",
        temperature=0.7
    )


@pytest.fixture
def rag_system(document_processor, retrieval_engine, context_augmenter, response_generator):
    """
    RAG system fixture.
    """
    system = RAGSystem(
        document_processor=document_processor,
        retrieval_engine=retrieval_engine,
        context_augmenter=context_augmenter,
        response_generator=response_generator
    )
    return system


@pytest.fixture
def evaluation_suite():
    """
    Evaluation suite fixture.
    """
    class MockEvaluationSuite(EvaluationSuite):
        def _initialize_judge(self):
            # Mock judge that doesn't require API keys
            return None
        
        def _evaluate_example(self, rag_system, example, metrics):
            # Simple mock evaluation
            return {
                "query": example["query"],
                "metrics": {
                    metric: {"score": 0.8, "explanation": f"Mock {metric} evaluation"} 
                    for metric in metrics
                }
            }
    
    return MockEvaluationSuite()


@pytest.fixture
def sample_test_dataset():
    """
    Sample test dataset for evaluation.
    """
    return [
        {
            "query": "What is RAG?",
            "reference": {
                "answer": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge sources before generating responses."
            }
        },
        {
            "query": "How does hybrid retrieval work?",
            "reference": {
                "answer": "Hybrid retrieval combines dense retrieval (using embeddings for semantic search) and sparse retrieval (like BM25 for keyword matching) to get the benefits of both approaches."
            }
        }
    ]


@pytest.fixture
def sample_config():
    """
    Sample configuration for testing.
    """
    return {
        "retrieval_config": {
            "type": "hybrid",
            "top_k": 3,
            "reranker_threshold": 0.7
        },
        "augmentation_config": {
            "max_context_length": 2000,
            "citation_format": "inline"
        },
        "generation_config": {
            "model": "mock-model",
            "temperature": 0.7
        },
        "evaluation_config": {
            "metrics": [
                "retrieval_precision",
                "answer_relevance",
                "factual_correctness"
            ]
        }
    }
