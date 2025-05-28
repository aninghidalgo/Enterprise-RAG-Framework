"""
Tests for the new retrieval components.
"""

import pytest
import numpy as np
import torch
from src.retrieval.vector_stores.milvus_store import MilvusVectorStore
from src.retrieval.sparse.colbert_retriever import ColBERTRetriever
from src.retrieval.reranker.mono_t5 import MonoT5Reranker


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "id": "1",
            "text": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"type": "test", "source": "sample"},
            "embedding": np.random.rand(768).tolist()
        },
        {
            "id": "2",
            "text": "A fast orange fox leaps across a sleepy canine.",
            "metadata": {"type": "test", "source": "sample"},
            "embedding": np.random.rand(768).tolist()
        },
        {
            "id": "3",
            "text": "The weather is beautiful today.",
            "metadata": {"type": "test", "source": "sample"},
            "embedding": np.random.rand(768).tolist()
        }
    ]


@pytest.fixture
def milvus_config():
    """Create Milvus configuration."""
    return {
        "host": "localhost",
        "port": 19530,
        "collection_name": "test_collection",
        "embedding_dimension": 768,
        "index_type": "IVF_FLAT",
        "index_params": {
            "nlist": 1024,
            "metric_type": "L2"
        }
    }


@pytest.fixture
def colbert_config():
    """Create ColBERT configuration."""
    return {
        "model_name": "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco",
        "max_length": 512,
        "batch_size": 32,
        "use_gpu": False
    }


@pytest.fixture
def monot5_config():
    """Create MonoT5 configuration."""
    return {
        "model_name": "castorini/monot5-base-msmarco",
        "max_length": 512,
        "batch_size": 8,
        "use_gpu": False
    }


def test_milvus_store(milvus_config, sample_documents):
    """Test Milvus vector store functionality."""
    # Skip if Milvus is not available
    try:
        store = MilvusVectorStore(milvus_config)
    except Exception as e:
        pytest.skip(f"Milvus not available: {str(e)}")
    
    # Test adding documents
    store.add_documents(sample_documents)
    assert store.get_document_count() == len(sample_documents)
    
    # Test search
    query_vector = np.random.rand(768).tolist()
    results = store.search(query_vector, top_k=2)
    assert len(results) == 2
    assert all("score" in doc for doc in results)
    
    # Test filters
    filtered_results = store.search(
        query_vector,
        top_k=2,
        filters={"type": "test"}
    )
    assert len(filtered_results) == 2
    
    # Test clearing
    store.clear()
    assert store.get_document_count() == 0


def test_colbert_retriever(colbert_config, sample_documents):
    """Test ColBERT retriever functionality."""
    retriever = ColBERTRetriever(colbert_config)
    
    # Test adding documents
    retriever.add_documents(sample_documents)
    assert retriever.get_document_count() == len(sample_documents)
    
    # Test search
    query = "fox jumping"
    results = retriever.search(query, top_k=2)
    assert len(results) == 2
    assert all("score" in doc for doc in results)
    
    # Test filters
    filtered_results = retriever.search(
        query,
        top_k=2,
        filters={"type": "test"}
    )
    assert len(filtered_results) == 2
    
    # Test clearing
    retriever.clear()
    assert retriever.get_document_count() == 0


def test_monot5_reranker(monot5_config, sample_documents):
    """Test MonoT5 reranker functionality."""
    reranker = MonoT5Reranker(monot5_config)
    
    # Test reranking
    query = "fox jumping"
    reranked_docs = reranker.rerank(query, sample_documents, top_k=2)
    assert len(reranked_docs) == 2
    assert all("score" in doc for doc in reranked_docs)
    
    # Verify scores are in descending order
    scores = [doc["score"] for doc in reranked_docs]
    assert scores == sorted(scores, reverse=True)


def test_integration(colbert_config, monot5_config, sample_documents):
    """Test integration of ColBERT and MonoT5."""
    # Initialize components
    retriever = ColBERTRetriever(colbert_config)
    reranker = MonoT5Reranker(monot5_config)
    
    # Add documents
    retriever.add_documents(sample_documents)
    
    # First stage: ColBERT retrieval
    query = "fox jumping"
    retrieved_docs = retriever.search(query, top_k=3)
    assert len(retrieved_docs) == 3
    
    # Second stage: MonoT5 reranking
    reranked_docs = reranker.rerank(query, retrieved_docs, top_k=2)
    assert len(reranked_docs) == 2
    
    # Verify scores are in descending order
    scores = [doc["score"] for doc in reranked_docs]
    assert scores == sorted(scores, reverse=True) 