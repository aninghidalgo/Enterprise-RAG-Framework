"""
Tests for the retrieval components of the RAG system.
"""

import pytest
from src.retrieval.engine import RetrievalEngine
from src.retrieval.vector_stores.faiss_store import FAISSVectorStore
from src.retrieval.sparse.bm25_retriever import BM25Retriever


def test_retrieval_engine_initialization(vector_store, sparse_retriever):
    """Test that the RetrievalEngine initializes correctly."""
    engine = RetrievalEngine(
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        retrieval_type="hybrid",
        top_k=3,
    )

    assert engine.retrieval_type == "hybrid"
    assert engine.top_k == 3
    assert engine.vector_store is not None
    assert engine.sparse_retriever is not None


def test_document_indexing(retrieval_engine, sample_documents):
    """Test indexing documents in the retrieval engine."""
    # Index the documents
    doc_ids = retrieval_engine.index_documents(sample_documents)

    # Check results
    assert len(doc_ids) == len(sample_documents)
    for doc_id in doc_ids:
        assert doc_id in [doc["id"] for doc in sample_documents]


def test_dense_retrieval(retrieval_engine, sample_documents):
    """Test dense retrieval functionality."""
    # First index the documents
    retrieval_engine.index_documents(sample_documents)

    # Set retrieval type to dense
    retrieval_engine.retrieval_type = "dense"

    # Perform retrieval
    query = "What is RAG?"
    results = retrieval_engine.retrieve(query)

    # Check results
    assert isinstance(results, list)
    assert len(results) <= retrieval_engine.top_k
    assert all("id" in doc for doc in results)
    assert all("text" in doc for doc in results)
    assert all("score" in doc for doc in results)


def test_sparse_retrieval(retrieval_engine, sample_documents):
    """Test sparse retrieval functionality."""
    # First index the documents
    retrieval_engine.index_documents(sample_documents)

    # Set retrieval type to sparse
    retrieval_engine.retrieval_type = "sparse"

    # Perform retrieval
    query = "Vector database FAISS"
    results = retrieval_engine.retrieve(query)

    # Check results
    assert isinstance(results, list)
    assert len(results) <= retrieval_engine.top_k
    assert all("id" in doc for doc in results)
    assert all("text" in doc for doc in results)
    assert all("score" in doc for doc in results)


def test_hybrid_retrieval(retrieval_engine, sample_documents):
    """Test hybrid retrieval functionality."""
    # First index the documents
    retrieval_engine.index_documents(sample_documents)

    # Set retrieval type to hybrid
    retrieval_engine.retrieval_type = "hybrid"

    # Perform retrieval
    query = "How does retrieval work?"
    results = retrieval_engine.retrieve(query)

    # Check results
    assert isinstance(results, list)
    assert len(results) <= retrieval_engine.top_k
    assert all("id" in doc for doc in results)
    assert all("text" in doc for doc in results)
    assert all("score" in doc for doc in results)


def test_filtered_retrieval(retrieval_engine, sample_documents):
    """Test retrieval with metadata filters."""
    # First index the documents
    retrieval_engine.index_documents(sample_documents)

    # Create a filter for documents with a specific title
    filters = {"metadata.title": "Vector Databases"}

    # Perform retrieval with filter
    query = "similarity search"
    results = retrieval_engine.retrieve(query, filters=filters)

    # Check that all results match the filter
    for doc in results:
        assert "metadata" in doc
        assert "title" in doc["metadata"]
        assert doc["metadata"]["title"] == "Vector Databases"
