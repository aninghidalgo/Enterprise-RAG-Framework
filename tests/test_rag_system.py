"""
Tests for the end-to-end RAG system functionality.
"""

import pytest
from src.enterprise_rag import RAGSystem


def test_rag_system_initialization(
    document_processor, retrieval_engine, context_augmenter, response_generator
):
    """Test that the RAG system initializes correctly with components."""
    system = RAGSystem(
        document_processor=document_processor,
        retrieval_engine=retrieval_engine,
        context_augmenter=context_augmenter,
        response_generator=response_generator,
    )

    assert system.document_processor is not None
    assert system.retrieval_engine is not None
    assert system.context_augmenter is not None
    assert system.response_generator is not None


def test_document_indexing(rag_system, sample_documents):
    """Test indexing documents in the RAG system."""
    # Index the documents
    doc_ids = rag_system.index_documents(sample_documents)

    # Check results
    assert len(doc_ids) == len(sample_documents)
    for doc_id in doc_ids:
        assert doc_id in [doc["id"] for doc in sample_documents]


def test_query_functionality(rag_system, sample_documents):
    """Test the main query functionality of the RAG system."""
    # First index the documents
    rag_system.index_documents(sample_documents)

    # Perform a query
    query = "What is RAG and how does it work?"
    response = rag_system.query(query)

    # Check response structure
    assert "query" in response
    assert response["query"] == query
    assert "answer" in response
    assert isinstance(response["answer"], str)
    assert len(response["answer"]) > 0
    assert "sources" in response
    assert isinstance(response["sources"], list)
    assert "metadata" in response


def test_query_with_options(rag_system, sample_documents):
    """Test querying with custom options."""
    # First index the documents
    rag_system.index_documents(sample_documents)

    # Set custom retrieval and generation options
    retrieval_options = {"top_k": 2, "retrieval_type": "dense"}

    generation_options = {"temperature": 0.5}

    # Perform query with options
    query = "Explain vector databases"
    response = rag_system.query(
        query,
        retrieval_options=retrieval_options,
        generation_options=generation_options,
    )

    # Check response
    assert "query" in response
    assert response["query"] == query
    assert "answer" in response
    assert "sources" in response
    assert len(response["sources"]) <= 2  # We set top_k to 2


def test_query_with_filters(rag_system, sample_documents):
    """Test querying with metadata filters."""
    # First index the documents
    rag_system.index_documents(sample_documents)

    # Create a filter
    filters = {"metadata.title": "Reranking"}

    # Perform query with filter
    query = "How do rerankers work?"
    response = rag_system.query(query, filters=filters)

    # Check that sources match the filter
    for source in response["sources"]:
        assert "metadata" in source
        assert "title" in source["metadata"]
        assert source["metadata"]["title"] == "Reranking"


def test_system_with_no_relevant_documents(rag_system, sample_documents):
    """Test system behavior when no relevant documents are found."""
    # First index the documents
    rag_system.index_documents(sample_documents)

    # Query about an unrelated topic
    query = "What is the capital of France?"
    response = rag_system.query(query)

    # System should still return an answer, but might indicate no relevant info
    assert "answer" in response
    assert isinstance(response["answer"], str)

    # There might be some sources, but they're likely not very relevant
    assert "sources" in response
