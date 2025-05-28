"""
Tests for the FastAPI deployment of the RAG system.
"""

import pytest
from fastapi.testclient import TestClient
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deployment.api import app, get_rag_system


# Create test client
client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy" or data["status"] == "unhealthy"


def test_add_document_endpoint():
    """Test adding a document via the API."""
    # Create a test document
    document = {
        "text": "This is a test document for the RAG API.",
        "metadata": {"source": "api_test", "title": "Test Document"},
    }

    # Send request
    response = client.post("/documents", json=document)

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["text"] == document["text"]
    assert data["metadata"] == document["metadata"]


def test_query_endpoint():
    """Test querying the RAG system via the API."""
    # Create query request
    query_request = {
        "query": "What is RAG?",
        "retrieval_options": {"top_k": 3},
        "generation_options": {"temperature": 0.7},
    }

    # Send request
    response = client.post("/query", json=query_request)

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert data["query"] == query_request["query"]
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert "sources" in data
    assert "metadata" in data


def test_batch_document_upload():
    """Test adding multiple documents in a batch."""
    # Create test documents
    documents = [
        {
            "text": "Document 1 for batch testing",
            "metadata": {"source": "batch_test", "index": 1},
        },
        {
            "text": "Document 2 for batch testing",
            "metadata": {"source": "batch_test", "index": 2},
        },
        {
            "text": "Document 3 for batch testing",
            "metadata": {"source": "batch_test", "index": 3},
        },
    ]

    # Send request
    response = client.post("/documents/batch", json=documents)

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3


def test_evaluation_endpoint():
    """Test starting an evaluation task."""
    # Create test dataset
    test_dataset = [
        {
            "query": "What is RAG?",
            "reference": {
                "answer": "Retrieval Augmented Generation (RAG) is a technique that enhances LLMs."
            },
        }
    ]

    # Create evaluation request
    evaluation_request = {
        "test_dataset": test_dataset,
        "metrics": ["retrieval_precision", "answer_relevance"],
    }

    # Send request
    response = client.post("/evaluate", json=evaluation_request)

    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"
    assert "task_id" in data
