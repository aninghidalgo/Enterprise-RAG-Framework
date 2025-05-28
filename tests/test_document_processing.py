"""
Tests for the document processing module.
"""

import pytest
from src.document_processing.processor import DocumentProcessor
from src.document_processing.text_processor import TextProcessor


def test_document_processor_initialization():
    """Test that the DocumentProcessor initializes correctly."""
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

    assert processor.chunk_size == 500
    assert processor.chunk_overlap == 50
    assert processor.chunker is not None


def test_process_text_document():
    """Test processing a text document."""
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    # Create a sample document
    doc = {
        "id": "test-doc",
        "text": "This is a test document. " * 20,  # Create text longer than chunk size
        "metadata": {"source": "test"},
    }

    # Process the document
    processed_doc = processor.process_document(doc)

    # Check results
    assert processed_doc["id"] == "test-doc"
    assert "chunks" in processed_doc
    assert len(processed_doc["chunks"]) > 1  # Should have multiple chunks

    # Verify chunks
    for chunk in processed_doc["chunks"]:
        assert "text" in chunk
        assert len(chunk["text"]) <= 100 + 20  # chunk_size + overlap
        assert "metadata" in chunk
        assert chunk["metadata"]["source"] == "test"
        assert "parent_id" in chunk["metadata"]
        assert chunk["metadata"]["parent_id"] == "test-doc"


def test_text_chunker():
    """Test the TextProcessor chunking functionality."""
    chunker = TextProcessor(config={'chunk_size': 50, 'chunk_overlap': 10})

    # Test text
    text = "This is a sample text. " * 10
    
    # Create a test document
    document = {
        "text": text,
        "metadata": {"source": "test", "id": "test-doc"}
    }

    # Generate chunks
    chunk_docs = chunker.process(document)

    # Check results
    assert len(chunk_docs) > 1

    # Extract text chunks for easier testing
    chunks = [doc["text"] for doc in chunk_docs]
    
    # Check first chunk
    assert chunks[0].startswith("This is a sample text.")
    
    # Check metadata
    assert chunk_docs[0]["metadata"]["source"] == "test"
    assert chunk_docs[0]["metadata"]["id"] == "test-doc"
    assert "chunk_index" in chunk_docs[0]["metadata"]
    
    # Check for overlap between chunks if using fixed strategy
    if chunker.config.get("chunking_strategy") == "fixed":
        for i in range(len(chunks) - 1):
            # The end of one chunk should overlap with the beginning of the next
            assert chunks[i][-10:] in chunks[i + 1]


def test_empty_document():
    """Test processing an empty document."""
    processor = DocumentProcessor()

    # Create an empty document
    doc = {"id": "empty-doc", "text": "", "metadata": {"source": "test"}}

    # Process the document
    processed_doc = processor.process_document(doc)

    # Check results
    assert processed_doc["id"] == "empty-doc"
    assert "chunks" in processed_doc
    assert len(processed_doc["chunks"]) == 0  # Should have no chunks
