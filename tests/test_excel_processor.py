"""
Tests for the Excel document processor.
"""

import os
import pytest
import pandas as pd
from src.document_processing.excel_processor import ExcelProcessor


@pytest.fixture
def sample_excel_file(tmp_path):
    """Create a sample Excel file for testing."""
    # Create a sample DataFrame
    df1 = pd.DataFrame({
        'Name': ['John', 'Alice', 'Bob'],
        'Age': [30, 25, 35],
        'City': ['New York', 'London', 'Paris']
    })
    
    df2 = pd.DataFrame({
        'Product': ['A', 'B', 'C'],
        'Price': [100, 200, 300],
        'Stock': [10, 20, 30]
    })
    
    # Create Excel file with multiple sheets
    file_path = tmp_path / "test.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
    
    return str(file_path)


@pytest.fixture
def processor():
    """Create an Excel processor instance."""
    config = {
        "chunk_size": 100,
        "chunk_overlap": 20,
        "extract_metadata": True
    }
    return ExcelProcessor(config)


def test_processor_initialization(processor):
    """Test processor initialization with config."""
    assert processor.chunk_size == 100
    assert processor.chunk_overlap == 20
    assert processor.extract_metadata is True


def test_process_excel_file(processor, sample_excel_file):
    """Test processing an Excel file."""
    # Create document dictionary
    document = {
        "content": sample_excel_file,
        "metadata": {
            "filename": "test.xlsx",
            "mimetype": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
    }
    
    # Process the document
    result = processor.process(document)
    
    # Verify the result structure
    assert isinstance(result, dict)
    assert "chunks" in result
    assert "metadata" in result
    
    # Verify metadata
    metadata = result["metadata"]
    assert metadata["document_type"] == "excel"
    assert metadata["sheet_count"] == 2
    assert "sheet_names" in metadata
    assert len(metadata["sheet_names"]) == 2
    
    # Verify chunks
    chunks = result["chunks"]
    assert len(chunks) > 0
    
    # Verify chunk structure
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "chunk_id" in chunk["metadata"]
        assert "sheet_name" in chunk["metadata"]


def test_process_empty_excel(processor, tmp_path):
    """Test processing an empty Excel file."""
    # Create empty Excel file
    file_path = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(file_path, index=False)
    
    document = {
        "content": str(file_path),
        "metadata": {
            "filename": "empty.xlsx",
            "mimetype": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
    }
    
    result = processor.process(document)
    assert len(result["chunks"]) == 0
    assert result["metadata"]["sheet_count"] == 1


def test_process_excel_with_formulas(processor, tmp_path):
    """Test processing Excel file with formulas."""
    # Create Excel file with formulas
    file_path = tmp_path / "formulas.xlsx"
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['=A1+B1', '=A2+B2', '=A3+B3']
    })
    df.to_excel(file_path, index=False)
    
    document = {
        "content": str(file_path),
        "metadata": {
            "filename": "formulas.xlsx",
            "mimetype": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
    }
    
    result = processor.process(document)
    assert len(result["chunks"]) > 0
    # Verify that formulas are included in the text
    assert any("=A1+B1" in chunk["text"] for chunk in result["chunks"])


def test_process_excel_with_special_characters(processor, tmp_path):
    """Test processing Excel file with special characters."""
    # Create Excel file with special characters
    file_path = tmp_path / "special.xlsx"
    df = pd.DataFrame({
        'Text': ['Hello, World!', 'Test & More', 'Price: $100'],
        'Numbers': [1.23, -4.56, 7.89]
    })
    df.to_excel(file_path, index=False)
    
    document = {
        "content": str(file_path),
        "metadata": {
            "filename": "special.xlsx",
            "mimetype": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
    }
    
    result = processor.process(document)
    assert len(result["chunks"]) > 0
    # Verify that special characters are preserved
    assert any("Hello, World!" in chunk["text"] for chunk in result["chunks"])
    assert any("Test & More" in chunk["text"] for chunk in result["chunks"])
    assert any("Price: $100" in chunk["text"] for chunk in result["chunks"])


def test_process_excel_with_dates(processor, tmp_path):
    """Test processing Excel file with dates."""
    # Create Excel file with dates
    file_path = tmp_path / "dates.xlsx"
    df = pd.DataFrame({
        'Date': ['2024-01-01', '2024-02-01', '2024-03-01'],
        'Event': ['New Year', 'Valentine\'s Day', 'Spring']
    })
    df.to_excel(file_path, index=False)
    
    document = {
        "content": str(file_path),
        "metadata": {
            "filename": "dates.xlsx",
            "mimetype": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
    }
    
    result = processor.process(document)
    assert len(result["chunks"]) > 0
    # Verify that dates are included in the text
    assert any("2024-01-01" in chunk["text"] for chunk in result["chunks"]) 