# Changelog

All notable changes to the Enterprise-RAG-Framework project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Document Processing Enhancements**
  - Added Excel document processor
    - Support for .xlsx and .xls formats
    - Enhanced metadata extraction
    - Sheet-level processing
    - Formula and special character handling
  - Enhanced PDF processor
    - OCR fallback support
    - Table extraction capabilities
    - Layout preservation options
    - Image extraction support
    - Binary PDF content handling
  - Multi-format document support
    - PDF, DOCX, HTML, text, markdown
    - Images (with OCR)
    - Audio (with transcription)
    - Excel files

- **Retrieval Engine Enhancements**
  - Added Milvus vector store for efficient similarity search
    - Support for multiple index types (IVF_FLAT, HNSW)
    - Efficient batch processing
    - Metadata filtering capabilities
    - Connection pooling and error handling
  - Implemented ColBERT retriever
    - Contextualized token embeddings
    - Late interaction scoring mechanism
    - Batch processing support
    - GPU acceleration
  - Added MonoT5 reranker
    - State-of-the-art reranking model
    - Query-document relevance scoring
    - Batch processing capabilities
    - GPU support

### Changed
- Updated `requirements.txt` with new dependencies:
  - Added `pymilvus>=2.3.0` for Milvus vector store
  - Added `torch>=2.0.0` for deep learning components
  - Added `transformers>=4.30.0` for transformer models
  - Added `sentence-transformers>=2.2.2` for embeddings
  - Added `rank-bm25>=0.2.2` for sparse retrieval
  - Added `pandas>=2.0.0` for Excel processing
  - Added `PyMuPDF>=1.22.0` for PDF processing
  - Added `pytesseract>=0.3.10` for OCR support
  - Added `tabula-py>=2.7.0` for table extraction

### Testing
- Added comprehensive test suite:
  - Unit tests for document processors
    - Excel processor tests
    - PDF processor tests
    - Text processor tests
  - Unit tests for retrieval components
    - Milvus vector store tests
    - ColBERT retriever tests
    - MonoT5 reranker tests
  - Integration tests for combined retrieval pipeline
  - Test fixtures and configurations

## [0.1.0] - 2024-03-19
### Added
- Initial project setup
- Basic RAG framework structure
- Core components:
  - Document processing
  - Basic retrieval
  - Generation support
  - Evaluation metrics
  - Monitoring capabilities 