#!/usr/bin/env python
"""
Example showing how to index documents with the Enterprise-Ready RAG System.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.enterprise_rag import RAGSystem
from src.document_processing.processor import DocumentProcessor

def main():
    """Run a document indexing example."""
    
    # Initialize the RAG system
    rag_system = RAGSystem(
        vector_store_config={
            "type": "faiss",
            "index_path": "data/index",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        }
    )
    
    # Define document paths (replace with your actual documents)
    documents_dir = Path("examples/sample_documents")
    if not documents_dir.exists():
        logger.warning(f"Sample documents directory not found: {documents_dir}")
        logger.info("Creating sample documents directory and a sample text file")
        
        # Create directory
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample text file
        sample_text = """
        # Enterprise-Ready RAG System
        
        Retrieval-Augmented Generation (RAG) combines the power of large language models with 
        information retrieval systems to provide accurate, up-to-date responses grounded in 
        your organization's data.
        
        ## Key Benefits
        
        1. **Reduced Hallucination**: By grounding responses in retrieved documents, RAG systems
           significantly reduce the problem of LLM hallucinations.
           
        2. **Domain Adaptation**: Enables LLMs to work with domain-specific knowledge without
           expensive fine-tuning.
           
        3. **Up-to-date Information**: Allows models to access the latest information that
           wasn't part of their training data.
           
        4. **Transparency**: Provides citations and references to source documents, increasing
           trust and verifiability.
           
        5. **Cost Efficiency**: More efficient than fine-tuning large models on domain-specific data.
        """
        
        with open(documents_dir / "rag_overview.txt", "w", encoding="utf-8") as f:
            f.write(sample_text)
    
    # Process documents
    document_processor = DocumentProcessor(
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Find documents
    document_paths = list(documents_dir.glob("**/*.*"))
    logger.info(f"Found {len(document_paths)} documents to process")
    
    # Process and index documents
    for doc_path in document_paths:
        try:
            logger.info(f"Processing document: {doc_path}")
            
            # Process document into chunks
            doc_chunks = document_processor.process_document(str(doc_path))
            
            logger.info(f"Created {len(doc_chunks)} chunks from {doc_path.name}")
            
            # Add document chunks to RAG system
            doc_ids = rag_system.add_documents(doc_chunks)
            
            logger.info(f"Indexed document with {len(doc_ids)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {str(e)}")
    
    # Save the index
    rag_system.save_index()
    logger.info(f"Index saved to {rag_system.vector_store.index_path}")
    
    # Print statistics
    stats = rag_system.get_stats()
    print("\n----- INDEX STATISTICS -----")
    print(f"Total documents indexed: {stats.get('total_documents', 0)}")
    print(f"Total chunks indexed: {stats.get('total_chunks', 0)}")
    print(f"Vector index size: {stats.get('vector_store_size_mb', 0):.2f} MB")
    print(f"Vocabulary size: {stats.get('vocabulary_size', 0)}")


if __name__ == "__main__":
    main()
