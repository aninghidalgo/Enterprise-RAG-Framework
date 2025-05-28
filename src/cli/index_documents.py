#!/usr/bin/env python
"""
CLI tool for indexing documents in the Enterprise-Ready RAG System.
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.enterprise_rag import RAGSystem
from src.document_processing.processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index documents in the Enterprise-Ready RAG System"
    )
    
    # Input options
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to document file or directory of documents"
    )
    parser.add_argument(
        "--recursive", "-r", action="store_true", 
        help="Process directories recursively"
    )
    parser.add_argument(
        "--file-types", type=str, default="pdf,docx,txt,md,html",
        help="Comma-separated list of file extensions to process (default: pdf,docx,txt,md,html)"
    )
    
    # Processing options
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Size of document chunks in tokens (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50,
        help="Overlap between chunks in tokens (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of documents to process in each batch (default: 10)"
    )
    
    # System configuration
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to RAG system configuration file"
    )
    parser.add_argument(
        "--index-path", type=str, default="data/index",
        help="Path to store/load vector index (default: data/index)"
    )
    
    # Output options
    parser.add_argument(
        "--output-metadata", type=str, default=None,
        help="Path to save document metadata JSON (optional)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def get_file_paths(input_path: str, file_types: List[str], recursive: bool = False) -> List[str]:
    """
    Get paths to all matching files.
    
    Args:
        input_path: Path to file or directory
        file_types: List of file extensions to include
        recursive: Whether to search directories recursively
        
    Returns:
        List of file paths
    """
    input_path = Path(input_path).resolve()
    
    if input_path.is_file():
        if input_path.suffix.lower().lstrip('.') in file_types:
            return [str(input_path)]
        return []
    
    if input_path.is_dir():
        file_paths = []
        
        if recursive:
            for ext in file_types:
                file_paths.extend([str(p) for p in input_path.glob(f"**/*.{ext}")])
        else:
            for ext in file_types:
                file_paths.extend([str(p) for p in input_path.glob(f"*.{ext}")])
        
        return file_paths
    
    return []

def process_files(
    file_paths: List[str],
    rag_system: RAGSystem,
    batch_size: int = 10,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Process files and index them in the RAG system.
    
    Args:
        file_paths: List of file paths to process
        rag_system: RAG system instance
        batch_size: Number of documents to process in each batch
        verbose: Whether to print verbose output
        
    Returns:
        List of document metadata
    """
    total_files = len(file_paths)
    logger.info(f"Processing {total_files} files")
    
    all_metadata = []
    start_time = time.time()
    
    # Process in batches
    for i in range(0, total_files, batch_size):
        batch_paths = file_paths[i:i+batch_size]
        batch_docs = []
        
        # Create document objects
        for file_path in batch_paths:
            try:
                path = Path(file_path)
                doc_id = str(path.stem)
                mime_type, _ = mimetypes.guess_type(file_path)
                
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Create document object
                doc = {
                    "id": doc_id,
                    "content": content,
                    "metadata": {
                        "source": str(path),
                        "filename": path.name,
                        "mime_type": mime_type or "application/octet-stream",
                        "extension": path.suffix.lower().lstrip('.'),
                        "size_bytes": path.stat().st_size,
                        "modified": path.stat().st_mtime
                    }
                }
                
                batch_docs.append(doc)
                
                if verbose:
                    logger.info(f"Added document: {path.name}")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Process batch
        try:
            # Process and index documents
            doc_ids = rag_system.document_processor.process_batch(batch_docs)
            indexed_ids = rag_system.index_documents(batch_docs)
            
            # Collect metadata
            for doc in batch_docs:
                doc["metadata"]["indexed"] = doc["id"] in indexed_ids
                all_metadata.append(doc["metadata"])
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1}: {len(batch_docs)} documents")
        
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Indexing complete: processed {total_files} documents in {elapsed_time:.2f} seconds")
    logger.info(f"Average processing time: {elapsed_time/total_files:.2f} seconds per document")
    
    return all_metadata

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse file types
    file_types = [ext.strip().lower().lstrip('.') for ext in args.file_types.split(',')]
    
    # Get file paths
    file_paths = get_file_paths(args.input, file_types, args.recursive)
    
    if not file_paths:
        logger.error(f"No matching files found in {args.input}")
        sys.exit(1)
    
    logger.info(f"Found {len(file_paths)} files to process")
    
    # Initialize RAG system
    try:
        if args.config:
            rag_system = RAGSystem.from_config(args.config)
        else:
            # Create with default settings
            rag_system = RAGSystem(
                document_processor=DocumentProcessor(
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap
                ),
                vector_store_config={
                    "index_path": args.index_path
                }
            )
        
        logger.info("Initialized RAG system")
    
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        sys.exit(1)
    
    # Process files
    metadata = process_files(
        file_paths=file_paths,
        rag_system=rag_system,
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    # Save metadata if requested
    if args.output_metadata and metadata:
        try:
            output_path = Path(args.output_metadata)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {output_path}")
        
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
    
    logger.info("Indexing completed successfully")

if __name__ == "__main__":
    main()
