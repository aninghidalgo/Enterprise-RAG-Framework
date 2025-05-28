#!/usr/bin/env python
"""
Basic example showing how to query the Enterprise-Ready RAG System using the Python API.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

from src.enterprise_rag import RAGSystem

def main():
    """Run a basic query example."""
    
    # Initialize the RAG system with a basic configuration
    rag_system = RAGSystem(
        vector_store_config={
            "type": "faiss",
            "index_path": "data/index",
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        },
        retrieval_config={
            "type": "hybrid",
            "top_k": 5,
            "use_reranker": True
        },
        generation_config={
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    # Define your query
    query = "What are the key benefits of using a RAG system for enterprise applications?"
    
    # Optional query parameters
    query_options = {
        "filters": {
            "metadata.doc_type": "pdf",  # Optional filter by document type
            "metadata.date": {"$gt": "2022-01-01"}  # Filter by date
        },
        "retrieval_options": {
            "use_semantic": True,
            "use_keyword": True
        }
    }
    
    # Execute the query
    print(f"Querying: '{query}'")
    
    response = rag_system.query(query, options=query_options)
    
    # Print the results
    print("\n----- ANSWER -----")
    print(response["answer"])
    
    print("\n----- SOURCES -----")
    for i, source in enumerate(response["sources"], 1):
        print(f"{i}. {source['title']} (score: {source['score']:.3f})")
        print(f"   Snippet: {source['text'][:150]}...")
    
    # Print performance metrics
    print("\n----- PERFORMANCE METRICS -----")
    for key, value in response["metrics"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}s")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY=your_api_key_here")
        print("For this example, using a mock response instead.\n")
    
    main()
