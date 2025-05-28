# Enterprise-Ready RAG System - Quick Start Guide

This guide will help you get up and running with the Enterprise-Ready RAG System quickly.

## Prerequisites

- Python 3.8+ installed
- pip or conda for package management
- At least 8GB RAM for optimal performance
- Optional: NVIDIA GPU for faster embedding generation and model inference

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/TaimoorKhan10/enterprise-rag.git
cd enterprise-rag

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .
```

### Option 2: Using Docker

```bash
# Clone the repository
git clone https://github.com/TaimoorKhan10/enterprise-rag.git
cd enterprise-rag

# Build the Docker image
docker build -t enterprise-rag -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here enterprise-rag
```

## Configuration

Create a configuration file at `config/rag_config.json`:

```json
{
  "retrieval_config": {
    "type": "hybrid",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "reranker_threshold": 0.4,
    "use_reranker": true
  },
  "augmentation_config": {
    "max_context_length": 4000,
    "max_documents": 10,
    "citation_format": "inline",
    "strategy": "relevance_weighted"
  },
  "generation_config": {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "provider": "openai"
  },
  "vector_store_config": {
    "type": "faiss",
    "index_name": "document_index",
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "index_path": "data/index"
  }
}
```

For API keys and other sensitive information, use environment variables:

```bash
# On Linux/macOS
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# On Windows
set OPENAI_API_KEY=your_key_here
set ANTHROPIC_API_KEY=your_key_here
```

## Basic Usage

### Using the Python API

```python
from enterprise_rag import RAGSystem

# Initialize the system
rag = RAGSystem.from_config("config/rag_config.json")

# Index documents
documents = [
    {
        "id": "doc1",
        "text": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models.",
        "metadata": {"source": "example", "author": "AI Team"}
    },
    {
        "id": "doc2",
        "text": "FAISS (Facebook AI Similarity Search) is an efficient library for similarity search.",
        "metadata": {"source": "example", "author": "Research Team"}
    }
]

rag.index_documents(documents)

# Query the system
response = rag.query("What is RAG?")
print(response["answer"])
print("Sources:", [src["id"] for src in response["sources"]])
```

### Running the API Server

```bash
# Start the API server
python -m src.deployment.api

# The API will be available at http://localhost:8000
# Swagger UI documentation at http://localhost:8000/docs
```

### Running the Demo UI

```bash
# Set the API URL
export API_URL=http://localhost:8000

# Start the Streamlit app
cd ui/demo_app
streamlit run app.py

# The UI will be available at http://localhost:8501
```

## Example Commands

### Document Processing

Process and index a collection of documents:

```bash
# Process a folder of documents
python -m src.cli.index_documents --input ./data/documents --recursive

# Process specific file types
python -m src.cli.index_documents --input ./data/documents --file-types pdf,docx,txt
```

### Querying

Query the system from the command line:

```bash
# Basic query
python -m src.cli.query "What is hybrid retrieval?"

# Query with advanced options
python -m src.cli.query "Explain vector databases" --top-k 3 --retrieval-type dense --temperature 0.5
```

### Evaluation

Run evaluation on a test dataset:

```bash
# Evaluate on a test dataset
python -m src.cli.evaluate --dataset ./data/test_dataset.json --metrics retrieval_precision,answer_relevance

# Export evaluation results
python -m src.cli.evaluate --dataset ./data/test_dataset.json --output ./results/evaluation.json
```

## Troubleshooting

### Common Issues

1. **API Key Issues**: Make sure you've set the required API keys as environment variables.

2. **Memory Errors**: If you encounter memory issues with large document collections, try:
   - Reducing the batch size with `--batch-size 10`
   - Using a smaller embedding model
   - Processing documents in smaller chunks

3. **Dependencies**: If you encounter issues with dependencies, try:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

4. **Index Not Found**: If you get "Index not found" errors, make sure you've indexed documents before querying.

### Getting Help

If you encounter any issues, please:
1. Check the documentation in the `docs/` folder
2. Look for similar issues in the GitHub repository
3. Open a new issue with a detailed description of your problem

## Next Steps

- Check out the [Architecture Overview](architecture.md) for a deeper understanding
- Explore the [API Documentation](api.md) for advanced usage
- See the [Benchmarks](benchmarks.md) for performance metrics
