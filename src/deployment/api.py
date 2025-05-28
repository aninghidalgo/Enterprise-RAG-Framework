"""
FastAPI application for the Enterprise-Ready RAG System.
Provides REST API endpoints for document management, querying, and evaluation.
"""

import logging
import time
import os
import json
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Query,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import RAG system
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enterprise_rag import RAGSystem


# Define API models
class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""

    query: str = Field(..., description="The user query text")
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata filters"
    )
    retrieval_options: Optional[Dict[str, Any]] = Field(
        None, description="Optional retrieval parameters"
    )
    generation_options: Optional[Dict[str, Any]] = Field(
        None, description="Optional generation parameters"
    )


class DocumentBase(BaseModel):
    """Base model for document operations."""

    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field({}, description="Document metadata")


class DocumentRequest(DocumentBase):
    """Request model for adding documents."""

    id: Optional[str] = Field(
        None, description="Optional document ID (will be generated if not provided)"
    )


class DocumentResponse(DocumentBase):
    """Response model for document operations."""

    id: str = Field(..., description="Document ID")


class QueryResponse(BaseModel):
    """Response model for query operations."""

    query: str = Field(..., description="Original query text")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field([], description="Source information")
    metadata: Dict[str, Any] = Field({}, description="Response metadata")


class EvaluationRequest(BaseModel):
    """Request model for evaluation operations."""

    test_dataset: List[Dict[str, Any]] = Field(
        ..., description="Test dataset for evaluation"
    )
    metrics: Optional[List[str]] = Field(
        None, description="List of metrics to evaluate"
    )


# Create FastAPI app
app = FastAPI(
    title="Enterprise-Ready RAG System API",
    description="API for document management, querying, and evaluation of a production-grade Retrieval Augmented Generation system",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "health", "description": "Health check and system status endpoints"},
        {
            "name": "documents",
            "description": "Document management operations (add, delete, query)",
        },
        {"name": "query", "description": "RAG system query operations"},
        {"name": "evaluation", "description": "System evaluation endpoints"},
    ],
    contact={
        "name": "Taimoor Khan",
        "url": "https://github.com/TaimoorKhan10",
        "email": "contact@example.com",
    },
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system = None

# Background tasks
running_tasks = {}


def get_rag_system() -> RAGSystem:
    """
    Get or initialize the RAG system.

    Returns:
        RAG system instance
    """
    global rag_system

    if rag_system is None:
        # Load configuration
        config_path = os.environ.get("RAG_CONFIG_PATH", "config/rag_config.json")

        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                # Default configuration
                config = {
                    "retrieval_config": {"type": "hybrid", "top_k": 5},
                    "augmentation_config": {"max_context_length": 4000},
                    "generation_config": {"model": "gpt-3.5-turbo"},
                    "vector_store_config": {
                        "type": "faiss",
                        "index_path": "data/index",
                    },
                }

            # Initialize RAG system
            rag_system = RAGSystem(
                retrieval_config=config.get("retrieval_config"),
                augmentation_config=config.get("augmentation_config"),
                generation_config=config.get("generation_config"),
                vector_store_config=config.get("vector_store_config"),
            )

            logger.info("Initialized RAG system")

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise

    return rag_system


# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to documentation."""
    return {"message": "Enterprise-Ready RAG System API", "docs_url": "/docs"}


@app.get(
    "/health",
    tags=["health"],
    summary="Check system health",
    description="Returns the health status of the RAG system and its components",
    response_description="Health status with detailed component information",
    responses={
        200: {
            "description": "Successful response with system health status",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "rag_status": {
                            "document_count": 120,
                            "vector_store_type": "faiss",
                            "retrieval_type": "hybrid",
                            "generation_model": "gpt-3.5-turbo",
                            "components": {
                                "document_processor": "healthy",
                                "retrieval_engine": "healthy",
                                "augmenter": "healthy",
                                "generator": "healthy",
                            },
                        },
                    }
                }
            },
        },
        500: {
            "description": "System is unhealthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "unhealthy",
                        "error": "Failed to initialize vector store",
                    }
                }
            },
        },
    },
)
async def health_check():
    """Check system health."""
    try:
        rag = get_rag_system()
        status = rag.get_status()
        return {"status": "healthy", "rag_status": status}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post(
    "/documents",
    response_model=DocumentResponse,
    tags=["documents"],
    summary="Add a single document to the system",
    description="Adds a document to the RAG system for indexing and retrieval",
    response_description="The indexed document with its assigned ID",
    responses={
        200: {
            "description": "Document successfully indexed",
            "content": {
                "application/json": {
                    "example": {
                        "id": "fb0c6a68-a26e-4737-9c76-4cf2eb3d9c0b",
                        "text": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models...",
                        "metadata": {
                            "source": "documentation",
                            "title": "RAG Overview",
                            "author": "AI Team",
                        },
                    }
                }
            },
        },
        500: {
            "description": "Failed to index document",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to index document: vector store error"
                    }
                }
            },
        },
    },
)
async def add_document(document: DocumentRequest):
    """Add a document to the system."""
    try:
        rag = get_rag_system()

        # Prepare document
        doc_dict = document.dict()

        if doc_dict.get("id") is None:
            doc_dict["id"] = str(uuid.uuid4())

        # Index document
        doc_ids = rag.index_documents([doc_dict])

        if not doc_ids:
            raise HTTPException(status_code=500, detail="Failed to index document")

        return doc_dict

    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/batch", response_model=List[str])
async def add_documents_batch(documents: List[DocumentRequest]):
    """Add multiple documents to the system."""
    try:
        rag = get_rag_system()

        # Prepare documents
        doc_dicts = []
        for doc in documents:
            doc_dict = doc.dict()

            if doc_dict.get("id") is None:
                doc_dict["id"] = str(uuid.uuid4())

            doc_dicts.append(doc_dict)

        # Index documents
        doc_ids = rag.index_documents(doc_dicts)

        return doc_ids

    except Exception as e:
        logger.error(f"Error adding documents batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...), document_type: Optional[str] = Form(None)
):
    """Upload document files to the system."""
    try:
        rag = get_rag_system()

        # Process uploaded files
        doc_ids = []

        for file in files:
            # Read file content
            content = await file.read()

            # Determine document type
            if document_type is None:
                if file.filename.endswith(".pdf"):
                    doc_type = "pdf"
                elif file.filename.endswith((".docx", ".doc")):
                    doc_type = "docx"
                elif file.filename.endswith((".txt", ".md")):
                    doc_type = "text"
                elif file.filename.endswith((".html", ".htm")):
                    doc_type = "html"
                else:
                    doc_type = "unknown"
            else:
                doc_type = document_type

            # Create document
            doc = {
                "id": str(uuid.uuid4()),
                "content": content,
                "metadata": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "type": doc_type,
                },
            }

            # Index document
            rag.document_processor.process_document(doc)
            doc_ids.append(doc["id"])

        return {"document_ids": doc_ids}

    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system."""
    try:
        rag = get_rag_system()

        # Delete document
        rag.retrieval_engine.delete_documents([document_id])

        return {"status": "success", "message": f"Document {document_id} deleted"}

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Query the RAG system",
    description="Submit a query to the RAG system to retrieve relevant information and generate an answer based on the retrieved documents",
    response_description="The generated answer along with source documents and metadata",
    responses={
        200: {
            "description": "Successful query response",
            "content": {
                "application/json": {
                    "example": {
                        "query": "What is hybrid retrieval in RAG systems?",
                        "answer": "Hybrid retrieval in RAG (Retrieval Augmented Generation) systems combines dense and sparse retrieval methods to get the best of both worlds. Dense retrieval uses embeddings to find semantically similar documents, capturing meaning even when different terms are used. Sparse retrieval like BM25 focuses on keyword matching, which works well for precise terminology. By combining these approaches, hybrid retrieval improves the overall quality of retrieved documents, especially for complex queries that contain both semantic meaning and specific terminology.",
                        "sources": [
                            {
                                "id": "doc2",
                                "text": "Dense retrieval uses embeddings to find semantically similar documents, while sparse retrieval like BM25 focuses on keyword matching. Hybrid retrieval combines both approaches.",
                                "score": 0.92,
                                "metadata": {
                                    "source": "documentation",
                                    "title": "Retrieval Methods",
                                },
                            },
                            {
                                "id": "doc7",
                                "text": "Hybrid retrieval approaches often outperform single-method retrievers, especially on complex queries.",
                                "score": 0.85,
                                "metadata": {
                                    "source": "research",
                                    "title": "RAG Performance Analysis",
                                },
                            },
                        ],
                        "metadata": {
                            "retrieval_time_ms": 42,
                            "generation_time_ms": 856,
                            "total_time_ms": 912,
                            "model": "gpt-3.5-turbo",
                            "documents_retrieved": 5,
                            "token_count": {
                                "prompt": 620,
                                "completion": 156,
                                "total": 776,
                            },
                        },
                    }
                }
            },
        },
        500: {
            "description": "Error processing query",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error querying system: Failed to generate response from LLM"
                    }
                }
            },
        },
    },
)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        rag = get_rag_system()

        # Execute query
        response = rag.query(
            query=request.query,
            filters=request.filters,
            retrieval_options=request.retrieval_options,
            generation_options=request.generation_options,
        )

        return response

    except Exception as e:
        logger.error(f"Error querying system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/evaluate",
    tags=["evaluation"],
    summary="Evaluate the RAG system performance",
    description="Start an asynchronous evaluation of the RAG system using a test dataset and specified metrics",
    response_description="Task ID for tracking the evaluation progress",
    responses={
        200: {
            "description": "Evaluation task started successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Evaluation started",
                        "task_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                    }
                }
            },
        },
        500: {
            "description": "Error starting evaluation",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error starting evaluation: Invalid dataset format"
                    }
                }
            },
        },
    },
)
async def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Evaluate the RAG system with a test dataset."""
    try:
        # Create task ID
        task_id = str(uuid.uuid4())

        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation,
            task_id=task_id,
            test_dataset=request.test_dataset,
            metrics=request.metrics,
        )

        return {
            "status": "success",
            "message": "Evaluation started",
            "task_id": task_id,
        }

    except Exception as e:
        logger.error(f"Error starting evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate/{task_id}")
async def get_evaluation_status(task_id: str):
    """Get the status of an evaluation task."""
    if task_id not in running_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = running_tasks[task_id]

    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress"),
        "result": task.get("result"),
    }


async def run_evaluation(
    task_id: str,
    test_dataset: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
):
    """
    Run evaluation in background.

    Args:
        task_id: Task ID
        test_dataset: Test dataset
        metrics: List of metrics to evaluate
    """
    try:
        # Initialize task
        running_tasks[task_id] = {"status": "running", "progress": 0}

        # Get RAG system
        rag = get_rag_system()

        # Run evaluation
        from evaluation.evaluator import EvaluationSuite

        evaluator = EvaluationSuite()
        result = evaluator.evaluate(rag, test_dataset, metrics)

        # Update task
        running_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "result": result,
        }

    except Exception as e:
        logger.error(f"Error in evaluation task: {str(e)}")

        # Update task with error
        running_tasks[task_id] = {"status": "failed", "error": str(e)}


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.

    Args:
        host: Host to listen on
        port: Port to listen on
        reload: Whether to enable auto-reload
    """
    uvicorn.run("deployment.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start_server()
