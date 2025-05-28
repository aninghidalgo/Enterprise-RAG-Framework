"""
Streamlit demo app for the Enterprise-Ready RAG System.
Provides an interactive UI for document management, querying, and evaluation.
"""

import streamlit as st
import requests
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO

# Configuration
# Initial API URL configuration - using session state instead of global variables
default_api_url = os.environ.get("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d7f9db;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .citation {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    .document-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
</style>
""",
    unsafe_allow_html=True,
)


# App functions
def check_api_health():
    """Check if the API is reachable and healthy."""
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def upload_document(file, document_type=None):
    """Upload a document to the API."""
    try:
        files = {"files": (file.name, file.getvalue(), file.type)}
        data = {}
        if document_type:
            data["document_type"] = document_type

        response = requests.post(
            f"{st.session_state.api_url}/documents/upload", files=files, data=data
        )

        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def add_text_document(text, metadata=None):
    """Add a text document to the API."""
    try:
        document = {"text": text, "metadata": metadata or {}}

        response = requests.post(f"{st.session_state.api_url}/documents", json=document)

        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def query_rag_system(
    query, filters=None, retrieval_options=None, generation_options=None
):
    """Query the RAG system."""
    try:
        request_data = {
            "query": query,
            "filters": filters,
            "retrieval_options": retrieval_options,
            "generation_options": generation_options,
        }

        response = requests.post(f"{st.session_state.api_url}/query", json=request_data)

        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def start_evaluation(test_dataset, metrics=None):
    """Start an evaluation task."""
    try:
        request_data = {"test_dataset": test_dataset, "metrics": metrics}

        response = requests.post(
            f"{st.session_state.api_url}/evaluate", json=request_data
        )

        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def get_evaluation_status(task_id):
    """Get the status of an evaluation task."""
    try:
        response = requests.get(f"{st.session_state.api_url}/evaluate/{task_id}")

        if response.status_code == 200:
            return True, response.json()
        return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


# Sidebar
st.sidebar.markdown(
    "<div class='main-header'>Enterprise RAG</div>", unsafe_allow_html=True
)

# Check API health
api_status, api_details = check_api_health()
if api_status:
    st.sidebar.markdown(
        "<div class='success-box'>✅ API Connected</div>", unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        "<div class='error-box'>❌ API Unreachable</div>", unsafe_allow_html=True
    )
    st.sidebar.error(f"Error: {api_details.get('error', 'Unknown error')}")

# Navigation
pages = {
    "🏠 Home": "home",
    "📄 Document Management": "documents",
    "🔍 RAG Query": "query",
    "📊 Evaluation": "evaluation",
    "⚙️ Settings": "settings",
}

selected_page = st.sidebar.radio("Navigation", list(pages.keys()))
page = pages[selected_page]

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "evaluation_tasks" not in st.session_state:
    st.session_state.evaluation_tasks = {}
if "settings" not in st.session_state:
    st.session_state.settings = {
        "retrieval_type": "hybrid",
        "top_k": 5,
        "use_reranker": True,
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "citation_format": "inline",
    }
# Store API URL in session state
if "api_url" not in st.session_state:
    st.session_state.api_url = default_api_url

# Home page
if page == "home":
    st.markdown(
        "<div class='main-header'>Enterprise-Ready RAG System</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class='info-box'>
        Welcome to the Enterprise-Ready RAG System demo! This application showcases a production-grade
        Retrieval Augmented Generation (RAG) system with comprehensive evaluation capabilities.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Features overview
    st.markdown("<div class='sub-header'>Key Features</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        **Document Processing**
        - Multi-format support (PDF, DOCX, HTML, etc.)
        - Intelligent chunking strategies
        - Metadata extraction
        """
        )

    with col2:
        st.markdown(
            """
        **Advanced Retrieval**
        - Hybrid search (dense + sparse)
        - Cross-encoder reranking
        - Metadata filtering
        """
        )

    with col3:
        st.markdown(
            """
        **Comprehensive Evaluation**
        - Retrieval precision metrics
        - Answer relevance scoring
        - Hallucination detection
        """
        )

    # System status
    if api_status:
        st.markdown(
            "<div class='sub-header'>System Status</div>", unsafe_allow_html=True
        )

        status = api_details.get("rag_status", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Documents", status.get("document_count", 0))

        with col2:
            st.metric("Vector Store", status.get("vector_store_type", "Unknown"))

        with col3:
            st.metric("Retrieval Type", status.get("retrieval_type", "Unknown"))

        with col4:
            st.metric("Model", status.get("generation_model", "Unknown"))

    # Quick links
    st.markdown("<div class='sub-header'>Quick Actions</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Manage Documents"):
            st.session_state.page = "documents"
            st.experimental_rerun()

    with col2:
        if st.button("🔍 Ask a Question"):
            st.session_state.page = "query"
            st.experimental_rerun()

    with col3:
        if st.button("📊 Run Evaluation"):
            st.session_state.page = "evaluation"
            st.experimental_rerun()

# Document Management page
elif page == "documents":
    st.markdown(
        "<div class='main-header'>Document Management</div>", unsafe_allow_html=True
    )

    # Tabs for different document operations
    doc_tab1, doc_tab2 = st.tabs(["📤 Upload Documents", "✏️ Add Text Document"])

    with doc_tab1:
        st.markdown(
            "<div class='sub-header'>Upload Documents</div>", unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a file", type=["pdf", "docx", "txt", "html", "md"]
        )

        doc_type = st.selectbox(
            "Document Type", ["Auto-detect", "PDF", "DOCX", "Text", "HTML"], index=0
        )

        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner("Uploading document..."):
                    type_map = {
                        "Auto-detect": None,
                        "PDF": "pdf",
                        "DOCX": "docx",
                        "Text": "text",
                        "HTML": "html",
                    }

                    success, result = upload_document(
                        uploaded_file, document_type=type_map[doc_type]
                    )

                    if success:
                        st.success(
                            f"Document uploaded successfully! Document IDs: {result.get('document_ids')}"
                        )

                        # Add to session state
                        for doc_id in result.get("document_ids", []):
                            st.session_state.documents.append(
                                {
                                    "id": doc_id,
                                    "name": uploaded_file.name,
                                    "type": (
                                        doc_type
                                        if doc_type != "Auto-detect"
                                        else "Unknown"
                                    ),
                                }
                            )
                    else:
                        st.error(
                            f"Error uploading document: {result.get('detail', 'Unknown error')}"
                        )

    with doc_tab2:
        st.markdown(
            "<div class='sub-header'>Add Text Document</div>", unsafe_allow_html=True
        )

        text_content = st.text_area("Document Content", height=200)

        col1, col2 = st.columns(2)

        with col1:
            doc_title = st.text_input("Document Title")

        with col2:
            doc_source = st.text_input("Source (optional)")

        if st.button("Add Document") and text_content:
            with st.spinner("Adding document..."):
                metadata = {
                    "title": doc_title or "Untitled Document",
                    "source": doc_source or "Manual Entry",
                }

                success, result = add_text_document(text_content, metadata)

                if success:
                    st.success(
                        f"Document added successfully! Document ID: {result.get('id')}"
                    )

                    # Add to session state
                    st.session_state.documents.append(
                        {
                            "id": result.get("id"),
                            "name": metadata["title"],
                            "type": "Text",
                        }
                    )
                else:
                    st.error(
                        f"Error adding document: {result.get('detail', 'Unknown error')}"
                    )

    # Document list
    st.markdown("<div class='sub-header'>Document List</div>", unsafe_allow_html=True)

    if not st.session_state.documents:
        st.info("No documents added yet. Upload or add documents using the tabs above.")
    else:
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(
                f"""<div class='document-card'>
                    <strong>{doc['name']}</strong> ({doc['type']})
                    <br>ID: <code>{doc['id']}</code>
                </div>""",
                unsafe_allow_html=True,
            )

# Query page
elif page == "query":
    st.markdown("<div class='main-header'>RAG Query</div>", unsafe_allow_html=True)

    # Query input
    query = st.text_area("Enter your question", height=100)

    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            retrieval_type = st.selectbox(
                "Retrieval Type", ["hybrid", "dense", "sparse"], index=0
            )

            top_k = st.slider("Number of documents to retrieve", 1, 20, 5)

            use_reranker = st.checkbox("Use reranker", value=True)

        with col2:
            model = st.selectbox(
                "Model", ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"], index=0
            )

            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

            citation_format = st.selectbox(
                "Citation Format", ["inline", "footnote"], index=0
            )

    # Execute query
    if st.button("Submit Query") and query:
        with st.spinner("Processing query..."):
            # Prepare options
            retrieval_options = {
                "retrieval_type": retrieval_type,
                "top_k": top_k,
                "rerank": use_reranker,
            }

            generation_options = {
                "model": model,
                "temperature": temperature,
                "prompt_template": "default",
            }

            augmentation_options = {"citation_format": citation_format}

            # Call API
            success, result = query_rag_system(
                query,
                retrieval_options=retrieval_options,
                generation_options=generation_options,
            )

            if success:
                # Save to history
                st.session_state.query_history.append(
                    {"query": query, "result": result, "timestamp": time.time()}
                )

                # Display result
                st.markdown(
                    "<div class='sub-header'>Answer</div>", unsafe_allow_html=True
                )
                st.markdown(result["answer"])

                # Display sources
                if result.get("sources"):
                    st.markdown(
                        "<div class='sub-header'>Sources</div>", unsafe_allow_html=True
                    )

                    for i, source in enumerate(result["sources"]):
                        source_id = source.get("id", f"source_{i}")
                        confidence = source.get("confidence", 0.0)

                        st.markdown(
                            f"""<div class='citation'>
                                <strong>Source {i+1}</strong> (ID: <code>{source_id}</code>, Confidence: {confidence:.2f})
                                <br>{source.get('title', 'Untitled')}
                            </div>""",
                            unsafe_allow_html=True,
                        )

                # Display metadata
                with st.expander("Response Metadata"):
                    st.json(result["metadata"])
            else:
                st.error(
                    f"Error processing query: {result.get('detail', 'Unknown error')}"
                )

    # Query history
    if st.session_state.query_history:
        st.markdown(
            "<div class='sub-header'>Query History</div>", unsafe_allow_html=True
        )

        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"Query: {item['query'][:50]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Answer:** {item['result']['answer']}")

                if item["result"].get("sources"):
                    st.markdown("**Sources:**")
                    for j, source in enumerate(item["result"]["sources"]):
                        st.markdown(f"- Source {j+1}: {source.get('id', 'Unknown')}")

# Evaluation page
elif page == "evaluation":
    st.markdown("<div class='main-header'>Evaluation</div>", unsafe_allow_html=True)

    # Tabs for different evaluation operations
    eval_tab1, eval_tab2 = st.tabs(["📊 Run Evaluation", "📈 View Results"])

    with eval_tab1:
        st.markdown(
            "<div class='sub-header'>Run Evaluation</div>", unsafe_allow_html=True
        )

        # Sample test dataset
        st.markdown("### Test Dataset")

        uploaded_file = st.file_uploader(
            "Upload test dataset (JSON/JSONL)", type=["json", "jsonl"]
        )

        if uploaded_file is None:
            st.markdown(
                """
            <div class='info-box'>
                No file uploaded. You can use our sample test dataset or upload your own.
                <br>The dataset should be a JSON array of objects with at least a "query" field.
            </div>
            """,
                unsafe_allow_html=True,
            )

            use_sample = st.checkbox("Use sample test dataset", value=True)

            if use_sample:
                # Sample dataset
                test_dataset = [
                    {
                        "query": "What is retrieval augmented generation?",
                        "answer": "Retrieval Augmented Generation (RAG) is an AI framework that enhances large language models by retrieving relevant information from external knowledge sources before generating responses. This approach improves accuracy, reduces hallucinations, and enables up-to-date information access.",
                    },
                    {
                        "query": "How does hybrid search work?",
                        "answer": "Hybrid search combines multiple retrieval methods, typically dense and sparse approaches. Dense retrieval uses embeddings to capture semantic meaning, while sparse retrieval (like BM25) focuses on keyword matching. Combining these methods provides better results by leveraging the strengths of each approach.",
                    },
                    {
                        "query": "What are the benefits of using a reranker?",
                        "answer": "Rerankers improve retrieval quality by taking an initial set of retrieved documents and reordering them based on relevance to the query. Benefits include better precision, reduced noise, improved answer quality, and compensation for weaknesses in the initial retrieval stage.",
                    },
                ]
        else:
            # Load dataset from file
            try:
                content = uploaded_file.getvalue().decode("utf-8")

                if uploaded_file.name.endswith(".jsonl"):
                    # JSONL format
                    test_dataset = [
                        json.loads(line)
                        for line in content.splitlines()
                        if line.strip()
                    ]
                else:
                    # JSON format
                    test_dataset = json.loads(content)

                st.success(f"Loaded test dataset with {len(test_dataset)} examples")
            except Exception as e:
                st.error(f"Error loading test dataset: {str(e)}")
                test_dataset = []

        # Metrics selection
        st.markdown("### Evaluation Metrics")

        metrics = st.multiselect(
            "Select metrics to evaluate",
            [
                "retrieval_precision",
                "answer_relevance",
                "factual_correctness",
                "hallucination_detection",
                "citation_accuracy",
                "latency",
            ],
            default=[
                "retrieval_precision",
                "answer_relevance",
                "factual_correctness",
                "latency",
            ],
        )

        # Start evaluation
        if st.button("Start Evaluation") and test_dataset and metrics:
            with st.spinner("Starting evaluation..."):
                success, result = start_evaluation(test_dataset, metrics)

                if success:
                    task_id = result.get("task_id")
                    st.success(f"Evaluation started! Task ID: {task_id}")

                    # Add to session state
                    st.session_state.evaluation_tasks[task_id] = {
                        "status": "running",
                        "metrics": metrics,
                        "dataset_size": len(test_dataset),
                        "timestamp": time.time(),
                    }
                else:
                    st.error(
                        f"Error starting evaluation: {result.get('detail', 'Unknown error')}"
                    )

    with eval_tab2:
        st.markdown(
            "<div class='sub-header'>Evaluation Results</div>", unsafe_allow_html=True
        )

        if not st.session_state.evaluation_tasks:
            st.info(
                "No evaluation tasks found. Run an evaluation using the 'Run Evaluation' tab."
            )
        else:
            # Task selection
            task_options = {
                f"{task_id} ({data['status']})": task_id
                for task_id, data in st.session_state.evaluation_tasks.items()
            }
            selected_task = st.selectbox("Select Task", list(task_options.keys()))
            task_id = task_options[selected_task]

            # Refresh button
            if st.button("Refresh Status"):
                with st.spinner("Refreshing..."):
                    success, result = get_evaluation_status(task_id)

                    if success:
                        task_status = result.get("status")
                        st.session_state.evaluation_tasks[task_id][
                            "status"
                        ] = task_status

                        if task_status == "completed":
                            st.session_state.evaluation_tasks[task_id]["result"] = (
                                result.get("result")
                            )
                    else:
                        st.error(
                            f"Error refreshing status: {result.get('detail', 'Unknown error')}"
                        )

            # Display status
            task_data = st.session_state.evaluation_tasks[task_id]

            if task_data["status"] == "running":
                st.info("Evaluation in progress...")
                st.progress(result.get("progress", 0) / 100)
            elif task_data["status"] == "completed" and "result" in task_data:
                st.success("Evaluation completed!")

                # Display results
                result = task_data["result"]

                # Overall metrics
                st.markdown("### Overall Metrics")

                metrics_data = {}
                for key, value in result.get("overall", {}).items():
                    if key.startswith("avg_") and key.endswith("_score"):
                        metric_name = key[4:-6]  # Remove avg_ and _score
                        metrics_data[metric_name] = value

                if metrics_data:
                    # Create bar chart
                    fig = px.bar(
                        x=list(metrics_data.keys()),
                        y=list(metrics_data.values()),
                        labels={"x": "Metric", "y": "Score"},
                        title="Evaluation Metrics",
                        color=list(metrics_data.values()),
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1],
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Latency metrics
                if "avg_total_latency_ms" in result.get("overall", {}):
                    st.markdown("### Latency Metrics")

                    latency_data = {
                        "Retrieval": result["overall"].get(
                            "avg_retrieval_latency_ms", 0
                        ),
                        "Augmentation": result["overall"].get(
                            "avg_augmentation_latency_ms", 0
                        ),
                        "Generation": result["overall"].get(
                            "avg_generation_latency_ms", 0
                        ),
                        "Total": result["overall"].get("avg_total_latency_ms", 0),
                    }

                    # Create bar chart
                    fig = px.bar(
                        x=list(latency_data.keys()),
                        y=list(latency_data.values()),
                        labels={"x": "Component", "y": "Latency (ms)"},
                        title="Average Latency by Component",
                        color=list(latency_data.keys()),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Example results
                st.markdown("### Example Results")

                examples = result.get("examples", [])

                if examples:
                    for i, example in enumerate(examples[:5]):  # Show first 5 examples
                        with st.expander(f"Example {i+1}: {example['query'][:50]}..."):
                            st.markdown(f"**Query:** {example['query']}")

                            if (
                                "reference" in example
                                and "answer" in example["reference"]
                            ):
                                st.markdown(
                                    f"**Reference Answer:** {example['reference']['answer']}"
                                )

                            if (
                                "response" in example
                                and "answer" in example["response"]
                            ):
                                st.markdown(
                                    f"**Generated Answer:** {example['response']['answer']}"
                                )

                            # Display metrics
                            if "metrics" in example:
                                st.markdown("**Metrics:**")

                                for metric, data in example["metrics"].items():
                                    if isinstance(data, dict) and "score" in data:
                                        st.markdown(f"- {metric}: {data['score']:.2f}")
            else:
                st.warning(f"Task status: {task_data['status']}")

# Settings page
elif page == "settings":
    st.markdown("<div class='main-header'>Settings</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='sub-header'>API Configuration</div>", unsafe_allow_html=True
    )

    api_url = st.text_input("API URL", value=st.session_state.api_url)

    if st.button("Update API URL"):
        # Update the API URL in session state
        st.session_state.api_url = api_url
        st.success(f"API URL updated to {st.session_state.api_url}")

    st.markdown(
        "<div class='sub-header'>Default Query Settings</div>", unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.session_state.settings["retrieval_type"] = st.selectbox(
            "Default Retrieval Type",
            ["hybrid", "dense", "sparse"],
            index=["hybrid", "dense", "sparse"].index(
                st.session_state.settings["retrieval_type"]
            ),
        )

        st.session_state.settings["top_k"] = st.slider(
            "Default Number of Documents", 1, 20, st.session_state.settings["top_k"]
        )

        st.session_state.settings["use_reranker"] = st.checkbox(
            "Use Reranker by Default", value=st.session_state.settings["use_reranker"]
        )

    with col2:
        st.session_state.settings["model"] = st.selectbox(
            "Default Model",
            ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"],
            index=["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"].index(
                st.session_state.settings["model"]
            ),
        )

        st.session_state.settings["temperature"] = st.slider(
            "Default Temperature",
            0.0,
            1.0,
            st.session_state.settings["temperature"],
            0.1,
        )

        st.session_state.settings["citation_format"] = st.selectbox(
            "Default Citation Format",
            ["inline", "footnote"],
            index=["inline", "footnote"].index(
                st.session_state.settings["citation_format"]
            ),
        )

    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>Enterprise-Ready RAG System by Taimoor Khan | GitHub: @TaimoorKhan10</div>",
    unsafe_allow_html=True,
)
