"""
Context augmentation module for optimizing retrieved documents before generation.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ContextAugmenter:
    """
    Optimizes retrieved documents to create the best context for generation.
    Handles context distillation, re-ordering, citation tracking, and more.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context augmenter with configuration options.

        Args:
            config: Augmentation configuration
        """
        self.config = config or {
            "max_context_length": 4000,
            "max_documents": 10,
            "citation_format": "inline",  # Options: inline, footnote, endnote
            "strategy": "relevance_weighted",  # Options: simple, relevance_weighted, semantic_clustering
            "include_metadata": True,
            "enable_compression": False,
        }

        logger.info("Initialized context augmenter with config: %s", self.config)

        # Initialize compression model if needed
        if self.config.get("enable_compression", False):
            self.compression_model = self._initialize_compression_model()
        else:
            self.compression_model = None

    def _initialize_compression_model(self) -> Any:
        """
        Initialize text compression model.

        Returns:
            Compression model instance
        """
        model_name = self.config.get("compression_model", "facebook/bart-large-cnn")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            logger.info("Initialized compression model: %s", model_name)

            return {
                "model": model,
                "tokenizer": tokenizer,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }

        except ImportError:
            logger.warning(
                "Failed to initialize compression model. Please install transformers library."
            )
            return None
        except Exception as e:
            logger.error("Error initializing compression model: %s", str(e))
            return None

    def augment(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Augment retrieved documents to create optimal context for generation.

        Args:
            query: Query text
            documents: List of retrieved documents with text, metadata, and scores
            options: Optional parameters to customize augmentation

        Returns:
            Dictionary with augmented context and related metadata
        """
        start_time = time.time()

        # Merge configuration with options
        config = self.config.copy()
        if options:
            config.update(options)

        # Limit to maximum number of documents
        max_documents = config.get("max_documents", 10)
        documents = documents[:max_documents]

        # Get strategy to use
        strategy = config.get("strategy", "relevance_weighted")

        if strategy == "simple":
            context_text, citations = self._simple_context(documents, config)
        elif strategy == "relevance_weighted":
            context_text, citations = self._relevance_weighted_context(
                documents, config
            )
        elif strategy == "semantic_clustering":
            context_text, citations = self._semantic_clustering_context(
                documents, query, config
            )
        else:
            logger.warning(
                "Unknown context strategy: %s, using simple context", strategy
            )
            context_text, citations = self._simple_context(documents, config)

        # Apply compression if enabled and available
        if config.get("enable_compression", False) and self.compression_model:
            context_text = self._compress_text(context_text, query)

        # Format citations
        if config.get("citation_format") == "inline":
            final_context = self._format_inline_citations(context_text, citations)
        else:
            final_context = context_text

        elapsed_time = time.time() - start_time

        return {
            "context": final_context,
            "citations": citations,
            "document_count": len(documents),
            "strategy": strategy,
            "latency_ms": int(elapsed_time * 1000),
        }

    def _simple_context(
        self, documents: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create a simple context by concatenating document texts.

        Args:
            documents: List of retrieved documents
            config: Augmentation configuration

        Returns:
            Tuple of (context text, citation list)
        """
        max_length = config.get("max_context_length", 4000)
        include_metadata = config.get("include_metadata", True)

        context_parts = []
        citations = []
        current_length = 0

        for i, doc in enumerate(documents):
            text = doc.get("text", "")
            text_length = len(text)

            # Check if adding this document would exceed max length
            if current_length + text_length > max_length:
                # If it's the first document, truncate it
                if i == 0:
                    truncated_text = text[:max_length]
                    context_parts.append(truncated_text)
                    current_length = max_length
                # Otherwise, skip this and remaining documents
                break

            # Add document text
            context_parts.append(text)
            current_length += text_length

            # Add citation
            citation = {
                "id": doc.get("id", f"doc_{i}"),
                "start_idx": current_length - text_length,
                "end_idx": current_length,
                "text": text,
            }

            # Add metadata if configured
            if include_metadata and "metadata" in doc:
                citation["metadata"] = doc["metadata"]

            citations.append(citation)

        # Join context parts with separators
        context_text = "\n\n".join(context_parts)

        return context_text, citations

    def _relevance_weighted_context(
        self, documents: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create a context by weighting document inclusion by relevance score.

        Args:
            documents: List of retrieved documents
            config: Augmentation configuration

        Returns:
            Tuple of (context text, citation list)
        """
        max_length = config.get("max_context_length", 4000)
        include_metadata = config.get("include_metadata", True)

        # Sort documents by score (highest first)
        # Use whatever score is available (reranker_score, score, etc.)
        def get_doc_score(doc):
            if "reranker_score" in doc:
                return doc["reranker_score"]
            elif "score" in doc:
                return doc["score"]
            else:
                return 0

        sorted_docs = sorted(documents, key=get_doc_score, reverse=True)

        # Calculate total score and normalized weights
        total_score = sum(get_doc_score(doc) for doc in sorted_docs)

        if total_score == 0:
            # If no scores, fall back to simple context
            return self._simple_context(documents, config)

        # Assign space budget to each document based on score
        doc_budgets = []
        remaining_budget = max_length

        for doc in sorted_docs:
            score = get_doc_score(doc)
            text = doc.get("text", "")
            text_length = len(text)

            if total_score > 0:
                # Calculate ideal budget based on score
                ideal_budget = int((score / total_score) * max_length)

                # Cap at actual text length and remaining budget
                budget = min(ideal_budget, text_length, remaining_budget)
                remaining_budget -= budget
            else:
                # Equal weighting if no scores
                equal_share = max_length // len(sorted_docs)
                budget = min(equal_share, text_length, remaining_budget)
                remaining_budget -= budget

            doc_budgets.append((doc, budget))

        # Redistribute remaining budget
        if remaining_budget > 0 and doc_budgets:
            # Give remaining budget to highest scored documents
            for i in range(len(doc_budgets)):
                doc, budget = doc_budgets[i]
                text_length = len(doc.get("text", ""))

                # If document can use more space
                if budget < text_length:
                    additional = min(remaining_budget, text_length - budget)
                    doc_budgets[i] = (doc, budget + additional)
                    remaining_budget -= additional

                if remaining_budget <= 0:
                    break

        # Build context from budgeted documents
        context_parts = []
        citations = []
        current_length = 0

        for i, (doc, budget) in enumerate(doc_budgets):
            if budget <= 0:
                continue

            text = doc.get("text", "")

            # Truncate text if needed
            if len(text) > budget:
                text = text[:budget]

            # Add document text
            context_parts.append(text)

            # Add citation
            citation = {
                "id": doc.get("id", f"doc_{i}"),
                "start_idx": current_length,
                "end_idx": current_length + len(text),
                "text": text,
            }

            # Add metadata if configured
            if include_metadata and "metadata" in doc:
                citation["metadata"] = doc["metadata"]

            citations.append(citation)
            current_length += len(text) + 2  # +2 for separator

        # Join context parts with separators
        context_text = "\n\n".join(context_parts)

        return context_text, citations

    def _semantic_clustering_context(
        self, documents: List[Dict[str, Any]], query: str, config: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create a context by clustering semantically similar passages.

        Args:
            documents: List of retrieved documents
            query: Query text
            config: Augmentation configuration

        Returns:
            Tuple of (context text, citation list)
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer

            max_length = config.get("max_context_length", 4000)
            include_metadata = config.get("include_metadata", True)
            num_clusters = min(config.get("num_clusters", 3), len(documents))

            # If only 1-2 documents, use relevance weighted approach
            if len(documents) <= 2:
                return self._relevance_weighted_context(documents, config)

            # Extract text from documents
            texts = [doc.get("text", "") for doc in documents]

            # Compute TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Cluster documents
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)

            # Group documents by cluster
            clustered_docs = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in clustered_docs:
                    clustered_docs[cluster_id] = []

                clustered_docs[cluster_id].append(documents[i])

            # Sort clusters by average score
            def get_cluster_score(cluster_id):
                docs = clustered_docs[cluster_id]
                scores = []

                for doc in docs:
                    if "reranker_score" in doc:
                        scores.append(doc["reranker_score"])
                    elif "score" in doc:
                        scores.append(doc["score"])

                return sum(scores) / len(scores) if scores else 0

            sorted_clusters = sorted(
                clustered_docs.keys(), key=get_cluster_score, reverse=True
            )

            # Allocate budget per cluster
            cluster_budgets = {}
            remaining_budget = max_length

            for cluster_id in sorted_clusters:
                docs = clustered_docs[cluster_id]
                score = get_cluster_score(cluster_id)

                # Assign proportional budget
                cluster_budget = int(
                    remaining_budget
                    * (score / sum(get_cluster_score(c) for c in sorted_clusters))
                )
                cluster_budgets[cluster_id] = min(cluster_budget, remaining_budget)
                remaining_budget -= cluster_budgets[cluster_id]

            # Build context from representative documents in each cluster
            context_parts = []
            citations = []
            current_length = 0

            for cluster_id in sorted_clusters:
                docs = clustered_docs[cluster_id]
                budget = cluster_budgets[cluster_id]

                # Sort docs in cluster by score
                sorted_docs = sorted(
                    docs,
                    key=lambda x: x.get("reranker_score", x.get("score", 0)),
                    reverse=True,
                )

                # Add documents until budget is exhausted
                cluster_parts = []
                cluster_length = 0

                for doc in sorted_docs:
                    text = doc.get("text", "")
                    text_length = len(text)

                    # If adding whole doc exceeds budget, truncate
                    if cluster_length + text_length > budget:
                        truncated_length = budget - cluster_length
                        if (
                            truncated_length > 50
                        ):  # Only add if we can include meaningful content
                            text = text[:truncated_length]
                            cluster_parts.append(text)

                            # Add citation
                            citation = {
                                "id": doc.get("id", f"doc_{len(citations)}"),
                                "start_idx": current_length + cluster_length,
                                "end_idx": current_length + cluster_length + len(text),
                                "text": text,
                            }

                            if include_metadata and "metadata" in doc:
                                citation["metadata"] = doc["metadata"]

                            citations.append(citation)

                            cluster_length += len(text)

                        break

                    # Otherwise add the whole document
                    cluster_parts.append(text)

                    # Add citation
                    citation = {
                        "id": doc.get("id", f"doc_{len(citations)}"),
                        "start_idx": current_length + cluster_length,
                        "end_idx": current_length + cluster_length + text_length,
                        "text": text,
                    }

                    if include_metadata and "metadata" in doc:
                        citation["metadata"] = doc["metadata"]

                    citations.append(citation)

                    cluster_length += text_length

                    if cluster_length >= budget:
                        break

                # Add cluster text to context
                if cluster_parts:
                    cluster_text = "\n".join(cluster_parts)
                    context_parts.append(cluster_text)
                    current_length += len(cluster_text) + 2  # +2 for separator

            # Join context parts with separators
            context_text = "\n\n".join(context_parts)

            return context_text, citations

        except ImportError:
            logger.warning(
                "scikit-learn not available, falling back to relevance weighted context"
            )
            return self._relevance_weighted_context(documents, config)
        except Exception as e:
            logger.error("Error in semantic clustering: %s", str(e))
            return self._relevance_weighted_context(documents, config)

    def _compress_text(self, text: str, query: str = "") -> str:
        """
        Compress text to reduce length while preserving meaning.

        Args:
            text: Text to compress
            query: Query for context-aware compression

        Returns:
            Compressed text
        """
        if not self.compression_model:
            return text

        try:
            model = self.compression_model["model"]
            tokenizer = self.compression_model["tokenizer"]
            device = self.compression_model["device"]

            import torch

            # Move model to appropriate device
            model = model.to(device)

            # Prepend query for context
            if query:
                input_text = f"Query: {query}\n\nText: {text}"
            else:
                input_text = text

            # Tokenize
            inputs = tokenizer(
                input_text, max_length=1024, truncation=True, return_tensors="pt"
            ).to(device)

            # Generate summary
            with torch.no_grad():
                output = model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    min_length=50,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True,
                )

            # Decode
            compressed = tokenizer.decode(output[0], skip_special_tokens=True)

            return compressed

        except Exception as e:
            logger.error("Error compressing text: %s", str(e))
            return text

    def _format_inline_citations(
        self, text: str, citations: List[Dict[str, Any]]
    ) -> str:
        """
        Format text with inline citations.

        Args:
            text: Context text
            citations: List of citation information

        Returns:
            Text with formatted citations
        """
        # If no citations, return original text
        if not citations:
            return text

        # Sort citations by start index (descending to avoid index shifting)
        sorted_citations = sorted(citations, key=lambda x: x["start_idx"], reverse=True)

        # Add citation markers
        result = text
        for i, citation in enumerate(sorted_citations):
            citation_id = citation.get("id", f"doc_{i}")

            # Get metadata for citation
            metadata = citation.get("metadata", {})
            citation_text = ""

            # Try to build an informative citation
            if "filename" in metadata:
                citation_text = f"[{metadata.get('filename')}]"
            elif "title" in metadata:
                citation_text = f"[{metadata.get('title')}]"
            elif "url" in metadata:
                citation_text = f"[Source {i+1}]"
            else:
                citation_text = f"[Source {i+1}]"

            # Insert citation marker at end of citation text
            start_idx = citation["start_idx"]
            end_idx = citation["end_idx"]

            # Ensure indices are within bounds
            if 0 <= start_idx < len(result) and 0 <= end_idx <= len(result):
                result = result[:end_idx] + f" {citation_text}" + result[end_idx:]

        return result
