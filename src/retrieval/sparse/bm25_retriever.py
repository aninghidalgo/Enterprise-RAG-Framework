"""
BM25 sparse retriever implementation for keyword-based search.
"""

import logging
import re
import string
import pickle
import os
from typing import Dict, List, Optional, Any, Set, Tuple
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Sparse retriever implementation using the BM25 algorithm.
    Provides keyword-based search that complements dense embedding retrieval.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BM25 retriever with configuration options.

        Args:
            config: Retrieval configuration
        """
        self.config = config
        self.k1 = config.get("bm25_k1", 1.5)  # Term frequency normalization
        self.b = config.get("bm25_b", 0.75)  # Document length normalization
        self.epsilon = config.get("bm25_epsilon", 0.25)  # Term frequency floor

        # Storage for corpus
        self.documents = {}
        self.doc_freqs = {}  # Document frequencies of terms
        self.doc_lengths = {}  # Document lengths
        self.average_doc_length = 0
        self.total_docs = 0
        self.all_tokens = set()  # All tokens in the corpus

        # Tokenization settings
        self.min_token_length = config.get("min_token_length", 2)
        self.tokenization_pattern = r"\b\w+\b"

        # Stopwords
        self.stopwords = self._load_stopwords(config.get("stopwords_path"))

        # Load existing index if path provided
        index_path = config.get("bm25_index_path")
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)

        logger.info("Initialized BM25 retriever")

    def _load_stopwords(self, stopwords_path: Optional[str] = None) -> Set[str]:
        """
        Load stopwords from a file or use default list.

        Args:
            stopwords_path: Path to stopwords file

        Returns:
            Set of stopwords
        """
        default_stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "if",
            "in",
            "into",
            "is",
            "it",
            "no",
            "not",
            "of",
            "on",
            "or",
            "such",
            "that",
            "the",
            "their",
            "then",
            "there",
            "these",
            "they",
            "this",
            "to",
            "was",
            "will",
            "with",
        }

        if not stopwords_path:
            return default_stopwords

        try:
            with open(stopwords_path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except Exception as e:
            logger.warning(
                f"Error loading stopwords file: {str(e)}. Using default stopwords."
            )
            return default_stopwords

    def _load_index(self, index_path: str) -> None:
        """
        Load BM25 index from disk.

        Args:
            index_path: Path to index file
        """
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.doc_freqs = data["doc_freqs"]
                self.doc_lengths = data["doc_lengths"]
                self.average_doc_length = data["average_doc_length"]
                self.total_docs = data["total_docs"]
                self.all_tokens = data["all_tokens"]

            logger.info(f"Loaded BM25 index with {self.total_docs} documents")
        except Exception as e:
            logger.error(f"Error loading BM25 index: {str(e)}")
            # Reset to empty state
            self.documents = {}
            self.doc_freqs = {}
            self.doc_lengths = {}
            self.average_doc_length = 0
            self.total_docs = 0
            self.all_tokens = set()

    def save_index(self, index_path: Optional[str] = None) -> None:
        """
        Save BM25 index to disk.

        Args:
            index_path: Path to save index file
        """
        if index_path is None:
            index_path = self.config.get("bm25_index_path")

        if not index_path:
            logger.warning("No index path specified, cannot save BM25 index")
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path), exist_ok=True)

            # Save index
            with open(index_path, "wb") as f:
                data = {
                    "documents": self.documents,
                    "doc_freqs": self.doc_freqs,
                    "doc_lengths": self.doc_lengths,
                    "average_doc_length": self.average_doc_length,
                    "total_docs": self.total_docs,
                    "all_tokens": self.all_tokens,
                }
                pickle.dump(data, f)

            logger.info(
                f"Saved BM25 index with {self.total_docs} documents to {index_path}"
            )
        except Exception as e:
            logger.error(f"Error saving BM25 index: {str(e)}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the BM25 index.

        Args:
            documents: List of document dictionaries with text and metadata

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        doc_ids = []

        for doc in documents:
            # Extract text and ID
            text = doc.get("text", "")

            if "id" not in doc:
                doc_id = str(uuid.uuid4())
                doc["id"] = doc_id
            else:
                doc_id = doc["id"]

            doc_ids.append(doc_id)

            # Tokenize text
            tokens = self._tokenize(text)

            # Update document frequencies
            token_freqs = {}
            for token in tokens:
                if token in token_freqs:
                    token_freqs[token] += 1
                else:
                    token_freqs[token] = 1

                # Add to all tokens set
                self.all_tokens.add(token)

            # Store document
            self.documents[doc_id] = {
                "text": text,
                "metadata": doc.get("metadata", {}),
                "token_freqs": token_freqs,
            }

            # Update document length
            self.doc_lengths[doc_id] = len(tokens)

            # Update document frequencies
            for token in token_freqs:
                if token in self.doc_freqs:
                    self.doc_freqs[token] += 1
                else:
                    self.doc_freqs[token] = 1

        # Update total documents and average length
        self.total_docs = len(self.documents)

        if self.total_docs > 0:
            total_length = sum(self.doc_lengths.values())
            self.average_doc_length = total_length / self.total_docs

        # Save index if auto_save is enabled
        if self.config.get("auto_save", False) and self.config.get("bm25_index_path"):
            self.save_index()

        logger.info(f"Added {len(documents)} documents to BM25 index")
        return doc_ids

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Find all tokens using regex
        tokens = re.findall(self.tokenization_pattern, text)

        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_token_length:
                continue

            # Skip stopwords
            if token in self.stopwords:
                continue

            filtered_tokens.append(token)

        return filtered_tokens

    def search(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching the query using BM25.

        Args:
            query: Query text
            top_k: Number of documents to return
            filters: Optional metadata filters

        Returns:
            List of document dictionaries with text, metadata, and relevance scores
        """
        if not self.total_docs:
            logger.warning("Empty BM25 index, no documents to search")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.warning("No valid tokens in query after tokenization")
            return []

        # Calculate BM25 scores for each document
        scores = {}

        for doc_id, doc_info in self.documents.items():
            score = self._calculate_bm25_score(query_tokens, doc_id, doc_info)
            scores[doc_id] = score

        # Sort documents by score
        sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Apply filters if specified
        if filters:
            sorted_doc_ids = self._apply_filters(sorted_doc_ids, filters)

        # Get top-k documents
        results = []
        for doc_id in sorted_doc_ids[:top_k]:
            doc_info = self.documents[doc_id]
            result = {
                "id": doc_id,
                "text": doc_info["text"],
                "metadata": doc_info["metadata"],
                "score": scores[doc_id],
            }
            results.append(result)

        return results

    def _calculate_bm25_score(
        self, query_tokens: List[str], doc_id: str, doc_info: Dict[str, Any]
    ) -> float:
        """
        Calculate BM25 score for a document given query tokens.

        Args:
            query_tokens: List of query tokens
            doc_id: Document ID
            doc_info: Document information

        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_id]
        token_freqs = doc_info["token_freqs"]

        # Length normalization factor
        length_norm = (1.0 - self.b) + self.b * (doc_length / self.average_doc_length)

        for token in query_tokens:
            # Skip tokens not in corpus
            if token not in self.doc_freqs:
                continue

            # Get document frequency
            df = self.doc_freqs[token]

            # Get term frequency in document
            tf = token_freqs.get(token, 0)

            # Calculate IDF
            idf = np.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

            # Calculate term score with smoothing
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm
            term_score = idf * (numerator / denominator)

            score += term_score

        return score

    def _apply_filters(self, doc_ids: List[str], filters: Dict[str, Any]) -> List[str]:
        """
        Apply metadata filters to a list of document IDs.

        Args:
            doc_ids: List of document IDs
            filters: Dictionary of metadata filters

        Returns:
            Filtered list of document IDs
        """
        if not filters:
            return doc_ids

        filtered_ids = []

        for doc_id in doc_ids:
            metadata = self.documents[doc_id]["metadata"]
            match = True

            for key, value in filters.items():
                # Handle nested keys with dot notation
                if "." in key:
                    parts = key.split(".")
                    current = metadata
                    for part in parts[:-1]:
                        if part not in current:
                            match = False
                            break
                        current = current[part]

                    if match and (
                        parts[-1] not in current or current[parts[-1]] != value
                    ):
                        match = False

                # Handle list membership
                elif isinstance(value, list):
                    if key not in metadata or metadata[key] not in value:
                        match = False

                # Handle range queries
                elif isinstance(value, dict) and (
                    "$gt" in value
                    or "$lt" in value
                    or "$gte" in value
                    or "$lte" in value
                ):
                    if key not in metadata:
                        match = False
                    else:
                        field_value = metadata[key]

                        if "$gt" in value and not field_value > value["$gt"]:
                            match = False
                        if "$lt" in value and not field_value < value["$lt"]:
                            match = False
                        if "$gte" in value and not field_value >= value["$gte"]:
                            match = False
                        if "$lte" in value and not field_value <= value["$lte"]:
                            match = False

                # Handle exact match
                elif key not in metadata or metadata[key] != value:
                    match = False

            if match:
                filtered_ids.append(doc_id)

        return filtered_ids

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the BM25 index.

        Args:
            document_ids: List of document IDs to delete
        """
        if not document_ids:
            return

        for doc_id in document_ids:
            if doc_id not in self.documents:
                continue

            # Get document info
            doc_info = self.documents[doc_id]
            token_freqs = doc_info["token_freqs"]

            # Update document frequencies
            for token, freq in token_freqs.items():
                if token in self.doc_freqs:
                    self.doc_freqs[token] -= 1

                    # Remove token from doc_freqs if frequency becomes 0
                    if self.doc_freqs[token] == 0:
                        del self.doc_freqs[token]

            # Remove document length
            if doc_id in self.doc_lengths:
                del self.doc_lengths[doc_id]

            # Remove document
            del self.documents[doc_id]

        # Update total documents and average length
        self.total_docs = len(self.documents)

        if self.total_docs > 0:
            total_length = sum(self.doc_lengths.values())
            self.average_doc_length = total_length / self.total_docs
        else:
            self.average_doc_length = 0

        # Recalculate all_tokens
        self.all_tokens = set()
        for doc_info in self.documents.values():
            for token in doc_info["token_freqs"]:
                self.all_tokens.add(token)

        # Save index if auto_save is enabled
        if self.config.get("auto_save", False) and self.config.get("bm25_index_path"):
            self.save_index()

        logger.info(f"Deleted {len(document_ids)} documents from BM25 index")

    def clear(self) -> None:
        """
        Clear all documents from the BM25 index.
        """
        self.documents = {}
        self.doc_freqs = {}
        self.doc_lengths = {}
        self.average_doc_length = 0
        self.total_docs = 0
        self.all_tokens = set()

        # Save empty index if auto_save is enabled
        if self.config.get("auto_save", False) and self.config.get("bm25_index_path"):
            self.save_index()

        logger.info("Cleared BM25 index")

    def get_document_count(self) -> int:
        """
        Get the number of documents in the BM25 index.

        Returns:
            Number of documents
        """
        return self.total_docs

    def supports_filters(self) -> bool:
        """
        Check if the retriever supports filters.

        Returns:
            True if filtering is supported
        """
        return True
