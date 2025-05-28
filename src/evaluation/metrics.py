"""
Evaluation metrics for RAG systems.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Metrics for evaluating retrieval performance.
    """

    @staticmethod
    def precision_at_k(
        relevant_docs: List[str], retrieved_docs: List[str], k: int
    ) -> float:
        """
        Calculate precision@k for retrieval.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of documents to consider

        Returns:
            Precision@k score
        """
        if not retrieved_docs or not k:
            return 0.0

        retrieved_k = retrieved_docs[:k]

        relevant_retrieved = set(retrieved_k).intersection(set(relevant_docs))

        return len(relevant_retrieved) / min(k, len(retrieved_k))

    @staticmethod
    def recall_at_k(
        relevant_docs: List[str], retrieved_docs: List[str], k: int
    ) -> float:
        """
        Calculate recall@k for retrieval.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of documents to consider

        Returns:
            Recall@k score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0

        retrieved_k = retrieved_docs[:k]

        relevant_retrieved = set(retrieved_k).intersection(set(relevant_docs))

        return len(relevant_retrieved) / len(relevant_docs)

    @staticmethod
    def mean_reciprocal_rank(
        relevant_docs: List[str], retrieved_docs: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs

        Returns:
            MRR score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0

        # Find the rank of the first relevant document
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def ndcg_at_k(relevant_docs: List[str], retrieved_docs: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at k.

        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of documents to consider

        Returns:
            NDCG@k score
        """
        if not relevant_docs or not retrieved_docs:
            return 0.0

        retrieved_k = retrieved_docs[:k]

        # Calculate DCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_k):
            if doc_id in relevant_docs:
                # Relevance is binary in this case (1 if relevant, 0 if not)
                rel = 1
                dcg += rel / np.log2(i + 2)  # +2 because i is 0-indexed

        # Calculate ideal DCG
        idcg = 0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1 / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg


class AnswerMetrics:
    """
    Metrics for evaluating answer quality.
    """

    @staticmethod
    def exact_match(reference_answer: str, generated_answer: str) -> float:
        """
        Calculate exact match score.

        Args:
            reference_answer: Reference answer
            generated_answer: Generated answer

        Returns:
            Exact match score (0 or 1)
        """
        # Normalize both answers
        ref_norm = AnswerMetrics._normalize_text(reference_answer)
        gen_norm = AnswerMetrics._normalize_text(generated_answer)

        return 1.0 if ref_norm == gen_norm else 0.0

    @staticmethod
    def token_overlap(reference_answer: str, generated_answer: str) -> float:
        """
        Calculate token overlap (F1 score).

        Args:
            reference_answer: Reference answer
            generated_answer: Generated answer

        Returns:
            F1 score for token overlap
        """
        # Tokenize both answers
        ref_tokens = AnswerMetrics._tokenize(reference_answer)
        gen_tokens = AnswerMetrics._tokenize(generated_answer)

        if not ref_tokens or not gen_tokens:
            return 0.0

        # Calculate token overlap
        ref_counter = Counter(ref_tokens)
        gen_counter = Counter(gen_tokens)

        # Calculate intersection
        intersection = sum((ref_counter & gen_counter).values())

        # Calculate precision and recall
        precision = intersection / sum(gen_counter.values())
        recall = intersection / sum(ref_counter.values())

        # Calculate F1
        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)

        return f1

    @staticmethod
    def semantic_similarity(
        reference_answer: str,
        generated_answer: str,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings.

        Args:
            reference_answer: Reference answer
            generated_answer: Generated answer
            model_name: Name of the sentence transformer model

        Returns:
            Cosine similarity score
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer(model_name)

            # Generate embeddings
            ref_embedding = model.encode(reference_answer, convert_to_numpy=True)
            gen_embedding = model.encode(generated_answer, convert_to_numpy=True)

            # Calculate cosine similarity
            similarity = np.dot(ref_embedding, gen_embedding) / (
                np.linalg.norm(ref_embedding) * np.linalg.norm(gen_embedding)
            )

            return float(similarity)

        except ImportError:
            logger.warning(
                "sentence-transformers not available, skipping semantic similarity"
            )
            return 0.0
        except Exception as e:
            logger.error("Error calculating semantic similarity: %s", str(e))
            return 0.0

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Normalize text
        text = AnswerMetrics._normalize_text(text)

        # Split into tokens
        return text.split()


class HallucinationMetrics:
    """
    Metrics for detecting and measuring hallucinations.
    """

    @staticmethod
    def entailment_score(
        context: str, answer: str, model_name: str = "cross-encoder/nli-deberta-v3-base"
    ) -> float:
        """
        Calculate entailment score using NLI model.

        Args:
            context: Ground truth context
            answer: Generated answer
            model_name: Name of the NLI model

        Returns:
            Entailment score (0-1)
        """
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)

            # Predict entailment
            scores = model.predict([(context, answer)])

            # Score interpretation depends on the model
            # For 3-class NLI models: [contradiction, neutral, entailment]
            # Higher score for the entailment class means less hallucination

            if isinstance(scores, list) and len(scores) == 3:
                # Return entailment score (last class)
                return float(scores[2])
            else:
                # Single score model
                return float(scores)

        except ImportError:
            logger.warning(
                "sentence-transformers not available, skipping entailment score"
            )
            return 0.0
        except Exception as e:
            logger.error("Error calculating entailment score: %s", str(e))
            return 0.0

    @staticmethod
    def fact_verification(
        context: str, answer: str, openai_client: Any = None, model: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Verify facts in the answer against the context using an LLM.

        Args:
            context: Ground truth context
            answer: Generated answer
            openai_client: OpenAI client instance
            model: Model to use for verification

        Returns:
            Dictionary with verification results
        """
        if not openai_client:
            return {"error": "No OpenAI client provided"}

        prompt = (
            "You are an expert fact-checker. Your task is to verify if the statements in the 'Answer' "
            "are supported by the information in the 'Context'. Identify any statements that are not "
            "supported (hallucinations) or contradict the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "List each factual claim in the answer and indicate whether it is:\n"
            "- SUPPORTED: Directly supported by the context\n"
            "- PARTIALLY_SUPPORTED: Partially supported but with some unsupported details\n"
            "- UNSUPPORTED: Not mentioned in the context (hallucination)\n"
            "- CONTRADICTORY: Contradicts information in the context\n\n"
            "For each claim, provide a brief explanation. Then provide a hallucination score "
            "from 0 to 10, where 0 means no hallucinations and 10 means completely hallucinated."
        )

        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content

            # Extract hallucination score
            score_match = re.search(
                r"hallucination score.*?(\d+(?:\.\d+)?)", response_text, re.IGNORECASE
            )
            score = float(score_match.group(1)) if score_match else 5.0

            # Normalize to 0-1 range
            normalized_score = score / 10.0

            return {
                "score": 1.0 - normalized_score,  # Invert so higher is better
                "explanation": response_text,
                "raw_score": score,
            }

        except Exception as e:
            logger.error("Error in fact verification: %s", str(e))
            return {"error": str(e)}


class CitationMetrics:
    """
    Metrics for evaluating citation accuracy.
    """

    @staticmethod
    def citation_recall(relevant_sources: List[str], cited_sources: List[str]) -> float:
        """
        Calculate citation recall.

        Args:
            relevant_sources: List of relevant source IDs
            cited_sources: List of cited source IDs

        Returns:
            Citation recall score
        """
        if not relevant_sources:
            return 1.0  # No relevant sources to cite

        if not cited_sources:
            return 0.0  # No citations

        # Calculate intersection
        cited_relevant = set(cited_sources).intersection(set(relevant_sources))

        return len(cited_relevant) / len(relevant_sources)

    @staticmethod
    def citation_precision(
        relevant_sources: List[str], cited_sources: List[str]
    ) -> float:
        """
        Calculate citation precision.

        Args:
            relevant_sources: List of relevant source IDs
            cited_sources: List of cited source IDs

        Returns:
            Citation precision score
        """
        if not cited_sources:
            return 1.0  # No incorrect citations

        if not relevant_sources:
            return 0.0  # All citations are incorrect

        # Calculate intersection
        cited_relevant = set(cited_sources).intersection(set(relevant_sources))

        return len(cited_relevant) / len(cited_sources)

    @staticmethod
    def citation_f1(relevant_sources: List[str], cited_sources: List[str]) -> float:
        """
        Calculate citation F1 score.

        Args:
            relevant_sources: List of relevant source IDs
            cited_sources: List of cited source IDs

        Returns:
            Citation F1 score
        """
        precision = CitationMetrics.citation_precision(relevant_sources, cited_sources)
        recall = CitationMetrics.citation_recall(relevant_sources, cited_sources)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def citation_verification(
        context: str,
        answer: str,
        sources: List[Dict[str, Any]],
        openai_client: Any = None,
        model: str = "gpt-4",
    ) -> Dict[str, Any]:
        """
        Verify citation accuracy using an LLM.

        Args:
            context: Ground truth context
            answer: Generated answer with citations
            sources: List of source information
            openai_client: OpenAI client instance
            model: Model to use for verification

        Returns:
            Dictionary with verification results
        """
        if not openai_client:
            return {"error": "No OpenAI client provided"}

        # Format sources
        sources_text = ""
        for i, source in enumerate(sources):
            source_id = source.get("id", f"source_{i}")
            title = source.get("title", "Unnamed source")

            sources_text += f"{source_id}: {title}\n"

        prompt = (
            "You are an expert evaluating citation accuracy. Your task is to verify if the citations in the 'Answer' "
            "correctly reference the appropriate sources and if the information attributed to each source is accurate.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer with citations:\n{answer}\n\n"
            f"Sources:\n{sources_text}\n\n"
            "Evaluate the citation accuracy by:\n"
            "1. Identifying each citation in the answer\n"
            "2. Checking if the cited source exists in the provided sources list\n"
            "3. Verifying if the information attributed to each source is accurate based on the context\n\n"
            "For each citation, indicate whether it is:\n"
            "- ACCURATE: Correctly cites an existing source and the information is accurate\n"
            "- MISATTRIBUTED: Cites an existing source but the information doesn't match\n"
            "- NONEXISTENT: Cites a source that doesn't exist in the provided list\n\n"
            "Then provide an overall citation accuracy score from 0 to 10, where 0 means completely inaccurate "
            "and 10 means perfectly accurate citations."
        )

        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content

            # Extract accuracy score
            score_match = re.search(
                r"accuracy score.*?(\d+(?:\.\d+)?)", response_text, re.IGNORECASE
            )
            score = float(score_match.group(1)) if score_match else 5.0

            # Normalize to 0-1 range
            normalized_score = score / 10.0

            return {
                "score": normalized_score,
                "explanation": response_text,
                "raw_score": score,
            }

        except Exception as e:
            logger.error("Error in citation verification: %s", str(e))
            return {"error": str(e)}
