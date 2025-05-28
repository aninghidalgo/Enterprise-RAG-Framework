"""
Additional evaluation functions for RAG system evaluation.
"""

import logging
import time
import re
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)


def evaluate_example(
    rag_system: Any,
    example: Dict[str, Any],
    metrics: List[str],
    judge_client: Any = None,
    judge_prompts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single example using the specified metrics.

    Args:
        rag_system: RAG system instance
        example: Test example with query and reference information
        metrics: List of metrics to evaluate
        judge_client: LLM client for judge-based evaluations
        judge_prompts: Dictionary of judge prompt templates

    Returns:
        Dictionary of evaluation results for this example
    """
    # Extract query and reference information
    query = example.get("query", "")
    reference_answer = example.get("answer", "")
    reference_context = example.get("context", "")
    relevant_docs = example.get("relevant_docs", [])

    if not query:
        logger.error("Example missing query")
        return {"error": "Missing query", "metrics": {}}

    # Run RAG system to get response
    start_time = time.time()
    response = rag_system.query(query)
    total_time = time.time() - start_time

    # Extract response information
    generated_answer = response.get("answer", "")
    retrieved_docs = [
        doc.get("id", "")
        for doc in response.get("metadata", {}).get("retrieved_docs", [])
    ]
    sources = response.get("sources", [])
    latency = response.get("latency", {})

    # Initialize result
    result = {
        "query": query,
        "reference": {
            "answer": reference_answer,
            "context": reference_context,
            "relevant_docs": relevant_docs,
        },
        "response": {
            "answer": generated_answer,
            "retrieved_docs": retrieved_docs,
            "sources": sources,
        },
        "metrics": {},
    }

    # Calculate metrics
    for metric in metrics:
        if metric == "retrieval_precision":
            result["metrics"][metric] = evaluate_retrieval_precision(
                relevant_docs, retrieved_docs
            )
        elif metric == "answer_relevance":
            result["metrics"][metric] = evaluate_answer_relevance(
                query, generated_answer, judge_client, judge_prompts
            )
        elif metric == "factual_correctness":
            result["metrics"][metric] = evaluate_factual_correctness(
                query, reference_context, generated_answer, judge_client, judge_prompts
            )
        elif metric == "hallucination_detection":
            result["metrics"][metric] = evaluate_hallucinations(
                query, reference_context, generated_answer, judge_client, judge_prompts
            )
        elif metric == "citation_accuracy":
            result["metrics"][metric] = evaluate_citation_accuracy(
                query,
                reference_context,
                generated_answer,
                sources,
                judge_client,
                judge_prompts,
            )
        elif metric == "latency":
            result["metrics"]["retrieval_latency_ms"] = latency.get("retrieval_ms", 0)
            result["metrics"]["augmentation_latency_ms"] = latency.get(
                "augmentation_ms", 0
            )
            result["metrics"]["generation_latency_ms"] = latency.get("generation_ms", 0)
            result["metrics"]["total_latency_ms"] = latency.get(
                "total_ms", int(total_time * 1000)
            )

    return result


def evaluate_retrieval_precision(
    relevant_docs: List[str], retrieved_docs: List[str]
) -> Dict[str, Any]:
    """
    Evaluate retrieval precision.

    Args:
        relevant_docs: List of relevant document IDs
        retrieved_docs: List of retrieved document IDs

    Returns:
        Dictionary with retrieval precision metrics
    """
    from .metrics import RetrievalMetrics

    # Calculate precision@k for different k values
    k_values = [1, 3, 5, 10]
    precision_at_k = {}

    for k in k_values:
        if k <= len(retrieved_docs):
            precision_at_k[f"p@{k}"] = RetrievalMetrics.precision_at_k(
                relevant_docs, retrieved_docs, k
            )

    # Calculate other retrieval metrics
    recall_at_k = {}
    for k in k_values:
        if k <= len(retrieved_docs):
            recall_at_k[f"r@{k}"] = RetrievalMetrics.recall_at_k(
                relevant_docs, retrieved_docs, k
            )

    mrr = RetrievalMetrics.mean_reciprocal_rank(relevant_docs, retrieved_docs)

    ndcg = {}
    for k in k_values:
        if k <= len(retrieved_docs):
            ndcg[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(
                relevant_docs, retrieved_docs, k
            )

    # Overall score (average of precision@k values)
    avg_precision = (
        sum(precision_at_k.values()) / len(precision_at_k) if precision_at_k else 0
    )

    return {
        "score": avg_precision,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def evaluate_answer_relevance(
    query: str,
    answer: str,
    judge_client: Any = None,
    judge_prompts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Evaluate answer relevance to the query.

    Args:
        query: Query text
        answer: Generated answer
        judge_client: LLM client for judge-based evaluations
        judge_prompts: Dictionary of judge prompt templates

    Returns:
        Dictionary with answer relevance metrics
    """
    from .metrics import AnswerMetrics

    # Calculate automatic metrics
    token_overlap = AnswerMetrics.token_overlap("", answer)  # No reference answer

    # Calculate semantic similarity if possible
    try:
        semantic_similarity = AnswerMetrics.semantic_similarity(query, answer)
    except:
        semantic_similarity = 0.0

    # LLM-based evaluation if available
    if judge_client and judge_prompts and "answer_relevance" in judge_prompts:
        prompt = judge_prompts["answer_relevance"].format(query=query, answer=answer)

        try:
            response = judge_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )

            response_text = response.choices[0].message.content

            # Extract score
            score_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5

            return {
                "score": score,
                "explanation": response_text,
                "token_overlap": token_overlap,
                "semantic_similarity": semantic_similarity,
            }
        except Exception as e:
            logger.error("Error in LLM-based answer relevance evaluation: %s", str(e))

    # Fallback to automatic metrics
    return {
        "score": semantic_similarity,
        "token_overlap": token_overlap,
        "semantic_similarity": semantic_similarity,
    }


def evaluate_factual_correctness(
    query: str,
    context: str,
    answer: str,
    judge_client: Any = None,
    judge_prompts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Evaluate factual correctness of the answer.

    Args:
        query: Query text
        context: Reference context
        answer: Generated answer
        judge_client: LLM client for judge-based evaluations
        judge_prompts: Dictionary of judge prompt templates

    Returns:
        Dictionary with factual correctness metrics
    """
    from .metrics import AnswerMetrics, HallucinationMetrics

    # Calculate automatic metrics
    token_overlap = AnswerMetrics.token_overlap(context, answer)

    # Calculate semantic similarity if possible
    try:
        semantic_similarity = AnswerMetrics.semantic_similarity(context, answer)
    except:
        semantic_similarity = 0.0

    # Calculate entailment score if possible
    try:
        entailment_score = HallucinationMetrics.entailment_score(context, answer)
    except:
        entailment_score = 0.0

    # LLM-based evaluation if available
    if judge_client and judge_prompts and "factual_correctness" in judge_prompts:
        prompt = judge_prompts["factual_correctness"].format(
            query=query, context=context, answer=answer
        )

        try:
            response = judge_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content

            # Extract score
            score_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5

            return {
                "score": score,
                "explanation": response_text,
                "token_overlap": token_overlap,
                "semantic_similarity": semantic_similarity,
                "entailment_score": entailment_score,
            }
        except Exception as e:
            logger.error(
                "Error in LLM-based factual correctness evaluation: %s", str(e)
            )

    # Fallback to automatic metrics
    combined_score = (token_overlap + semantic_similarity + entailment_score) / 3

    return {
        "score": combined_score,
        "token_overlap": token_overlap,
        "semantic_similarity": semantic_similarity,
        "entailment_score": entailment_score,
    }


def evaluate_hallucinations(
    query: str,
    context: str,
    answer: str,
    judge_client: Any = None,
    judge_prompts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Evaluate hallucinations in the answer.

    Args:
        query: Query text
        context: Reference context
        answer: Generated answer
        judge_client: LLM client for judge-based evaluations
        judge_prompts: Dictionary of judge prompt templates

    Returns:
        Dictionary with hallucination metrics
    """
    from .metrics import HallucinationMetrics

    # Calculate entailment score if possible
    try:
        entailment_score = HallucinationMetrics.entailment_score(context, answer)
    except:
        entailment_score = 0.0

    # LLM-based evaluation if available
    if judge_client and judge_prompts and "hallucination_detection" in judge_prompts:
        prompt = judge_prompts["hallucination_detection"].format(
            query=query, context=context, answer=answer
        )

        try:
            response = judge_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
            )

            response_text = response.choices[0].message.content

            # Extract score
            score_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
            hallucination_score = (
                float(score_match.group(1)) / 10.0 if score_match else 0.5
            )

            # Invert score (higher is better, meaning less hallucination)
            score = 1.0 - hallucination_score

            return {
                "score": score,
                "explanation": response_text,
                "entailment_score": entailment_score,
                "hallucination_score": hallucination_score,
            }
        except Exception as e:
            logger.error("Error in LLM-based hallucination evaluation: %s", str(e))

    # Fallback to automatic metrics
    return {"score": entailment_score, "entailment_score": entailment_score}


def evaluate_citation_accuracy(
    query: str,
    context: str,
    answer: str,
    sources: List[Dict[str, Any]],
    judge_client: Any = None,
    judge_prompts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """
    Evaluate citation accuracy in the answer.

    Args:
        query: Query text
        context: Reference context
        answer: Generated answer
        sources: List of source information
        judge_client: LLM client for judge-based evaluations
        judge_prompts: Dictionary of judge prompt templates

    Returns:
        Dictionary with citation accuracy metrics
    """
    from .metrics import CitationMetrics

    # Extract source IDs
    source_ids = [s.get("id", "") for s in sources]

    # No reference for relevant sources, use all sources as relevant
    # In a real evaluation, this would come from ground truth
    relevant_sources = source_ids

    # Calculate citation metrics
    citation_precision = CitationMetrics.citation_precision(
        relevant_sources, source_ids
    )
    citation_recall = CitationMetrics.citation_recall(relevant_sources, source_ids)
    citation_f1 = CitationMetrics.citation_f1(relevant_sources, source_ids)

    # LLM-based evaluation if available
    if judge_client and judge_prompts and "citation_accuracy" in judge_prompts:
        # Format sources for prompt
        sources_text = "\n".join(
            [f"{s.get('id', 'unknown')}: {s.get('title', 'Untitled')}" for s in sources]
        )

        prompt = judge_prompts["citation_accuracy"].format(
            query=query, context=context, answer=answer, sources=sources_text
        )

        try:
            response = judge_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
            )

            response_text = response.choices[0].message.content

            # Extract score
            score_match = re.search(r"(\d+(?:\.\d+)?)", response_text)
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5

            return {
                "score": score,
                "explanation": response_text,
                "citation_precision": citation_precision,
                "citation_recall": citation_recall,
                "citation_f1": citation_f1,
            }
        except Exception as e:
            logger.error("Error in LLM-based citation accuracy evaluation: %s", str(e))

    # Fallback to automatic metrics
    return {
        "score": citation_f1,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,
    }
