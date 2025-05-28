"""
Evaluation suite for assessing RAG system performance.
"""

import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """
    Comprehensive evaluation suite for RAG systems.
    Measures retrieval precision, answer relevance, factual correctness, and more.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation suite with configuration options.

        Args:
            config: Evaluation configuration
        """
        self.config = config or {
            "metrics": [
                "retrieval_precision",
                "answer_relevance",
                "factual_correctness",
                "hallucination_detection",
                "citation_accuracy",
                "latency",
            ],
            "openai_model": "gpt-4",
            "judge_prompt_template": "default",
        }

        # Initialize LLM client for judge-based evaluations
        self.judge_client = self._initialize_judge()

        # Load prompt templates
        self.judge_prompts = self._load_judge_prompts()

        logger.info("Initialized evaluation suite with config: %s", self.config)

    def _initialize_judge(self) -> Any:
        """
        Initialize the LLM client for judge-based evaluations.

        Returns:
            LLM client instance
        """
        try:
            from openai import OpenAI

            # Get API key from config or environment
            api_key = self.config.get("openai_api_key")
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")

            if not api_key:
                logger.warning("OpenAI API key not found in config or environment")
                return None

            client = OpenAI(api_key=api_key)
            logger.info("Initialized OpenAI client for evaluation judge")
            return client

        except ImportError:
            logger.error(
                "Failed to import OpenAI client. Install with: pip install openai"
            )
            return None

    def _load_judge_prompts(self) -> Dict[str, str]:
        """
        Load judge prompt templates.

        Returns:
            Dictionary of prompt templates
        """
        # Check if templates are provided in config
        if "judge_prompts" in self.config:
            return self.config["judge_prompts"]

        # Default templates
        return {
            "retrieval_precision": (
                "You are an expert evaluator assessing the relevance of retrieved documents to a query. "
                "Query: {query}\n\n"
                "Retrieved Document: {document}\n\n"
                "Rate the relevance of this document to the query on a scale from 0 to 10, "
                "where 0 means completely irrelevant and 10 means perfectly relevant. "
                "Provide your rating as a single number and a brief explanation."
            ),
            "answer_relevance": (
                "You are an expert evaluator assessing the relevance of an answer to a query. "
                "Query: {query}\n\n"
                "Answer: {answer}\n\n"
                "Rate how relevant and responsive this answer is to the query on a scale from 0 to 10, "
                "where 0 means completely irrelevant and 10 means perfectly relevant and responsive. "
                "Provide your rating as a single number and a brief explanation."
            ),
            "factual_correctness": (
                "You are an expert evaluator assessing the factual correctness of an answer. "
                "Query: {query}\n\n"
                "Context (ground truth): {context}\n\n"
                "Answer: {answer}\n\n"
                "Rate the factual correctness of this answer on a scale from 0 to 10, "
                "where 0 means completely incorrect and 10 means perfectly correct according to the context. "
                "Focus ONLY on factual correctness, not completeness or relevance. "
                "Provide your rating as a single number and a brief explanation."
            ),
            "hallucination_detection": (
                "You are an expert evaluator detecting hallucinations in an answer. "
                "Query: {query}\n\n"
                "Context (ground truth): {context}\n\n"
                "Answer: {answer}\n\n"
                "Identify any statements in the answer that are not supported by the context or contradict it. "
                "Rate the hallucination level on a scale from 0 to 10, "
                "where 0 means no hallucinations and 10 means completely hallucinated. "
                "Provide your rating as a single number and list the specific hallucinations."
            ),
            "citation_accuracy": (
                "You are an expert evaluator assessing citation accuracy in an answer. "
                "Query: {query}\n\n"
                "Context (ground truth): {context}\n\n"
                "Answer with citations: {answer}\n\n"
                "Sources provided: {sources}\n\n"
                "Rate the citation accuracy on a scale from 0 to 10, "
                "where 0 means completely inaccurate citations and 10 means perfectly accurate. "
                "Consider both whether the information is cited and whether the citations are correct. "
                "Provide your rating as a single number and a brief explanation."
            ),
        }

    def evaluate(
        self,
        rag_system: Any,
        test_dataset: Union[str, List[Dict[str, Any]]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG system using the specified metrics.

        Args:
            rag_system: RAG system instance
            test_dataset: Path to test dataset file or list of test examples
            metrics: List of metrics to evaluate (default: all configured metrics)

        Returns:
            Dictionary of evaluation results by metric
        """
        # Load test dataset if it's a file path
        if isinstance(test_dataset, str):
            dataset = self._load_dataset(test_dataset)
        else:
            dataset = test_dataset

        if not dataset:
            logger.error("Empty test dataset")
            return {"error": "Empty test dataset"}

        # Use configured metrics if none specified
        if not metrics:
            metrics = self.config.get(
                "metrics",
                [
                    "retrieval_precision",
                    "answer_relevance",
                    "factual_correctness",
                    "hallucination_detection",
                    "citation_accuracy",
                    "latency",
                ],
            )

        # Initialize results
        results = {
            "overall": {},
            "examples": [],
            "metadata": {"dataset_size": len(dataset), "metrics": metrics},
        }

        # Evaluate each example
        start_time = time.time()

        for i, example in enumerate(dataset):
            logger.info("Evaluating example %d/%d", i + 1, len(dataset))

            example_result = self._evaluate_example(rag_system, example, metrics)
            results["examples"].append(example_result)

        # Calculate aggregate results
        for metric in metrics:
            if metric == "latency":
                # Average latencies
                retrieval_latencies = [
                    ex["metrics"].get("retrieval_latency_ms", 0)
                    for ex in results["examples"]
                ]
                augmentation_latencies = [
                    ex["metrics"].get("augmentation_latency_ms", 0)
                    for ex in results["examples"]
                ]
                generation_latencies = [
                    ex["metrics"].get("generation_latency_ms", 0)
                    for ex in results["examples"]
                ]
                total_latencies = [
                    ex["metrics"].get("total_latency_ms", 0)
                    for ex in results["examples"]
                ]

                results["overall"]["avg_retrieval_latency_ms"] = (
                    sum(retrieval_latencies) / len(retrieval_latencies)
                    if retrieval_latencies
                    else 0
                )
                results["overall"]["avg_augmentation_latency_ms"] = (
                    sum(augmentation_latencies) / len(augmentation_latencies)
                    if augmentation_latencies
                    else 0
                )
                results["overall"]["avg_generation_latency_ms"] = (
                    sum(generation_latencies) / len(generation_latencies)
                    if generation_latencies
                    else 0
                )
                results["overall"]["avg_total_latency_ms"] = (
                    sum(total_latencies) / len(total_latencies)
                    if total_latencies
                    else 0
                )
            else:
                # Average scores
                scores = [
                    ex["metrics"].get(metric, {}).get("score", 0)
                    for ex in results["examples"]
                    if metric in ex["metrics"]
                ]

                if scores:
                    results["overall"][f"avg_{metric}_score"] = sum(scores) / len(
                        scores
                    )

        # Add overall execution time
        results["metadata"]["evaluation_time_seconds"] = time.time() - start_time

        return results

    def _load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Load a test dataset from a file.

        Args:
            dataset_path: Path to dataset file (JSON or JSONL)

        Returns:
            List of test examples
        """
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                if dataset_path.endswith(".jsonl"):
                    # JSONL format (one example per line)
                    return [json.loads(line) for line in f if line.strip()]
                else:
                    # JSON format (array of examples)
                    return json.load(f)
        except Exception as e:
            logger.error("Error loading dataset: %s", str(e))
            return []

    def _evaluate_example(
        self, rag_system: Any, example: Dict[str, Any], metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a single example using the specified metrics.

        Args:
            rag_system: RAG system instance
            example: Test example with query and reference information
            metrics: List of metrics to evaluate

        Returns:
            Dictionary of evaluation results for this example
        """
        from .example_evaluator import evaluate_example

        return evaluate_example(
            rag_system=rag_system,
            example=example,
            metrics=metrics,
            judge_client=self.judge_client,
            judge_prompts=self.judge_prompts,
        )
