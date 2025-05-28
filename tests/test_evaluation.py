"""
Tests for the evaluation components of the RAG system.
"""

import pytest
from src.evaluation.evaluator import EvaluationSuite


def test_evaluation_suite_initialization():
    """Test that the EvaluationSuite initializes correctly."""
    suite = EvaluationSuite()
    
    # Check default configuration
    assert isinstance(suite.config, dict)
    assert "metrics" in suite.config
    
    # Default metrics should include standard evaluation metrics
    default_metrics = suite.config["metrics"]
    assert "retrieval_precision" in default_metrics
    assert "answer_relevance" in default_metrics
    assert "factual_correctness" in default_metrics


def test_evaluate_example(evaluation_suite, rag_system, sample_test_dataset):
    """Test evaluating a single example."""
    # Get a single example from the test dataset
    example = sample_test_dataset[0]
    
    # Evaluate the example
    metrics = ["retrieval_precision", "answer_relevance"]
    result = evaluation_suite._evaluate_example(rag_system, example, metrics)
    
    # Check results
    assert "query" in result
    assert result["query"] == example["query"]
    assert "metrics" in result
    
    # Check that requested metrics are present
    assert "retrieval_precision" in result["metrics"]
    assert "answer_relevance" in result["metrics"]
    
    # Each metric should have a score
    for metric_name, metric_data in result["metrics"].items():
        assert "score" in metric_data
        assert 0 <= metric_data["score"] <= 1  # Scores should be normalized


def test_full_evaluation(evaluation_suite, rag_system, sample_test_dataset):
    """Test running a full evaluation on a test dataset."""
    # Run evaluation
    metrics = ["retrieval_precision", "answer_relevance"]
    results = evaluation_suite.evaluate(rag_system, sample_test_dataset, metrics)
    
    # Check overall results structure
    assert "metadata" in results
    assert "examples" in results
    assert "overall" in results
    
    # Check metadata
    assert "timestamp" in results["metadata"]
    assert "num_examples" in results["metadata"]
    assert results["metadata"]["num_examples"] == len(sample_test_dataset)
    
    # Check examples
    assert len(results["examples"]) == len(sample_test_dataset)
    for example in results["examples"]:
        assert "query" in example
        assert "metrics" in example
    
    # Check overall metrics
    assert f"avg_retrieval_precision_score" in results["overall"]
    assert f"avg_answer_relevance_score" in results["overall"]


def test_evaluation_with_custom_metrics(evaluation_suite, rag_system, sample_test_dataset):
    """Test evaluation with custom metrics."""
    # Define custom metrics subset
    custom_metrics = ["factual_correctness", "hallucination_detection"]
    
    # Run evaluation with custom metrics
    results = evaluation_suite.evaluate(rag_system, sample_test_dataset, custom_metrics)
    
    # Check that custom metrics were used
    for example in results["examples"]:
        for metric in custom_metrics:
            assert metric in example["metrics"]
    
    # Check overall metrics
    for metric in custom_metrics:
        assert f"avg_{metric}_score" in results["overall"]
