#!/usr/bin/env python
"""
CLI tool for evaluating the Enterprise-Ready RAG System.
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import colorama
from colorama import Fore, Style

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.enterprise_rag import RAGSystem
from src.evaluation.evaluator import EvaluationSuite

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Enterprise-Ready RAG System using a test dataset"
    )
    
    # Input options
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to test dataset file (JSON or JSONL format)"
    )
    
    # Evaluation options
    parser.add_argument(
        "--metrics", "-m", type=str, default="retrieval_precision,answer_relevance,factual_correctness",
        help="Comma-separated list of metrics to evaluate (default: retrieval_precision,answer_relevance,factual_correctness)"
    )
    parser.add_argument(
        "--subset", "-s", type=int, default=0,
        help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--random-sample", action="store_true",
        help="Randomly sample examples if subset is specified"
    )
    
    # System configuration
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to RAG system configuration file"
    )
    parser.add_argument(
        "--index-path", type=str, default="data/index",
        help="Path to vector index (default: data/index)"
    )
    parser.add_argument(
        "--eval-config", type=str, default=None,
        help="Path to evaluation configuration file"
    )
    
    # Model options (for judge-based evaluation)
    parser.add_argument(
        "--judge-model", type=str, default="gpt-4",
        help="Model to use for judge-based evaluation (default: gpt-4)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to save evaluation results (JSON format)"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Include detailed results for each example"
    )
    parser.add_argument(
        "--format", choices=["text", "json", "csv"], default="text",
        help="Output format (default: text)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load test dataset from file.
    
    Args:
        dataset_path: Path to dataset file (JSON or JSONL)
        
    Returns:
        List of test examples
    """
    try:
        path = Path(dataset_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.jsonl':
                # JSONL format (one example per line)
                return [json.loads(line) for line in f if line.strip()]
            else:
                # JSON format (array of examples)
                return json.load(f)
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return []

def format_evaluation_results(
    results: Dict[str, Any],
    format_type: str = "text",
    detailed: bool = False
) -> str:
    """
    Format evaluation results for display.
    
    Args:
        results: Evaluation results dictionary
        format_type: Output format (text, json, csv)
        detailed: Whether to include detailed results for each example
        
    Returns:
        Formatted results string
    """
    if format_type == "json":
        if not detailed:
            # Only include overall results
            simplified = {
                "metadata": results["metadata"],
                "overall": results["overall"]
            }
            return json.dumps(simplified, indent=2)
        return json.dumps(results, indent=2)
    
    elif format_type == "csv":
        lines = []
        
        # Header
        lines.append("Metric,Score")
        
        # Overall metrics
        for key, value in results["overall"].items():
            if isinstance(value, (int, float)):
                lines.append(f"{key},{value}")
        
        if detailed and "examples" in results:
            lines.append("")
            lines.append("Example,Query,Metric,Score")
            
            for i, example in enumerate(results["examples"]):
                query = example.get("query", "").replace(",", " ")
                for metric_name, metric_data in example.get("metrics", {}).items():
                    score = metric_data.get("score", 0)
                    lines.append(f"{i+1},\"{query}\",{metric_name},{score}")
        
        return "\n".join(lines)
    
    else:  # text format
        output = []
        
        # Title
        output.append(f"{Fore.CYAN}Evaluation Results{Style.RESET_ALL}")
        output.append("=" * 80)
        
        # Metadata
        output.append(f"{Fore.MAGENTA}Metadata:{Style.RESET_ALL}")
        output.append(f"  Dataset: {results['metadata'].get('dataset_name', 'Unknown')}")
        output.append(f"  Number of examples: {results['metadata'].get('num_examples', 0)}")
        output.append(f"  Evaluation time: {results['metadata'].get('evaluation_time_seconds', 0):.2f} seconds")
        output.append(f"  Timestamp: {results['metadata'].get('timestamp', '')}")
        output.append("")
        
        # Overall results
        output.append(f"{Fore.GREEN}Overall Results:{Style.RESET_ALL}")
        
        # Group metrics by type
        retrieval_metrics = {}
        answer_metrics = {}
        latency_metrics = {}
        other_metrics = {}
        
        for key, value in results["overall"].items():
            if isinstance(value, (int, float)):
                if key.startswith("avg_retrieval_"):
                    retrieval_metrics[key] = value
                elif key.startswith("avg_answer_") or key.startswith("avg_factual_") or key.startswith("avg_hallucination_"):
                    answer_metrics[key] = value
                elif key.endswith("_latency_ms") or key.endswith("_time_ms"):
                    latency_metrics[key] = value
                else:
                    other_metrics[key] = value
        
        # Retrieval metrics
        if retrieval_metrics:
            output.append(f"  {Fore.YELLOW}Retrieval Metrics:{Style.RESET_ALL}")
            for key, value in retrieval_metrics.items():
                output.append(f"    {key}: {value:.3f}")
        
        # Answer metrics
        if answer_metrics:
            output.append(f"  {Fore.YELLOW}Answer Quality Metrics:{Style.RESET_ALL}")
            for key, value in answer_metrics.items():
                output.append(f"    {key}: {value:.3f}")
        
        # Latency metrics
        if latency_metrics:
            output.append(f"  {Fore.YELLOW}Latency Metrics:{Style.RESET_ALL}")
            for key, value in latency_metrics.items():
                output.append(f"    {key}: {value:.2f} ms")
        
        # Other metrics
        if other_metrics:
            output.append(f"  {Fore.YELLOW}Other Metrics:{Style.RESET_ALL}")
            for key, value in other_metrics.items():
                output.append(f"    {key}: {value}")
        
        # Detailed results for each example
        if detailed and "examples" in results:
            output.append("")
            output.append(f"{Fore.BLUE}Detailed Example Results:{Style.RESET_ALL}")
            
            for i, example in enumerate(results["examples"]):
                output.append(f"  {Fore.CYAN}Example {i+1}:{Style.RESET_ALL}")
                output.append(f"    Query: {example.get('query', '')}")
                
                if "metrics" in example:
                    output.append(f"    Metrics:")
                    for metric_name, metric_data in example["metrics"].items():
                        score = metric_data.get("score", 0)
                        output.append(f"      {metric_name}: {score:.3f}")
                        
                        if "explanation" in metric_data:
                            explanation = metric_data["explanation"]
                            if len(explanation) > 100:
                                explanation = explanation[:97] + "..."
                            output.append(f"        Explanation: {explanation}")
                
                output.append("")
        
        return "\n".join(output)

def run_evaluation(
    rag_system: RAGSystem,
    test_dataset: List[Dict[str, Any]],
    metrics: List[str],
    eval_config: Dict[str, Any] = None,
    subset: int = 0,
    random_sample: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on the RAG system.
    
    Args:
        rag_system: RAG system instance
        test_dataset: List of test examples
        metrics: List of metrics to evaluate
        eval_config: Evaluation configuration
        subset: Number of examples to evaluate (0 for all)
        random_sample: Whether to randomly sample examples
        verbose: Whether to print verbose output
        
    Returns:
        Evaluation results
    """
    # Initialize evaluation suite
    evaluator = EvaluationSuite(config=eval_config)
    
    # Select subset of examples if specified
    if subset > 0 and subset < len(test_dataset):
        if random_sample:
            import random
            test_dataset = random.sample(test_dataset, subset)
        else:
            test_dataset = test_dataset[:subset]
    
    # Run evaluation
    start_time = time.time()
    
    if verbose:
        logger.info(f"Running evaluation on {len(test_dataset)} examples with metrics: {', '.join(metrics)}")
    
    results = evaluator.evaluate(
        rag_system=rag_system,
        test_dataset=test_dataset,
        metrics=metrics
    )
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    
    return results

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load test dataset
    test_dataset = load_dataset(args.dataset)
    
    if not test_dataset:
        logger.error(f"No examples found in dataset: {args.dataset}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(test_dataset)} examples from dataset")
    
    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Load evaluation config if specified
    eval_config = None
    if args.eval_config:
        try:
            with open(args.eval_config, 'r', encoding='utf-8') as f:
                eval_config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading evaluation config: {str(e)}")
    else:
        # Use default config with specified judge model
        eval_config = {
            "metrics": metrics,
            "openai_model": args.judge_model,
            "judge_prompt_template": "default"
        }
    
    # Initialize RAG system
    try:
        if args.config:
            rag_system = RAGSystem.from_config(args.config)
        else:
            # Create with default settings and specified index path
            rag_system = RAGSystem(
                vector_store_config={
                    "index_path": args.index_path
                }
            )
        
        logger.info("Initialized RAG system")
    
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = run_evaluation(
            rag_system=rag_system,
            test_dataset=test_dataset,
            metrics=metrics,
            eval_config=eval_config,
            subset=args.subset,
            random_sample=args.random_sample,
            verbose=args.verbose
        )
        
        # Format and display results
        formatted_results = format_evaluation_results(
            results,
            format_type=args.format,
            detailed=args.detailed
        )
        
        print(formatted_results)
        
        # Save results if requested
        if args.output:
            try:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if args.format == "json" or output_path.suffix.lower() == ".json":
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2)
                else:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(formatted_results)
                
                logger.info(f"Saved evaluation results to {output_path}")
            
            except Exception as e:
                logger.error(f"Error saving results: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
