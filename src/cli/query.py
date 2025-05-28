#!/usr/bin/env python
"""
CLI tool for querying the Enterprise-Ready RAG System.
"""

import os
import sys
import argparse
import json
import logging
import time
from typing import Dict, Any, Optional
import colorama
from colorama import Fore, Style

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.enterprise_rag import RAGSystem

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
        description="Query the Enterprise-Ready RAG System"
    )

    # Query options
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Query text (if not provided, will enter interactive mode)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        action="append",
        default=[],
        help="Metadata filters in the format key=value (can be used multiple times)",
    )

    # Retrieval options
    parser.add_argument(
        "--retrieval-type",
        choices=["hybrid", "dense", "sparse"],
        default="hybrid",
        help="Retrieval method to use (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking of retrieved documents",
    )

    # Generation options
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use for response generation (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens in the generated response (default: 500)",
    )

    # System configuration
    parser.add_argument(
        "--config", type=str, default=None, help="Path to RAG system configuration file"
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="data/index",
        help="Path to vector index (default: data/index)",
    )

    # Output options
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show source documents in the output",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Show response metadata in the output",
    )
    parser.add_argument(
        "--raw-json", action="store_true", help="Output raw JSON response"
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Path to save query results (JSON format)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def parse_filters(filter_strings: list) -> Dict[str, Any]:
    """
    Parse filter strings into a filter dictionary.

    Args:
        filter_strings: List of filter strings in the format key=value

    Returns:
        Filter dictionary
    """
    filters = {}

    for filter_str in filter_strings:
        if "=" in filter_str:
            key, value = filter_str.split("=", 1)

            # Try to convert value to appropriate type
            try:
                # Try as number
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Try as boolean
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                # Otherwise keep as string

            # Handle metadata prefix
            if not key.startswith("metadata.") and key != "id":
                key = f"metadata.{key}"

            filters[key] = value

    return filters


def format_query_response(
    response: Dict[str, Any], show_sources: bool = False, show_metadata: bool = False
) -> str:
    """
    Format query response for display.

    Args:
        response: Query response dictionary
        show_sources: Whether to show source documents
        show_metadata: Whether to show response metadata

    Returns:
        Formatted response string
    """
    output = []

    # Query
    output.append(f"{Fore.CYAN}Query:{Style.RESET_ALL} {response['query']}\n")

    # Answer
    output.append(f"{Fore.GREEN}Answer:{Style.RESET_ALL}\n{response['answer']}\n")

    # Sources
    if show_sources and "sources" in response and response["sources"]:
        output.append(f"{Fore.YELLOW}Sources:{Style.RESET_ALL}")

        for i, source in enumerate(response["sources"]):
            output.append(
                f"{Fore.YELLOW}[{i+1}]{Style.RESET_ALL} {source.get('id', 'Unknown')}"
            )

            if "score" in source:
                output.append(f"    Score: {source['score']:.3f}")

            if "text" in source:
                text = source["text"]
                if len(text) > 200:
                    text = text[:197] + "..."
                output.append(f"    Text: {text}")

            if "metadata" in source:
                output.append(
                    f"    Metadata: {json.dumps(source['metadata'], indent=2)}"
                )

            output.append("")

    # Metadata
    if show_metadata and "metadata" in response:
        output.append(f"{Fore.MAGENTA}Metadata:{Style.RESET_ALL}")

        for key, value in response["metadata"].items():
            if isinstance(value, dict):
                output.append(f"  {key}:")
                for subkey, subvalue in value.items():
                    output.append(f"    {subkey}: {subvalue}")
            else:
                output.append(f"  {key}: {value}")

    return "\n".join(output)


def query_rag_system(
    rag_system: RAGSystem,
    query: str,
    filters: Dict[str, Any] = None,
    retrieval_options: Dict[str, Any] = None,
    generation_options: Dict[str, Any] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Query the RAG system.

    Args:
        rag_system: RAG system instance
        query: Query text
        filters: Metadata filters
        retrieval_options: Retrieval options
        generation_options: Generation options
        verbose: Whether to print verbose output

    Returns:
        Query response
    """
    start_time = time.time()

    if verbose:
        logger.info(f"Querying: {query}")
        if filters:
            logger.info(f"Filters: {filters}")
        if retrieval_options:
            logger.info(f"Retrieval options: {retrieval_options}")
        if generation_options:
            logger.info(f"Generation options: {generation_options}")

    # Execute query
    response = rag_system.query(
        query=query,
        filters=filters,
        retrieval_options=retrieval_options,
        generation_options=generation_options,
    )

    elapsed_time = time.time() - start_time

    if verbose:
        logger.info(f"Query completed in {elapsed_time:.2f} seconds")

    return response


def interactive_mode(
    rag_system: RAGSystem,
    filters: Dict[str, Any] = None,
    retrieval_options: Dict[str, Any] = None,
    generation_options: Dict[str, Any] = None,
    show_sources: bool = False,
    show_metadata: bool = False,
    verbose: bool = False,
):
    """
    Run in interactive query mode.

    Args:
        rag_system: RAG system instance
        filters: Metadata filters
        retrieval_options: Retrieval options
        generation_options: Generation options
        show_sources: Whether to show source documents
        show_metadata: Whether to show response metadata
        verbose: Whether to print verbose output
    """
    print(f"{Fore.CYAN}Enterprise-Ready RAG System Interactive Mode{Style.RESET_ALL}")
    print(
        f"Type {Fore.YELLOW}'exit'{Style.RESET_ALL} or {Fore.YELLOW}'quit'{Style.RESET_ALL} to end the session"
    )
    print(f"Type {Fore.YELLOW}'help'{Style.RESET_ALL} to see available commands")
    print("")

    while True:
        try:
            # Get query
            query = input(f"{Fore.GREEN}Query>{Style.RESET_ALL} ")

            # Check for exit command
            if query.lower() in ["exit", "quit"]:
                print("Exiting interactive mode")
                break

            # Check for help command
            if query.lower() == "help":
                print(f"\n{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
                print(
                    f"  {Fore.YELLOW}exit, quit{Style.RESET_ALL} - Exit interactive mode"
                )
                print(f"  {Fore.YELLOW}help{Style.RESET_ALL} - Show this help message")
                print(
                    f"  {Fore.YELLOW}sources on/off{Style.RESET_ALL} - Toggle display of source documents"
                )
                print(
                    f"  {Fore.YELLOW}metadata on/off{Style.RESET_ALL} - Toggle display of response metadata"
                )
                print(
                    f"  {Fore.YELLOW}filter add key=value{Style.RESET_ALL} - Add a metadata filter"
                )
                print(
                    f"  {Fore.YELLOW}filter clear{Style.RESET_ALL} - Clear all filters"
                )
                print(
                    f"  {Fore.YELLOW}filter list{Style.RESET_ALL} - List current filters"
                )
                print(
                    f"  {Fore.YELLOW}topk N{Style.RESET_ALL} - Set number of documents to retrieve"
                )
                print(
                    f"  {Fore.YELLOW}temp N.N{Style.RESET_ALL} - Set temperature for generation"
                )
                print("")
                continue

            # Check for configuration commands
            if query.lower().startswith("sources "):
                value = query.lower().split(" ", 1)[1]
                if value in ["on", "true", "yes"]:
                    show_sources = True
                    print("Source display enabled")
                elif value in ["off", "false", "no"]:
                    show_sources = False
                    print("Source display disabled")
                continue

            if query.lower().startswith("metadata "):
                value = query.lower().split(" ", 1)[1]
                if value in ["on", "true", "yes"]:
                    show_metadata = True
                    print("Metadata display enabled")
                elif value in ["off", "false", "no"]:
                    show_metadata = False
                    print("Metadata display disabled")
                continue

            if query.lower().startswith("filter "):
                parts = query.split(" ", 2)
                if len(parts) < 2:
                    print("Invalid filter command")
                    continue

                cmd = parts[1].lower()

                if cmd == "clear":
                    filters = {}
                    print("Filters cleared")
                    continue

                if cmd == "list":
                    if not filters:
                        print("No filters set")
                    else:
                        print("Current filters:")
                        for key, value in filters.items():
                            print(f"  {key}: {value}")
                    continue

                if cmd == "add" and len(parts) >= 3:
                    filter_str = parts[2]
                    new_filters = parse_filters([filter_str])
                    filters.update(new_filters)
                    print(f"Filter added: {filter_str}")
                    continue

            if query.lower().startswith("topk "):
                try:
                    value = int(query.lower().split(" ", 1)[1])
                    retrieval_options["top_k"] = value
                    print(f"Set top_k to {value}")
                    continue
                except (ValueError, IndexError):
                    print("Invalid topk value")
                    continue

            if query.lower().startswith("temp "):
                try:
                    value = float(query.lower().split(" ", 1)[1])
                    if 0 <= value <= 1:
                        generation_options["temperature"] = value
                        print(f"Set temperature to {value}")
                    else:
                        print("Temperature must be between 0 and 1")
                    continue
                except (ValueError, IndexError):
                    print("Invalid temperature value")
                    continue

            # Execute query
            if not query.strip():
                continue

            response = query_rag_system(
                rag_system=rag_system,
                query=query,
                filters=filters,
                retrieval_options=retrieval_options,
                generation_options=generation_options,
                verbose=verbose,
            )

            # Format and display response
            formatted_response = format_query_response(
                response, show_sources=show_sources, show_metadata=show_metadata
            )

            print("\n" + formatted_response + "\n")

        except KeyboardInterrupt:
            print("\nExiting interactive mode")
            break

        except Exception as e:
            print(f"{Fore.RED}Error:{Style.RESET_ALL} {str(e)}")


def main():
    """Main entry point."""
    args = parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize RAG system
    try:
        if args.config:
            rag_system = RAGSystem.from_config(args.config)
        else:
            # Create with default settings and specified index path
            rag_system = RAGSystem(vector_store_config={"index_path": args.index_path})

        logger.info("Initialized RAG system")

    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        sys.exit(1)

    # Parse filters
    filters = parse_filters(args.filter)

    # Prepare retrieval options
    retrieval_options = {
        "retrieval_type": args.retrieval_type,
        "top_k": args.top_k,
        "rerank": not args.no_rerank,
    }

    # Prepare generation options
    generation_options = {
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Run in interactive mode if no query provided
    if not args.query:
        interactive_mode(
            rag_system=rag_system,
            filters=filters,
            retrieval_options=retrieval_options,
            generation_options=generation_options,
            show_sources=args.show_sources,
            show_metadata=args.show_metadata,
            verbose=args.verbose,
        )
        return

    # Execute single query
    try:
        response = query_rag_system(
            rag_system=rag_system,
            query=args.query,
            filters=filters,
            retrieval_options=retrieval_options,
            generation_options=generation_options,
            verbose=args.verbose,
        )

        # Output raw JSON if requested
        if args.raw_json:
            print(json.dumps(response, indent=2))
        else:
            # Format and display response
            formatted_response = format_query_response(
                response,
                show_sources=args.show_sources,
                show_metadata=args.show_metadata,
            )

            print(formatted_response)

        # Save output if requested
        if args.save_output:
            try:
                with open(args.save_output, "w", encoding="utf-8") as f:
                    json.dump(response, f, indent=2)

                logger.info(f"Saved output to {args.save_output}")

            except Exception as e:
                logger.error(f"Error saving output: {str(e)}")

    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
