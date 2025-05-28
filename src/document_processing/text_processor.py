"""
Text processor module for chunking text documents.
"""

import logging
import re
from typing import Dict, List, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles text documents and implements various chunking strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text processor with configuration options.

        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "chunking_strategy": "fixed",  # Options: fixed, semantic, recursive
        }

        # Download NLTK data if needed
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer data")
            nltk.download("punkt", quiet=True)

    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a text document into chunks.

        Args:
            document: Document dictionary with text content

        Returns:
            List of document chunks with text and metadata
        """
        # Extract text content from document
        if "content" in document:
            text = document["content"]
        elif "text" in document:
            text = document["text"]
        else:
            raise ValueError("Document must contain 'content' or 'text' field")

        # Use appropriate chunking strategy
        chunking_strategy = self.config.get("chunking_strategy", "fixed")

        if chunking_strategy == "fixed":
            chunks = self._fixed_size_chunking(text)
        elif chunking_strategy == "semantic":
            chunks = self._semantic_chunking(text)
        elif chunking_strategy == "recursive":
            chunks = self._recursive_chunking(text)
        else:
            logger.warning(
                "Unknown chunking strategy: %s, using fixed size chunking",
                chunking_strategy,
            )
            chunks = self._fixed_size_chunking(text)

        # Create document dictionaries for each chunk
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_doc = {
                "text": chunk_text,
                "metadata": {
                    "chunk_index": i,
                    "chunk_strategy": chunking_strategy,
                    "source_type": "text",
                },
            }

            # Copy original document metadata if it exists
            if "metadata" in document:
                for key, value in document["metadata"].items():
                    if key not in chunk_doc["metadata"]:
                        chunk_doc["metadata"][key] = value

            chunk_docs.append(chunk_doc)

        return chunk_docs

    def _fixed_size_chunking(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with optional overlap.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        chunk_size = self.config.get("chunk_size", 500)
        chunk_overlap = self.config.get("chunk_overlap", 100)

        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split into chunks
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # If this is not the last chunk and we're not at the end of the text,
            # try to break at a sentence boundary or space
            if end < len(text):
                # Try to find the last sentence boundary within the chunk
                last_period = text.rfind(". ", start, end)
                if last_period != -1 and last_period > start + chunk_size // 2:
                    end = last_period + 1  # Include the period
                else:
                    # If no good sentence boundary, try to break at a space
                    last_space = text.rfind(" ", start, end)
                    if last_space != -1:
                        end = last_space

            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move the start position for the next chunk, accounting for overlap
            start = end - chunk_overlap if end - start > chunk_overlap else end

        return chunks

    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Split text into semantic chunks based on sentence and paragraph boundaries.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        max_chunk_size = self.config.get("chunk_size", 500)

        # Split text into paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        # Split paragraphs into sentences
        sentences = []
        for paragraph in paragraphs:
            paragraph_sentences = sent_tokenize(paragraph)
            # Add paragraph marker to the last sentence
            if paragraph_sentences:
                paragraph_sentences[-1] += " [EOP]"  # End of paragraph marker
            sentences.extend(paragraph_sentences)

        # Combine sentences into chunks respecting max chunk size
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds max chunk size and we already have content,
            # finalize the current chunk
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            # If a single sentence exceeds max chunk size, it needs to be split
            if sentence_size > max_chunk_size:
                if current_chunk:  # Finalize any existing chunk first
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split the long sentence using fixed size chunking
                sentence_chunks = self._fixed_size_chunking(sentence)
                chunks.extend(sentence_chunks)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for the space

        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Clean up end of paragraph markers for the final output
        chunks = [chunk.replace(" [EOP]", "") for chunk in chunks]

        return chunks

    def _recursive_chunking(self, text: str) -> List[str]:
        """
        Split text recursively by sections, paragraphs, and sentences as needed.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        max_chunk_size = self.config.get("chunk_size", 500)

        # Try to identify sections first (by headings)
        section_pattern = r"(?:^|\n)#+\s+.+?(?=(?:^|\n)#+\s+|\Z)"
        sections = re.findall(section_pattern, text, re.DOTALL | re.MULTILINE)

        # If no clear sections found, fall back to paragraphs
        if not sections:
            sections = [text]

        chunks = []

        for section in sections:
            if len(section) <= max_chunk_size:
                chunks.append(section.strip())
                continue

            # Split section into paragraphs
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]

            for paragraph in paragraphs:
                if len(paragraph) <= max_chunk_size:
                    chunks.append(paragraph)
                    continue

                # Split long paragraphs into sentences
                sentences = sent_tokenize(paragraph)

                current_chunk = []
                current_size = 0

                for sentence in sentences:
                    sentence_size = len(sentence)

                    if current_size + sentence_size <= max_chunk_size:
                        current_chunk.append(sentence)
                        current_size += sentence_size + 1  # +1 for space
                    else:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            current_size = 0

                        # Handle sentences longer than max chunk size
                        if sentence_size > max_chunk_size:
                            sentence_chunks = self._fixed_size_chunking(sentence)
                            chunks.extend(sentence_chunks)
                        else:
                            current_chunk.append(sentence)
                            current_size = sentence_size + 1

                if current_chunk:
                    chunks.append(" ".join(current_chunk))

        return chunks

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a text file.

        Args:
            file_path: Path to text file

        Returns:
            Extracted text content
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
