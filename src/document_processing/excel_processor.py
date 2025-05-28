"""
Excel document processor for handling Excel files with enhanced metadata extraction.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import hashlib
import time

logger = logging.getLogger(__name__)

class ExcelProcessor:
    """
    Processes Excel files into chunks suitable for embedding and retrieval.
    Supports both .xlsx and .xls formats with enhanced metadata extraction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Excel processor with configuration options.

        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 100)

    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an Excel document into chunks.

        Args:
            document: Document dictionary containing Excel data and metadata

        Returns:
            List of processed document chunks with text and metadata
        """
        try:
            # Read Excel file
            if "content" in document:
                df = pd.read_excel(document["content"])
            elif "file_path" in document:
                df = pd.read_excel(document["file_path"])
            else:
                raise ValueError("No Excel content or file path provided")

            chunks = []
            metadata = document.get("metadata", {})

            # Extract enhanced metadata
            enhanced_metadata = self._extract_metadata(df, metadata)
            metadata.update(enhanced_metadata)

            # Process each sheet
            for sheet_name in df.sheet_names:
                sheet_df = pd.read_excel(document.get("content") or document.get("file_path"), 
                                       sheet_name=sheet_name)
                
                # Convert sheet to text representation
                sheet_text = self._sheet_to_text(sheet_df)
                
                # Create chunks from sheet text
                sheet_chunks = self._create_chunks(sheet_text, metadata, sheet_name)
                chunks.extend(sheet_chunks)

            return chunks

        except Exception as e:
            logger.error(f"Error processing Excel document: {str(e)}")
            raise

    def _extract_metadata(self, df: pd.DataFrame, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract enhanced metadata from Excel file.

        Args:
            df: Pandas DataFrame containing Excel data
            base_metadata: Base metadata dictionary

        Returns:
            Dictionary containing enhanced metadata
        """
        metadata = {
            "document_type": "excel",
            "sheet_count": len(df.sheet_names),
            "sheet_names": df.sheet_names,
            "processed_at": datetime.now().isoformat(),
        }

        # Extract column statistics
        for sheet_name in df.sheet_names:
            sheet_df = pd.read_excel(df, sheet_name=sheet_name)
            metadata[f"{sheet_name}_columns"] = list(sheet_df.columns)
            metadata[f"{sheet_name}_row_count"] = len(sheet_df)
            metadata[f"{sheet_name}_column_count"] = len(sheet_df.columns)

        return metadata

    def _sheet_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert Excel sheet to text representation.

        Args:
            df: Pandas DataFrame containing sheet data

        Returns:
            String representation of the sheet
        """
        text_parts = []
        
        # Add column headers
        text_parts.append(" | ".join(df.columns))
        text_parts.append("-" * len(text_parts[0]))
        
        # Add data rows
        for _, row in df.iterrows():
            text_parts.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(text_parts)

    def _create_chunks(self, text: str, metadata: Dict[str, Any], sheet_name: str) -> List[Dict[str, Any]]:
        """
        Create chunks from sheet text.

        Args:
            text: Text representation of the sheet
            metadata: Document metadata
            sheet_name: Name of the current sheet

        Returns:
            List of document chunks
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "sheet_name": sheet_name,
                "chunk_id": len(chunks),
                "chunk_start_word": i,
                "chunk_end_word": i + len(chunk_words),
            })
            
            # Generate unique chunk ID
            chunk_id = hashlib.md5(
                f"{chunk_text}_{sheet_name}_{i}_{time.time()}".encode()
            ).hexdigest()
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks 