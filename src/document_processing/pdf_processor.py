"""
PDF processor module for extracting and processing text from PDF documents.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF documents, extracting text and metadata.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF processor with configuration options.
        
        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {
            "extract_images": False,
            "ocr_fallback": False,
            "preserve_layout": True,
            "extract_tables": False,
        }
        
        # Initialize OCR if needed
        if self.config.get("ocr_fallback", False):
            try:
                import pytesseract
                self.ocr_available = True
                logger.info("OCR fallback enabled for PDF processing")
            except ImportError:
                logger.warning("pytesseract not found, OCR fallback disabled")
                self.ocr_available = False
        else:
            self.ocr_available = False
            
        # Initialize table extraction if needed
        if self.config.get("extract_tables", False):
            try:
                import tabula
                self.table_extraction_available = True
                logger.info("Table extraction enabled for PDF processing")
            except ImportError:
                logger.warning("tabula-py not found, table extraction disabled")
                self.table_extraction_available = False
        else:
            self.table_extraction_available = False
    
    def process(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a PDF document into chunks.
        
        Args:
            document: Document dictionary with PDF content
            
        Returns:
            List of document chunks with text and metadata
        """
        # Get file path from metadata if available
        file_path = None
        if "metadata" in document and "filepath" in document["metadata"]:
            file_path = document["metadata"]["filepath"]
        
        # Extract text from PDF
        if file_path and os.path.isfile(file_path):
            pdf_text, pdf_metadata = self.extract_text_and_metadata(file_path)
        elif "content" in document and isinstance(document["content"], bytes):
            # Handle binary PDF content
            import io
            pdf_text, pdf_metadata = self.extract_text_from_binary(document["content"])
        else:
            raise ValueError("PDF document must have a valid filepath or binary content")
        
        # Merge extracted metadata with document metadata
        metadata = document.get("metadata", {}).copy()
        metadata.update(pdf_metadata)
        
        # Create document with extracted text for the text processor
        text_document = {
            "text": pdf_text,
            "metadata": metadata,
        }
        
        # Use the text processor to chunk the text
        from .text_processor import TextProcessor
        text_processor = TextProcessor(self.config)
        chunks = text_processor.process(text_document)
        
        return chunks
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text, _ = self.extract_text_and_metadata(file_path)
        return text
    
    def extract_text_and_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"PDF file not found: {file_path}")
        
        extracted_text = []
        metadata = {}
        
        try:
            # Open the PDF
            pdf_document = fitz.open(file_path)
            
            # Extract document metadata
            metadata = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "keywords": pdf_document.metadata.get("keywords", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "page_count": len(pdf_document),
                "file_size_bytes": os.path.getsize(file_path),
            }
            
            # Process each page
            for page_num, page in enumerate(pdf_document):
                # Extract text from the page
                if self.config.get("preserve_layout", True):
                    page_text = page.get_text("text")
                else:
                    page_text = page.get_text()
                
                # If page has no text or very little text, try OCR if enabled
                if len(page_text.strip()) < 50 and self.ocr_available and self.config.get("ocr_fallback", False):
                    page_text = self._ocr_page(page)
                
                # Add page number information
                page_header = f"\n--- Page {page_num + 1} ---\n"
                
                # Add to extracted text
                extracted_text.append(page_header + page_text)
                
                # Extract images if configured
                if self.config.get("extract_images", False):
                    self._extract_images(page, page_num, file_path)
            
            # Extract tables if configured and available
            if self.config.get("extract_tables", False) and self.table_extraction_available:
                table_text = self._extract_tables(file_path)
                if table_text:
                    extracted_text.append("\n--- Tables ---\n" + table_text)
            
            # Close the PDF
            pdf_document.close()
            
        except Exception as e:
            logger.error("Error extracting text from PDF %s: %s", file_path, str(e))
            raise
        
        return "\n".join(extracted_text), metadata
    
    def extract_text_from_binary(self, pdf_binary: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from binary PDF content.
        
        Args:
            pdf_binary: Binary PDF content
            
        Returns:
            Tuple of (extracted text, metadata dictionary)
        """
        extracted_text = []
        metadata = {}
        
        try:
            # Open the PDF from binary
            pdf_document = fitz.open(stream=pdf_binary, filetype="pdf")
            
            # Extract document metadata
            metadata = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "keywords": pdf_document.metadata.get("keywords", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "page_count": len(pdf_document),
            }
            
            # Process each page
            for page_num, page in enumerate(pdf_document):
                # Extract text from the page
                if self.config.get("preserve_layout", True):
                    page_text = page.get_text("text")
                else:
                    page_text = page.get_text()
                
                # If page has no text or very little text, try OCR if enabled
                if len(page_text.strip()) < 50 and self.ocr_available and self.config.get("ocr_fallback", False):
                    page_text = self._ocr_page(page)
                
                # Add page number information
                page_header = f"\n--- Page {page_num + 1} ---\n"
                
                # Add to extracted text
                extracted_text.append(page_header + page_text)
            
            # Close the PDF
            pdf_document.close()
            
        except Exception as e:
            logger.error("Error extracting text from binary PDF: %s", str(e))
            raise
        
        return "\n".join(extracted_text), metadata
    
    def _ocr_page(self, page) -> str:
        """
        Perform OCR on a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text from OCR
        """
        if not self.ocr_available:
            return ""
        
        try:
            import pytesseract
            from PIL import Image
            import numpy as np
            
            # Convert page to image
            pix = page.get_pixmap(alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(img)
            
            return ocr_text
            
        except Exception as e:
            logger.error("OCR error: %s", str(e))
            return ""
    
    def _extract_images(self, page, page_num: int, file_path: str) -> None:
        """
        Extract images from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            file_path: Path to PDF file
        """
        try:
            # Create directory for images if it doesn't exist
            pdf_name = os.path.splitext(os.path.basename(file_path))[0]
            image_dir = f"{pdf_name}_images"
            os.makedirs(image_dir, exist_ok=True)
            
            # Extract images
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_path = os.path.join(image_dir, f"page{page_num+1}_img{img_index+1}.{image_ext}")
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    logger.debug("Extracted image: %s", image_path)
            
        except Exception as e:
            logger.error("Error extracting images from page %d: %s", page_num + 1, str(e))
    
    def _extract_tables(self, file_path: str) -> str:
        """
        Extract tables from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted table text
        """
        if not self.table_extraction_available:
            return ""
        
        try:
            import tabula
            import pandas as pd
            import io
            
            # Extract tables
            tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            
            if not tables:
                return ""
            
            # Convert tables to text
            table_texts = []
            
            for i, table in enumerate(tables):
                buffer = io.StringIO()
                table.to_csv(buffer)
                table_text = f"Table {i+1}:\n{buffer.getvalue()}\n"
                table_texts.append(table_text)
            
            return "\n".join(table_texts)
            
        except Exception as e:
            logger.error("Table extraction error: %s", str(e))
            return ""
