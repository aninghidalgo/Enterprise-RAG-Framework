"""
Response generator module for creating answers from context.
"""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Union
import json

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates responses to queries using retrieved context.
    Supports multiple LLM providers with configurable prompts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the response generator with configuration options.
        
        Args:
            config: Generation configuration
        """
        self.config = config or {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500,
            "provider": "openai",  # Options: openai, anthropic, local
            "prompt_template": "default",
            "streaming": False,
            "use_system_prompt": True,
        }
        
        # Initialize provider client
        self.client = self._initialize_client()
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info("Initialized response generator with config: %s", self.config)
    
    def _initialize_client(self) -> Any:
        """
        Initialize the LLM client based on the provider.
        
        Returns:
            LLM client instance
        """
        provider = self.config.get("provider", "openai")
        
        try:
            if provider == "openai":
                return self._initialize_openai_client()
            elif provider == "anthropic":
                return self._initialize_anthropic_client()
            elif provider == "local":
                return self._initialize_local_client()
            else:
                logger.warning("Unknown provider: %s, falling back to OpenAI", provider)
                return self._initialize_openai_client()
        except Exception as e:
            logger.error("Error initializing %s client: %s", provider, str(e))
            return None
    
    def _initialize_openai_client(self) -> Any:
        """
        Initialize the OpenAI client.
        
        Returns:
            OpenAI client instance
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
            logger.info("Initialized OpenAI client")
            return client
            
        except ImportError:
            logger.error("Failed to import OpenAI client. Install with: pip install openai")
            return None
    
    def _initialize_anthropic_client(self) -> Any:
        """
        Initialize the Anthropic client.
        
        Returns:
            Anthropic client instance
        """
        try:
            from anthropic import Anthropic
            
            # Get API key from config or environment
            api_key = self.config.get("anthropic_api_key")
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
            
            if not api_key:
                logger.warning("Anthropic API key not found in config or environment")
                return None
            
            client = Anthropic(api_key=api_key)
            logger.info("Initialized Anthropic client")
            return client
            
        except ImportError:
            logger.error("Failed to import Anthropic client. Install with: pip install anthropic")
            return None
    
    def _initialize_local_client(self) -> Any:
        """
        Initialize the local model client.
        
        Returns:
            Local model client instance
        """
        model_path = self.config.get("local_model_path")
        if not model_path:
            logger.error("Local model path not specified in config")
            return None
        
        try:
            from .local.vllm_client import VLLMClient
            
            client = VLLMClient(model_path=model_path)
            logger.info("Initialized local vLLM client with model: %s", model_path)
            return client
            
        except ImportError:
            logger.error("Failed to import vLLM client. Install with: pip install vllm")
            
            try:
                from .local.basic_client import BasicLocalClient
                
                client = BasicLocalClient(model_path=model_path)
                logger.info("Initialized basic local client with model: %s", model_path)
                return client
                
            except ImportError:
                logger.error("Failed to initialize any local client")
                return None
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load prompt templates from configuration or default.
        
        Returns:
            Dictionary of prompt templates
        """
        # Check if templates are provided in config
        if "prompt_templates" in self.config:
            return self.config["prompt_templates"]
        
        # Check if a template file is specified
        template_file = self.config.get("prompt_template_file")
        if template_file and os.path.exists(template_file):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error loading prompt templates from file: %s", str(e))
        
        # Default templates
        return {
            "default": {
                "system": "You are a helpful assistant that answers questions based on the provided context. "
                         "If the answer is not in the context, say you don't know. "
                         "Always include source citations when referencing specific information.",
                "user": "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                "assistant": ""
            },
            "concise": {
                "system": "You are a concise assistant that gives brief, accurate answers based on the provided context.",
                "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer:",
                "assistant": ""
            },
            "comprehensive": {
                "system": "You are a comprehensive assistant that provides detailed, thorough answers based on the provided context. "
                         "Include all relevant information and properly cite sources.",
                "user": "Context:\n{context}\n\nQuestion: {query}\n\nProvide a detailed answer with citations:",
                "assistant": ""
            },
            "extractive": {
                "system": "You are an assistant that extracts precise answers directly from the provided context. "
                         "Use exact quotes from the context and provide citations.",
                "user": "Context:\n{context}\n\nQuestion: {query}\n\nExtract the answer directly from the context with citations:",
                "assistant": ""
            }
        }
    
    def generate(
        self,
        query: str,
        context: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response to a query using the provided context.
        
        Args:
            query: Query text
            context: Context dictionary with text and citations
            options: Optional parameters to customize generation
            
        Returns:
            Dictionary with generated text, sources, and metadata
        """
        start_time = time.time()
        
        # Merge configuration with options
        config = self.config.copy()
        if options:
            config.update(options)
        
        # Get model and parameters
        model = config.get("model", "gpt-3.5-turbo")
        temperature = config.get("temperature", 0.7)
        max_tokens = config.get("max_tokens", 500)
        streaming = config.get("streaming", False)
        
        # Select prompt template
        template_name = config.get("prompt_template", "default")
        if template_name not in self.prompt_templates:
            logger.warning("Unknown prompt template: %s, using default", template_name)
            template_name = "default"
        
        template = self.prompt_templates[template_name]
        
        # Format prompt
        context_text = context.get("context", "")
        citations = context.get("citations", [])
        
        prompt = self._format_prompt(template, query, context_text)
        
        # Generate response
        if self.client is None:
            error_msg = "LLM client not initialized"
            logger.error(error_msg)
            return {
                "text": f"Error: {error_msg}",
                "sources": [],
                "confidence": 0.0,
                "latency": {
                    "retrieval_ms": context.get("latency_ms", 0),
                    "augmentation_ms": 0,
                    "generation_ms": 0,
                    "total_ms": context.get("latency_ms", 0),
                }
            }
        
        try:
            generation_start = time.time()
            
            provider = config.get("provider", "openai")
            
            if provider == "openai":
                response_text = self._generate_openai(
                    prompt, model, temperature, max_tokens, streaming
                )
            elif provider == "anthropic":
                response_text = self._generate_anthropic(
                    prompt, model, temperature, max_tokens, streaming
                )
            elif provider == "local":
                response_text = self._generate_local(
                    prompt, temperature, max_tokens, streaming
                )
            else:
                logger.warning("Unknown provider: %s, falling back to OpenAI", provider)
                response_text = self._generate_openai(
                    prompt, model, temperature, max_tokens, streaming
                )
            
            generation_time = time.time() - generation_start
            
            # Extract sources used in the response
            sources = self._extract_sources(response_text, citations)
            
            # Calculate confidence based on sources
            confidence = self._calculate_confidence(response_text, sources)
            
            total_time = time.time() - start_time
            
            return {
                "text": response_text,
                "sources": sources,
                "confidence": confidence,
                "latency": {
                    "retrieval_ms": context.get("latency_ms", 0),
                    "augmentation_ms": context.get("latency_ms", 0),
                    "generation_ms": int(generation_time * 1000),
                    "total_ms": int(total_time * 1000),
                }
            }
            
        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            
            return {
                "text": f"Error generating response: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "latency": {
                    "retrieval_ms": context.get("latency_ms", 0),
                    "augmentation_ms": 0,
                    "generation_ms": 0,
                    "total_ms": int((time.time() - start_time) * 1000),
                }
            }
    
    def _format_prompt(
        self,
        template: Dict[str, str],
        query: str,
        context: str
    ) -> Dict[str, str]:
        """
        Format prompt using template.
        
        Args:
            template: Prompt template with system, user, and assistant parts
            query: Query text
            context: Context text
            
        Returns:
            Formatted prompt dictionary
        """
        use_system_prompt = self.config.get("use_system_prompt", True)
        
        # Format system prompt if present
        system_prompt = ""
        if "system" in template and use_system_prompt:
            system_prompt = template["system"]
        
        # Format user prompt
        user_prompt = template["user"].format(
            query=query,
            context=context
        )
        
        # Format assistant prompt
        assistant_prompt = template.get("assistant", "")
        
        return {
            "system": system_prompt,
            "user": user_prompt,
            "assistant": assistant_prompt
        }
    
    def _generate_openai(
        self,
        prompt: Dict[str, str],
        model: str,
        temperature: float,
        max_tokens: int,
        streaming: bool
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            prompt: Formatted prompt
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream the response
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized")
        
        messages = []
        
        # Add system message if present
        if prompt["system"]:
            messages.append({"role": "system", "content": prompt["system"]})
        
        # Add user message
        messages.append({"role": "user", "content": prompt["user"]})
        
        # Add assistant message if present
        if prompt["assistant"]:
            messages.append({"role": "assistant", "content": prompt["assistant"]})
        
        if streaming:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Collect chunks
            chunks = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            
            return "".join(chunks)
        else:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: Dict[str, str],
        model: str,
        temperature: float,
        max_tokens: int,
        streaming: bool
    ) -> str:
        """
        Generate response using Anthropic API.
        
        Args:
            prompt: Formatted prompt
            model: Model name
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream the response
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("Anthropic client not initialized")
        
        # Combine prompts in Claude format
        system = prompt["system"]
        user = prompt["user"]
        
        # Claude v2 format
        if "claude-2" in model:
            message = f"{system}\n\n{user}"
        # Claude 3 format 
        else:
            messages = []
            
            if system:
                messages.append({
                    "role": "system",
                    "content": system
                })
            
            messages.append({
                "role": "user",
                "content": user
            })
            
            if prompt["assistant"]:
                messages.append({
                    "role": "assistant",
                    "content": prompt["assistant"]
                })
        
        if streaming:
            if "claude-2" in model:
                response = self.client.completions.create(
                    prompt=message,
                    model=model,
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                    stream=True
                )
            else:
                response = self.client.messages.create(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
            
            # Collect chunks
            chunks = []
            for chunk in response:
                if "claude-2" in model:
                    if chunk.completion:
                        chunks.append(chunk.completion)
                else:
                    if chunk.delta.text:
                        chunks.append(chunk.delta.text)
            
            return "".join(chunks)
        else:
            if "claude-2" in model:
                response = self.client.completions.create(
                    prompt=message,
                    model=model,
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return response.completion
            else:
                response = self.client.messages.create(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text
    
    def _generate_local(
        self,
        prompt: Dict[str, str],
        temperature: float,
        max_tokens: int,
        streaming: bool
    ) -> str:
        """
        Generate response using local model.
        
        Args:
            prompt: Formatted prompt
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            streaming: Whether to stream the response
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("Local model client not initialized")
        
        # Format prompt based on client type
        if hasattr(self.client, "generate_with_messages"):
            # Client supports message format
            messages = []
            
            if prompt["system"]:
                messages.append({"role": "system", "content": prompt["system"]})
            
            messages.append({"role": "user", "content": prompt["user"]})
            
            if prompt["assistant"]:
                messages.append({"role": "assistant", "content": prompt["assistant"]})
            
            return self.client.generate_with_messages(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=streaming
            )
        else:
            # Client expects raw text input
            formatted_prompt = ""
            
            if prompt["system"]:
                formatted_prompt += f"<system>\n{prompt['system']}\n</system>\n\n"
            
            formatted_prompt += f"<user>\n{prompt['user']}\n</user>\n\n"
            
            if prompt["assistant"]:
                formatted_prompt += f"<assistant>\n{prompt['assistant']}</assistant>\n\n"
            
            formatted_prompt += "<assistant>\n"
            
            return self.client.generate(
                prompt=formatted_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=streaming
            )
    
    def _extract_sources(
        self,
        response_text: str,
        citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract sources used in the response.
        
        Args:
            response_text: Generated response text
            citations: List of citation information
            
        Returns:
            List of sources used in the response
        """
        sources = []
        
        # Check each citation to see if it was referenced
        for citation in citations:
            citation_id = citation.get("id", "")
            
            # Skip if no ID
            if not citation_id:
                continue
            
            # Extract metadata
            metadata = citation.get("metadata", {})
            
            # Check different citation formats
            source_info = {
                "id": citation_id,
                "confidence": 0.0
            }
            
            # Add metadata if available
            if "filename" in metadata:
                source_info["filename"] = metadata["filename"]
            if "title" in metadata:
                source_info["title"] = metadata["title"]
            if "author" in metadata:
                source_info["author"] = metadata["author"]
            if "page" in metadata:
                source_info["page"] = metadata["page"]
            if "url" in metadata:
                source_info["url"] = metadata["url"]
            
            # Check if this source was cited
            cited = False
            
            # Check for filename citation
            if "filename" in metadata and metadata["filename"] in response_text:
                cited = True
                source_info["confidence"] = 1.0
            
            # Check for title citation
            elif "title" in metadata and metadata["title"] in response_text:
                cited = True
                source_info["confidence"] = 1.0
            
            # Check for numeric citation
            elif f"[{citation_id}]" in response_text or f"[Source {citation_id}]" in response_text:
                cited = True
                source_info["confidence"] = 1.0
            
            # Check for inline citation with document number
            elif "Source " in response_text and citation_id in response_text:
                cited = True
                source_info["confidence"] = 0.9
            
            if cited:
                sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(
        self,
        response_text: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            response_text: Generated response text
            sources: List of sources used
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Simple heuristic for confidence based on sources
        if not sources:
            # No sources = low confidence
            return 0.3
        
        # More sources = higher confidence
        source_count_score = min(len(sources) / 5, 0.5)  # Up to 0.5 for 5+ sources
        
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "I don't know",
            "I'm not sure",
            "It's unclear",
            "The context doesn't provide",
            "The context doesn't mention",
            "The context doesn't specify",
            "I cannot determine",
            "There is no information",
            "uncertain",
            "might be",
            "could be",
            "possibly"
        ]
        
        uncertainty_score = 1.0
        for phrase in uncertainty_phrases:
            if phrase.lower() in response_text.lower():
                uncertainty_score = 0.5
                break
        
        # Average source confidence
        avg_source_confidence = sum(s.get("confidence", 0.0) for s in sources) / len(sources)
        
        # Combined score
        confidence = (source_count_score * 0.4 + uncertainty_score * 0.3 + avg_source_confidence * 0.3)
        
        return min(max(confidence, 0.1), 1.0)  # Ensure between 0.1 and 1.0
