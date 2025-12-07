"""LLM client abstraction for LCoT2Tree pipeline

Updated to support:
- 4-tuple return: (content, in_tokens, out_tokens, cache_hits)
- DeepSeek prefix caching statistics
- Backward compatibility with other providers
"""
import os
from typing import Optional, Dict, Any, Tuple
from openai import OpenAI


class LLMClient:
    """Base class for LLM clients"""
    
    def __init__(self, model_name: str, temperature: float = 0.6, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(
        self, 
        prompt: str = None, 
        messages: list = None, 
        **kwargs
    ) -> Tuple[str, int, int, int]:
        """
        Generate response from prompt or messages.
        
        Args:
            prompt: Input prompt (deprecated, use messages instead)
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters
        
        Returns:
            Tuple of (response_text, input_tokens, output_tokens, cache_hit_tokens)
        """
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client (for GPT models and DeepSeek)"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        base_url: Optional[str] = None
    ):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def generate(
        self, 
        prompt: str = None, 
        messages: list = None, 
        **kwargs
    ) -> Tuple[str, int, int, int]:
        """
        Generate response using OpenAI API.
        
        Returns:
            (content, input_tokens, output_tokens, cache_hit_tokens)
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Handle both prompt (legacy) and messages (preferred) formats
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        
        try:
            # Newer models (gpt-5) use max_completion_tokens and temperature=1 only
            if "gpt-5" in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                    max_completion_tokens=max_tokens,
                    stream=False
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Extract token usage
            in_tokens = response.usage.prompt_tokens
            out_tokens = response.usage.completion_tokens
            
            # SAFELY EXTRACT CACHE HITS (DeepSeek specific)
            # DeepSeek API returns 'prompt_cache_hit_tokens' in usage
            # Other providers don't have this field, so we default to 0
            cache_hits = getattr(response.usage, 'prompt_cache_hit_tokens', 0)
            
            return content, in_tokens, out_tokens, cache_hits
        
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            raise


class HuggingFaceClient(LLMClient):
    """HuggingFace self-hosted model client (for Qwen3, etc.)"""
    
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        device: str = "cuda"
    ):
        super().__init__(model_name, temperature, max_tokens)
        self.model_path = model_path or model_name
        self.device = device
        
        # Lazy import to avoid requiring transformers if not using this backend
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for HuggingFaceClient. "
                "Install with: pip install transformers torch"
            )
        
        print(f"Loading model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def generate(
        self, 
        prompt: str = None, 
        messages: list = None, 
        **kwargs
    ) -> Tuple[str, int, int, int]:
        """
        Generate response using HuggingFace model.
        
        Returns:
            (content, input_tokens, output_tokens, 0)
            Note: cache_hits always 0 for HuggingFace models
        """
        import torch
        
        temperature = kwargs.get("temperature", self.temperature)
        max_new_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Handle both prompt (legacy) and messages (preferred) formats
        if messages is None:
            if prompt is None:
                raise ValueError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        
        try:
            # Format as chat if tokenizer supports it
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback: just use the last message content
                formatted_prompt = messages[-1]["content"]
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            input_length = inputs.input_ids.shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Count tokens
            in_tokens = input_length
            out_tokens = len(generated_tokens)
            
            # HuggingFace models don't support caching, so cache_hits is always 0
            cache_hits = 0
            
            return response, in_tokens, out_tokens, cache_hits
        
        except Exception as e:
            print(f"HuggingFace generation failed: {e}")
            raise


def create_llm_client(
    backend: str = "openai",
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.
    
    Args:
        backend: One of "openai", "huggingface", "gpt5-nano", "gpt5-mini", 
                 "qwen3-4b", "qwen3-32b", "deepseek", "deepseek-v3.2"
        model_name: Explicit model name (overrides backend defaults)
        config: Configuration dict
        **kwargs: Additional parameters passed to client constructor
    
    Returns:
        LLMClient instance
    """
    config = config or {}
    
    # Handle named backends
    if backend == "deepseek-v3.2" or backend == "deepseek":
        return OpenAIClient(
            model_name=model_name or config.get("model_id", "deepseek-chat"),
            api_key=config.get("api_key") or kwargs.get("api_key"),
            temperature=config.get("temperature", kwargs.get("temperature", 1.0)),
            max_tokens=config.get("max_tokens", kwargs.get("max_tokens", 4096)),
            base_url=config.get("url") or kwargs.get("base_url")
        )
    
    elif backend == "gpt5-nano":
        return OpenAIClient(
            model_name=model_name or "gpt-5-nano",
            api_key=config.get("api_key") or kwargs.get("api_key"),
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 4096)
        )

    elif backend == "gpt5-mini":
        return OpenAIClient(
            model_name=model_name or "gpt-5-mini",
            api_key=config.get("api_key") or kwargs.get("api_key"),
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
    
    elif backend == "qwen3-4b":
        return HuggingFaceClient(
            model_name=model_name or "Qwen/Qwen3-3B-Instruct",
            model_path=config.get("model_path") or kwargs.get("model_path"),
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 4096),
            device=kwargs.get("device", "cuda")
        )
    
    elif backend == "qwen3-32b":
        return HuggingFaceClient(
            model_name=model_name or "Qwen/Qwen3-32B-Instruct",
            model_path=config.get("model_path") or kwargs.get("model_path"),
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 4096),
            device=kwargs.get("device", "cuda")
        )
    
    elif backend == "openai":
        return OpenAIClient(
            model_name=model_name or config.get("model_id", "gpt-4o-mini"),
            api_key=config.get("api_key") or kwargs.get("api_key"),
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 4096)
        )
    
    elif backend == "huggingface":
        if not model_name:
            raise ValueError("model_name must be provided for huggingface backend")
        return HuggingFaceClient(
            model_name=model_name,
            model_path=config.get("model_path") or kwargs.get("model_path"),
            temperature=kwargs.get("temperature", 0.6),
            max_tokens=kwargs.get("max_tokens", 4096),
            device=kwargs.get("device", "cuda")
        )
    
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Must be one of: openai, huggingface, gpt5-nano, gpt5-mini, "
            "qwen3-4b, qwen3-32b, deepseek, deepseek-v3.2"
        )