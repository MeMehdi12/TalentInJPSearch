"""
OpenAI Configuration Module
===========================

Production-ready OpenAI client setup with:
- Environment variable management
- Connection pooling
- Retry logic with exponential backoff
- Cost tracking
- Error handling
- Timeout management

Author: Talentin AI Search Team
Version: 1.0.0
"""

import os
import logging
from typing import Optional
from functools import lru_cache
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env from backend directory (where the .env file actually lives)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class OpenAIConfig:
    """OpenAI configuration with production defaults"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        self.timeout = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.daily_limit_usd = float(os.getenv("OPENAI_DAILY_LIMIT_USD", "10.0"))
        
        # Validate API key
        if not self.api_key or self.api_key == "your-openai-api-key-here":
            env_file = Path(__file__).parent.parent / ".env"
            raise ValueError(
                f"\n{'='*70}\n"
                f"❌ OPENAI_API_KEY not configured!\n"
                f"{'='*70}\n"
                f"Please add your OpenAI API key to: {env_file.absolute()}\n\n"
                f"Edit the line:\n"
                f"  OPENAI_API_KEY=your-openai-api-key-here\n\n"
                f"Replace with your actual key from: https://platform.openai.com/api-keys\n"
                f"{'='*70}\n"
            )
        
        # Validate model
        valid_models = [
            "gpt-5.2", "gpt-5.2-mini", "gpt-5.2-nano",  # GPT-5.2 family
            "gpt-5-mini", "gpt-5-mini-2025-08-07",  # GPT-5 mini
            "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"  # Legacy
        ]
        if self.model not in valid_models:
            logger.warning(
                f"Model '{self.model}' not in recommended list: {valid_models}. "
                "Proceeding anyway, but costs may vary."
            )

    
    def __repr__(self):
        return (
            f"OpenAIConfig(model={self.model}, max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}, timeout={self.timeout}s)"
        )


@lru_cache(maxsize=1)
def get_openai_config() -> OpenAIConfig:
    """
    Get OpenAI configuration (cached).
    
    Returns:
        OpenAIConfig: Configuration object
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    return OpenAIConfig()


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Get OpenAI client (cached, singleton pattern).
    
    Returns:
        OpenAI: Configured OpenAI client
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    config = get_openai_config()
    
    client = OpenAI(
        api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries
    )
    
    logger.info(f"OpenAI client initialized: {config}")
    return client


@retry(
    retry=retry_if_exception_type((APITimeoutError, RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_openai_with_retry(
    client: OpenAI,
    model: str,
    messages: list,
    temperature: float = 0.1,
    max_tokens: int = 1000,
    response_format: Optional[dict] = None,
    tools: Optional[list] = None,
    tool_choice: Optional[str] = None
) -> dict:
    """
    Call OpenAI API with automatic retry on transient failures.
    
    Args:
        client: OpenAI client instance
        model: Model name (e.g., "gpt-4o-mini")
        messages: List of message dicts
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        response_format: Optional response format specification
        tools: Optional list of tool definitions
        tool_choice: Optional tool choice strategy
    
    Returns:
        dict: API response
    
    Raises:
        APIError: On API errors after retries
        APITimeoutError: On timeout after retries
        RateLimitError: On rate limit after retries
    """
    try:
        kwargs = {
            "model": model,
            "messages": messages,
        }
        
        # GPT-5 models use max_completion_tokens and don't support custom temperature
        if model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
            # GPT-5 uses default temperature, custom values not supported
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        
        if response_format:
            kwargs["response_format"] = response_format
        
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        
        response = client.chat.completions.create(**kwargs)
        
        logger.info(
            f"OpenAI API call successful: model={model}, "
            f"tokens={response.usage.total_tokens}"
        )
        
        return response
        
    except APITimeoutError as e:
        logger.error(f"OpenAI API timeout: {e}")
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        raise
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        raise


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost of OpenAI API call.
    
    Pricing (as of Jan 2026):
    - gpt-5.2-nano: $0.10/1M input, $0.40/1M output (very cheap)
    - gpt-5.2-mini: $0.25/1M input, $1.00/1M output (cheap)
    - gpt-5.2: $2.00/1M input, $8.00/1M output (best)
    - gpt-4o-mini: $0.150/1M input, $0.600/1M output
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    
    Returns:
        float: Estimated cost in USD
    """
    pricing = {
        # GPT-5.2 family
        "gpt-5.2-nano": {"input": 0.10, "output": 0.40},
        "gpt-5.2-mini": {"input": 0.25, "output": 1.00},
        "gpt-5.2": {"input": 2.00, "output": 8.00},
        # GPT-5 family
        "gpt-5-mini": {"input": 0.20, "output": 0.80},
        "gpt-5-mini-2025-08-07": {"input": 0.20, "output": 0.80},
        # Legacy models
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50}
    }
    
    if model not in pricing:
        logger.warning(f"Unknown model '{model}', using gpt-5.2-nano pricing")
        model = "gpt-5.2-nano"
    
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output"]
    
    return input_cost + output_cost


class CostTracker:
    """Track OpenAI API costs"""
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def add_call(self, model: str, input_tokens: int, output_tokens: int):
        """Add a call to the tracker"""
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        logger.info(
            f"OpenAI call tracked: ${cost:.6f} "
            f"(total: ${self.total_cost:.4f}, calls: {self.total_calls})"
        )
    
    def get_stats(self) -> dict:
        """Get cost statistics"""
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_cost_per_call": round(self.total_cost / max(self.total_calls, 1), 6)
        }
    
    def reset(self):
        """Reset tracker"""
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0


# Global cost tracker instance
_cost_tracker = CostTracker()


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance"""
    return _cost_tracker
