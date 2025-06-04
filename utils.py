from typing import List, Tuple, Dict, Optional, Union
import threading

def _format_messages(messages):
    """Format messages into a single string for token counting"""
    formatted_text = ""
    
    # Handle different message formats
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "")
                
                # Handle content which could be a string or a list of content parts
                content = message.get("content", "")
                if isinstance(content, list):
                    # Extract text from content parts
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            formatted_text += f"{role}: {item['text']}\n"
                else:
                    # Direct string content
                    formatted_text += f"{role}: {content}\n"
    
    return formatted_text

def check_for_token_limit(input_ids: List[int]):
    input_token_count = len(input_ids)
            
    # Gemma-3-1b-it has a 32K token input context and 8192 token output context
    input_context_limit = 32768  # 32K for 1B model
    output_context_limit = 8192  # 8K for all Gemma models
    
    # Make sure we don't exceed input context
    if input_token_count > input_context_limit:
        raise HTTPException(
            status_code=400, 
            detail=f"Input too long: {input_token_count} tokens exceeds the model's {input_context_limit} token limit"
        )
    
    return input_context_limit, output_context_limit, input_token_count


class RequestLimiter:
    def __init__(self, max_concurrent=4):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.current_requests = 0
        self.lock = threading.Lock()  # This lock protects the current_requests counter
    
    def acquire(self, blocking=True, timeout=None):
        """Acquire a request slot. Returns True if successful, False otherwise."""
        result = self.semaphore.acquire(blocking=blocking, timeout=timeout)
        if result:
            with self.lock:
                self.current_requests += 1
        return result
    
    def release(self):
        """Release a request slot."""
        with self.lock:
            if self.current_requests > 0:
                self.current_requests -= 1
        self.semaphore.release()
    
    def get_stats(self):
        """Get current usage statistics."""
        with self.lock:
            return {
                "max_concurrent": self.max_concurrent,
                "current_requests": self.current_requests,
                "available_slots": self.max_concurrent - self.current_requests
            }

# Utility functions for vLLM-specific operations
def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for planning purposes
    """
    # Rough approximation: ~4 characters per token
    return len(text) // 4

def validate_sampling_params(temperature: float, max_tokens: int) -> Tuple[float, int]:
    """
    Validate and adjust sampling parameters
    """
    # Clamp temperature
    temperature = max(0.1, min(1.0, temperature))
    
    # Ensure reasonable max_tokens
    max_tokens = max(1, min(8192, max_tokens))
    
    return temperature, max_tokens

def format_error_response(error: Exception) -> Dict:
    """
    Format error responses consistently
    """
    return {
        "error": True,
        "message": str(error),
        "type": type(error).__name__
    }