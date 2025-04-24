import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from transformers import pipeline, TextIteratorStreamer
import gc
import uvicorn
import os
import threading
import time
from utils import _format_messages, check_for_token_limit, RequestLimiter
# Set PyTorch memory allocation settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Lock for thread safety
model_lock = threading.Lock()

# Model and pipeline initialization
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.initialize_model()
        return cls._instance
    
    def initialize_model(self):
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize the pipeline with memory-efficient settings
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-3-1b-it",
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = self.pipe.tokenizer
        self.model = self.pipe.model
        self.last_used = time.time()
        self.processing = False #This is for lock
        self.concurrent_tasks = 0

    
    def check_memory(self, required_memory_gb=2.0):
        """Check if there's enough GPU memory available"""
        if not torch.cuda.is_available():
            return True
            
        # Get current memory stats
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated
        
        # Log memory status
        print(f"GPU Memory: {allocated:.2f}GB used, {free:.2f}GB free out of {total:.2f}GB total")
        
        # Check if we have enough free memory with some buffer
        return free >= required_memory_gb
    
    def cleanup_if_needed(self):
        """Perform cleanup if memory usage is high"""
        if not self.check_memory(required_memory_gb=3.0):
            print("Memory running low, performing cleanup...")
            # Force cleanup to release memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # If still not enough memory, reload the model
            if not self.check_memory(required_memory_gb=5.0):
                print("Reinitializing model to clear memory...")
                del self.pipe
                torch.cuda.empty_cache()
                gc.collect()
                
                # Reinitialize
                self.pipe = pipeline(
                    "text-generation",
                    model="google/gemma-3-1b-it",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.bfloat16
                )
    def stream_generate(self, messages, max_tokens, temperature=0.7):
        """Generate text with streaming support"""
        try:
            self.concurrent_tasks += 1
            
            # Check memory before generation
            if not self.check_memory(2.0):
                self.cleanup_if_needed()
            
            # Get tokenizer and model from the pipeline
            tokenizer = self.pipe.tokenizer
            model = self.pipe.model
            
            # Format messages
            formatted_prompt = _format_messages(messages)
            
            # Count tokens in the input
            input_ids = tokenizer.encode(formatted_prompt)
            input_context_limit, output_context_limit, input_token_count = check_for_token_limit(input_ids)
            
            # Set safe token limit
            gpu_memory_limit = 2048 if self.check_memory(4.0) else 1024
            safe_max_tokens = max(1, min(max_tokens, output_context_limit, gpu_memory_limit))
            
            print(f"Streaming generation - Input tokens: {input_token_count}, Max new tokens: {safe_max_tokens}")
            
            # Create a streamer object
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Prepare inputs
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Create a thread for generation
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": safe_max_tokens,
                "do_sample": True,
                "temperature": temperature,
                "repetition_penalty": 1.1,
                "streamer": streamer
            }
            
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Return the streamer iterator
            return streamer
            
        except Exception as e:
            print(f"Streaming generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def generate(self, messages, max_tokens, use_lock):
        try:
            if use_lock == True:
                self.processing = True
            else:
                self.concurrent_tasks += 1
            
            # Check memory before generation
            if not self.check_memory(2.0):
                self.cleanup_if_needed()
            
            # Get tokenizer from the pipeline
            tokenizer = self.pipe.tokenizer
            
            # Convert messages to the format expected by the tokenizer
            formatted_prompt = _format_messages(messages)
            
            # Count tokens in the input
            input_ids = tokenizer.encode(formatted_prompt)

            input_context_limit, output_context_limit, input_token_count = check_for_token_limit(input_ids)
            
            
            # Ensure we don't exceed user request or output context limit
            # Also apply a reasonable default limit to prevent OOM errors
            gpu_memory_limit = 7680 if self.check_memory(4.0) else 3072  # Adaptive based on available memory
            safe_max_tokens = max(1, min(max_tokens, output_context_limit, gpu_memory_limit))
            
            print(f"Input tokens: {input_token_count}, Generating up to: {safe_max_tokens} tokens")
            
            # Perform generation with memory-efficient settings
            output = self.pipe(
                messages, 
                max_new_tokens=safe_max_tokens,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                clean_up_tokenization_spaces=True,
                repetition_penalty=1.1
            )
            
            result = output[0]['generated_text']
            
            # Get the last message content which is the assistant's response
            if isinstance(result, list) and result and isinstance(result[-1], dict) and 'content' in result[-1]:
                return result[-1]["content"]
            else:
                # Fallback handling for different output formats
                return str(result)
                
        except Exception as e:
            # Log the error
            print(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if use_lock == True:
                self.processing = False
            else:
                self.concurrent_tasks -= 1

            self.last_used = time.time()
            # Force memory cleanup after generation
            torch.cuda.empty_cache()
            gc.collect()
    
    def full_cleanup(self):
        """Complete cleanup of model resources"""
        if hasattr(self, 'pipe'):
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()

# Background task to monitor and clean memory
async def memory_monitor():
    while True:
        try:
            model_manager = ModelManager()
            with model_lock:
                if not model_manager.processing and (time.time() - model_manager.last_used > 300):  # 5 minutes
                    # No activity for 5 minutes, do deeper cleanup
                    print("Performing deep memory cleanup after inactivity")
                    model_manager.full_cleanup()
                    model_manager.initialize_model()
        except Exception as e:
            print(f"Memory monitor error: {e}")
        
        # Run every 60 seconds
        await asyncio.sleep(60)

# Pydantic models for request validation
class GenerateRequest(BaseModel):
    system_prompt: str = Field(default="You are a helpful assistant.")
    user_prompt: str
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]
    max_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)
    stream: bool = Field(default=False)

class ServerConfig(BaseModel):
    max_concurrent_requests: int = Field(default=3, ge=1, le=10, 
                                         description="Maximum number of concurrent requests")


# FastAPI app initialization
app = FastAPI(
    title="Gemma Text Generation API",
    description="Memory-optimized text generation API using Gemma model"
)

# Global model manager
model_manager = ModelManager()

# Initialize with default of 3 concurrent requests
request_limiter = RequestLimiter(max_concurrent=10)

@app.post("/generate")
async def generate_text(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Generate text based on system and user prompts
    """
    # Check if we have enough memory
    if not model_manager.check_memory(2.0):
        background_tasks.add_task(model_manager.cleanup_if_needed)
        raise HTTPException(status_code=503, detail="Server is low on GPU memory. Please try again in a few moments.")
    
    # Acquire lock to prevent concurrent use
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is currently processing another request. Please try again shortly.")
    
    try:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": request.system_prompt}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": request.user_prompt}]
            }
        ]
        
        generated_text = model_manager.generate(messages, request.max_tokens, True)
        
        return {"generated_text": generated_text}
    finally:
        # Always release the lock
        model_lock.release()
        # Schedule cleanup
        background_tasks.add_task(lambda: torch.cuda.empty_cache())

# Stream generator function
async def stream_generator(streamer):
    """Convert TextIteratorStreamer to an async generator for FastAPI streaming"""
    try:
        for text in streamer:
            # Yield chunks in the format expected by the client
            yield f"{text}"
            await asyncio.sleep(0)  # Yield control to the event loop
        
        # Send end of stream marker
        yield "data: [DONE]\n\n"
    finally:
        # Ensure cleanup happens
        model_manager.concurrent_tasks -= 1
        model_manager.last_used = time.time()
        torch.cuda.empty_cache()
        gc.collect()
        request_limiter.release()

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint supporting multiple message contexts
    """
    # Check if we have enough memory
    # if not model_manager.check_memory(2.0):
    #     background_tasks.add_task(model_manager.cleanup_if_needed)
    #     raise HTTPException(status_code=503, detail="Server is low on GPU memory. Please try again in a few moments.")
    
    # Try to acquire a request slot with timeout
    if not request_limiter.acquire(blocking=True, timeout=1.0):
        raise HTTPException(
            status_code=429, 
            detail=f"Server is at capacity ({request_limiter.max_concurrent} concurrent requests). Please try again shortly."
        )
    
    try:
        if request.stream:
            # Streaming mode
            streamer = model_manager.stream_generate(
                request.messages, 
                request.max_tokens,
                request.temperature
            )
            
            # Return a streaming response
            return StreamingResponse(
                stream_generator(streamer),
                media_type="text/event-stream"
            )
        else:
            generated_text = model_manager.generate(request.messages, request.max_tokens, False)
            
            return {"generated_text": generated_text}
    except Exception as e:
        # Always release the request slot in case of error
        request_limiter.release()
        raise e
    finally:
        # For non-streaming responses, release here
        if not request.stream:
            request_limiter.release()
        
        # Schedule cleanup
        background_tasks.add_task(lambda: torch.cuda.empty_cache())

@app.get("/status")
async def get_status():
    """
    Get server status and memory usage
    """
    limiter_stats = request_limiter.get_stats()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - allocated
        
        return {
            "status": "running",
            "gpu_memory": {
                "total_gb": round(total, 2),
                "used_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(free, 2),
                "processing": model_manager.processing
            },
            "processing": model_manager.processing,
            "concurrency": limiter_stats
        }
    else:
        return {
            "status": "running", 
            "device": "cpu", 
            "processing": model_manager.processing,
            "concurrency": limiter_stats
        }

# New endpoint to configure concurrency
@app.post("/config")
async def set_configuration(config: ServerConfig):
    """
    Update server configuration including maximum concurrent requests
    """
    global request_limiter
    
    # Create a new request limiter with the updated max_concurrent value
    new_limiter = RequestLimiter(max_concurrent=config.max_concurrent_requests)
    
    # Wait until all current requests are processed before switching
    while request_limiter.current_requests > 0:
        await asyncio.sleep(0.1)
    
    # Replace the limiter
    request_limiter = new_limiter
    
    return {"success": True, "message": f"Server now accepts {config.max_concurrent_requests} concurrent requests"}


@app.on_event("shutdown")
def shutdown_event():
    """
    Cleanup resources on server shutdown
    """
    model_manager.full_cleanup()

import asyncio

@app.on_event("startup")
async def startup_event():
    """
    Setup model and start memory monitoring
    """
    # Start memory monitoring task
    asyncio.create_task(memory_monitor())

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8080))

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker to prevent loading multiple model copies
    )