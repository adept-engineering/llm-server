import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from transformers import pipeline
import gc
import uvicorn
import os
import threading
import time

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
            torch_dtype=torch.bfloat16,
            model_kwargs={'low_cpu_mem_usage': True}
        )
        self.last_used = time.time()
        self.processing = False
    
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
                    torch_dtype=torch.bfloat16,
                    model_kwargs={'low_cpu_mem_usage': True}
                )
    
    def generate(self, messages, max_tokens):
        try:
            self.processing = True
            
            # Check memory before generation
            if not self.check_memory(2.0):
                self.cleanup_if_needed()
                
            # Calculate a reasonable amount of tokens based on input
            input_tokens = len(str(messages)) // 4  # Rough estimate
            safe_max_tokens = min(max_tokens, 2048, 4096 - input_tokens)
            
            # Perform generation with memory-efficient settings
            output = self.pipe(
                messages, 
                max_new_tokens=safe_max_tokens,
                do_sample=True,
                temperature=0.7,  # Lower temperature for more deterministic outputs
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
            self.processing = False
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
    max_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]
    max_tokens: int = Field(default=1024, ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)

# FastAPI app initialization
app = FastAPI(
    title="Gemma Text Generation API",
    description="Memory-optimized text generation API using Gemma model"
)

# Global model manager
model_manager = ModelManager()

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
        
        generated_text = model_manager.generate(messages, request.max_tokens)
        
        return {"generated_text": generated_text}
    finally:
        # Always release the lock
        model_lock.release()
        # Schedule cleanup
        background_tasks.add_task(lambda: torch.cuda.empty_cache())

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Chat endpoint supporting multiple message contexts
    """
    # Check if we have enough memory
    if not model_manager.check_memory(2.0):
        background_tasks.add_task(model_manager.cleanup_if_needed)
        raise HTTPException(status_code=503, detail="Server is low on GPU memory. Please try again in a few moments.")
    
    # Acquire lock to prevent concurrent use
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is currently processing another request. Please try again shortly.")
    
    try:
        generated_text = model_manager.generate(request.messages, request.max_tokens)
        
        return {"generated_text": generated_text}
    finally:
        # Always release the lock
        model_lock.release()
        # Schedule cleanup
        background_tasks.add_task(lambda: torch.cuda.empty_cache())

@app.get("/status")
async def get_status():
    """
    Get server status and memory usage
    """
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
            }
        }
    else:
        return {"status": "running", "device": "cpu", "processing": model_manager.processing}

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