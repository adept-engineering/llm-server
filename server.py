import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from transformers import pipeline
import gc
import uvicorn

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
            device="cuda",
            torch_dtype=torch.bfloat16,
            # Optionally add model-specific memory optimization kwargs
            # model_kwargs={'use_cache': True}
        )
    
    def generate(self, messages, max_tokens):
        try:
            # Perform generation with memory-efficient settings
            output = self.pipe(
                messages, 
                max_new_tokens=max_tokens,
                # Additional generation optimization
                num_return_sequences=1,
                clean_up_tokenization_spaces=True
            )
            return output[0]['generated_text'][-1]["content"]
        except Exception as e:
            # Log the error (in a production setup, use proper logging)
            print(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def cleanup(self):
        # Manual memory cleanup
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()

# Pydantic models for request validation
class GenerateRequest(BaseModel):
    system_prompt: str = Field(default="You are a helpful assistant.")
    user_prompt: str
    max_tokens: int = Field(default=2048, ge=1, le=4096)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]
    max_tokens: int = Field(default=1024, ge=1, le=4096)

# FastAPI app initialization
app = FastAPI(
    title="Gemma Text Generation API",
    description="Optimized text generation API using Gemma model"
)

# Global model manager
model_manager = ModelManager()

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Generate text based on system and user prompts
    """
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

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint supporting multiple message contexts
    """
    generated_text = model_manager.generate(request.messages, request.max_tokens)
    
    return {"generated_text": generated_text}

@app.on_event("shutdown")
def shutdown_event():
    """
    Cleanup resources on server shutdown
    """
    model_manager.cleanup()

# Optional: Add startup event for initial model loading
@app.on_event("startup")
def startup_event():
    """
    Ensure model is loaded and cached on startup
    """
    # The ModelManager constructor will initialize the model
    pass

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = 8080

    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Single worker to prevent loading multiple model copies
    )


# Additional performance and memory optimization hints:
# 1. Use uvicorn for ASGI server: uvicorn main:app --workers 1 --host 0.0.0.0 --port 8000
# 2. Consider using torch.compile() for further optimization if using PyTorch 2.0+
# 3. Monitor GPU memory usage and adjust max_tokens accordingly