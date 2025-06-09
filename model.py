import requests
import sys
from typing import List, Any

#model_name = "gemma3:4b-it-qat"
model_name = "llama3.2:latest"

session = requests.Session()
session.timeout = (300, 300)

def model_inference(messages: List[Any], max_tokens: int, stream=True, model: str = model_name):
    resp = session.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model, 
            "messages": messages, 
            "stream": stream,
            "num_predict": max_tokens
        },
        stream=stream,
    )

    return resp