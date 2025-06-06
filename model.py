import requests
import sys
from typing import List, Any

model_name = "gemma3:4b-it-qat"

def model_inference(messages: List[Any], max_tokens: int, stream=True):
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name, 
            "messages": messages, 
            "stream": stream,
            "num_predict": max_tokens
        },
        stream=stream,
    )

    return resp