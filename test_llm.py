import requests

MODEL_NAME = "qwen3:4b"  # or "qwen3:4b" if that is what you have loaded
OLLAMA_URL = "http://localhost:11434/api/generate"

print(f"Testing connection to Ollama ({MODEL_NAME})...")
try:
    res = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            
            "prompt": "Say: LLM is working properly!",
            "stream": False
        },
        timeout=15
    )
    print("HTTP Status Code:", res.status_code)
    print("Response Body:", res.text)
except Exception as e:
    print("Connection Failed:", e)