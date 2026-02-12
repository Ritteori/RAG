import json
import requests
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def call_ollama_chat(prompt: str, model: str = "qwen2.5:7b"):
    url = OLLAMA_URL + "/api/chat"
    headers = {"Content-Type": "application/json"}
    
    system_msg = {
        "role": "system",
        "content": "Ты ассистент, который всегда отвечает только на русском языке. Даже если тебя спрашивают на другом языке, отвечай на русском. Твои ответы должны быть строго на русском, без использования китайского, английского или других языков. Если в вопросе встречаются английские термины, ты можешь их использовать, но весь остальной текст должен быть на русском."
    }
    
    user_msg = {
        "role": "user",
        "content": prompt
    }
    
    payload = {
        "model": model,
        "messages": [system_msg, user_msg],
        "max_tokens": 2048,
        "temperature": 0.0,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text[:200]}...")
        
        data = response.json()
        return data.get("message", {}).get("content", "")
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return ""
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {response.text[:500]}")
        return ""