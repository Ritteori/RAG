import json
import requests

def call_ollama_chat(prompt: str, model: str = "qwen2.5:7b"):
    url = "http://localhost:11434/api/chat"
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
        "stream": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    data = response.json()
    return data.get("message", {}).get("content", "")
