from prompt_builder import inference_mvp

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import json
import re

app = FastAPI(title='RAG implementation')

class Query(BaseModel):
    prompts: List[str]

def normalize_text(text):
    # удаляем непечатные символы и заменяем битые UTF-8
    text = text.replace("\ufffd", "")  # битые символы
    text = text.replace("\x00", "")
    text = text.replace("\n\n", "\n") # лишние переносы
    return text

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

CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')
def contains_chinese(text: str) -> bool:
    return bool(CHINESE_RE.search(text))

@app.post("/query")
def query_rag(query: Query):
    final_prompt_with_thrash = inference_mvp(query.prompts)
    final_prompt = normalize_text(final_prompt_with_thrash)
    answer = call_ollama_chat(final_prompt)
    
    MAX_RETRIES = 5
    REGENERATE_PREFIX = (
    "В ПРЕДЫДУЩЕМ ОТВЕТЕ БЫЛ НЕРУССКИЙ ТЕКСТ.\n"
    "ЭТО ЗАПРЕЩЕНО.\n\n"
    "Перепиши ответ полностью.\n"
    "Используй ТОЛЬКО русский язык.\n\n"
)
    retries = 0
    while contains_chinese(answer) and retries < MAX_RETRIES:
        retries += 1
        answer = call_ollama_chat(REGENERATE_PREFIX + final_prompt)

    if contains_chinese(answer):
        return {
            "answer": "Ошибка генерации: модель нарушает языковое ограничение."
        }

    return {"answer": answer}