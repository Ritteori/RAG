from prompt_builder import inference_mvp

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import json
import re
import random

app = FastAPI(title='RAG implementation')
with open('questions.txt','r',encoding='utf-8') as f:
    ALL_QUESTIONS = f.readlines()
class Answer(BaseModel):
    user_answer: str
class QueryRAG(BaseModel):
    question: str
    user_answer: str

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

@app.get("/get_question")
def get_random_question():
    q = random.choice(ALL_QUESTIONS).strip()
    return {"question": q}

@app.post("/query")
def query_rag(payload: QueryRAG):
    question = payload.question
    answer = payload.user_answer

    final_prompt = normalize_text(inference_mvp(question, answer))
    print(final_prompt)

    response_text = call_ollama_chat(final_prompt)

    MAX_RETRIES = 5
    REGENERATE_PREFIX = "В предыдущем ответе был не русский текст. Перепиши ответ полностью, только на русском."
    retries = 0
    while contains_chinese(response_text) and retries < MAX_RETRIES:
        retries += 1
        response_text = call_ollama_chat(REGENERATE_PREFIX + final_prompt)

    if contains_chinese(response_text):
        return {"answer": "Ошибка генерации: модель нарушает языковое ограничение."}

    return {"answer": response_text}
