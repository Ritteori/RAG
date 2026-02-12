from fastapi import FastAPI
from pydantic import BaseModel
import random

from app.core.container import create_rag_service

app = FastAPI(title='RAG implementation')

rag_service, logger = create_rag_service()

with open('questions.txt','r',encoding='utf-8') as f:
    ALL_QUESTIONS = f.readlines()
class Answer(BaseModel):
    user_answer: str
class QueryRAG(BaseModel):
    question: str
    user_answer: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/get_question")
def get_random_question():
    q = random.choice(ALL_QUESTIONS).strip()
    logger.info(f"Вытянули вопрос длиной={len(q)}")
    logger.debug(f"Question: {q}")
    return {"question": q}

@app.post("/query")
def query_rag(payload: QueryRAG):
    logger.info("Получен запрос пользователя")
    question = payload.question
    answer = payload.user_answer

    response_text = rag_service.core(question,answer)

    return {"answer": response_text}