from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import random

from app.core.container import create_rag_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service, logger = create_rag_service()

    with open('questions.txt','r',encoding='utf-8') as f:
        ALL_QUESTIONS = f.readlines()   

    app.state.rag_service = rag_service
    app.state.logger = logger
    app.state.questions = ALL_QUESTIONS

    app.state.logger.info("RAG Service and Questions loaded successfully")

    yield

    app.state.logger.info("Shutting down...")

app = FastAPI(title='RAG implementation', lifespan=lifespan)

class QueryRAG(BaseModel):
    question: str = Field(...,min_length=3,max_length=400)
    user_answer: str = Field(...,min_length=1,max_length=3000)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/get_question")
def get_random_question(request: Request):
    q = random.choice(request.app.state.questions).strip()
    request.app.state.logger.info(f"Вытянули вопрос длиной={len(q)}")
    request.app.state.logger.debug(f"Question: {q}")
    return {"question": q}

@app.post("/query")
def query_rag(payload: QueryRAG, request: Request):
    request.app.state.logger.info("Получен запрос пользователя")
    question = payload.question
    answer = payload.user_answer

    response_text = request.app.state.rag_service.core(question,answer)

    return {"answer": response_text}