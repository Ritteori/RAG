from prompt_builder import inference_mvp
from retriever import retrive
from load_index import category_indices, category_id_maps, chunked_texts
from logger import setup_logger, timed
from utils.ollama_client import call_ollama_chat

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import re
import random

logger = setup_logger('RAG')
logger.info('Сервер запущен')

embed_model = SentenceTransformer( "sentence-transformers/all-MiniLM-L6-v2", cache_folder="/data/models_cache" )

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

CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')
def contains_chinese(text: str) -> bool:
    return bool(CHINESE_RE.search(text))

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

    with timed(logger, 'Top k contexts'):
        top_k_contexts = retrive(embed_model, category_indices, category_id_maps, chunked_texts, logger, question)
    logger.info(f"Найдено {len(top_k_contexts)} контекстов")
    logger.debug(f"Top-k scores: {top_k_contexts}")

    with timed(logger, 'Final prompt'):
        final_prompt = normalize_text(inference_mvp(top_k_contexts, question, answer))
    logger.debug(f'Final prompt: {final_prompt}')

    with timed(logger, 'Model response'):
        response_text = call_ollama_chat(final_prompt)
    logger.debug(f'Model response: {response_text}')

    MAX_RETRIES = 5
    REGENERATE_PREFIX = "В предыдущем ответе был не русский текст. Перепиши ответ полностью, только на русском."
    retries = 0
    while contains_chinese(response_text) and retries < MAX_RETRIES:
        logger.info('Regenerating answer due to incorrect output...')
        retries += 1
        response_text = call_ollama_chat(REGENERATE_PREFIX + final_prompt)

    if contains_chinese(response_text):
        return {"answer": "Ошибка генерации: модель нарушает языковое ограничение."}

    logger.info("Ответ успешно сгенерирован")
    return {"answer": response_text}