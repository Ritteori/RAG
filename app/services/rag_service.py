from app.core.logger import timed
from app.utils.ollama_client import call_ollama_chat
from app.prompts.prompt_builder import inference_mvp

import re

class RAGService():
    def __init__(self, config, logger, retriever):

        self.CHINESE_RE = re.compile(r'[\u4e00-\u9fff]')

        self.logger = logger
        self.logger.info('Сервер запущен')

        self.OLLAMA_MODEL = config.OLLAMA_MODEL
        self.MAX_RETRIES = config.limits.max_retries

        self.retriever = retriever

    def normalize_text(self,text):

        text = text.replace("\ufffd", "")
        text = text.replace("\x00", "")
        text = text.replace("\n\n", "\n")

        return text
    
    def contains_chinese(self, text: str) -> bool:
        return bool(self.CHINESE_RE.search(text))
    
    def core(self, question, answer):

        with timed(self.logger, 'Top k contexts'):
            top_k_contexts = self.retriever.retrieve(question)

        self.logger.info(f"Найдено {len(top_k_contexts)} контекстов")
        self.logger.debug(f"Top-k scores: {top_k_contexts}")

        with timed(self.logger, 'Final prompt'):
            final_prompt = self.normalize_text(inference_mvp(top_k_contexts, question, answer))
        self.logger.debug(f'Final prompt: {final_prompt}')

        with timed(self.logger, 'Model response'):
            response_text = call_ollama_chat(final_prompt,self.OLLAMA_MODEL)
        self.logger.debug(f'Model response: {response_text}')

        REGENERATE_PREFIX = "В предыдущем ответе был не русский текст. Перепиши ответ полностью, только на русском."
        retries = 0
        while self.contains_chinese(response_text) and retries < self.MAX_RETRIES:
            self.logger.info('Regenerating answer due to incorrect output...')
            retries += 1
            response_text = call_ollama_chat(REGENERATE_PREFIX + final_prompt,self.OLLAMA_MODEL)

        if self.contains_chinese(response_text):
            return {"answer": "Ошибка генерации: модель нарушает языковое ограничение."}

        self.logger.info("Ответ успешно сгенерирован")
        return response_text