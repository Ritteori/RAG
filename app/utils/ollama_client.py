import json
import requests
import os
import time

class OllamaCLient():
    def __init__(self, logger):

        OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.url = OLLAMA_URL + "/api/chat"
        self.MAX_RETRIES = 3
        self.MAX_FAILURES = 3
        self.failures_count = 0
        self.is_broken = False
        self.last_failure_time = 0
        self.refresh_time = 300

        self.logger = logger
    
    def _get_content(self, prompt: str, model: str = "qwen2.5:7b"):
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

        response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        
        raw_data = response.json()
        content_str = raw_data.get("message", {}).get("content", "").strip()
        
        if content_str.startswith("```"):
            content_str = content_str.strip("`").replace("json", "", 1).strip()

        content = json.loads(content_str)

        return content
    
    def _get_empty_response(self):

        return {
            "score": 0,
            "weak_points": [],
            "missed_topics": [],
            "correct_points": [],
            "full_correct_answer": "Ошибка сервиса",
            "final_feedback": "Превышено время ожидания или сервер недоступен."
        }    
                    
    def _execute_with_retry(self, prompt: str, model: str = "qwen2.5:7b"):
        
        backoff = 5
        for retry in range(self.MAX_RETRIES):
            try:
                content = self._get_content(prompt,model)
                if content:
                    return content
                
            except Exception as e:
                self.logger.debug(f"Retry {retry + 1}/{self.MAX_RETRIES}:{e}")
                time.sleep(backoff)

        raise RuntimeError("LLM Unavailable or invalid response")

    def _circuit_breaker(self, prompt: str, model: str = "qwen2.5:7b"):
        
        if self.is_broken:
            if time.time() - self.last_failure_time < self.refresh_time:
                self.logger.debug(f"Ollama client is broken. Wait {int(self.refresh_time - (time.time() - self.last_failure_time))}s")
                return self._get_empty_response()
            else:
                self.logger.info("Trying to recover Ollama connection...")
                self.is_broken = False

        try:
            content = self._execute_with_retry(prompt,model)
            self.failures_count = 0 
            return content

        except Exception as e:

            self.failures_count += 1
            self.last_failure_time = time.time()
            self.logger.error(f"Global failure {self.failures_count}/{self.MAX_FAILURES}: {e}")

            if self.failures_count >= self.MAX_FAILURES:
                self.is_broken = True
                self.logger.critical("CIRCUIT BREAKER TRIPPED!")
            
            return self._get_empty_response()
        
    def call_ollama_chat(self, prompt: str, model: str = "qwen2.5:7b"):

        return self._circuit_breaker(prompt, model)