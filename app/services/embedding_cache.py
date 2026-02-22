from app.api.metrics import cache_count

from pathlib import Path
import numpy as np
import json
import time
import os

class EmbeddingCache():
    def __init__(self, logger, cache_path: str = "cache/embeddings.json", ttl_seconds: int = 3600):
        
        ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        self.full_cache_path = ROOT_DIR/cache_path
        self.full_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.logger = logger
        self.cache = {}
        self._load()

        cache_count.set(len(self.cache))

    def _load(self):
        
        if os.path.exists(self.full_cache_path):
            
            try:
                with open(self.full_cache_path,'r',encoding='utf-8') as f:
                    raw = json.load(f)
            except Exception as e:
                self.logger.debug(f"Файл {self.full_cache_path} пуст или поврежден. Инициализируем пустой кэш.")
                raw = {}

            now_time = time.time()
            for question, info in raw.items():
                embedding, ts = info.values()
                if now_time - ts >= self.ttl_seconds:
                    continue
                self.cache[question] = {
                    "embedding":np.array(embedding, dtype="float32"),
                    "created_at":ts}
                
    def save(self):
        
        to_json = {
            question: {
                "embedding" : info.get('embedding').tolist() if isinstance(info.get('embedding'),np.ndarray) else info.get('embedding'),
                "created_at" : info.get('created_at')
            }
            for question, info in self.cache.items()
        }

        with open(self.full_cache_path,mode="w",encoding='utf-8') as f:
            json.dump(to_json,f,ensure_ascii=False,indent=2)

    def get(self, question):
        info = self.cache.get(question)
        now_time = time.time()
        if info:
            embedding, ts = info.values()
            if now_time - ts >= self.ttl_seconds:
                del(self.cache[question])
                cache_count.set(len(self.cache))
            else:
                return embedding

    def set(self, question, embedding):
        self.cache[question] = {
            "embedding":embedding,
            "created_at":time.time()
        }
        self.save()
        cache_count.set(len(self.cache))

    def clear_expired(self):

        to_delete = []
        now_time = time.time()
        for question, info in self.cache.items():

            ts = info.get('created_at', 0)

            if now_time - ts >= self.ttl_seconds:
                to_delete.append(question)

        for question in to_delete:
            del(self.cache[question])
            cache_count.set(len(self.cache))
