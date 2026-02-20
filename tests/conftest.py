import pytest
from fastapi.testclient import TestClient
from app.api.inference import app
from app.utils.ollama_client import OllamaCLient
from app.services.embedding_cache import EmbeddingCache
from types import SimpleNamespace
from unittest.mock import MagicMock
import logging

class FakeRAGService:
    def query(self, question: str):
        return "fake answer"

@pytest.fixture
def client(test_logger):
    client = TestClient(app)

    app.state.rag_service = FakeRAGService()
    app.state.logger = test_logger
    app.state.questions = ["Test question?"]

    return client

@pytest.fixture
def sample_question():
    return "Что такое градиентный спуск."

@pytest.fixture
def sample_answer():
    return "Это метод оптмизации для минимизации ошибки."

@pytest.fixture
def test_config():
    config = SimpleNamespace(
        retrieval = SimpleNamespace(
            search_k = 3,
            top_k_best_contexts = 10,
            neighbour_window = 1
        ),
        limits = SimpleNamespace(
            max_retries = 5
        ),
        keywords = SimpleNamespace(
            math = ['aboba'],
            statistics_probabilities = ['aboba'],
            ml = ['aboba'],
            python = ['aboba'],
            ops = ['aboba'],
            softskills = ['aboba']
        ),
        logger = SimpleNamespace(
            logger_name = 'RAG'
        ),

        categories = ['math','ops','statistics_probabilities','ml','python','softskills'],
        OLLAMA_MODEL='qwen2.5:7b',
        ENV = 'dev',
        API_PORT = 8000,
        OLLAMA_EXTERNAL_PORT = 11435,
        OLLAMA_URL = 'http://localhost:11434',
        LOG_LEVEL = 'DEBUG',
        MODELS_CACHE_PATH = './models_cache',
        EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    )

    return config

@pytest.fixture
def test_logger():
    return logging.getLogger("test")

@pytest.fixture
def ollama_client(test_logger):
    return OllamaCLient(logger=test_logger)

@ pytest.fixture
def chunked_texts():
    return {
  "data/math/vectors_matrix_definition.txt::0": {
    "text": "1. что такое вектор в математике? \nв математике вектор — это упорядоченный набор чисел, представляющий \nэлемент векторного пространства. \nон имеет направление и величину и может быть записан как столбец или строка \nиз чисел. векторы часто интерпретируют как точки или стрелки в пространстве \nℝⁿ, где каждое число — это координата в соответствующем измерении. \n",
    "path": "/data/projects/RAG/data/math/vectors_matrix_definition.txt",
    "category": "math",
    "chunk_index": 0,
    "source_file": "/data/math/vectors_matrix_definition.txt"
  },
  "data/math/vectors_matrix_definition.txt::1": {
    "text": "1. что такое вектор в математике? \nв математике вектор — это упорядоченный набор чисел, представляющий \nэлемент векторного пространства. \nон имеет направление и величину и может быть записан как столбец или строка \nиз чисел. векторы часто интерпретируют как точки или стрелки в пространстве \nℝⁿ, где каждое число — это координата в соответствующем измерении. \n",
    "path": "/data/projects/RAG/data/math/vectors_matrix_definition.txt",
    "category": "math",
    "chunk_index": 1,
    "source_file": "/data/math/vectors_matrix_definition.txt"
  },
    "data/math/vectors_matrix_definition.txt::2": {
    "text": "1. что такое вектор в математике? \nв математике вектор — это упорядоченный набор чисел, представляющий \nэлемент векторного пространства. \nон имеет направление и величину и может быть записан как столбец или строка \nиз чисел. векторы часто интерпретируют как точки или стрелки в пространстве \nℝⁿ, где каждое число — это координата в соответствующем измерении. \n",
    "path": "/data/projects/RAG/data/math/vectors_matrix_definition.txt",
    "category": "math",
    "chunk_index": 2,
    "source_file": "/data/math/vectors_matrix_definition.txt"
  }
}

@pytest.fixture
def embedding_cache(test_logger):
    return EmbeddingCache(logger=test_logger, cache_path="cache/test_embeddings.json", ttl_seconds=1)