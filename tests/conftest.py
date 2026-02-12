import pytest
from fastapi.testclient import TestClient
from app.api.inference import app
from types import SimpleNamespace
from unittest.mock import MagicMock
import logging


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

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