import pytest
from fastapi.testclient import TestClient
from inference import app

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