import json
import time


def test_set(embedding_cache):
    question = "q1"
    embedding = [1, 2, 3]

    embedding_cache.set(question, embedding)

    assert question in embedding_cache.cache
    assert embedding_cache.get(question) == embedding


def test_save(embedding_cache):
    question = "q2"
    embedding = [1, 2, 3]

    embedding_cache.set(question, embedding)

    with open(embedding_cache.full_cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert question in data
    assert data[question]["embedding"] == embedding


def test_clear_expired(embedding_cache):
    question = "q3"
    embedding = [1, 2, 3]

    embedding_cache.set(question, embedding)

    time.sleep(2)
    embedding_cache.clear_expired()

    assert question not in embedding_cache.cache


def test_get(embedding_cache):
    question = "q4"
    embedding = [1, 2, 3]

    embedding_cache.set(question, embedding)

    cached_embedding = embedding_cache.get(question)

    assert cached_embedding == embedding
