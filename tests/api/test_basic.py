def test_healthcheck(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"ok"}

def test_get_question(client):
    response = client.get("/get_question")
    assert response.status_code == 200
    data = response.json()
    assert 'question' in data
    assert isinstance(data["question"],str)

def test_query_rag_empty_answer(client,sample_question):
    test_data = {
        "question": sample_question,
        "user_answer": ""
    }
    response = client.post("/query",json=test_data)
    assert response.status_code == 422

def test_query_rag_empty_question(client,sample_answer):
    test_data = {
        "question" : "",
        "user_answer" : sample_answer
    }

    response = client.post("/query",json=test_data)
    assert response.status_code == 422

def test_query_rag_long_answer_error(client,sample_question):
    test_data = {
        'questions' : sample_question,
        'user_answer' : 'a' * 10000
    }

    response = client.post('/query',json=test_data)
    assert response.status_code == 422

def test_query_rag_long_question_error(client,sample_answer):
    test_data = {
        'questions' : 'a' * 6000,
        'user_answer' : sample_answer
    }

    response = client.post('/query',json=test_data)
    assert response.status_code == 422
