def test_healthcheck(client):  ### ДОБАВИТЬ В INFERENCE АСИНХРОННЫЙ ОБРАБОТЧИК GET!!!!!!!!!!!1
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"ok"}

def test_get_question(client):
    response = client.get("/get_question")
    assert response.status_code == 200
    data = response.json()
    assert 'question' in data
    assert isinstance(data["question"],str)

def test_query_rag_with_empty_answer(client,sample_question):
    test_data = {
        "question": sample_question,
        "user_answer": ""
    }
    response = client.post("/query",json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data

    answer_data = data["answer"]
    structure = ["score","weak_points","missed_topics","correct_points","full_correct_answer","final_feedback"]
    for field in structure:
        assert field in answer_data
