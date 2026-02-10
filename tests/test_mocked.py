import pytest
from unittest.mock import patch, MagicMock
import json

@patch('inference.call_ollama_chat')
def test_rag_with_mocked_llm(mock_ollama,client,sample_answer,sample_question):
    mock_response = {
        "score":6,
        "weak_points": ["не указал regularization"],
        "missed_topics": ["dropout", "batch normalization"],
        "correct_points": ["правильно объяснил backpropagation"],
        "full_correct_answer": "Полный правильный ответ...",
        "final_feedback": "Хорошо, но можно лучше"
    }

    mock_ollama.return_value = json.dumps(mock_response)

    test_data = {
        "question": sample_question,
        "user_answer": sample_answer   
    }

    response = client.post('/query',json=(test_data))
    assert response.status_code == 200

    mock_ollama.assert_called_once()