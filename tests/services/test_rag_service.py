from app.services.rag_service import RAGService
from unittest.mock import MagicMock

def test_api(
    mocker,
    test_config,
    test_logger,
    sample_answer,
    sample_question,
    ollama_client
):
    fake_retriever = MagicMock()
    fake_retriever.retrieve.return_value = [
        ("fake context", 0.5),
        ("aboba", 0.8),
    ]

    service = RAGService(test_config, test_logger, fake_retriever, ollama_client)
    mock_llm = mocker.patch.object(
        service.ollama_client,
        "call_ollama_chat",
        return_value={
            "score": 8,
            "weak_points": [],
            "missed_topics": [],
            "correct_points": [],
            "full_correct_answer": "text",
            "final_feedback": "good"
        }
    )

    response_text = service.core(sample_question, sample_answer)

    assert "score" in response_text
    assert "weak_points" in response_text
    assert "missed_topics" in response_text
    assert "correct_points" in response_text
    assert "full_correct_answer" in response_text
    assert "final_feedback" in response_text

    fake_retriever.retrieve.assert_called_once()
    mock_llm.assert_called_once()