from unittest.mock import MagicMock
import time

def test_ollama_empty_response(mocker, ollama_client, sample_question):
    mock_response = mocker.patch.object(
        ollama_client,
        "_get_content",
        side_effect = RuntimeError("LLM Unavailable or invalid response")
    )

    response = ollama_client.call_ollama_chat(sample_question)
    empty_response = ollama_client._get_empty_response()
    assert empty_response == response
    assert ollama_client.failures_count == 1

def test_ollama_circuit_break(mocker, ollama_client, sample_question):
    mock_response = mocker.patch.object(
        ollama_client,
        "_get_content",
        side_effect = RuntimeError("LLM Unavailable or invalid response")
    )
    
    for _ in range(3):
        ollama_client.call_ollama_chat(sample_question)
    
    assert ollama_client.is_broken == True
    assert ollama_client.failures_count >= 3

def test_broken_ollama(ollama_client, sample_question):

    ollama_client.is_broken = True
    ollama_client.last_failure_time = time.time()

    response = ollama_client.call_ollama_chat(sample_question)
    empty_response = ollama_client._get_empty_response()

    assert response == empty_response

def test_retry_success_covery(mocker, ollama_client, sample_question):

    mock_response = mocker.patch.object(
        ollama_client,
        "_get_content",
        side_effect = [
            RuntimeError("LLM Unavailable or invalid response"),
            RuntimeError("LLM Unavailable or invalid response"),
            {"content":"Valid answer"}
        ]
    )

    response = ollama_client.call_ollama_chat(sample_question)

    assert response["content"] == "Valid answer"
    assert ollama_client.failures_count == 0
    assert ollama_client.is_broken == False