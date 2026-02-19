import subprocess
import time
import requests
import pytest

pytestmark = pytest.mark.integration

def test_docker_containers_up():

    result = subprocess.run(
        ["docker","ps","--format","{{.Names}}"],
        capture_output=True,text=True
    )
    containers = result.stdout.split()

    assert any("rag-api" in c for c in containers), "Can't find container rag-rag-api-1"
    assert any("rag-ollama-1" in c for c in containers), "Can't find container rag-ollama-1"

def test_ollama_health():
    time.sleep(5)
    response = requests.get("http://localhost:11434/api/tags", timeout=10)
    assert response.status_code == 200
