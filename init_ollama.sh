#!/bin/bash

echo "Starting Ollama server..."
/bin/ollama serve &

echo "Waiting for Ollama to initialize..."
sleep 35  # Даем больше времени на инициализацию

echo "Pulling model qwen2.5:7b..."
ollama pull qwen2.5:7b || echo "Model pull completed or failed"

echo "Available models:"
ollama list || echo "Cannot list models"

echo "Ollama is ready!"
tail -f /dev/null