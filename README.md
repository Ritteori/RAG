# Production-Ready RAG API with GPU, Monitoring & Resilience

End-to-end Retrieval-Augmented Generation backend built with production architecture principles.

This project demonstrates:

* Clean service architecture
* LLM orchestration with circuit breaker
* GPU-based inference
* Observability (metrics + dashboards)
* Dockerized multi-service deployment
* CI/CD
* Config-driven design

Вот аккуратный, GitHub-friendly вариант 👇
(коротко, понятно, без лишнего шума)

---

## Run with GPU support

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

> Requires NVIDIA GPU and `nvidia-container-toolkit` installed on the host machine.

---

## What This Project Demonstrates

It is a production-style backend system designed to showcase:

* System design for LLM services
* Dependency injection patterns
* Resilience engineering (retry + circuit breaker)
* Monitoring & metrics engineering
* GPU orchestration in containers
* Configuration-driven infrastructure

---

## Architecture Overview

```
app/
 ├── api/            # FastAPI routes
 ├── services/       # RAGService, LLM client, Retriever
 ├── core/           # DI container, settings, logging
 ├── prompts/
 ├── indexing/ 
 └── utils/

docker-compose.yml
prometheus.yml
grafana dashboards
```

---

## Tech Stack

* FastAPI
* FAISS (vector search)
* Ollama (LLM inference)
* Sentence Transformers (embeddings)
* Prometheus (metrics)
* Grafana (dashboards)
* Docker Compose (multi-container setup)
* GitHub Actions (CI/CD)
* NVIDIA GPU support (optional)

---

## RAG Pipeline

1. User query
2. Embedding generation
3. FAISS vector retrieval
4. Context construction
5. LLM generation
6. Metrics logging
7. Circuit breaker update

---

## Resilience Features

* Circuit breaker for LLM failures
* Retry logic
* Health checks
* Graceful container dependency orchestration
* Embedding cache with TTL

---

## Observability

Exposed metrics include:

* Request latency
* LLM generation latency
* Embedding latency
* Cache hit rate
* Circuit breaker state

Prometheus scrapes `/metrics`.

Grafana dashboard visualizes:

* Latency trends
* Failure rate
* Cache efficiency
* System health

---

## Docker Setup

Multi-service architecture:

* rag_api
* ollama (GPU-enabled optional)
* model pre-puller
* prometheus
* grafana

GPU support available via docker-compose override.

---

## Testing

* Unit tests
* Integration tests
* Fixtures
* CI pipeline validates build + tests on every push

---

## Why This Project Matters

This repository demonstrates the ability to:

* Design LLM systems beyond notebooks
* Build observable ML services
* Handle GPU inference in containers
* Implement fault-tolerant service communication
* Structure real-world backend ML applications

---

## Status

Production-ready architecture for single-node LLM deployment.

Future improvements (optional):

* Hybrid search
* Reranking
* Distributed inference
* Load testing