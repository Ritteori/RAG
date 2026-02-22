from prometheus_client import Counter, Gauge, Histogram

retrieval_latency = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent retrieving contexts",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]
)

generate_latency = Histogram(
    "rag_generation_latency_seconds",
    "Time spent generating LLM response",
    buckets=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 90.0]
)

retrieved_contexts = Histogram(
    "rag_retrieved_contexts",
    "Number of contexts retrieved per request",
    buckets=[0,1,2,3,5,10]
)

circuit_breaker_state = Gauge(
    "rag_ollama_circuit_breaker_state",
    "1 if circuit breaker is open, 0 otherwise"
)

cache_count = Gauge(
    "cached_questions_count",
    "Number of cached questions"
)