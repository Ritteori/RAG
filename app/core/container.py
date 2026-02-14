from app.core.logger import setup_logger
from app.core.settings import load_config
from app.services.rag_service import RAGService
from app.services.index_loader import IndexLoader
from app.utils.ollama_client import OllamaCLient
from app.services.retriever import Retriever
from sentence_transformers import SentenceTransformer
import os

def create_rag_service():
    """
    return: (rag_service, logger)
    """

    config = load_config()

    LOGGER_NAME = config.logger.logger_name
    EMBEDDING_MODEL = config.EMBEDDING_MODEL
    MATH = config.keywords.math
    STATISTICS_PROBABILITIES = config.keywords.statistics_probabilities
    ML = config.keywords.ml
    PYTHON = config.keywords.python
    OPS = config.keywords.ops
    SOFTSKILLS = config.keywords.softskills

    OLLAMA_MODEL = config.OLLAMA_MODEL

    SEARCH_K = config.retrieval.search_k
    TOP_K_BEST_CONTEXTS = config.retrieval.top_k_best_contexts

    if os.path.exists("/app/models_cache"):
        cache_folder = "/app/models_cache"
    elif os.path.exists("models_cache"):
        cache_folder = "models_cache"
    else:
        home_cache = os.path.expanduser("~/.cache/rag_models")
        os.makedirs(home_cache, exist_ok=True)
        cache_folder = home_cache

    embed_model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=cache_folder)

    logger = setup_logger(LOGGER_NAME)
    logger.info('Сервер запущен')

    index_loader = IndexLoader(config)
    category_indices, category_id_maps, chunked_texts = index_loader.load()

    ollama_client = OllamaCLient(logger)

    retriever = Retriever(
        embed_model,category_indices,category_id_maps,chunked_texts,logger,
        MATH,ML,OPS,PYTHON,SOFTSKILLS,STATISTICS_PROBABILITIES,ollama_client,
        OLLAMA_MODEL, TOP_K_BEST_CONTEXTS, SEARCH_K
    )

    rag_service = RAGService(config, logger, retriever, ollama_client)

    return rag_service, logger