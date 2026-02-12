from pydantic import BaseModel
from pydantic_settings import BaseSettings
import yaml

class RetrievalConfig(BaseModel):
    search_k: int
    top_k_best_contexts: int
    neighbour_window: int

class Limits(BaseModel):
    max_retries: int

class Logger(BaseModel):
    logger_name: str

class KeyWord(BaseModel):
    math: list[str]
    ml: list[str]
    ops: list[str]
    python: list[str]
    statistics_probabilities: list[str]
    softskills: list[str]

class AppConfig(BaseSettings):
    ENV: str = "dev"
    API_PORT: int
    OLLAMA_EXTERNAL_PORT: int
    OLLAMA_URL: str
    OLLAMA_MODEL: str
    LOG_LEVEL: str="INFO"
    MODELS_CACHE_PATH: str
    EMBEDDING_MODEL: str

    retrieval: RetrievalConfig
    limits: Limits
    logger: Logger
    categories: list[str]
    keywords: KeyWord

    class Config:
        env_file ='.env'


class ChunkingConfig(BaseModel):
    chunk_size: int
    overlap: int
    max_length_before_division: int
    minimal_length: int
    encoder_model: str
    encoder_model_cache: str

class ChunksConfig(BaseSettings):

    chunking:ChunkingConfig


def load_config(path="config.yaml")->AppConfig:

    with open(path,"r",encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)

def load_chunk_config(path="config_chunker.yaml")->ChunksConfig:

    with open(path,"r",encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ChunksConfig(**data)