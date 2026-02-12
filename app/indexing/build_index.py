from sentence_transformers import SentenceTransformer
from app.indexing.chunker import build_chunks
from app.indexing.embedder import embed
from app.core.settings  import load_chunk_config

from storage.save import save

chunks_config = load_chunk_config()

chunk_size = chunks_config.chunking.chunk_size
overlap = chunks_config.chunking.overlap
max_length_before_division = chunks_config.chunking.max_length_before_division
minimal_length = chunks_config.chunking.minimal_length
encoder_model = chunks_config.chunking.encoder_model
encoder_model_cache = chunks_config.chunking.encoder_model_cache

embed_model = SentenceTransformer(
    encoder_model,
    cache_folder=encoder_model_cache
)

chunked_texts = build_chunks(overlap,chunk_size,max_length_before_division,minimal_length)
embedded_texts, chunks_by_category, category_indices, category_id_maps = embed(embed_model, chunked_texts)
save(chunked_texts, category_indices, category_id_maps)