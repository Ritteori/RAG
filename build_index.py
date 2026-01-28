from sentence_transformers import SentenceTransformer
from chunker import build_chunks
from embedder import embed

from storage.save import save

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/data/models_cache"
)

chunked_texts = build_chunks()
embedded_texts, chunks_by_category, category_indices, category_id_maps = embed(model, chunked_texts)
save(chunked_texts, category_indices, category_id_maps)