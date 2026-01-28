from storage.load import load_indices, load_chunks

CATEGORIES = [
    "math", "ml", "ops",
    "python", "softskills",
    "statistics_probabilities"
]

category_indices, category_id_maps = load_indices(CATEGORIES)
chunked_texts = load_chunks('chunked_texts.json')

if category_indices is None:
    raise RuntimeError("FAISS индексы не найдены. Сначала запусти build_index.py")
