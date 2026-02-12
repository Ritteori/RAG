import faiss
import numpy as np

def embed(embed_model, chunked_texts):
    """
    Compute embeddings for text chunks and build FAISS indices by category.

    Encodes all chunk texts using the provided sentence-transformer model,
    groups chunks by category, and creates a separate FAISS index for
    each category.

    Args:
        embed_model: SentenceTransformer model used for encoding.
        chunked_texts (dict): Mapping chunk_id -> chunk metadata.

    Returns:
        tuple:
            embedded_texts (dict): chunk_id -> embedding vector
            chunks_by_category (dict): category -> list of chunk_ids
            category_indices (dict): category -> FAISS index
            category_id_maps (dict): category -> faiss_id -> chunk_id
    """
    
    keys = list(chunked_texts.keys())
    texts = [chunked_texts[k]['text'] for k in keys]

    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embedded_texts = {keys[i]: {'embeddings': embeddings[i]} for i in range(len(keys))}

    chunks_by_category = {
        'math':[],
        'ml':[],
        'ops':[],
        'python':[],
        'softskills':[],
        'statistics_probabilities':[]
    }

    for key, chunk_info in chunked_texts.items():
        
        topic = chunk_info['category']
        chunks_by_category[topic].append(key)

    category_indices = {}
    category_id_maps = {}

    for category, keys in chunks_by_category.items():

        embeddings = np.array(
            [embedded_texts[k]["embeddings"] for k in keys],
            dtype="float32"
        )

        faiss.normalize_L2(embeddings)

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        category_indices[category] = index
        category_id_maps[category] = {
            i: keys[i] for i in range(len(keys))
        }

    return embedded_texts, chunks_by_category, category_indices, category_id_maps