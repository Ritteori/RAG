import numpy as np
from guess_cat import guess_categories

def search(prompts, embed_model, category_indices, category_id_maps, model="qwen2.5:7b", k=3):
    """
    Perform semantic search over FAISS indices with optional category routing.

    For each prompt, the function:
    - encodes the query
    - selects a category via keyword matching (if possible)
    - searches either a single category index or all indices (fallback)

    Args:
        prompts (list[str]): List of user queries.
        model: SentenceTransformer model.
        category_indices (dict): category -> FAISS index.
        category_id_maps (dict): category -> faiss_id -> chunk_id.
        k (int): Number of top results to retrieve.

    Returns:
        dict: prompt_id -> list of retrieved chunks with scores.
    """

    if isinstance(prompts, str):
        prompts = [prompts]

    results = {}

    for idx, prompt in enumerate(prompts):
        emb = embed_model.encode(prompt, normalize_embeddings=True)
        emb = np.array([emb], dtype="float32")

        results[idx] = []

        cat = guess_categories(prompt,model)

        # ---- CASE 1: категория определена ----
        if cat and cat in category_indices:
            index = category_indices[cat]
            distances, ids = index.search(emb, k)

            for score, i in zip(distances[0], ids[0]):
                results[idx].append({
                    "chunk_id": category_id_maps[cat][i],
                    "score": float(score),
                    "category": cat
                })

        # ---- CASE 2: fallback (по всем категориям) ----
        else:
            all_candidates = []

            for cat_name, index in category_indices.items():
                distances, ids = index.search(emb, k)

                for score, i in zip(distances[0], ids[0]):
                    all_candidates.append({
                        "chunk_id": category_id_maps[cat_name][i],
                        "score": float(score),
                        "category": cat_name
                    })

            # глобальная сортировка
            all_candidates.sort(key=lambda x: x["score"], reverse=True)

            # берём top-k глобально
            results[idx] = all_candidates[:k]

    return results

def group_by_files(searches, chunked_texts):
    """
    Group retrieved chunks by their source document.

    Args:
        searches (dict): Output of the search() function.
        chunked_texts (dict): chunk_id -> chunk metadata.

    Returns:
        dict: source_file -> sorted list of chunk metadata.
    """

    grouped = {}

    for objects in searches.values():
        for obj in objects:
            chunk_id = obj["chunk_id"]
            meta = chunked_texts[chunk_id]

            if meta["source_file"] not in grouped:
                grouped[meta["source_file"]] = []

            grouped[meta["source_file"]].append({
                "chunk_id": chunk_id,
                "chunk_index": meta["chunk_index"],
                "char_start": meta["char_start"],
                "char_end": meta["char_end"],
                "score": obj["score"]
            })

    for source in grouped:
        grouped[source].sort(key=lambda x: x["char_start"])

    return grouped

def find_neighbours(groups,chunked_texts):
    """
    Find neighbouring chunks around retrieved anchor chunks.

    For each anchor chunk, includes its previous and next chunks
    (if they exist) to preserve local context.

    Args:
        groups (dict): Output of group_by_files().
        chunked_texts (dict): chunk_id -> chunk metadata.

    Returns:
        dict: source_file -> anchor chunks and their context chunks.
    """

    neighbours = {}

    for source_file, chunks_info in groups.items():
        
        if source_file not in neighbours:
            neighbours[source_file] = {
                'anchor_chunks':[],
                'context_chunks':[]
            }

        for info in chunks_info:
            chunk_index = info['chunk_index']
            chunk_indices = [i for i in range(chunk_index-1,chunk_index+2)]

            neighbours[source_file]['anchor_chunks'].append(chunk_index)
            for index in chunk_indices:
                chunk_name = info['chunk_id'].split('::')[0] + f'::{index}'
                if chunk_name in chunked_texts:
                    neighbours[source_file]['context_chunks'].append(index)

    return neighbours

def build_context_texts(neighbours, chunked_texts):
    """
    Build contiguous context texts from neighbouring chunks.

    Merges consecutive chunks into coherent text blocks
    based on their chunk indices.

    Args:
        neighbours (dict): Output of find_neighbours().
        chunked_texts (dict): chunk_id -> chunk metadata.

    Returns:
        list[str]: List of merged context texts.
    """

    texts = []

    for file, info in neighbours.items():
        file_key = file[1:]

        context_indices = sorted(info["context_chunks"])

        current_text = ""
        prev_idx = None

        for idx in context_indices:
            chunk_path = f"{file_key}::{idx}"

            if chunk_path not in chunked_texts:
                continue

            if prev_idx is not None and idx != prev_idx + 1:
                texts.append(current_text)
                current_text = ""

            current_text += chunked_texts[chunk_path]["text"]
            prev_idx = idx

        if current_text:
            texts.append(current_text)

    return texts

def find_anchor_chunks_scores(searches, neighbours):
    """
    Extract similarity scores for anchor chunks.

    Args:
        searches (dict): Output of the search() function.
        neighbours (dict): Output of find_neighbours().

    Returns:
        tuple:
            chunk_score (dict): chunk_id -> similarity score
            best_contexts (list[float]): scores of anchor contexts
    """

    chunk_score = {}

    for chunks in searches.values():
        for c in chunks:
            chunk_score[c["chunk_id"]] = c["score"]

    best_contexts = []

    for file, neighbour in neighbours.items():
        file_key = file[1:]

        for anchor in neighbour["anchor_chunks"]:
            chunk_id = file_key + f"::{anchor}"
            best_contexts.append(chunk_score[chunk_id])

    return chunk_score, best_contexts

def find_top_k_contexts(contexts,best_contexts,k):
    """
    Select top-k context texts based on similarity scores.

    Args:
        contexts (list[str]): Context texts.
        best_contexts (list[float]): Corresponding scores.
        k (int): Number of contexts to select.

    Returns:
        list[tuple]: (context_text, score) sorted by score.
    """
    
    pairs = [(context,score) for context,score in zip(contexts,best_contexts)]
    pairs.sort(key=lambda x:x[1],reverse=True)
    top_pairs = pairs[:k]

    return top_pairs
