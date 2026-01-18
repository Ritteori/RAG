import numpy as np
from collections import Counter

from config import MATH_KEYWORDS,ML_KEYWORDS,OPS_KEYWORDS,PYTHON_KEYWORDS,SOFTSKILLS_KEYWORDS,STAT_KEYWORDS

def guess_categories(prompt):

    reps = []

    p = prompt.lower()
    for word in p.strip().split():
        if word in OPS_KEYWORDS:
            reps.append("ops")
        if word in MATH_KEYWORDS:
            reps.append("math")
        if word in SOFTSKILLS_KEYWORDS:
            reps.append("softskills")
        if word in STAT_KEYWORDS:
            reps.append("statistics_probabilities")
        if word in ML_KEYWORDS:
            reps.append("ml")
        if word in PYTHON_KEYWORDS:
            reps.append("python")
    
    counter = Counter(reps)
    if len(counter) == 0:
        return None
    else:
        return counter.most_common(1)[0][0]

def search(prompts, model, category_indices, category_id_maps, k=3):
    results = {}

    for idx, prompt in enumerate(prompts):
        emb = model.encode(prompt, normalize_embeddings=True)
        emb = np.array([emb], dtype="float32")

        results[idx] = []

        cat = guess_categories(prompt)

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
