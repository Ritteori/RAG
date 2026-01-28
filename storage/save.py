import pickle
import faiss
import json
import os

def save(chunked_texts, category_indices, category_id_maps, dir='storage'):
    os.makedirs(dir, exist_ok=True)

    for cat, index in category_indices.items():
        faiss.write_index(index, f"{dir}/{cat}.faiss")
    
    with open(f"{dir}/category_id_maps.pkl", "wb") as f:
        pickle.dump(category_id_maps, f)

    with open(f"{dir}/chunked_texts.json", "w", encoding="utf-8") as f:
        json.dump(chunked_texts, f, ensure_ascii=False, indent=2)

    print('Indices and chunked texts has been saved.')