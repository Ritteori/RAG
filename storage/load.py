import os
import json
import pickle
import faiss

def load_indices(categories, dir="storage"):
    category_indices = {}
    category_id_maps = {}

    for cat in categories:
        index_dir = f"{dir}/{cat}.faiss"
        if not os.path.exists(index_dir):
            return None, None

        category_indices[cat] = faiss.read_index(index_dir)

    with open(f"{dir}/category_id_maps.pkl", "rb") as f:
        category_id_maps = pickle.load(f)

    return category_indices, category_id_maps

def load_chunks(filename, dir="storage"):
        with open(f"{dir}/{filename}", "r", encoding="utf-8") as f:
            return json.load(f)
