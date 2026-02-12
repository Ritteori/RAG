from storage.load import load_indices, load_chunks

class IndexLoader():
    def __init__(self,config):
        self.categories = config.categories

    def load(self):
        
        category_indices, category_id_maps = load_indices(self.categories)

        chunked_texts = load_chunks('chunked_texts.json')

        if category_indices is None:
            raise RuntimeError("FAISS индексы не найдены. Сначала запусти build_index.py")
        
        return category_indices, category_id_maps, chunked_texts
