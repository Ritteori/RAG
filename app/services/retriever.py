from app.services.retrieval import (
    search,
    group_by_files,
    find_neighbours,
    build_context_texts,
    find_anchor_chunks_scores,
    find_top_k_contexts
)

class Retriever:
    def __init__(
        self,
        embed_model,
        category_indices,
        category_id_maps,
        chunked_texts,
        logger,
        math,
        ml,
        ops,
        python,
        softskills,
        stat,
        ollama_client,
        model="qwen2.5:7b",
        top_k=5,
        search_k=3,
    ):
        self.embed_model = embed_model
        self.category_indices = category_indices
        self.category_id_maps = category_id_maps
        self.chunked_texts = chunked_texts
        self.logger = logger
        self.ollama_client = ollama_client

        self.math = math
        self.ml = ml
        self.ops = ops
        self.python = python
        self.softskills = softskills
        self.stat = stat

        self.model = model
        self.top_k = top_k
        self.search_k = search_k

    def retrieve(self, question: str):
        
        searches = self._search(question)
        groups = self._group_by_files(searches)
        neighbours = self._find_neighbours(groups)
        contexts = self._build_context_texts(neighbours)
        chunk_score, best_contexts = self._find_anchor_chunks_scores(searches,neighbours)
        top_k_contexts = self._find_top_k_contexts(contexts,best_contexts)

        return top_k_contexts
    
    def _search(self,question):
        searches = search(
            question, self.embed_model, self.category_indices, self.category_id_maps, self.search_k,
            self.math, self.ml, self.ops, self.python, self.softskills, self.stat, self.ollama_client, self.model
        )

        for prompt_id, results in searches.items():
            for i, result in enumerate(results):
                self.logger.debug(f"Result {i}: cat={result['category']}, score={result['score']:.3f}, chunk={result['chunk_id'][:50]}...")

        return searches
    
    def _group_by_files(self,searches):
        groups = group_by_files(searches, self.chunked_texts)
        return groups

    def _find_neighbours(self,groups):
        neighbours = find_neighbours(groups, self.chunked_texts)
        return neighbours

    def _build_context_texts(self,neighbours):
        contexts = build_context_texts(neighbours, self.chunked_texts)
        return contexts

    def _find_anchor_chunks_scores(self,searches, neighbours):
        chunk_score, best_contexts = find_anchor_chunks_scores(searches, neighbours)
        return chunk_score, best_contexts
    
    def _find_top_k_contexts(self,contexts, best_contexts):
        top_k_contexts = find_top_k_contexts(contexts, best_contexts, self.top_k)
        return top_k_contexts