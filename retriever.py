from retrieval import (
    search,
    group_by_files,
    find_neighbours,
    build_context_texts,
    find_anchor_chunks_scores,
    find_top_k_contexts
)
from config import COUNT_OF_BEST_CONTEXTS

def retrive(model, category_indices, category_id_maps, chunked_texts, logger, question: str, top_k: int = COUNT_OF_BEST_CONTEXTS):

    searches = search(question, model, category_indices, category_id_maps, k=10)
 
    for prompt_id, results in searches.items():
        for i, result in enumerate(results):
            logger.debug(f"Result {i}: cat={result['category']}, score={result['score']:.3f}, chunk={result['chunk_id'][:50]}...")

    groups = group_by_files(searches, chunked_texts)
    
    neighbours = find_neighbours(groups, chunked_texts)
    
    contexts = build_context_texts(neighbours, chunked_texts)
    
    chunk_score, best_contexts = find_anchor_chunks_scores(searches, neighbours)
    top_k_contexts = find_top_k_contexts(contexts, best_contexts, top_k)

    return top_k_contexts