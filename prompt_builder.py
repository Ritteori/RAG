from sentence_transformers import SentenceTransformer
from chunker import build_chunks
from embedder import embed
from retrieval import (
    search,
    group_by_files,
    find_neighbours,
    build_context_texts,
    find_anchor_chunks_scores,
    find_top_k_contexts
)
from config import COUNT_OF_BEST_CONTEXTS

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/data/models_cache"
)

chunked_texts = build_chunks()
embedded_texts, chunks_by_category, category_indices, category_id_maps = embed(model, chunked_texts)


def inference_mvp(user_prompt: str, top_k: int = COUNT_OF_BEST_CONTEXTS) -> str:
    """
    Create final prompt for llm

    Args:
        user_prompt: list(str) — user questions
        top_k: how many contexts include (по умолчанию COUNT_OF_BEST_CONTEXTS)

    Returns:
        final_prompt: finished text(str)
    """
    searches = search(user_prompt, model, category_indices, category_id_maps, k=10)
    
    groups = group_by_files(searches, chunked_texts)
    
    neighbours = find_neighbours(groups, chunked_texts)
    
    contexts = build_context_texts(neighbours, chunked_texts)
    
    chunk_score, best_contexts = find_anchor_chunks_scores(searches, neighbours)
    top_k_contexts = find_top_k_contexts(contexts, best_contexts, top_k)
    
    system_part = (
        "SYSTEM:\n"
        "Ты экзаменатор по ML. Ты должен отвечать ТОЛЬКО на русском языке. "
        "Использование китайского, английского или любого другого языка ЗАПРЕЩЕНО. "
        "Если в контексте есть английские термины, ты можешь их использовать, "
        "но весь ответ должен быть строго на русском. "
        "Если ты не можешь ответить на русском - лучше ничего не отвечай.\n"
    )

    context_part = "CONTEXT:\n"
    # for idx, (context, score) in enumerate(top_k_contexts):
    #    context_part += f"[{idx}] {context}\n"

    user_part = f"USER:\n{user_prompt}\n"

    final_prompt = system_part + context_part + user_part
    return final_prompt

