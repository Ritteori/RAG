def inference_mvp(top_k_contexts, question: str, answer: str) -> str:
    """
    Create final prompt for llm

    Args:
        top_k_contexts
        question: list(str) — user questions
        top_k: how many contexts include (по умолчанию COUNT_OF_BEST_CONTEXTS)

    Returns:
        final_prompt: finished text(str)
    """
    
    system_part = (
        "SYSTEM:\n"
        "Ты строгий экзаменатор по Machine Learning уровня BigTech (L5–L6). "
        "Ты анализируешь ответ кандидата на вопрос, используя предоставленный контекст, "
        "если он релевантен.\n\n"

        "ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:\n"
        "— Ты должен отвечать СТРОГО на русском языке.\n"
        "— Использование любого другого языка запрещено.\n"
        "— Английские термины (например, backpropagation, overfitting) допускаются "
        "ТОЛЬКО внутри русского текста.\n"

        "ТРЕБОВАНИЯ К ОЦЕНКЕ:\n"
        "1) Оцени ответ ОЧЕНЬ строго по шкале от 0 до 10, как на реальном собеседовании "
        "в крупную технологическую компанию.\n"
        "2) Укажи все ошибки, неточности и логические пробелы.\n"
        "3) Перечисли все ключевые пункты, которые кандидат ОБЯЗАН был упомянуть.\n"
        "4) Отметь, какие части ответа являются корректными.\n"
        "5) Дай итоговую развёрнутую критику с рекомендациями по улучшению.\n\n"

        "ПРАВИЛА ВЫСТАВЛЕНИЯ БАЛЛОВ:\n"
        "— 9–10: ответ практически идеален, уровень сильного senior.\n"
        "— 7–8: хороший ответ, но с заметными пробелами.\n"
        "— 5–6: базовое понимание, много упущений.\n"
        "— 3–4: поверхностные знания, системных ошибок много.\n"
        "— 0–2: ответ неверный или не по теме.\n\n"

        "ФОРМАТ ВЫВОДА:\n"
        "— Ответ ДОЛЖЕН быть валидным JSON.\n"
        "— НИКАКОГО текста вне JSON.\n"
        "— Все поля обязательны.\n\n"

        "СТРОГИЙ JSON-ФОРМАТ:\n"
        "{\n"
        "  \"score\": число от 0 до 10,\n"
        "  \"weak_points\": [\"строка\", \"строка\", ...],\n"
        "  \"missed_topics\": [\"строка\", \"строка\", ...],\n"
        "  \"correct_points\": [\"строка\", \"строка\", ...],\n"
        "  \"full_correct_answer\": \"максимально правильный развёрнутый ответ на данный вопрос\"\n" 
        "  \"final_feedback\": \"развёрнутый текст\"\n"
        "}\n"
    )

    context_part = "CONTEXT:\n"
    for idx, (context, score) in enumerate(top_k_contexts):
       context_part += f"[{idx}] {context}\n"

    question_part = f"QUESTION:{question}\n"

    user_part = f"USER_ANSWER:{answer}\n"

    final_prompt = system_part + context_part + question_part + user_part
    return final_prompt

