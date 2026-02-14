from collections import Counter

def guess_categories(prompt, math, ml, ops, python, softskills, stat, ollama_client, model=None):
    """
    Guess the most relevant category for a user prompt using keyword matching.

    Args:
        prompt (str): User query.

    Returns:
        str | None: Category name if detected, otherwise None.
    """
    if model:

        final_prompt = f"""
            Ты — классификатор вопросов для ML-интервью.

            Твоя задача — определить, к какой категории относится вопрос.

            Доступные категории (выбери ОДНУ):
            - math
            - statistics_probabilities
            - ml
            - python
            - ops
            - softskills

            Правила:
            1. Верни ТОЛЬКО название категории.
            2. Не добавляй пояснений, текста, комментариев.
            3. Не используй кавычки, точки, двоеточия.
            4. Если вопрос касается вероятностей, случайных величин, распределений, матожидания — выбирай statistics_probabilities.
            5. Если вопрос касается производных, градиентов, линейной алгебры, оптимизации — выбирай math.
            6. Если вопрос касается моделей машинного обучения, нейросетей, обучения, loss, backprop — выбирай ml.
            7. Если вопрос про код, синтаксис, библиотеки Python — выбирай python.
            8. Если вопрос про инфраструктуру, Docker, Linux, CI/CD — выбирай ops.
            9. Если вопрос про коммуникацию, опыт, командную работу — выбирай softskills.

            Вопрос:
            {prompt}

            Верни ответ строго в JSON формате:
            {{
                "category": "math"
            }}
            Без комментариев, Без markdown, Без ```json
        """

        response = ollama_client.call_ollama_chat(final_prompt,model)
        if isinstance(response, dict):
            category = response.get("category")
        else:
            category = response

        return category
    
    else:
        reps = []

        p = prompt.lower()
        for word in p.strip().split():
            if word in ops:
                reps.append("ops")
            if word in math:
                reps.append("math")
            if word in softskills:
                reps.append("softskills")
            if word in stat:
                reps.append("statistics_probabilities")
            if word in ml:
                reps.append("ml")
            if word in python:
                reps.append("python")
        
        counter = Counter(reps)
        if len(counter) == 0:
            return None
        else:
            return counter.most_common(1)[0][0]

