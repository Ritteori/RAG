from collections import Counter
from utils.ollama_client import call_ollama_chat

from config import MATH_KEYWORDS,ML_KEYWORDS,OPS_KEYWORDS,PYTHON_KEYWORDS,SOFTSKILLS_KEYWORDS,STAT_KEYWORDS

def guess_categories(prompt, model=None):
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

            Ответ в виде строки:
            """

        return call_ollama_chat(final_prompt,model)
    
    else:
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

