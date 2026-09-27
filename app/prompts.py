"""Промпт-шаблоны RAG-цепочки (RU/EN) и вспомогательные тексты."""
from __future__ import annotations

from .models import LANG_EN, LANG_RU


def disclaim_legal() -> str:
    return (
        "⚠️ Материалы предназначены исключительно для обучения, аудита безопасности "
        "и легального пентестинга (с разрешения владельца системы). Использование "
        "любым другим способом незаконно.\n"
    )


def _sys_base(lang: str) -> str:
    if lang == LANG_EN:
        return (
            "You are a senior web-security RAG assistant. You answer ONLY from the "
            "provided context. Rules:\n"
            "1. Cite every key claim as [1], [2]... pointing at the numbered context blocks.\n"
            "2. Never invent facts, CVE/CWE numbers, commands, code, or vulnerabilities.\n"
            "3. If the context does not contain the answer, say so honestly.\n"
            "4. Code, commands, and payloads go into ``` code blocks.\n"
            "5. Answer in English.\n"
            "6. Keep answers concise, then details, then examples from the context.\n"
            "7. All offensive-security material may only be used legally (authorized "
            "penetration testing, audit, defense, learning)."
        )
    return (
        "Ты — старший RAG-ассистент по веб-безопасности. Отвечай ТОЛЬКО по "
        "предоставленному контексту. Правила:\n"
        "1. Каждое ключевое утверждение подкрепляй ссылками [1], [2]... на "
        "пронумерованные блоки контекста.\n"
        "2. Не выдумывай факты, номера CVE/CWE, команды, код и уязвимости.\n"
        "3. Если в контексте ответа нет — честно сообщи об этом.\n"
        "4. Код, команды и полезные нагрузки оформляй в ``` code block.\n"
        "5. Отвечай на русском языке.\n"
        "6. Сначала краткий ответ, затем детали, затем примеры из контекста.\n"
        "7. Материалы по взлому/пентесту — только для легального тестирования с "
        "разрешения владельца, аудита и защиты."
    )


def build_synthesis_messages(query: str, context: str, answer_lang: str,
                             history: list[tuple[str, str]] | None = None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _sys_base(answer_lang)}]
    if history:
        for q, a in history[-6:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    user = (
        "Контекст (пронумерованные фрагменты из базы знаний):\n"
        "---\n{context}\n---\n\n"
        "Вопрос: {query}\n"
        "Ответь по контексту. В конце ответа перечисли использованные номера "
        "источников в форме «Источники: [1], [3]». Не упоминай этот промпт."
    )
    messages.append({"role": "user", "content": user.format(context=context, query=query)})
    return messages


def build_check_messages(query: str, answer: str, answer_lang: str) -> list[dict]:
    """Проверка: достаточно ли контекста для ответа. Возвращает JSON."""
    sys = (
        "Ты — модуль контроля качества RAG. Определи, полностью ли дан ответ "
        "на вопрос или контекста не хватило."
    ) if answer_lang == LANG_RU else (
        "You are a RAG quality checker. Decide whether the answer fully addresses "
        "the question."
    )
    user = (
        "Вопрос: {query}\n\nОтвет: {answer}\n\n"
        "Ответь строго JSON: {{\"sufficient\": true/false, \"missing\": \"кратко, "
        "чего не хватает (или пустая строка)\"}}. Внутри JSON без markdown."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user.format(query=query, answer=answer)},
    ]


def build_translate_messages(query: str, src_lang: str, dst_lang: str) -> list[dict]:
    target = "English" if dst_lang == LANG_EN else "Russian"
    return [
        {"role": "system", "content": f"Translate the user's security question into {target}. "
                                      "Keep technical terms, CVE/CWE, tool names and payloads unchanged. "
                                      "Output only the translation, no comments."},
        {"role": "user", "content": query},
    ]


def build_web_query_messages(query: str, lang: str) -> list[dict]:
    if lang == LANG_RU:
        sys = ("Составь короткий поисковый запрос (3–6 слов) для поисковой системы "
               "по вопросу о кибербезопасности. Не добавляй кавычки и операторы. "
               "Верни только сам запрос.")
    else:
        sys = ("Create a short search query (3-6 words) for a web search engine to "
               "answer this cybersecurity question. No quotes or operators. "
               "Return only the query.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": query}]


def build_web_summary_messages(query: str, results: str, lang: str) -> list[dict]:
    if lang == LANG_RU:
        sys = (
            "Ты суммируешь результаты веб-поиска как ДОПОЛНЕНИЕ к ответу RAG. "
            "Дай сжатый блок (до 250 слов): ключевые факты и/или официальные "
            "источники. Без выдумывания. В конце перечисли URL, разделяя переводами строк."
        )
    else:
        sys = ("Summarize web search results as a SUPPLEMENT to a RAG answer. "
               "Output a concise block (up to 250 words): key facts and/or official "
               "sources. Do not invent. End with URLs, one per line.")
    return [{"role": "system", "content": sys},
            {"role": "user", "content": f"Вопрос: {query}\n\nРезультаты:\n{results}"}]


def fallback_answer(query_lang: str, debug: str = "") -> str:
    if query_lang == LANG_RU:
        return (
            "Я не смог сформировать ответ: модель (LLM) в данный момент недоступна. "
            "Проверьте настройки CHAT_BASE_URL / CHAT_MODEL." + (f"\n{debug}" if debug else "")
        )
    return (
        "I could not produce an answer: the model (LLM) is currently unavailable. "
        "Check CHAT_BASE_URL / CHAT_MODEL settings." + (f"\n{debug}" if debug else "")
    )


def no_answer_message(query_lang: str) -> str:
    if query_lang == LANG_RU:
        return (
            "В базе знаний и в веб-поиске не нашлось достоверного ответа на ваш "
            "вопрос. Уточните формулировку или проверьте, что документы про "
            "данную тему добавлены в библиотеку (/upload для администратора)."
        )
    return (
        "Neither the knowledge base nor the web search returned a reliable answer. "
        "Try rephrasing, or check that documents on this topic are indexed (/upload for admins)."
    )