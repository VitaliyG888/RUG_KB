"""RAG-цепочка (LCEL-подобный пайплайн):
language detect -> retriever -> format_docs -> prompt -> LLM -> parser,
с порогом релевантности, веб-фолбэком, проверкой полноты и пометками источников.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .config import Config
from .lang import LANG_EN, LANG_RU, detect_language, dominant_language
from .llm_client import ChatClient, Embedder, EndpointError
from .models import LANG_MIXED, RAGAnswer, Retrieved, SourceCitation, WebImage, WebResult
from .prompts import (
    build_check_messages,
    build_synthesis_messages,
    build_web_query_messages,
    build_web_summary_messages,
    fallback_answer,
    no_answer_message,
)
from .retriever import Retriever
from .vectorstore import VectorStore
from .websearch import WebSearch

log = logging.getLogger("rug.rag")

_LEGAL_TOPIC_HINT = re.compile(
    r"взлом|hack|взлома|инструмент|метод|эксплуатац|exploit|пентест|pentest|payload|"
    r"bypass|обход|атака|attack|уязвимост|vulnerab|привилег|privilege|крэк|crack",
    re.IGNORECASE,
)


class RAGChain:
    def __init__(self, cfg: Config, store: VectorStore, embedder: Embedder,
                 chat: Optional[ChatClient], web: Optional[WebSearch] = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.chat = chat
        self.web = web or (WebSearch(cfg) if cfg.web.enabled else None)
        self.retriever = Retriever(cfg, store, embedder, chat)

    # =================================================================== API
    def ask(self, query: str, history: Optional[list[tuple[str, str]]] = None,
            filters: Optional[dict] = None) -> RAGAnswer:
        """Полный цикл ответа на вопрос."""
        q_lang = detect_language(query)
        a_lang = dominant_language(query)
        try:
            hits = self.retriever.retrieve(query, language=q_lang)
        except EndpointError as e:
            return RAGAnswer(answer=fallback_answer(a_lang, str(e)), language=a_lang,
                             query_language=q_lang, error=str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка поиска")
            return RAGAnswer(answer=fallback_answer(a_lang, str(e)), language=a_lang,
                             query_language=q_lang, error=str(e))

        if not hits:
            return self._web_only(query, a_lang, q_lang)

        max_score = max(h.score for h in hits)
        if max_score < self.cfg.retrieval.threshold:
            return self._web_only(query, a_lang, q_lang)

        # --- RAG-путь ---
        context, citations = self._format_docs(hits)
        try:
            answer = self._synthesize(query, context, a_lang, history)
        except EndpointError as e:
            answer = fallback_answer(a_lang, str(e))

        need_web = max_score < self.cfg.retrieval.high_confidence
        supplement = ""
        web_results: list[WebResult] = []
        web_images: list[WebImage] = []
        pipeline = "rag"
        if need_web and self.web is not None:
            ok, supplement, web_results = self._try_web_supplement(query, a_lang, q_lang)
            if ok:
                pipeline = "rag+web" if supplement else "rag"

        if self.chat is not None and not _looks_like_fallback(answer):
            # Проверка полноты (только когда RAG — единственный источник частичных данных)
            verdict = self._check_sufficiency(query, answer, a_lang)
            if verdict is False:
                ok, s2, w2 = self._try_web_supplement(query, a_lang, q_lang)
                if ok and s2:
                    supplement = s2
                    web_results = w2 or web_results
                    pipeline = "rag+web"
                    try:
                        answer = self._synthesize(query, context, a_lang, history, supplement=supplement)
                    except EndpointError:
                        pass

        # Поиск схем / графиков / диаграмм / картинок
        if self.web is not None and (pipeline in ("rag+web", "web") or _has_visual_intent(query)):
            web_images = self._search_visuals(query, a_lang)

        return RAGAnswer(
            answer=answer,
            language=a_lang,
            sources=citations,
            web_supplement=supplement,
            web_results=web_results,
            web_images=web_images,
            pipeline=pipeline,
            query_language=q_lang,
        )

    # ================================================================ helpers
    def _format_docs(self, hits: list[Retrieved]) -> tuple[str, list[SourceCitation]]:
        citations: list[SourceCitation] = []
        blocks: list[str] = []
        for i, h in enumerate(hits, 1):
            blocks.append(f"[{i}] (source: {h.source}, page: {h.page or '-'}, "
                          f"topic: {h.topic}, lang: {h.language})\n{h.content}")
            citations.append(
                SourceCitation(index=i, source=h.source, page=h.page, topic=h.topic,
                               language=h.language, content=h.content[:400],
                               score=h.score, url=h.url)
            )
        return "\n\n".join(blocks), citations

    def _synthesize(self, query: str, context: str, lang: str,
                    history: Optional[list[tuple[str, str]]] = None,
                    supplement: str = "") -> str:
        if self.chat is None:
            raise EndpointError("CHAT не настроен")
        if supplement:
            context += (
                "\n\n--- ДОПОЛНИТЕЛЬНЫЕ ВЕБ-ДАННЫЕ (pomетить как 'Веб-дополнение' в ответе) ---\n"
                + supplement
            )
        messages = build_synthesis_messages(query, context, lang, history)
        return self.chat.complete(messages, temperature=self.cfg.chat.temperature,
                                  max_tokens=self.cfg.chat.max_tokens)

    def _check_sufficiency(self, query: str, answer: str, lang: str) -> bool:
        try:
            raw = self.chat.complete(build_check_messages(query, answer, lang),
                                     temperature=0.0, max_tokens=200, json_mode=True)
            data = json.loads(raw)
            return bool(data.get("sufficient", False))
        except Exception as e:  # noqa: BLE001
            log.info("Проверка полноты недоступна: %s", e)
            return True

    # ---------------------------------------------------------------- web-part
    def _web_query(self, query: str, lang: str) -> str:
        if self.chat is None:
            return query
        try:
            q = self.chat.complete(build_web_query_messages(query, lang),
                                   temperature=0.0, max_tokens=60).strip().strip('"')
            return q or query
        except EndpointError:
            return query

    def _try_web_supplement(self, query: str, a_lang: str, q_lang: str) -> tuple[bool, str, list[WebResult]]:
        """Пытается получить веб-дополнение. (ok, supplement_text, results)"""
        if self.web is None:
            return False, "", []
        wq = self._web_query(query, a_lang)
        results = self.web.search(wq, lang=a_lang)
        return self._web_results_to_supplement(query, results, a_lang)

    def _search_visuals(self, query: str, lang: str) -> list[WebImage]:
        """Поиск картинок: схем, архитектур, графиков, примеров кода."""
        if self.web is None:
            return []
        # Если в запросе нет слов 'схема', 'diagram' и т.д. — уточняем запрос для картинок
        iq = query
        if not _has_visual_intent(query):
            iq = f"{query} diagram architecture scheme code"
        try:
            return self.web.search_images(iq, lang=lang, n=3)
        except Exception as e:
            log.info("Ошибка поиска картинок: %s", e)
            return []

    def _web_results_to_supplement(self, query: str, results: list[WebResult],
                                   a_lang: str) -> tuple[bool, str, list[WebResult]]:
        if not results:
            return False, "", []
        # Фильтр мусора: у релевантного результата есть лексическое пересечение с запросом
        def _toks(s: str) -> set[str]:
            return {t[:5] for t in re.findall(r"[а-яёa-z0-9]{3,}", s.lower())}
        qtoks = _toks(query)
        relevant: list[WebResult] = []
        for r in results:
            hay = _toks(f"{r.title} {r.snippet}")
            if qtoks & hay:
                relevant.append(r)
        if not relevant:
            return False, "", []
        results = relevant[:5]
        blurb = "\n\n".join(
            f"- {r.title}\n  {r.url}\n  {r.snippet[:300]}" for r in results
        )
        if self.chat is not None:
            try:
                summary = self.chat.complete(
                    build_web_summary_messages(query, blurb, a_lang),
                    temperature=0.2, max_tokens=400,
                )
                if summary.strip():
                    return True, summary.strip(), results
            except EndpointError:
                pass
        # Без LLM — сырой список
        head = ("Веб-дополнение (результаты поиска):\n" if a_lang == LANG_RU
                else "Web supplement (search results):\n")
        return True, head + blurb, results

    def _web_only(self, query: str, a_lang: str, q_lang: str) -> RAGAnswer:
        if self.web is None:
            return RAGAnswer(answer=no_answer_message(a_lang), language=a_lang,
                             query_language=q_lang, pipeline="none")
        wq = self._web_query(query, a_lang)
        results = self.web.search(wq, lang=a_lang)
        web_images = self._search_visuals(query, a_lang)
        if not results:
            return RAGAnswer(answer=no_answer_message(a_lang), language=a_lang,
                             web_images=web_images,
                             query_language=q_lang, pipeline="none")
        ok, supplement, _ = self._web_results_to_supplement(query, results, a_lang)
        if not ok:
            return RAGAnswer(answer=no_answer_message(a_lang), language=a_lang,
                             web_images=web_images,
                             query_language=q_lang, pipeline="none")
        answer = supplement
        return RAGAnswer(answer=answer, language=a_lang, web_supplement=supplement,
                         web_results=results, web_images=web_images,
                         pipeline="web", query_language=q_lang)


_VISUAL_RE = re.compile(
    r"\b(схем[аеыу]|график[а-я]*|диаграмм[а-я]*|картинк[а-я]*|скриншот[а-я]*|"
    r"иллюстрац[а-я]*|рисунок|рисунк[а-я]*|код[а-я]*|пример\s+кода|"
    r"diagram[s]?|chart[s]?|schema[s]?|flowchart[s]?|screenshot[s]?|"
    r"architecture|visual|image[s]?|picture[s]?)\b",
    flags=re.IGNORECASE,
)


def _has_visual_intent(query: str) -> bool:
    return bool(_VISUAL_RE.search(query))


def _looks_like_fallback(text: str) -> bool:
    return text.startswith(("Я не смог", "I could not"))


def answer_with_sources(result: RAGAnswer) -> str:
    """Финальный текст ответа для пользователя: ответ + источники + пометка."""
    parts: list[str] = [result.answer]
    if result.sources and result.pipeline in ("rag", "rag+web"):
        lines = ["", "Источники:"]
        for s in result.sources:
            loc = f", стр. {s.page}" if s.page else ""
            src = re.sub(r"\.pdf$|\.txt$", "", s.source, flags=re.IGNORECASE)
            lang_note = f" ({s.language})" if s.language and s.language != LANG_MIXED else ""
            lines.append(f"[{s.index}] {src}{loc}{lang_note} — {s.topic}")
        parts.append("\n".join(lines))
    if result.pipeline == "rag":
        label = "Источник: RAG" if result.language == LANG_RU else "Source: RAG"
        parts.append("\n" + label)
    elif result.pipeline == "rag+web":
        parts.append("\nВеб-дополнение:\n" + result.web_supplement)
        if result.web_results:
            parts.append("Ссылки: " + " ".join(
                f"[{r.engine}]({r.url})" for r in result.web_results[:3] if r.url))
        label = "Источник: RAG + веб-дополнение" if result.language == LANG_RU \
            else "Source: RAG + web supplement"
        parts.append("\n" + label)
    elif result.pipeline == "web":
        if result.web_supplement and result.web_supplement != result.answer:
            parts.append(result.web_supplement)
        label = "Веб-дополнение (в базе знаний ответ не найден)" if result.language == LANG_RU \
            else "Web supplement (no answer in the knowledge base)"
        parts.append("\n" + label)
        if result.web_results:
            parts.append("Ссылки: " + " ".join(
                f"[{r.engine}]({r.url})" for r in result.web_results[:3] if r.url))
    if result.web_images:
        img_lines = ["\n🖼 Изображения / Схемы:"]
        for img in result.web_images[:3]:
            title = img.title or "Иллюстрация"
            img_lines.append(f"- [{title}]({img.image_url})")
        parts.append("\n".join(img_lines))
    if _LEGAL_TOPIC_HINT.search(result.answer + " " + " ".join(
            s.topic for s in result.sources)):
        from .prompts import disclaim_legal
        parts.append("\n" + disclaim_legal())
    return "\n".join(parts)