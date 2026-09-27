"""Retrieval-слой: векторный/гибридный поиск, кросс-языковой повтор,
приоритет точным совпадениям CVE/CWE."""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from .config import Config
from .lang import LANG_EN, LANG_RU, detect_language
from .llm_client import ChatClient, Embedder, EndpointError
from .models import Retrieved
from .topics import extract_cve, extract_cwe
from .vectorstore import VectorStore

log = logging.getLogger("rug.retriever")
_CVE_IN_Q = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_CWE_IN_Q = re.compile(r"CWE-\d{1,5}", re.IGNORECASE)

TranslateFn = Callable[[str, str, str], str]


class Retriever:
    def __init__(self, cfg: Config, store: VectorStore, embedder: Embedder,
                 chat: Optional[ChatClient] = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.chat = chat

    # ------------------------------------------------------------------ main
    def retrieve(self, query: str, language: Optional[str] = None) -> list[Retrieved]:
        """Поиск по запросу; при малочисленной выдаче — кросс-язычный повтор."""
        lang = language or detect_language(query)
        qvec = None
        if self.embedder is not None:
            try:
                qvec = self.embedder.embed_one(query)
            except EndpointError as e:
                log.info("Эмбеддер временно недоступен (%s), используем полнотекстовый поиск", e)

        hits = self._search(query, qvec, lang)
        if hits:
            return hits

        # --- кросс-язычный повтор (если запрос был на одном языке) ---
        if self.cfg.retrieval.cross_lingual and lang in (LANG_RU, LANG_EN):
            other = LANG_EN if lang == LANG_RU else LANG_RU
            tquery = self._translate(query, lang, other)
            if tquery and tquery.lower() != query.lower():
                log.info("Кросс-язычный поиск: %r -> %r", query[:60], tquery[:60])
                qvec2 = None
                if self.embedder is not None and qvec is not None:
                    try:
                        qvec2 = self.embedder.embed_one(tquery)
                    except EndpointError:
                        pass
                hits2 = self._search(tquery, qvec2, other)
                if hits2:
                    # объединяем с результатами исходного поиска (их могло быть мало)
                    merged = {h.chunk_id: h for h in (hits + hits2) if h}
                    return sorted(merged.values(), key=lambda h: h.score, reverse=True)[: self.cfg.retrieval.top_k]
        return hits

    def _search(self, query: str, qvec: Optional[list[float]], lang: str) -> list[Retrieved]:
        cve = extract_cve(query)
        cwe = extract_cwe(query)
        hits_cve: list[Retrieved] = []
        # Точные совпадения CVE/CWE поднимаем отдельным запросом (без порога по score)
        if cve or cwe:
            try:
                hits_cve = self.store.search(
                    query_text=query, query_vec=qvec,
                    top_k=max(2, self.cfg.retrieval.top_k // 2),
                    threshold=0.0,
                    language=None, topic=None,
                    cve=cve[0] if cve else None,
                    cwe=cwe[0] if cwe else None,
                    hybrid=False, use_mmr=False,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Поиск по CVE/CWE не удался: %s", e)

        eff_threshold = self.cfg.retrieval.threshold if qvec is not None else min(0.01, self.cfg.retrieval.threshold)
        hits = self.store.search(
            query_text=query, query_vec=qvec,
            top_k=self.cfg.retrieval.top_k,
            threshold=eff_threshold,
            min_hits=self.cfg.retrieval.min_hits,
            hybrid=True,
            use_mmr=self.cfg.retrieval.use_mmr,
        )
        if hits_cve:
            known = {h.chunk_id for h in hits}
            extra = hits_cve if not hits else [h for h in hits_cve if h.chunk_id not in known]
            return (extra + hits)[: self.cfg.retrieval.top_k]
        return hits

    # ------------------------------------------------------------- translate
    def _translate(self, query: str, src: str, dst: str) -> str:
        if self.chat is None:
            return ""
        from .prompts import build_translate_messages
        try:
            out = self.chat.complete(
                build_translate_messages(query, src, dst),
                temperature=0.0, max_tokens=180,
            ).strip().strip('"')
            return out
        except EndpointError as e:
            log.warning("LLM-перевод недоступен: %s", e)
            return ""