"""Smoke-test: чанкинг -> (fake) эмбеддинги -> pgvector -> поиск -> RAG-цепочка.

Использует отдельную БД rug_kb_test и детерминированный «эмбеддер» на основе
хэшей токенов, чтобы проверить весь конвейер без живого LLM-эндпоинта.

Запуск:  python scripts/smoke_test.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2

from app.chunking import chunk_text, estimate_tokens
from app.config import Config, load_config
from app.db import init_schema, reset_schema, test_connection
from app.lang import detect_language
from app.models import LANG_RU
from app.rag import RAGChain
from app.topics import classify_topic, extract_cve, extract_cwe
from app.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parent.parent
N00B = ROOT / "library_text" / "N00B"
DIM = 64  # быстрый тест


class FakeEmbedder:
    """Детерминированный content-aware эмбеддер для тестов (не семантика!)."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._cache: dict[str, list[float]] = {}

    def _tok_vec(self, tok: str) -> list[float]:
        v = self._cache.get(tok)
        if v is None:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            rnd = random.Random(h)
            v = [rnd.uniform(-1.0, 1.0) for _ in range(self.dim)]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / n for x in v]
            self._cache[tok] = v
        return v

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            toks = re.findall(r"[a-zа-яё0-9_]+", t.lower())
            vec = [0.0] * self.dim
            for tok in toks:
                tv = self._tok_vec(tok)
                for i in range(self.dim):
                    vec[i] += tv[i]
            n = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / n for x in vec])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimensions(self) -> int:
        return self.dim


def _test_cfg(base: Config) -> Config:
    cfg = dataclasses.replace(base)
    cfg = dataclasses.replace(cfg, db=dataclasses.replace(base.db, dbname="rug_kb_test"))
    cfg = dataclasses.replace(cfg, embed=dataclasses.replace(base.embed, dimensions=DIM))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(base.web, enabled=False))
    return cfg


def _ensure_test_db(cfg: Config) -> None:
    c = psycopg2.connect(host=cfg.db.host, port=cfg.db.port, dbname="postgres",
                         user=cfg.db.user, password=cfg.db.password)
    try:
        c.autocommit = True
        cur = c.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname='rug_kb_test'")
        if not cur.fetchone():
            cur.execute("CREATE DATABASE rug_kb_test")
        cur.close()
    finally:
        c.close()
    print("OK: база rug_kb_test готова")


def main() -> int:
    cfg = _test_cfg(load_config())
    _ensure_test_db(cfg)
    print("DB:", test_connection(cfg))

    # ------------------------------------------------------------------ 1. язык
    assert detect_language("Это русский текст про SQL инъекции") in ("ru", "mixed")
    assert detect_language("This is an English text about XSS") == "en"
    print("OK: определение языка")

    # ------------------------------------------------------------------ 2. темы и сущности
    t = classify_topic("Используйте UNION SELECT для обхода фильтров при SQL injection")
    assert t == "SQL injection", t
    assert extract_cve("CVE-2021-44228 log4j plus cve-2023-12345") == ["CVE-2021-44228", "CVE-2023-12345"]
    assert extract_cwe("CWE-89 sql injection") == ["CWE-89"]
    print(f"OK: рубрикатор -> {t!r}; CVE/CWE парсятся")

    # ------------------------------------------------------------------ 3. чанкинг (EN + RU)
    sample = N00B / "8. SQL Injection Fundamentals.txt"
    text = sample.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text, sample.name)
    assert chunks, "не получили чанков"
    too_big = [c for c in chunks if estimate_tokens(c.content) > 1500]
    assert not too_big, f"чанки слишком большие: {len(too_big)}"
    sizes = [len(c.content) for c in chunks]
    langs = {c.language for c in chunks}
    print(f"OK: чанкинг '{sample.name}': {len(chunks)} чанков, "
          f"сред. длина {sum(sizes)//len(sizes)}, языки {langs}")

    ru_file = ROOT / "library_text" / "2015" / "Хакер 2015 01(192).txt"
    assert ru_file.exists(), f"нет файла {ru_file}"
    ru_chunks = chunk_text(ru_file.read_text(encoding="utf-8", errors="replace"), ru_file.name)
    assert ru_chunks, "RU-чанкинг не выдал чанков"
    ru_langs = {c.language for c in ru_chunks}
    assert (LANG_RU in ru_langs) or ("mixed" in ru_langs), ru_langs
    ru_topics = {c.topic for c in ru_chunks}
    print(f"OK: RU-чанкинг '{ru_file.name}': {len(ru_chunks)} чанков, языки={ru_langs}, "
          f"тем={len(ru_topics)} (напр. {sorted(ru_topics)[:3]})")

    # ------------------------------------------------------------------ 4. индексирование (fake emb)
    reset_schema(cfg)
    init_schema(cfg, dimensions=DIM)
    emb = FakeEmbedder()
    store = VectorStore(cfg, dimensions=DIM)
    files = [
        N00B / "4. Information Gathering.txt",
        N00B / "8. SQL Injection Fundamentals.txt",
        N00B / "7. Cross-Site Scripting (XSS).txt",
        ROOT / "library_text" / "2015" / "Хакер 2015 01(192).txt",
    ]
    files = [f for f in files if f.exists()]
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        chs = chunk_text(txt, f.name)
        store.insert(chs, emb.embed([c.content for c in chs]))
        print(f"   insert {f.name}: {len(chs)} chunks")
    st = store.stats()
    assert st["chunks"] > 0
    print(f"OK: индексация, всего чанков в тестовой БД: {st['chunks']}, языки: {st['languages']}")

    # ------------------------------------------------------------------ 5. поиск
    hits = store.search("SQL injection payload examples", emb.embed_one("SQL injection payload examples"))
    assert hits, "поиск не вернул результатов"
    topics = {h.topic for h in hits}
    print(f"OK: поиск по вектору: {len(hits)} hits, темы={topics}")
    for h in hits[:3]:
        print(f"    [{h.score:.3f}] {h.source} | {h.topic} | {h.content[:80]!r}")

    # CVE-повышенный поиск
    store2 = VectorStore(cfg, dimensions=DIM)
    cve_hits = store2.search("log4j CVE-2021-44228", emb.embed_one("log4j CVE-2021-44228"))
    print(f"OK: поиск с CVE в запросе: {len(cve_hits)} hits")

    # ------------------------------------------------------------------ 6. RAG без LLM (честный фолбэк)
    rag = RAGChain(cfg, store, emb, chat=None, web=None)
    res = rag.ask("Что такое SQL injection?")
    assert res.pipeline in ("rag", "none", "rag+web")
    assert res.answer, "пустой ответ"
    print(f"OK: RAG-цепочка без LLM: pipeline={res.pipeline}, "
          f"sources={len(res.sources)}, answer={res.answer[:60]!r}")

    print("\nSMOKE-TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())