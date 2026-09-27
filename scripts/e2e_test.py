"""Интеграционный тест полного конвейера против фейкового OpenAI-эндпоинта.

Требует запущенного: python scripts/fake_openai_server.py --port 8765
Запуск:  python scripts/e2e_test.py
"""
from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("EMBED_BASE_URL", "http://127.0.0.1:8765/v1")
os.environ.setdefault("CHAT_BASE_URL", "http://127.0.0.1:8765/v1")
os.environ.setdefault("RUG_PG_DB", "rug_kb_test")
os.environ.setdefault("WEB_ENABLED", "false")

from app.config import load_config  # noqa: E402
from app.db import init_schema, reset_schema  # noqa: E402
from app.ingest import ingest_file  # noqa: E402
from app.rag import RAGChain, answer_with_sources  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402
from app.llm_client import Embedder, ChatClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N00B = ROOT / "library_text" / "N00B"


def main() -> int:
    cfg = load_config()
    init_schema(cfg, dimensions=cfg.embed.dimensions)
    store = VectorStore(cfg, dimensions=cfg.embed.dimensions)
    embedder = Embedder(cfg.embed.base_url, cfg.embed.api_key, cfg.embed.model,
                        cfg.embed.batch_size, expected_dim=cfg.embed.dimensions)
    chat = ChatClient(cfg.chat.base_url, cfg.chat.api_key, cfg.chat.model,
                      cfg.chat.temperature, cfg.chat.max_tokens)

    # --- 1. реальный клиент эмбеддингов против fake endpoint ---
    dim = embedder.probe()
    assert dim == cfg.embed.dimensions, (dim, cfg.embed.dimensions)
    print(f"OK: Embedder -> {dim}-мерные векторы (модель {embedder.model!r})")

    # --- 2. модели на сервере ---
    models = chat.check_models()
    print(f"OK: chat /models -> {models}")

    # --- 3. индексация реальным потоком ---
    reset_schema(cfg)
    init_schema(cfg, dimensions=cfg.embed.dimensions)
    from app.ingest import ingest_file
    pick = ["4. Information Gathering.txt",
            "8. SQL Injection Fundamentals.txt",
            "7. Cross-Site Scripting (XSS).txt"]
    results = [ingest_file(N00B / name, cfg, store, embedder) for name in pick]
    ok = [r for r in results if not r.error and r.chunks > 0]
    assert ok, [(r.source, r.error) for r in results]
    print(f"OK: ingest -> {len(ok)} файлов, "
          f"всего {sum(r.chunks for r in ok)} чанков")
    st = store.stats()
    print("     stats:", st)

    # --- 4. поиск с настоящим клиентом ---
    hits = store.search("sql injection union select payload",
                        embedder.embed_one("sql injection union select payload"),
                        threshold=0.0)
    assert hits
    print(f"OK: search -> {len(hits)} hits; top: {hits[0].topic} / {hits[0].source} "
          f"(score={hits[0].score:.3f})")

    # --- 5. RAG ask: полный цикл с LLM-шагами (EN-запрос — сильный матчинг) ---
    rag = RAGChain(cfg, store, embedder, chat, web=None)
    res = rag.ask("What is SQL injection and how is it exploited?")
    txt = answer_with_sources(res)
    assert res.answer and res.pipeline == "rag", (res.pipeline, res.answer)
    assert "[1]" in txt or "Источник" in txt
    print(f"OK: ask (RAG) -> pipeline={res.pipeline}, sources={len(res.sources)}")
    print("-" * 60)
    print(txt[:600])
    print("-" * 60)

    # --- 6. RU-запрос: кросс-язычный повтор выполнен, pipeline честный ---
    res2 = rag.ask("Что такое SQL инъекция и чем она опасна?")
    print(f"OK: ask RU -> pipeline={res2.pipeline}, sources={len(res2.sources)}, "
          f"query_lang={res2.query_language}")

    # --- 7. переключение порога: ниже => web-pipeline (web выключен -> none) ---
    low_cfg = dataclasses.replace(cfg, retrieval=dataclasses.replace(
        cfg.retrieval, threshold=0.99, high_confidence=1.0))
    rag_strict = RAGChain(low_cfg, store, embedder, chat, web=None)
    res3 = rag_strict.ask("What is SQL injection and how is it exploited?")
    assert res3.pipeline == "none", res3.pipeline
    print(f"OK: порог завышен -> pipeline='none' (честный ответ), "
          f"answer={res3.answer[:50]!r}")

    print("\nE2E TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())