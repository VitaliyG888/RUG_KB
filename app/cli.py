"""CLI RUG_KB.

Примеры:
  python -m app.cli doctor
  python -m app.cli ingest library_text/N00B --limit 2
  python -m app.cli ingest "C:/path/to/file.pdf"
  python -m app.cli search "ssrf обход фильтров"
  python -m app.cli ask "что такое SQL injection?"
  python -m app.cli topics | sources | stats
  python -m app.cli bot
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .db import init_schema, test_connection
from .ingest import ingest_directory, ingest_file
from .llm_client import ChatClient, Embedder, EndpointError
from .rag import RAGChain, answer_with_sources
from .vectorstore import VectorStore
from .websearch import WebSearch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rug.cli")


# ---------------------------------------------------------------------------
def _make_parts(cfg):
    init_schema(cfg, dimensions=cfg.embed.dimensions)
    store = VectorStore(cfg, dimensions=cfg.embed.dimensions)
    embedder = Embedder(cfg.embed.base_url, cfg.embed.api_key, cfg.embed.model,
                        cfg.embed.batch_size, expected_dim=cfg.embed.dimensions)
    chat = ChatClient(cfg.chat.base_url, cfg.chat.api_key, cfg.chat.model,
                      cfg.chat.temperature, cfg.chat.max_tokens)
    web = WebSearch(cfg) if cfg.web.enabled else None
    rag = RAGChain(cfg, store, embedder, chat, web)
    return store, embedder, chat, web, rag


def cmd_doctor(cfg, args=None) -> int:
    print("== База данных ==")
    try:
        print("  ", test_connection(cfg))
    except Exception as e:  # noqa: BLE001
        print("  ОШИБКА:", e)
        return 1
    print("== Эмбеддинги ==")
    print(f"   base_url={cfg.embed.base_url} model={cfg.embed.model} dims={cfg.embed.dimensions}")
    emb = Embedder(cfg.embed.base_url, cfg.embed.api_key, cfg.embed.model,
                   cfg.embed.batch_size, expected_dim=cfg.embed.dimensions)
    try:
        dim = emb.probe()
        print(f"   OK: доступен, размерность {dim}")
    except EndpointError as e:
        print("   ПРЕДУПРЕЖДЕНИЕ:", e, "(поиск по векторам не заработает)")
    print("== Chat LLM ==")
    chat = ChatClient(cfg.chat.base_url, cfg.chat.api_key, cfg.chat.model)
    try:
        models = chat.check_models()
        if models:
            print(f"   OK: {len(models)} моделей на сервере, напр. {models[:5]}")
        else:
            print("   Сервер отвечает, но /models пуст — укажите CHAT_MODEL явно.")
    except EndpointError as e:
        print("   ПРЕДУПРЕЖДЕНИЕ (LLM недоступен):", e)
    print("== Веб-поиск ==")
    print("   провайдеры:", ", ".join(cfg.web.providers),
          "| tavily:", bool(cfg.web.tavily_key), "| serpapi:", bool(cfg.web.serpapi_key))
    print("== Бот ==")
    print("   BOT_TOKEN:", "задан" if cfg.bot.token else "НЕ задан",
          "| admin:", cfg.bot.admin_ids)
    return 0


def cmd_ingest(cfg, args) -> int:
    path = Path(args.path)
    store, embedder, _chat, _web, _rag = _make_parts(cfg)
    incremental = getattr(args, "incremental", False)
    if path.is_dir():
        results = ingest_directory(path, cfg, store, embedder, limit=args.limit,
                                   reindex=args.reindex, incremental=incremental)
    else:
        results = [ingest_file(path, cfg, store, embedder, reindex=args.reindex)]
    ok = sum(1 for r in results if not r.error and r.chunks > 0)
    err = [r for r in results if r.error]
    print(f"Готово: успешно {ok}/{len(results)}")
    for r in err:
        print(f"  FAIL {r.source}: {r.error}")
    for r in results:
        if not r.error and r.chunks:
            print(f"  OK  {r.source}: {r.chunks} чанков, {r.chars} символов")
    return 0 if not err else 1


def cmd_watch(cfg, args) -> int:
    from .ingest import watch_directory
    path = Path(args.path)
    if not path.is_dir():
        print(f"Путь {path} не является директорией.")
        return 1
    store, embedder, _chat, _web, _rag = _make_parts(cfg)
    watch_directory(path, cfg, store, embedder, interval_sec=args.interval)
    return 0


def cmd_search(cfg, args) -> int:
    store, embedder, _chat, _web, rag = _make_parts(cfg)
    from .lang import detect_language
    hits = rag.retriever.retrieve(args.query, detect_language(args.query))
    if not hits:
        print("Ничего не найдено по порогу релевантности.")
        return 0
    if args.json:
        data = [{"chunk_id": h.chunk_id, "score": round(h.score, 3),
                 "source": h.source, "page": h.page, "topic": h.topic,
                 "language": h.language, "content": h.content} for h in hits]
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h.score:.3f} (vec {h.vec_score:.3f} / txt {h.text_score:.3f}) "
              f"| {h.source} стр.{h.page or '-'} | {h.topic} | lang={h.language}")
        print("    " + h.content[:220].replace("\n", " "))
        print()
    return 0


def cmd_ask(cfg, args) -> int:
    store, embedder, chat, web, rag = _make_parts(cfg)
    result = rag.ask(args.query)
    if args.json:
        print(json.dumps({
            "answer": result.answer,
            "pipeline": result.pipeline,
            "language": result.language,
            "sources": [{"index": s.index, "source": s.source, "page": s.page,
                         "topic": s.topic} for s in result.sources],
            "web_supplement": result.web_supplement,
            "error": result.error,
        }, ensure_ascii=False, indent=2))
        return 0
    print(answer_with_sources(result))
    return 0


def cmd_list(cfg, args) -> int:
    store = _make_parts(cfg)[0]
    if args.what == "topics":
        for t, n in store.topics():
            print(f"{n:6d}  {t}")
    elif args.what == "sources":
        for s, ft, n, p in store.sources():
            print(f"{n:6d}  [{ft}] {s}  (pages: {p})")
    elif args.what == "stats":
        st = store.stats()
        print(json.dumps(st, ensure_ascii=False, indent=2))
    return 0


def cmd_reset_db(cfg, args=None) -> int:
    from .db import reset_schema
    reset_schema(cfg)
    print("Таблица chunks удалена. Перезапустите ingest для пересоздания схемы.")
    return 0


def cmd_bot(cfg, args=None) -> int:
    from .bot import run as run_bot
    store, embedder, chat, web, rag = _make_parts(cfg)
    run_bot(cfg, rag)
    return 0


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="rug-kb", description="RAG-база знаний по кибербезопасности")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="диагностика окружения")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("ingest", help="индексация файла или папки")
    p.add_argument("path")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--incremental", action="store_true", help="пропускать уже проиндексированные файлы")
    p.add_argument("--no-reindex", action="store_true", help="не удалять старые чанки источника")
    p.set_defaults(func=cmd_ingest, reindex=True)

    p = sub.add_parser("sync", help="инкрементальная синхронизация папки (только новые файлы)")
    p.add_argument("path", default="Библиотека", nargs="?")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_ingest, incremental=True, reindex=True)

    p = sub.add_parser("watch", help="фоновый мониторинг папки: автоиндексация новых файлов")
    p.add_argument("path", default="Библиотека", nargs="?")
    p.add_argument("--interval", type=int, default=10, help="интервал проверки в секундах")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("search", help="поиск фрагментов по векторам")
    p.add_argument("query")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="полный RAG-ответ на вопрос")
    p.add_argument("query")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("topics", help="список тем")
    p.set_defaults(func=lambda c, a: cmd_list(c, argparse.Namespace(what="topics")))

    p = sub.add_parser("sources", help="список источников")
    p.set_defaults(func=lambda c, a: cmd_list(c, argparse.Namespace(what="sources")))

    p = sub.add_parser("stats", help="статистика базы")
    p.set_defaults(func=lambda c, a: cmd_list(c, argparse.Namespace(what="stats")))

    p = sub.add_parser("bot", help="запуск Telegram-бота")
    p.set_defaults(func=cmd_bot)

    p = sub.add_parser("reset-db", help="удалить таблицу chunks (пересоздание с новой размерностью)")
    p.set_defaults(func=cmd_reset_db)

    args = ap.parse_args(argv)
    if hasattr(args, "no_reindex"):
        args.reindex = not args.no_reindex
    try:
        return args.func(cfg, args) if "func" in args else 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())