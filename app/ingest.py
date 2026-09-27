"""Ingestion: файл (TXT/PDF) -> текст -> чанки -> эмбеддинги -> pgvector.

TXT: автоопределение кодировки (UTF-8, UTF-16 LE/BE, CP1251, CP866, latin-1).
PDF: постранично через pymupdf (каждая страница = page в метаданных).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_text
from .config import Config
from .llm_client import Embedder, EndpointError
from .models import ChunkIn
from .vectorstore import VectorStore

log = logging.getLogger("rug.ingest")


@dataclass(slots=True)
class IngestResult:
    source: str
    file_type: str
    chunks: int
    chars: int
    error: str = ""


_ENC_CANDIDATES = ("utf-8-sig", "utf-8", "utf-16", "cp1251", "cp866", "latin-1")


def _read_txt(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    for enc in ("utf-8", "cp1251", "cp866", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_pdf(path: Path, max_pages: int = 0) -> tuple[str, list[tuple[int, str]]]:
    """Возвращает (весь текст, [(page, page_text), ...])."""
    import pymupdf
    doc = pymupdf.open(str(path))
    pages: list[tuple[int, str]] = []
    all_text: list[str] = []
    total = doc.page_count if max_pages <= 0 else min(max_pages, doc.page_count)
    for i in range(total):
        t = doc[i].get_text("text", sort=True)
        t = t.replace("\xa0", " ").replace("\xad", "")
        pages.append((i + 1, t))
        all_text.append(t)
    doc.close()
    return "\n".join(all_text), pages


def _read_html(path: Path) -> str:
    """Извлекает чистый текст из HTML-документа, удаляя скрипты и стили."""
    from bs4 import BeautifulSoup
    raw = _read_txt(path)
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        tag.decompose()
    # Удаляем служебные блоки навигации «Хакера» если есть
    for el in soup.select(".article-header, .article-footer, .comments, .sidebar"):
        el.decompose()
    return soup.get_text("\n", strip=True)


def extract_text(path: Path) -> tuple[str, str, list[tuple[int, str]]]:
    """(file_type, full_text, pages) — pages пуст для TXT/HTML."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        full, pages = _read_pdf(path)
        return "pdf", full, pages
    if ext in (".html", ".htm"):
        text = _read_html(path)
        return "html", text, []
    text = _read_txt(path)
    return "txt" if ext == ".txt" else ext.lstrip("."), text, []


def chunk_document(path: Path, version: str = "0.1.0",
                   min_tokens: int = 0, max_tokens: int = 0) -> tuple[str, list[ChunkIn]]:
    """Разбивает документ на чанки с метаданными. Возвращает (тип, чанки)."""
    from .chunking import DEFAULT_MAX_TOKENS, DEFAULT_MIN_TOKENS
    min_t = min_tokens or DEFAULT_MIN_TOKENS
    max_t = max_tokens or DEFAULT_MAX_TOKENS
    file_type, _full, pages = extract_text(path)
    source = path.name
    if file_type == "pdf":
        chunks: list[ChunkIn] = []
        for page_no, page_text in pages:
            if not page_text.strip():
                continue
            chunks.extend(
                chunk_text(page_text, source, file_type="pdf", page=page_no,
                           version=version, min_tokens=min_t, max_tokens=max_t)
            )
        return file_type, chunks
    chunks = chunk_text(_full, source, "txt", None, version,
                        min_tokens=min_t, max_tokens=max_t)
    return file_type, chunks


def ingest_file(path: Path, cfg: Config, store: VectorStore, embedder: Embedder,
                reindex: bool = True, min_tokens: int = 0, max_tokens: int = 0) -> IngestResult:
    """Полный цикл: разбор -> чанкинг -> эмбеддинги -> вставка в БД."""
    try:
        file_type, chunks = chunk_document(path, cfg.version, min_tokens, max_tokens)
    except Exception as e:  # noqa: BLE001
        log.exception("Parse failed: %s", path)
        return IngestResult(source=path.name, file_type=path.suffix.lstrip(".") or "?", chunks=0, chars=0, error=str(e))
    if not chunks:
        return IngestResult(source=path.name, file_type=file_type, chunks=0, chars=0, error="Нет извлекаемого текста")

    if reindex:
        try:
            deleted = store.delete_source(path.name)
            if deleted:
                log.info("Удалены старые чанки источника %s: %s", path.name, deleted)
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось удалить старые чанки: %s", e)

    try:
        embeddings = embedder.embed([c.content for c in chunks])
    except EndpointError as e:
        return IngestResult(source=path.name, file_type=file_type, chunks=0,
                            chars=sum(len(c.content) for c in chunks), error=f"Embedding: {e}")

    store.insert(chunks, embeddings)
    chars = sum(len(c.content) for c in chunks)
    log.info("Ingested %s: %d chunks, %d chars", path.name, len(chunks), chars)
    return IngestResult(source=path.name, file_type=file_type, chunks=len(chunks), chars=chars)


SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".pdf", ".html", ".htm", ".md")


def ingest_directory(dir_path: Path, cfg: Config, store: VectorStore, embedder: Embedder,
                     patterns: tuple[str, ...] = SUPPORTED_EXTENSIONS,
                     reindex: bool = True, incremental: bool = False,
                     limit: int = 0) -> list[IngestResult]:
    # Исключаем служебные папки _files и временные файлы saved_resource
    files = sorted(
        p for p in dir_path.rglob("*")
        if p.suffix.lower() in patterns and p.is_file()
        and "_files" not in p.parts
        and not p.name.startswith("saved_resource")
    )
    if incremental:
        # Пропускаем файлы, которые уже есть в базе
        existing_sources = {s[0] for s in store.sources()}
        files = [p for p in files if p.name not in existing_sources]
        log.info("Инкрементальный режим: найдено %d новых файлов (из %d существующих в БД)",
                 len(files), len(existing_sources))
    if limit:
        files = files[:limit]
    results: list[IngestResult] = []
    total = len(files)
    for i, p in enumerate(files, 1):
        log.info("[%d/%d] %s", i, total, p.name)
        r = ingest_file(p, cfg, store, embedder, reindex=reindex)
        results.append(r)
    return results


def watch_directory(dir_path: Path, cfg: Config, store: VectorStore, embedder: Embedder,
                    interval_sec: int = 10, patterns: tuple[str, ...] = SUPPORTED_EXTENSIONS) -> None:
    """Непрерывный мониторинг директории (например Библиотека):
    при появлении новых файлов автоматически их индексирует."""
    import time
    log.info("Запущен мониторинг папки: %s (интервал %d сек). Выход: Ctrl+C", dir_path, interval_sec)
    seen_mtimes: dict[str, float] = {}
    while True:
        try:
            for p in sorted(dir_path.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in patterns:
                    continue
                mtime = p.stat().st_mtime
                last = seen_mtimes.get(str(p))
                if last is None or mtime > last:
                    seen_mtimes[str(p)] = mtime
                    log.info("Обнаружен новый/изменённый файл: %s", p.name)
                    r = ingest_file(p, cfg, store, embedder, reindex=True)
                    if r.error:
                        log.warning("Ошибка индексации %s: %s", p.name, r.error)
                    else:
                        log.info("Успешно проиндексирован: %s (%d чанков)", p.name, r.chunks)
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            log.warning("Ошибка в цикле мониторинга: %s", e)
            time.sleep(interval_sec)