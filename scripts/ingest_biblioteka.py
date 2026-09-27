"""Полная индексация всех файлов из папки ./Библиотека/ и всех её подпапок (2015..2026)."""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import init_schema
from app.ingest import ingest_directory
from app.llm_client import Embedder
from app.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("rug.biblio_ingest")


def main():
    cfg = load_config()
    init_schema(cfg, dimensions=cfg.embed.dimensions)
    store = VectorStore(cfg, dimensions=cfg.embed.dimensions)
    embedder = Embedder(
        cfg.embed.base_url,
        cfg.embed.api_key,
        cfg.embed.model,
        cfg.embed.batch_size,
        expected_dim=cfg.embed.dimensions
    )

    biblio = ROOT / "Библиотека"
    if not biblio.is_dir():
        log.error("Папка %s не найдена!", biblio)
        return

    log.info("=== Запуск полной индексации ./Библиотека/ (включая все подпапки) ===")
    
    # Рекурсивный обход папки Библиотека со всеми подпапками 2015..2026
    results = ingest_directory(
        biblio,
        cfg,
        store,
        embedder,
        patterns=(".pdf", ".txt", ".html", ".htm", ".md"),
        incremental=True,  # Пропускать то, что уже в базе
        reindex=True
    )

    ok = [r for r in results if not r.error and r.chunks > 0]
    errors = [r for r in results if r.error]
    total_chunks = sum(r.chunks for r in ok)

    log.info("=== Итог индексации Библиотеки ===")
    log.info("Успешно: %d/%d файлов", len(ok), len(results))
    log.info("Добавлено чанков: %d", total_chunks)
    if errors:
        log.warning("Ошибок: %d", len(errors))
        for e in errors[:10]:
            log.warning("  - %s: %s", e.source, e.error)


if __name__ == "__main__":
    main()
