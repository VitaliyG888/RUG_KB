"""Массовая индексация всех папок 2015..2025 и папки Библиотека."""
import sys
import logging
from pathlib import Path

# Добавляем корень проекта в sys.path
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
log = logging.getLogger("rug.bulk_ingest")


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

    years = [str(y) for y in range(2015, 2026)]
    total_chunks = 0
    total_files = 0

    # 1. Индексация текстовых архивов по годам library_text/2015..2025
    for year in years:
        ydir = ROOT / "library_text" / year
        if not ydir.is_dir():
            log.warning("Папка не найдена: %s", ydir)
            continue
        log.info("=== Начинаем индексацию года: %s ===", year)
        results = ingest_directory(ydir, cfg, store, embedder, incremental=True, reindex=True)
        ok = [r for r in results if not r.error and r.chunks > 0]
        total_files += len(ok)
        chunks_added = sum(r.chunks for r in ok)
        total_chunks += chunks_added
        log.info("=== Год %s завершён: %d/%d файлов, +%d чанков ===",
                 year, len(ok), len(results), chunks_added)

    # 2. Индексация папки Библиотека (PDF-файлы)
    biblio_dir = ROOT / "Библиотека"
    if biblio_dir.is_dir():
        log.info("=== Начинаем индексацию папки Библиотека (PDF) ===")
        b_results = ingest_directory(biblio_dir, cfg, store, embedder, incremental=True, reindex=True)
        b_ok = [r for r in b_results if not r.error and r.chunks > 0]
        total_files += len(b_ok)
        total_chunks += sum(r.chunks for r in b_ok)
        log.info("=== Библиотека завершена: %d/%d файлов ===", len(b_ok), len(b_results))

    log.info("🎉 Все папки проиндексированы! Всего добавлено файлов: %d, чанков: %d",
             total_files, total_chunks)


if __name__ == "__main__":
    main()
