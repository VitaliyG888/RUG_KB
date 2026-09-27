"""Подключение к PostgreSQL + pgvector: пул соединений и схема БД."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
import psycopg2.pool

from .config import Config

log = logging.getLogger("rug.db")

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_schema_checked = False


def get_pool(cfg: Config) -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            1, max(cfg.db.pool_size, 2), cfg.db.dsn
        )
    return _pool


@contextmanager
def conn(cfg: Config) -> Iterator[psycopg2.extensions.connection]:
    """Получить соединение из пула (не забывайте закрывать курсоры)."""
    pool = get_pool(cfg)
    c = pool.getconn()
    try:
        yield c
    finally:
        pool.putconn(c)


def test_connection(cfg: Config) -> str:
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute("SELECT version()")
            ver = cur.fetchone()[0]
            cur.execute("SELECT default_version, installed_version "
                        "FROM pg_available_extensions WHERE name='vector'")
            row = cur.fetchone()
    pgv = ver.split(",")[0]
    if row:
        if row[1]:
            return f"PostgreSQL {pgv}; pgvector установлена v{row[1]}"
        return f"PostgreSQL {pgv}; pgvector доступна (v{row[0]}, включится автоматически)"
    return f"PostgreSQL {pgv}; pgvector НЕ НАЙДЕНА на сервере"


def init_schema(cfg: Config, dimensions: int = 1536) -> None:
    """Создаёт таблицу chunks и индексы (HNSW + GIN tsvector + справочные)."""
    global _schema_checked
    if _schema_checked:
        return
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          BIGSERIAL PRIMARY KEY,
                    content     TEXT NOT NULL,
                    embedding   vector({dimensions}),
                    metadata    JSONB NOT NULL DEFAULT '{{}}',
                    source      TEXT NOT NULL,
                    file_type   TEXT NOT NULL DEFAULT 'txt',
                    page        INTEGER,
                    topic       TEXT,
                    cve         TEXT[],
                    cwe         TEXT[],
                    url         TEXT,
                    language    TEXT NOT NULL DEFAULT 'mixed',
                    version     TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_hnsw_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks (source)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_topic_idx ON chunks (topic)")
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_language_idx ON chunks (language)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks "
                "USING gin (to_tsvector('simple', content))"
            )
            # Проверка фактической размерности столбца
            cur.execute(
                "SELECT atttypmod FROM pg_attribute WHERE attrelid='chunks'::regclass AND attname='embedding'"
            )
            row = cur.fetchone()
            if row and row[0]:
                # pgvector: биты 16-24 = размерность, бит 30 = samedim
                typmod = int(row[0])
                if typmod & 0x40000000:  # VECTOR_SAMEDIM — фиксированная размерность
                    col_dim = (typmod >> 16) & 0x3FF
                    if col_dim != dimensions:
                        raise RuntimeError(
                            f"Столбец embedding имеет размерность {col_dim}, "
                            f"а конфиг задаёт {dimensions}. Удалите таблицу или выровняйте конфиг."
                        )
        c.commit()
    _schema_checked = True
    log.info("Schema ready (dimensions=%s)", dimensions)


def reset_schema(cfg: Config) -> None:
    """Удаляет таблицу (для пересоздания индекса с другими параметрами)."""
    global _schema_checked
    with conn(cfg) as c:
        with c.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS chunks")
        c.commit()
    _schema_checked = False