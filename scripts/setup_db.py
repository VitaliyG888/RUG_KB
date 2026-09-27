"""Создание БД rug_kb и проверка pgvector на локальном PostgreSQL.

Использует параметры из .env (RUG_PG_*). Подключение к БД 'postgres',
создаёт базу rug_kb и расширение vector.

Запуск:  python scripts/setup_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2

from app.config import load_config


def main() -> int:
    cfg = load_config().db
    c = psycopg2.connect(host=cfg.host, port=cfg.port, dbname="postgres",
                         user=cfg.user, password=cfg.password)
    try:
        c.autocommit = True
        cur = c.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (cfg.dbname,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{cfg.dbname}"')
            print(f"OK: база {cfg.dbname} создана")
        else:
            print(f"OK: база {cfg.dbname} уже есть")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("OK: расширение pgvector создано")
        cur.close()
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())