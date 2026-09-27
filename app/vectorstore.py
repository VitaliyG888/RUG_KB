"""Хранилище векторов на PostgreSQL + pgvector: вставка, гибридный поиск,
MMR-дедупликация, фильтры, статистика."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict
from typing import Any, Iterable, Optional

import psycopg2.extras

from .config import Config
from .db import conn
from .models import ChunkIn, Retrieved

log = logging.getLogger("rug.vectorstore")

_TOKEN_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9_]+")


def vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def sanitize_ts_query(text: str) -> str:
    """Превращает произвольный запрос в гибкое OR-выражение для tsquery (поиск любого из ключевых слов)."""
    toks = _TOKEN_RE.findall(text or "")
    # Отсекаем слишком короткие служебные слова (меньше 3 символов)
    words = [t for t in toks if len(t) >= 3] or toks
    return " | ".join(words[:12]) if words else "none"


def norm_ts(rank: float) -> float:
    # Нормализация ранга ts_rank_cd в диапазон 0..1 для гибридного скоринга
    return min(1.0, rank * 3.0) if rank > 0 else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class VectorStore:
    def __init__(self, cfg: Config, dimensions: int = 1536, hybrid_weight: float | None = None):
        self.cfg = cfg
        self.dim = dimensions
        self.hybrid_w = float(cfg.retrieval.hybrid_weight) if hybrid_weight is None else hybrid_weight

    # ------------------------------------------------------------------ write
    def delete_source(self, source: str) -> int:
        with conn(self.cfg) as c:
            with c.cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE source = %s", (source,))
                n = cur.rowcount
            c.commit()
        return n

    def insert(self, chunks: Iterable[ChunkIn], embeddings: Iterable[list[float]]) -> int:
        n = 0
        with conn(self.cfg) as c:
            with c.cursor() as cur:
                for ch, emb in zip(chunks, embeddings):
                    if len(emb) != self.dim:
                        raise ValueError(
                            f"Размерность эмбеддинга {len(emb)} != {self.dim} (EMBED_DIMENSIONS). "
                            "Выровняйте конфиг под модель эмбеддингов."
                        )
                    meta = asdict(ch)
                    meta["created_at"] = ch.created_at.isoformat()
                    cur.execute(
                        """
                        INSERT INTO chunks
                            (content, embedding, metadata, source, file_type, page,
                             topic, cve, cwe, url, language, version)
                        VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            ch.content, vec_literal(emb),
                            psycopg2.extras.Json(meta), ch.source, ch.file_type, ch.page,
                            ch.topic, ch.cve or None, ch.cwe or None, ch.url,
                            ch.language, ch.version,
                        ),
                    )
                    n += 1
            c.commit()
        return n

    # ---------------------------------------------------------------- reading
    def search(
        self,
        query_text: str,
        query_vec: list[float],
        top_k: Optional[int] = None,
        threshold: float = 0.25,
        min_hits: int = 2,
        hybrid: bool = True,
        use_mmr: bool = True,
        language: Optional[str] = None,
        topic: Optional[str] = None,
        source_contains: Optional[str] = None,
        cve: Optional[str] = None,
        cwe: Optional[str] = None,
    ) -> list[Retrieved]:
        top_k = top_k or self.cfg.retrieval.top_k
        filters, fparams = self._build_filters(
            language=language, topic=topic, source_contains=source_contains,
            cve=cve, cwe=cwe,
        )
        limit = max(top_k * 4, 20)
        qv = vec_literal(query_vec) if query_vec is not None else None
        tsq = sanitize_ts_query(query_text)

        vec_rows: dict[int, dict] = {}
        text_rows: dict[int, dict] = {}
        with conn(self.cfg) as c:
            with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # --- векторные кандидаты ---
                if qv is not None:
                    cur.execute(
                        f"""
                        SELECT id, content, source, file_type, page, topic, cve, cwe,
                               url, language, version,
                               (1 - (embedding <=> %s::vector)) AS vec_score,
                               embedding
                        FROM chunks
                        WHERE (1 - (embedding <=> %s::vector)) >= {threshold * 0.4:.3f}
                              {filters}
                        ORDER BY embedding <=> %s::vector
                        LIMIT {limit}
                        """,
                        [qv, qv] + fparams + [qv],
                    )
                    for r in cur.fetchall():
                        vec_rows[int(r["id"])] = r

                if hybrid or qv is None:
                    # --- tsvector кандидаты (подзапрос с фильтрами) ---
                    cur.execute(
                        f"""
                        SELECT ch.id, ch.content, ch.source, ch.file_type, ch.page,
                               ch.topic, ch.cve, ch.cwe, ch.url, ch.language, ch.version,
                               ts_rank_cd(to_tsvector('simple', ch.content),
                                          to_tsquery('simple', %s)) AS text_score,
                               ch.embedding
                        FROM chunks ch
                        JOIN (
                            SELECT id FROM chunks
                            WHERE to_tsvector('simple', content) @@ to_tsquery('simple', %s)
                            {filters}
                        ) sel ON sel.id = ch.id
                        ORDER BY text_score DESC
                        LIMIT {limit}
                        """,
                        [tsq] + [tsq] + fparams,
                    )
                    for r in cur.fetchall():
                        text_rows[int(r["id"])] = r

        # --- объединение и нормализация ---
        merged: dict[int, tuple[Retrieved, list[float]]] = {}
        for cid, row in vec_rows.items():
            vscore = max(0.0, min(1.0, float(row.get("vec_score") or 0.0)))
            tscore = norm_ts(float(text_rows[cid].get("text_score") or 0.0)) if cid in text_rows else 0.0
            score = (1 - self.hybrid_w) * vscore + self.hybrid_w * tscore if hybrid else vscore
            merged[cid] = (self._row_to_retrieved(row, vscore, tscore, score), self._row_vec(row))
        for cid, row in text_rows.items():
            if cid in merged:
                continue
            vscore = 0.0
            tscore = norm_ts(float(row.get("text_score") or 0.0))
            score = (1 - self.hybrid_w) * vscore + self.hybrid_w * tscore if (hybrid and qv is not None) else tscore
            merged[cid] = (self._row_to_retrieved(row, vscore, tscore, score), self._row_vec(row))

        hits = sorted(merged.values(), key=lambda kv: kv[0].score, reverse=True)[: limit]
        hits = [kv for kv in hits if kv[0].score >= threshold]
        if not hits:
            return []

        if use_mmr and len(hits) > 1 and query_vec is not None:
            picked = self._mmr(query_vec, hits, top_k, self.cfg.retrieval.mmr_lambda)
        else:
            picked = [kv[0] for kv in hits[:top_k]]
        return picked

    @staticmethod
    def _row_vec(row: dict) -> list[float]:
        emb = row.get("embedding")
        if emb is None:
            return []
        if isinstance(emb, str):
            s = emb.strip().strip("[]").strip()
            return [float(x) for x in s.split(",")] if s else []
        return [float(x) for x in emb]

    @staticmethod
    def _row_to_retrieved(row: dict, vscore: float, tscore: float, score: float) -> Retrieved:
        return Retrieved(
            chunk_id=int(row["id"]),
            content=str(row["content"]),
            source=str(row["source"]),
            page=int(row["page"]) if row.get("page") is not None else None,
            topic=str(row["topic"] or ""),
            cve=list(row.get("cve") or []),
            cwe=list(row.get("cwe") or []),
            url=str(row.get("url") or ""),
            language=str(row.get("language") or "mixed"),
            score=score,
            vec_score=vscore,
            text_score=tscore,
        )

    @staticmethod
    def _mmr(query_vec: list[float],
             candidates: list[tuple[Retrieved, list[float]]],
             top_k: int, lam: float) -> list[Retrieved]:
        selected: list[Retrieved] = []
        sel_vecs: list[list[float]] = []
        remaining = candidates[:]
        while remaining and len(selected) < top_k:
            best_i, best_score = -1, -1e18
            for i, (cand, vec) in enumerate(remaining):
                relevance = cand.vec_score
                max_sim = 0.0
                for sv in sel_vecs:
                    if vec and sv:
                        s = max(0.0, cosine(vec, sv))
                        max_sim = max(max_sim, s)
                mmr_score = lam * relevance - (1 - lam) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_i = i
            if best_i < 0:
                break
            cand, vec = remaining.pop(best_i)
            selected.append(cand)
            sel_vecs.append(vec)
        return selected

    # ------------------------------------------------------------------ facets
    def topics(self) -> list[tuple[str, int]]:
        with conn(self.cfg) as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT topic, count(*) FROM chunks WHERE topic IS NOT NULL "
                    "GROUP BY topic ORDER BY count(*) DESC"
                )
                return [(r[0], r[1]) for r in cur.fetchall()]

    def sources(self) -> list[tuple[str, str, int, int]]:
        """(source, file_type, chunk_count, distinct_pages)"""
        with conn(self.cfg) as c:
            with c.cursor() as cur:
                cur.execute(
                    "SELECT source, file_type, count(*) AS n, count(DISTINCT page) AS p "
                    "FROM chunks GROUP BY source, file_type ORDER BY n DESC"
                )
                return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]

    def stats(self) -> dict[str, Any]:
        with conn(self.cfg) as c:
            with c.cursor() as cur:
                cur.execute("SELECT count(*), count(DISTINCT source) FROM chunks")
                total, sources_n = cur.fetchone()
                cur.execute(
                    "SELECT language, count(*) FROM chunks GROUP BY language ORDER BY count(*) DESC"
                )
                langs = dict(cur.fetchall())
                cur.execute(
                    "SELECT file_type, count(*) FROM chunks GROUP BY file_type ORDER BY count(*) DESC"
                )
                ftypes = dict(cur.fetchall())
                cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname='chunks'")
        return {
            "chunks": total,
            "sources": sources_n,
            "languages": langs,
            "file_types": ftypes,
        }

    # ------------------------------------------------------------------ utils
    def _build_filters(
        self,
        language: Optional[str] = None,
        topic: Optional[str] = None,
        source_contains: Optional[str] = None,
        cve: Optional[str] = None,
        cwe: Optional[str] = None,
    ) -> tuple[str, list]:
        clauses: list[str] = []
        params: list[Any] = []
        if language:
            clauses.append("language = %s")
            params.append(language)
        if topic:
            clauses.append("topic = %s")
            params.append(topic)
        if source_contains:
            clauses.append("source ILIKE %s")
            params.append(f"%{source_contains}%")
        if cve:
            clauses.append("%s = ANY(cve)")
            params.append(cve.upper())
        if cwe:
            clauses.append("%s = ANY(cwe)")
            params.append(cwe.upper())
        return (" AND " + " AND ".join(clauses)) if clauses else "", params