"""Pydantic-модели данных приложения."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

LANG_RU = "ru"
LANG_EN = "en"
LANG_MIXED = "mixed"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class ChunkIn:
    """Чанк, готовый к эмбеддингу и записи в pgvector."""
    content: str
    source: str                       # имя файла/источника
    file_type: str = "txt"            # txt | pdf
    page: int | None = None
    topic: str = ""
    cve: list[str] = field(default_factory=list)
    cwe: list[str] = field(default_factory=list)
    url: str = ""
    language: str = LANG_MIXED
    version: str = "0.1.0"
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Retrieved:
    """Один найденный фрагмент из векторной БД."""
    chunk_id: int
    content: str
    source: str
    page: int | None
    topic: str
    cve: list[str]
    cwe: list[str]
    url: str
    language: str
    score: float          # гибридный (или векторный) score, 0..1
    vec_score: float = 0.0
    text_score: float = 0.0


@dataclass(slots=True)
class WebResult:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""


@dataclass(slots=True)
class WebImage:
    """Картинка из веб-поиска (схема, график, скриншот кода)."""
    image_url: str
    title: str = ""
    page_url: str = ""
    engine: str = ""


@dataclass(slots=True)
class SourceCitation:
    """Источник для цитирования [1], [2]..."""
    index: int
    source: str
    page: int | None
    topic: str
    language: str
    content: str = ""
    score: float | None = None
    url: str = ""


@dataclass(slots=True)
class RAGAnswer:
    """Итоговый ответ RAG-цепочки."""
    answer: str
    language: str
    sources: list[SourceCitation] = field(default_factory=list)
    web_supplement: str = ""          # текст блока «Веб-дополнение»
    web_results: list[WebResult] = field(default_factory=list)
    web_images: list[WebImage] = field(default_factory=list)
    pipeline: str = "none"            # rag | rag+web | web | none
    error: str = ""
    query_language: str = LANG_MIXED