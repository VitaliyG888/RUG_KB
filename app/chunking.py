"""Структурно-осознанный чанкинг: 500–1000 токенов, overlap 100–200.

Блоки (параграфы) атомарны: код, таблицы, списки и определения не
разрываются. Если блок больше целевого размера (огромный код/таблица),
он режется по строкам. Блоки упаковываются в чанки жадным алгоритмом
с перекрытием за счёт повтора хвоста предыдущего чанка.
"""
from __future__ import annotations

import re
from typing import Optional

from .lang import detect_language
from .models import ChunkIn
from .topics import classify_topic, extract_cve, extract_cwe, extract_url

DEFAULT_MIN_TOKENS = 500
DEFAULT_MAX_TOKENS = 1000
OVERLAP_MIN = 100
OVERLAP_MAX = 200

# ---------------------------------------------------------------------------
# Оценка размера в токенах (BPE-эвристика; tiktoken, если доступен)
# ---------------------------------------------------------------------------
_CYR = re.compile(r"[а-яёА-ЯЁ]")
_LAT = re.compile(r"[a-zA-Z]")

_tiktoken_enc = None
_tiktoken_tried = False


def _get_tokenizer():
    global _tiktoken_enc, _tiktoken_tried
    if not _tiktoken_tried:
        _tiktoken_tried = True
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = None
    return _tiktoken_enc


def estimate_tokens(text: str) -> int:
    enc = _get_tokenizer()
    if enc is not None:
        try:
            return max(1, len(enc.encode(text)))
        except Exception:
            pass
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    other = max(0, len(text) - cyr - lat)
    return max(1, int(cyr / 2.9 + lat / 4.2 + other / 5))


# ---------------------------------------------------------------------------
# Классификация и разбиение на атомарные единицы
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$|^\s*[+\-|][\-\s]")
_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)]|\b[a-z]\))\s+")
_CODE_RE = re.compile(
    r"^\s{4,}|\t|^>>>|^\.\.\.|^import |^from |^def |^class |^SELECT |^INSERT "
    r"|^UPDATE |^DELETE |^CREATE |^-- |^/\*|^#include|^<\?php|^#!/|^<script"
)
_HEADING_SINGLE_RE = re.compile(r"^#{1,6}\s+")

Block = tuple[str, str]  # (kind, text); kind: para|code|table|list|heading


def _classify_block(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return "para"
    first = lines[0]
    if _FENCE_RE.match(first):
        return "code"
    if _TABLE_RE.match(first):
        pipes = sum(1 for ln in lines if "|" in ln) / max(1, len(lines))
        return "table" if pipes >= 0.6 else "para"
    if any(_LIST_RE.match(ln) for ln in lines[:3]):
        return "list"
    if _HEADING_SINGLE_RE.match(first):
        return "heading"
    if len(lines) == 1 and len(first) <= 70 and first.endswith(":"):
        return "heading"
    if _CODE_RE.match(first):
        return "code"
    # Эвристика кода: строки с ';', '(', '=' или длиннее 90 символов
    if len(lines) >= 2 and sum(
        1 for ln in lines[:3] if ("(" in ln or ";" in ln or "=" in ln or len(ln) > 90)
    ) >= 2:
        return "code"
    return "para"


def _split_large_block(kind: str, text: str, max_tok: int) -> list[str]:
    """Разбивает слишком большой блок по строкам (не по смыслу внутри строки)."""
    if estimate_tokens(text) <= max_tok:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in text.splitlines():
        tok = max(1, estimate_tokens(line) + 1)
        if cur_tok + tok > max_tok and cur:
            chunks.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += tok
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _blocks_from_text(text: str, max_tokens: int) -> list[Block]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n", text)
    units: list[Block] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        kind = _classify_block(p)
        if kind == "heading":
            units.append((kind, p))
            continue
        for piece in _split_large_block(kind, p, max_tokens):
            if piece.strip():
                units.append((kind, piece.strip()))
    return units


def _pack(units: list[Block], source: str, file_type: str, page: Optional[int],
          version: str, min_tokens: int, max_tokens: int) -> list[ChunkIn]:
    chunks: list[ChunkIn] = []
    window: list[str] = []
    window_tok = 0

    def flush():
        nonlocal window, window_tok
        if not window:
            return
        content = "\n\n".join(window)
        chunks.append(
            ChunkIn(
                content=content,
                source=source,
                file_type=file_type,
                page=page,
                topic=classify_topic(content),
                cve=extract_cve(content),
                cwe=extract_cwe(content),
                url=extract_url(content),
                language=detect_language(content),
                version=version,
            )
        )
        window, window_tok = [], 0

    for kind, text in units:
        tok = max(1, estimate_tokens(text))

        if kind == "heading":
            # Граница темы: хедер всегда начинает новый чанк
            flush()
            window.append(text)
            window_tok = tok
            continue

        if window and window_tok + tok > max_tokens:
            if len(window) == 1:
                # Один гигантский параграф (важное определение) — не резать
                flush()
                window.append(text)
                window_tok = tok
                continue
            # Перекрытие: хвост прошлого чанка повторяем в начале нового
            tail = window[-1]
            tail_tok = estimate_tokens(tail)
            flush()
            if OVERLAP_MIN <= tail_tok <= OVERLAP_MAX:
                window.append(tail)
                window_tok = tail_tok
            elif tail_tok < OVERLAP_MIN:
                shift = int(len(tail) * (OVERLAP_MIN / max(1, tail_tok)))
                piece = ("…" + tail[-shift:]) if shift < len(tail) else tail
                window.append(piece)
                window_tok += estimate_tokens(piece)
            window.append(text)
            window_tok += tok
            continue

        window.append(text)
        window_tok += tok

        # Не держим очень маленькие хвостовые чанки: если следующий блок уже
        # известен заранее нельзя, поэтому закрываем по верхней границе выше.
    flush()
    return chunks


def chunk_text(text: str, source: str, file_type: str = "txt",
               page: Optional[int] = None, version: str = "0.1.0",
               min_tokens: int = DEFAULT_MIN_TOKENS,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> list[ChunkIn]:
    """Разбивает текст документа на чанки с метаданными."""
    if not text or not text.strip():
        return []
    units = _blocks_from_text(text, max_tokens=max_tokens)
    return _pack(units, source, file_type, page, version, min_tokens, max_tokens)