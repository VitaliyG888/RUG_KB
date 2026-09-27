"""Определение языка текста без внешних зависимостей.

Тексты русского/английского пентест-домена определяются по соотношению
кириллицы и латиницы (плюс служебные символы исключаются).
"""
from __future__ import annotations

import re

from .models import LANG_EN, LANG_MIXED, LANG_RU

_CYR = re.compile(r"[а-яёА-ЯЁ]")
_LAT = re.compile(r"[a-zA-Z]")
_CODE_LIKE = re.compile(r"(^|[^a-zA-Zа-яА-ЯёЁ])(http|www\.|CVE-|CWE-|[A-Z]{2,}|\b0x[0-9a-fA-F]+|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")


def detect_language(text: str) -> str:
    """Определяет язык фрагмента: ru | en | mixed."""
    if not text:
        return LANG_MIXED
    cyr = len(_CYR.findall(text))
    lat = len(_LAT.findall(text))
    total = cyr + lat
    if total == 0:
        return LANG_MIXED
    ru_ratio = cyr / total
    if ru_ratio > 0.35:
        return LANG_RU
    if ru_ratio < 0.05:
        return LANG_EN
    # Между 5% и 35% кириллицы — проверяем, не технический ли это текст
    code_hits = len(_CODE_LIKE.findall(text))
    if code_hits > 3 and ru_ratio < 0.15:
        return LANG_EN
    return LANG_MIXED


def dominant_language(text: str) -> str:
    lang = detect_language(text)
    return LANG_RU if lang == LANG_MIXED else lang