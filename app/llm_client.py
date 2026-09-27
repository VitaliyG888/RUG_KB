"""OpenAI-совместимый клиент эмбеддингов и чат-LLM через HTTP.

Работает с любым сервером, реализующим OpenAI wire-протокол:
OpenAI, DeepSeek, Ollama (/v1), LM Studio, vLLM, llama.cpp и т.д.
Используем requests (без зависимостей от оф. SDK), с таймаутами и ретраями.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

log = logging.getLogger("rug.llm")

DEFAULT_TIMEOUT = 60.0
HEADERS = {"Content-Type": "application/json", "User-Agent": "rug-kb/0.1"}


class EndpointError(RuntimeError):
    """Сервер недоступен или вернул ошибку — приложение должно сказать об этом честно."""


class _BaseClient:
    def __init__(self, base_url: str, api_key: str = "none", timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        if api_key and api_key.lower() not in ("none", "empty", "sk-nokey", ""):
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_err: Exception | None = None
        for attempt in (0, 1):
            try:
                r = self.session.post(url, data=json.dumps(payload), timeout=timeout or self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    raise EndpointError(
                        f"Маршрут {path} не найден на {self.base_url} "
                        "(эндпоинт не OpenAI-совместимый или вернул 404)"
                    )
                raise EndpointError(f"HTTP {r.status_code}: {r.text[:400]}")
            except EndpointError:
                raise
            except Exception as e:  # network-level
                last_err = e
                if attempt == 0:
                    time.sleep(0.5)
        raise EndpointError(f"Не удалось связаться с {self.base_url}: {last_err}")

    def _get(self, path: str, timeout: Optional[float] = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            r = self.session.get(url, timeout=timeout or min(self.timeout, 15))
            if r.status_code == 200:
                return r.json()
            raise EndpointError(f"HTTP {r.status_code}: {r.text[:300]}")
        except EndpointError:
            raise
        except Exception as e:
            raise EndpointError(f"Не удалось связаться с {self.base_url}: {e}")

    def check_models(self) -> list[str]:
        """Список моделей на сервере. При недоступности endpoint — EndpointError."""
        data = self._get("models")
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


class Embedder(_BaseClient):
    def __init__(self, base_url: str, api_key: str = "none", model: str = "text-embedding-3-small",
                 batch_size: int = 16, expected_dim: Optional[int] = None, timeout: float = DEFAULT_TIMEOUT):
        super().__init__(base_url, api_key, timeout)
        self.model = model
        self.batch_size = max(1, batch_size)
        self._dim: Optional[int] = None
        self._expected = expected_dim

    @property
    def dimensions(self) -> Optional[int]:
        return self._dim or self._expected

    def _embed_one(self, texts: list[str]) -> list[list[float]]:
        data = self._post("embeddings", {"model": self.model, "input": texts})
        out = data.get("data") or []
        if len(out) != len(texts):
            raise EndpointError("Сервер вернул неверное число эмбеддингов")
        return [list(map(float, d["embedding"])) for d in out]

    @staticmethod
    def _is_context_error(err: Exception) -> bool:
        s = str(err).lower()
        return "exceeds the maximum context" in s or "context length" in s or "too long" in s

    def _embed_adaptive(self, batch: list[str]) -> list[list[float]]:
        """Пробует батч; при ошибке длины контекста делит его пополам."""
        try:
            return self._embed_one(batch)
        except EndpointError as e:
            if len(batch) == 1 or not self._is_context_error(e):
                raise
            mid = len(batch) // 2
            left = self._embed_adaptive(batch[:mid])
            right = self._embed_adaptive(batch[mid:])
            return left + right

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Батч-эмбеддинг списка текстов. Возвращает списки float."""
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t for t in texts[i:i + self.batch_size] if t and t.strip()] or [" "]
            vectors = self._embed_adaptive(batch)
            if self._dim is None and vectors:
                self._dim = len(vectors[0])
            for v in vectors:
                if self._expected and len(v) != self._expected:
                    raise EndpointError(
                        f"Модель '{self.model}' вернула эмбеддинг {len(v)}-мерный, "
                        f"а EMBED_DIMENSIONS={self._expected}. Исправьте конфиг."
                    )
            results.extend(vectors)
        return results

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def probe(self) -> int:
        """Проверяет доступность и возвращает размерность эмбеддингов."""
        _ = self.embed(["rug-kb probe"])
        if self._dim is None:
            raise EndpointError("Не удалось определить размерность эмбеддинга")
        return self._dim


class ChatClient(_BaseClient):
    def __init__(self, base_url: str, api_key: str = "none", model: str = "gpt-4o-mini",
                 temperature: float = 0.2, max_tokens: int = 1400, timeout: float = DEFAULT_TIMEOUT):
        super().__init__(base_url, api_key, timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, json_mode: bool = False) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise EndpointError("Сервер вернул пустой choices")
        content = choices[0].get("message", {}).get("content")
        if content is None or content == "":
            raise EndpointError("Модель вернула пустой ответ")
        return str(content)

    def ping(self) -> bool:
        try:
            return bool(self.check_models()) or True
        except EndpointError:
            return False