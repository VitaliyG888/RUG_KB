"""Фейковый OpenAI-совместимый эндпоинт для интеграционных тестов.

POST /v1/embeddings      -> детерминированные векторы (хэш текста)
POST /v1/chat/completions -> canned-ответы (synthesis/checker/translate/web-query)
GET  /v1/models           -> список моделей

Запуск:  python scripts/fake_openai_server.py --port 8765 --dims 1536
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def embed_text(text: str, dim: int) -> list[float]:
    toks = re.findall(r"[a-zа-яё0-9_]+", text.lower())
    vec = [0.0] * dim
    for tok in toks:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        rnd = random.Random(h)
        for i in range(dim):
            vec[i] += rnd.uniform(-1.0, 1.0)
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / n, 6) for x in vec]


class Handler(BaseHTTPRequestHandler):
    dim = 1536

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": "test-embed", "object": "model"},
                {"id": "test-chat", "object": "model"},
            ]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        if path == "/v1/embeddings":
            texts = payload.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            data = [{"object": "embedding", "index": i,
                     "embedding": embed_text(t, self.dim)} for i, t in enumerate(texts)]
            self._json({"object": "list", "data": data,
                        "model": payload.get("model", ""), "usage": {"total_tokens": 1}})
        elif path == "/v1/chat/completions":
            messages = payload.get("messages", [])
            last = messages[-1]["content"] if messages else ""
            # Simple canned logic keyed on the system prompt
            sysp = messages[0]["content"] if messages else ""
            if "качества RAG" in sysp or "RAG quality checker" in sysp or "response_format" in str(payload.get("response_format", "")):
                answer = json.dumps({"sufficient": True, "missing": ""})
            elif "переведи" in sysp.lower() or "translate" in sysp.lower():
                answer = last  # echo: для тестов перевод = исходный текст
            elif "поисковый запрос" in sysp or "search query" in sysp:
                answer = "sql injection mitigation"
            elif "Веб-дополнение" in sysp or "Web supplement" in sysp:
                answer = "Дополнение: проверено по официальной документации портсвигер (см. ссылки)."
            else:
                refs = [m for m in re.findall(r"\[(\d+)\]", last)]
                answer = ("Fake-ответ RAG. Ключевые факты из контекста: [%s]. "
                          "Подробнее в источниках. " % (", ".join(dict.fromkeys(refs)) or "нет"))
            self._json({
                "id": "fake", "object": "chat.completion", "model": payload.get("model", ""),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                             "finish_reason": "stop"}],
            })
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dims", type=int, default=1536)
    args = ap.parse_args()
    Handler.dim = args.dims
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fake-openai listening on http://127.0.0.1:{args.port}/v1 (dims={args.dims})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()