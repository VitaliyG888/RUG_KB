"""Веб-поиск как ДОПОЛНЕНИЕ к RAG.

Провайдеры: duckduckgo (ключ не нужен), bing (ключ не нужен), tavily, serpapi.
Порядок попыток задаётся конфигом WEB_PROVIDERS; при сбое одного провайдера
автоматически пробуем следующий.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .config import Config
from .lang import LANG_RU
from .models import WebImage, WebResult

log = logging.getLogger("rug.websearch")
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}
TIMEOUT = 15.0


class WebSearch:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tavily = bool(cfg.web.tavily_key)
        self._serpapi = bool(cfg.web.serpapi_key)

    def search(self, query: str, lang: str = LANG_RU,
               n: Optional[int] = None) -> list[WebResult]:
        """Возвращает до n результатов; при сбоях идёт к следующему провайдеру."""
        if not self.cfg.web.enabled:
            return []
        n = n or self.cfg.web.top_n
        results: list[WebResult] = []
        used: list[str] = []
        for name in self.cfg.web.providers:
            pname = name.strip().lower()
            if pname not in ("duckduckgo", "bing", "tavily", "serpapi"):
                log.warning("Неизвестный провайдер: %s", pname)
                continue
            if pname == "tavily" and not self._tavily:
                continue
            if pname == "serpapi" and not self._serpapi:
                continue
            try:
                got = self._run(pname, query, lang, n - len(results))
            except Exception as e:  # noqa: BLE001 — провайдер может упасть по-разному
                log.info("Провайдер %s недоступен: %s", pname, e)
                continue
            used.append(pname)
            results.extend(got)
            if len(results) >= n:
                break
        if not results:
            log.info("Веб-поиск не дал результатов (query=%r)", query[:60])
        return results[:n]

    def search_images(self, query: str, lang: str = LANG_RU,
                      n: int = 3) -> list[WebImage]:
        """Поиск картинок (схемы, диаграммы, графики, примеры кода)."""
        if not self.cfg.web.enabled or n <= 0:
            return []
        # Пробуем Tavily (если есть ключ) -> Bing Images (keyless) -> SerpAPI
        images: list[WebImage] = []
        if self._tavily:
            try:
                images = self._tavily_images(query, n)
            except Exception as e:
                log.info("Tavily images error: %s", e)
        if not images:
            try:
                images = self._bing_images(query, lang, n)
            except Exception as e:
                log.info("Bing images error: %s", e)
        if not images and self._serpapi:
            try:
                images = self._serpapi_images(query, lang, n)
            except Exception as e:
                log.info("SerpAPI images error: %s", e)
        return images[:n]

    # ------------------------------------------------------------------ impl
    def _run(self, name: str, query: str, lang: str, n: int) -> list[WebResult]:
        if n <= 0:
            return []
        if name == "duckduckgo":
            return self._duckduckgo(query, lang, n)
        if name == "bing":
            return self._bing(query, lang, n)
        if name == "tavily":
            return self._tavily_api(query, n)
        if name == "serpapi":
            return self._serpapi(query, lang, n)
        return []

    def _duckduckgo(self, query: str, lang: str, n: int) -> list[WebResult]:
        params = {"q": query, "kl": "ru-ru" if lang == LANG_RU else "us-en",
                  "o": None, "kp": "-1"}
        params.pop("o")
        r = requests.get("https://html.duckduckgo.com/html/", params=params,
                         headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        out: list[WebResult] = []
        for a in soup.select("a.result__a")[:n]:
            title = a.get_text(" ", strip=True)
            url = a.get("href", "")
            if url.startswith("//"):
                url = "https:" + url
            snip_el = a.find_next("a", class_="result__snippet")
            snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
            if title and url:
                out.append(WebResult(title=title, url=url, snippet=snippet, engine="duckduckgo"))
        return out

    def _bing(self, query: str, lang: str, n: int) -> list[WebResult]:
        params = {"q": query}
        if lang == LANG_RU:
            params["mkt"] = "ru-RU"
            params["setlang"] = "ru"
        else:
            params["mkt"] = "en-US"
            params["setlang"] = "en"
        headers = {**UA, "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5" if lang == LANG_RU else "en-US,en;q=0.9"}
        r = requests.get("https://www.bing.com/search", params=params,
                         headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        out: list[WebResult] = []
        for li in soup.select("li.b_algo")[:n]:
            h = li.select_one("h2 a")
            if not h:
                continue
            title = h.get_text(" ", strip=True)
            url = self._decode_bing_url(h.get("href", ""))
            p = li.select_one(".b_caption p, .b_lineclamp p, p")
            snippet = p.get_text(" ", strip=True) if p else ""
            if title and url:
                out.append(WebResult(title=title, url=url, snippet=snippet, engine="bing"))
        return out

    @staticmethod
    def _decode_bing_url(href: str) -> str:
        """Bing отдаёт /ck/a?u=a1<base64url> — декодируем реальный URL."""
        if "bing.com/ck/a" not in href:
            return href
        m = re.search(r"[?&]u=a1([A-Za-z0-9_\-=]+)", href)
        if not m:
            return href
        import base64
        b64 = m.group(1)
        b64 += "=" * (-len(b64) % 4)
        try:
            dec = base64.urlsafe_b64decode(b64).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return href
        return dec if dec.startswith("http") else href

    def _tavily_api(self, query: str, n: int) -> list[WebResult]:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": self.cfg.web.tavily_key, "query": query, "max_results": n},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return [
            WebResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                engine="tavily",
            )
            for item in data.get("results", [])
            if item.get("url")
        ][:n]

    def _serpapi(self, query: str, lang: str, n: int) -> list[WebResult]:
        params = {
            "engine": self.cfg.web.serpapi_engine or "google",
            "q": query,
            "api_key": self.cfg.web.serpapi_key,
            "num": str(n),
        }
        if lang == LANG_RU:
            params["gl"] = "ru"
            params["hl"] = "ru"
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("organic_results", []) or []
        return [
            WebResult(
                title=item.get("title", ""),
                url=item.get("link", item.get("url", "")),
                snippet=item.get("snippet", "") or item.get("description", ""),
                engine=self.cfg.web.serpapi_engine or "serpapi",
            )
            for item in results
            if item.get("url") or item.get("link")
        ][:n]

    # ----------------------------------------------------------- images impl
    def _bing_images(self, query: str, lang: str, n: int) -> list[WebImage]:
        import json
        params = {"q": query, "first": "1", "count": str(max(n * 4, 15))}
        if lang == LANG_RU:
            params["mkt"] = "ru-RU"
            params["setlang"] = "ru"
        else:
            params["mkt"] = "en-US"
            params["setlang"] = "en"
        headers = {
            **UA,
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7" if lang == LANG_RU else "en-US,en;q=0.9",
        }
        r = requests.get("https://www.bing.com/images/search", params=params,
                         headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        images: list[WebImage] = []
        seen: set[str] = set()
        q_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9а-яА-ЯёЁ]{3,}", query)}
        for a in soup.select("a.iusc"):
            m_raw = a.get("m")
            if not m_raw:
                continue
            try:
                m = json.loads(m_raw)
                img_url = m.get("murl", "")
                page_url = m.get("purl", "")
                title = m.get("t") or a.get("aria-label") or ""
                if not (img_url and img_url.startswith("http") and img_url not in seen):
                    continue
                # Фильтр релевантности: совпадение по заголовку или URL страницы
                hay = (title + " " + page_url).lower()
                if q_tokens and not any(t in hay for t in q_tokens):
                    continue
                seen.add(img_url)
                images.append(WebImage(
                    image_url=img_url,
                    title=title.strip(),
                    page_url=page_url,
                    engine="bing_images",
                ))
                if len(images) >= n:
                    break
            except Exception:
                continue
        if not images and lang == LANG_RU:
            # Большинство схем, графиков и кода по ИБ индексируются в международном сегменте
            return self._bing_images(query, lang="en", n=n)
        return images

    def _tavily_images(self, query: str, n: int) -> list[WebImage]:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.cfg.web.tavily_key,
                "query": query,
                "max_results": n,
                "include_images": True,
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        raw_imgs = data.get("images", []) or []
        out: list[WebImage] = []
        for img in raw_imgs:
            if isinstance(img, str) and img.startswith("http"):
                out.append(WebImage(image_url=img, engine="tavily_images"))
            elif isinstance(img, dict) and img.get("url"):
                out.append(WebImage(
                    image_url=img["url"],
                    title=img.get("description", ""),
                    engine="tavily_images",
                ))
            if len(out) >= n:
                break
        return out

    def _serpapi_images(self, query: str, lang: str, n: int) -> list[WebImage]:
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": self.cfg.web.serpapi_key,
            "num": str(n),
        }
        if lang == LANG_RU:
            params["gl"] = "ru"
            params["hl"] = "ru"
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("images_results", []) or []
        out: list[WebImage] = []
        for item in results:
            img = item.get("original") or item.get("thumbnail")
            if img:
                out.append(WebImage(
                    image_url=img,
                    title=item.get("title", ""),
                    page_url=item.get("link", ""),
                    engine="serpapi_images",
                ))
            if len(out) >= n:
                break
        return out