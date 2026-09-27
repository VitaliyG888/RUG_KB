"""Telegram-бот RUG_KB на aiogram 3.

Команды: /start /ask /topics /sources /upload /stats /help.
- обычные текстовые сообщения тоже трактуются как вопрос.
- /upload (администратор): загрузка PDF/TXT -> ingestion.
- Rate limiting + история диалога.
"""
from __future__ import annotations

import asyncio
import html
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import FSInputFile, Message

from .config import Config
from .ingest import IngestResult, ingest_file
from .rag import RAGChain, answer_with_sources

log = logging.getLogger("rug.bot")

MAX_REPLY = 3800  # безопасная длина сообщения Telegram


class UploadStates(StatesGroup):
    waiting_document = State()


def escape_html(text: str) -> str:
    return html.escape(text, quote=False)


def mdlinks_to_html(text: str) -> str:
    """[label](url) -> <a href=...>label</a>, затем экранирование остального."""
    import re
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", text):
        out.append(escape_html(text[pos:m.start()]))
        out.append(f'<a href="{m.group(2)}">{escape_html(m.group(1))}</a>')
        pos = m.end()
    out.append(escape_html(text[pos:]))
    return "".join(out)


def split_reply(text: str, limit: int = MAX_REPLY) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    for paragraph in text.split("\n\n"):
        if not paragraph:
            continue
        if len(parts) > 0 and len(parts[-1]) + len(paragraph) + 2 <= limit:
            parts[-1] += "\n\n" + paragraph
        else:
            # нарезаем гигантский параграф посимвольно
            while len(paragraph) > limit:
                cut = paragraph[:limit]
                nl = cut.rfind("\n")
                cut = cut[:nl] if nl > limit // 2 else cut
                parts.append(cut)
                paragraph = paragraph[len(cut):]
            if paragraph:
                parts.append(paragraph)
    return parts


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = max(1, per_minute)
        self.window = 60.0
        self._hits: dict[int, deque[float]] = {}

    def allow(self, user_id: int) -> bool:
        now = time.monotonic()
        q = self._hits.setdefault(user_id, deque())
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True

    def retry_in(self, user_id: int) -> int:
        q = self._hits.get(user_id)
        if not q or len(q) < self.per_minute:
            return 0
        return max(1, int(self.window - (time.monotonic() - q[0])))


class HistoryStore:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._turns: dict[int, deque[tuple[str, str]]] = {}
        self._ttl: dict[int, float] = {}
        self._live = 3600.0

    def get(self, chat_id: int) -> list[tuple[str, str]]:
        if time.monotonic() - self._ttl.get(chat_id, 0) > self._live:
            self._turns.pop(chat_id, None)
            return []
        return list(self._turns.get(chat_id, []))

    def push(self, chat_id: int, q: str, a: str) -> None:
        dq = self._turns.setdefault(chat_id, deque(maxlen=self.max_turns))
        dq.append((q, a))
        self._ttl[chat_id] = time.monotonic()


class RugBot:
    def __init__(self, cfg: Config, rag: RAGChain, upload_dir: Path):
        self.cfg = cfg
        self.rag = rag
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.rate = RateLimiter(cfg.bot.rate_limit_per_minute)
        self.history = HistoryStore(cfg.bot.history_turns)
        self.bot: Optional[Bot] = None
        self.dp: Optional[Dispatcher] = None

    # ------------------------------------------------------------- helpers
    def _is_admin(self, user_id: int) -> bool:
        return user_id in self.cfg.bot.admin_ids

    async def _reply(self, message: Message, text: str) -> None:
        """Отправка обычного текста: md-ссылки [t](url) превращаются в HTML-ссылки,
        остальное экранируется — безопасно для произвольного текста LLM."""
        for part in split_reply(text):
            await message.answer(mdlinks_to_html(part), parse_mode=ParseMode.HTML)

    # ---------------------------------------------------------------- routes
    def build(self) -> None:
        storage = MemoryStorage()
        self.bot = Bot(
            token=self.cfg.bot.token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self.dp = Dispatcher(storage=storage)
        r = Router()
        r.message.register(self.cmd_start, CommandStart())
        r.message.register(self.cmd_help, Command("help"))
        r.message.register(self.cmd_topics, Command("topics"))
        r.message.register(self.cmd_sources, Command("sources"))
        r.message.register(self.cmd_stats, Command("stats"))
        r.message.register(self.cmd_upload, Command("upload"))
        r.message.register(self.cmd_cancel, Command("cancel"))
        r.message.register(self.on_document, F.document, StateFilter(UploadStates.waiting_document))
        r.message.register(self.on_text, F.text)
        self.dp.include_router(r)

    # ------------------------------------------------------------------ cmd
    async def cmd_start(self, message: Message) -> None:
        uid = message.from_user.id
        adm = " (админ: /upload)" if self._is_admin(uid) else ""
        hint = ""
        if not self.cfg.bot.admin_ids:
            hint = (
                "\n\n📌 Ваш Telegram ID для прав администратора (/upload): "
                f"{uid}\nПропишите его в .env: BOT_ADMIN_IDS={uid}"
            )
        await self._reply(message, (
            "🔐 RUG_KB — справочный бот по компьютерной безопасности\n\n"
            "Пишите вопрос текстом или используйте команды:\n"
            "/ask вопрос — спросить\n"
            "/topics — список тем\n"
            "/sources — источники\n"
            "/stats — статистика базы\n"
            "/upload — добавить PDF/TXT" + adm + "\n"
            "/help — помощь\n\n"
            "Ответы берутся из локальной базы знаний; веб-поиск используется "
            "только как дополнение." + hint
        ))

    async def cmd_help(self, message: Message) -> None:
        await self.cmd_start(message)

    async def cmd_topics(self, message: Message) -> None:
        topics = self.rag.store.topics()
        if not topics:
            await self._reply(message, "База знаний пуста. Добавьте документы через /upload.")
            return
        rows = [f"• {t} — {n}" for t, n in topics[:40]]
        await self._reply(message, "Темы в базе знаний:\n" + "\n".join(rows))

    async def cmd_sources(self, message: Message) -> None:
        srcs = self.rag.store.sources()
        if not srcs:
            await self._reply(message, "База знаний пуста.")
            return
        rows = [f"• {s} ({ft}) — {n} чанков" for s, ft, n, _p in srcs[:40]]
        await self._reply(message, "Источники:\n" + "\n".join(rows))

    async def cmd_stats(self, message: Message) -> None:
        st = self.rag.store.stats()
        langs = ", ".join(f"{k}: {v}" for k, v in st["languages"].items()) or "—"
        await self._reply(message, (
            f"📊 Статистика базы знаний\n"
            f"Чанков: {st['chunks']}\n"
            f"Источников: {st['sources']}\n"
            f"Языки: {langs}"
        ))

    async def cmd_upload(self, message: Message, state: FSMContext) -> None:
        user = message.from_user
        if not self._is_admin(user.id):
            await self._reply(message, "⛔ Загрузка доступна только администратору.")
            return
        await state.set_state(UploadStates.waiting_document)
        await self._reply(message,
                          "📎 Пришлите PDF или TXT файл (до 20 МБ). "
                          "Для отмены — /cancel.")

    async def cmd_cancel(self, message: Message, state: FSMContext) -> None:
        await state.clear()
        await self._reply(message, "Отменено.")

    # ---------------------------------------------------------------- upload
    async def on_document(self, message: Message, state: FSMContext) -> None:
        user = message.from_user
        if not self._is_admin(user.id):
            await state.clear()
            await self._reply(message, "⛔ Доступ запрещён.")
            return
        doc = message.document
        fname = doc.file_name or "document.bin"
        ext = Path(fname).suffix.lower()
        if ext not in (".txt", ".pdf"):
            await self._reply(message, "Принимаются только .txt и .pdf. Попробуйте ещё раз.")
            return
        if doc.file_size and doc.file_size > 20 * 1024 * 1024:
            await self._reply(message, "Файл больше 20 МБ.")
            return

        dest = self.upload_dir / fname
        await self.bot.download(doc, destination=dest)
        await state.clear()
        ok = await message.answer("⏳ Индексирую файл…")
        result = await asyncio.to_thread(
            self._ingest, dest
        )
        if result.error:
            await ok.edit_text(f"⚠️ Ошибка индексации: {escape_html(result.error)}",
                               parse_mode=ParseMode.HTML)
        else:
            await ok.edit_text(
                f"✅ Проиндексировано: <b>{escape_html(fname)}</b>\n"
                f"Тип: {escape_html(result.file_type)} | Чанков: <b>{result.chunks}</b> "
                f"| Символов: {result.chars}",
                parse_mode=ParseMode.HTML,
            )

    def _ingest(self, dest: Path) -> IngestResult:
        store = self.rag.store
        return ingest_file(dest, self.cfg, store, self.rag.embedder, reindex=True)

    # ------------------------------------------------------------------ ask
    async def on_text(self, message: Message, state: FSMContext) -> None:
        user = message.from_user
        uid = user.id
        if not self.rate.allow(uid):
            await self._reply(message, "⏳ Слишком много запросов в минуту. Секунду…")
            return
        query = (message.text or "").strip()
        if query.startswith("/ask"):
            query = query[4:].strip()
        if not query:
            await self._reply(message, "Задайте вопрос, например: /ask Что такое SSRF?")
            return

        history = self.history.get(message.chat.id)
        typing = await message.answer("🤔 Думаю…")
        try:
            result = await asyncio.to_thread(self.rag.ask, query, history)
        except Exception as e:  # noqa: BLE001
            log.exception("Ошибка RAG")
            await typing.edit_text("⚠️ Внутренняя ошибка при обработке вопроса.")
            return
        text = answer_with_sources(result)
        parts = split_reply(text)
        if parts:
            try:
                await typing.edit_text(mdlinks_to_html(parts[0]), parse_mode=ParseMode.HTML)
            except Exception:
                await message.answer(mdlinks_to_html(parts[0]), parse_mode=ParseMode.HTML)
            for part in parts[1:]:
                await message.answer(mdlinks_to_html(part), parse_mode=ParseMode.HTML)
        self.history.push(message.chat.id, query, result.answer[:800])

        # Отправка картинок (схемы, диаграммы, примеры кода)
        if result.web_images:
            for img in result.web_images[:3]:
                try:
                    caption = f"🖼 <b>{escape_html(img.title[:150])}</b>" if img.title else "🖼 <b>Схема / Диаграмма</b>"
                    if img.page_url:
                        caption += f'\n<a href="{escape_html(img.page_url)}">Источник</a>'
                    await message.answer_photo(
                        photo=img.image_url,
                        caption=caption,
                        parse_mode=ParseMode.HTML,
                    )
                except Exception as img_err:  # noqa: BLE001
                    log.info("Не удалось отправить картинку %s: %s", img.image_url, img_err)

    # ------------------------------------------------------------------ run
    async def run(self) -> None:
        if not self.cfg.bot.token:
            raise RuntimeError(
                "BOT_TOKEN не задан. Укажите его в .env (см. .env.example) "
                "и повторите запуск бота."
            )
        self.build()
        log.info("Бот запущен (polling). Выход: Ctrl+C")
        await self.dp.start_polling(self.bot)


def run(cfg: Config, rag: RAGChain) -> None:
    app = RugBot(cfg, rag, cfg.upload_dir)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        log.info("Остановка бота")