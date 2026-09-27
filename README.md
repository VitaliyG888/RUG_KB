# RUG_KB — справочный Telegram-бот по компьютерной безопасности на базе RAG

RAG-приложение: локальная база знаний (PostgreSQL + pgvector) из PDF/TXT
документов по кибербезопасности; ответы формирует LLM строго по контексту;
веб-поиск используется **только** как дополнение.

Поддерживаются **русские и английские** документы и запросы (мультиязычный
поиск с кросс-язычным повтором), загрузка новых файлов через Telegram,
тематический рубрикатор, извлечение CVE/CWE, гибридный поиск
(vector + BM25/tsvector), дедупликация выдачи (MMR), история диалога и
rate limiting.

---

## Архитектура

```
PDF/TXT (Библиотека/, library_text/, uploads/)
   │  pdf_extract.py / bulk_extract.py   (уже сделано: 161 TXT в library_text/)
   ▼
[Ingestion]  app/ingest.py + app/chunking.py
   текст → кодировка/страницы → атомарные блоки (код/таблицы/списки не рвутся)
   → чанки 500–1000 токенов (overlap 100–200) → язык (ru/en/mixed) → topic/CVE/CWE
   │  эмбеддинги: OpenAI-совместимый endpoint (по умолчанию text-embedding-3-small, 1536)
   ▼
[Store]  PostgreSQL + pgvector
   таблица chunks(id, content, embedding vector(1536), metadata jsonb,
                  source, file_type, page, topic, cve[], cwe[], url,
                  language, version, created_at)
   индексы: HNSW (cosine), GIN to_tsvector('simple'), topic/source/language
   ▼
[Retrieval]  app/retriever.py + app/vectorstore.py
   гибрид: (1-w)·vec_cosine + w·BM25(tsvector) → MMR-дедупликация
   кросс-язычный повтор (ru↔en) при малочисленной выдаче
   точный приоритет чанкам с запрошенным CVE/CWE
   ▼
[RAG-цепочка]  app/rag.py  (LCEL-подобный пайплайн)
   detect_language → retriever → format_docs(нумерованные [1]..[n])
   → prompt → LLM → checker полноты
   └—— если max score < порога → веб-поиск (только дополнение)
   └—— если ответ неполный → веб-дополнение + пометка
   ▼
[Слои]  Telegram-бот (aiogram 3) · CLI · /upload → re-install в БД
```

## Структура

```
app/
  config.py       чтение .env (все настройки имеют рабочие дефолты)
  db.py           пул соединений, schema/индексы, проверка размерности
  vectorstore.py  pgvector: insert/search(hybrid+MMR)/filters/topics/stats
  chunking.py     структурно-осознанный чанкинг + оценка токенов
  topics.py       рубрикатор тем + regex CVE/CWE/URL
  lang.py         определение языка ru/en/mixed (без внешних библиотек)
  llm_client.py   OpenAI-совместимый клиент (эмбеддинги + чат) на requests
  prompts.py      шаблоны: синтез, проверка полноты, перевод, веб-запрос
  websearch.py    дакduckduckgo (ключ не нужен), bing (ключ не нужен),
                  tavily/serpapi (по ключам) — автофолбэк между ними
  retriever.py    гибридный поиск + кросс-язычный повтор + CVE/CWE
  rag.py          RAG-цепочка и форматирование ответа с источниками
  bot.py          Telegram-бот aiogram 3 (команды/FSM/rate limit/history)
  cli.py          CLI: doctor/ingest/search/ask/topics/sources/stats/bot
  models.py       pydantic-модели
scripts/
  smoke_test.py   конвейер на тестовой БД rug_kb_test + фейковый эмбеддер
  fake_openai_server.py  фейковый OpenAI-совместимый эндпоинт для тестов
  e2e_test.py     интеграционный тест против fake-эндпоинта
  setup_db.py     создать БД rug_kb + расширение pgvector
pdf_extract.py / bulk_extract.py — извлечение текста из PDF (в т.ч. «Хакер»,
  шрифты CID Identity-H с частичными ToUnicode — посимвольная реконструкция)
```

## Быстрый старт

```bash
pip install -r requirements.txt

# 0) настроить .env (cp .env.example .env)
#    Обязательно указать:
#    EMBED_BASE_URL / EMBED_MODEL и CHAT_BASE_URL / CHAT_MODEL — любой
#    OpenAI-совместимый endpoint (OpenAI, Ollama /v1, LM Studio, vLLM...)
#    BOT_TOKEN и BOT_ADMIN_IDS — для Telegram

python -m app.cli doctor          # диагностика окружения

# 1) БД (PostgreSQL 18 на 127.0.0.1:5433, user/pass postgres/postgres)
python scripts/setup_db.py

# 2) индексация одного файла или каталога
python -m app.cli ingest "Библиотека/10. Wireshark for Security Professionals.pdf"
python -m app.cli ingest library_text/N00B

# 3) поиск и вопросы
python -m app.cli search "ssrf обход фильтров"
python -m app.cli ask   "что такое SQL injection?"

# 4) Telegram-бот
python -m app.cli bot
```

## Переменные окружения (см. .env.example)

| Группа | Ключи | Назначение |
|---|---|---|
| БД | `RUG_PG_HOST/PORT/DB/USER/PASSWORD/POOL_SIZE` | PostgreSQL, дефолт `127.0.0.1:5433/rug_kb` |
| Эмбеддинги | `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL`, `EMBED_DIMENSIONS`, `EMBED_BATCH_SIZE` | dims по умолчанию 1536 (text-embedding-3-small) |
| LLM | `CHAT_BASE_URL`, `CHAT_API_KEY`, `CHAT_MODEL`, `CHAT_TEMPERATURE`, `CHAT_MAX_TOKENS` | |
| Retrieval | `RETRIEVE_TOP_K` (8), `RETRIEVE_SCORE_THRESHOLD` (0.25: ниже → веб), `RETRIEVE_HIGH_CONFIDENCE` (0.55: выше → без веб), `RETRIEVE_MIN_HITS`, `RETRIEVE_MMR`, `RETRIEVE_MMR_LAMBDA`, `RETRIEVE_HYBRID_WEIGHT`, `CROSS_LINGUAL_FALLBACK` | |
| Веб | `WEB_ENABLED`, `WEB_PROVIDERS` (duckduckgo,bing), `WEB_TOP_N`, `TAVILY_API_KEY`, `SERPAPI_API_KEY`, `SERPAPI_ENGINE` | |
| Бот | `BOT_TOKEN`, `BOT_ADMIN_IDS` (числа через запятую), `RATE_LIMIT_PER_MINUTE` (10), `HISTORY_TURNS` (6), `UPLOAD_DIR` | |

## Логика ответа

1. Сначала RAG: гибридный поиск по порогу `RETRIEVE_SCORE_THRESHOLD`.
2. Нет релевантного контекста → веб-поиск (`pipeline="web"`, пометка).
3. Есть, но `max score < RETRIEVE_HIGH_CONFIDENCE` → LLM-проверка полноты;
   неполный ответ → веб-дополнение (`pipeline="rag+web"`).
4. Мало выдачи и включён `CROSS_LINGUAL_FALLBACK` → перевод запроса на другой
   язык и повторный поиск, результаты объединяются.
5. CVE/CWE в запросе поднимают точные совпадения вверх выдачи.
6. Нет ответа нигде → честное сообщение. Галлюцинации запрещены промптом.
7. Ответ на языке запроса; код/команды — в code block.

## Telegram-бот

| Команда | Описание |
|---|---|
| `/start` `/help` | приветствие/помощь |
| текст или `/ask вопрос` | RAG-ответ с источниками `[1]..[n]`, пометкой «Источник: RAG» / «RAG + веб-дополнение» |
| `/topics` `/sources` `/stats` | темы/источники/статистика базы |
| `/upload` | (админ) загрузка PDF/TXT → ingestion (чанкинг → эмбеддинги → pgvector) |
| `/cancel` | отмена загрузки |

История диалога (последние `HISTORY_TURNS` ходов) подаётся в LLM-контекст;
rate limiting — `RATE_LIMIT_PER_MINUTE` сообщений на пользователя в минуту.
Расшифровка вида «[engine](url)» в выдачах превращается в кликабельные ссылки.

## Тестирование

```bash
python scripts/smoke_test.py            # БД rug_kb_test + фейковый эмбеддер
python scripts/fake_openai_server.py --port 8765
python scripts/e2e_test.py              # полный конвейер против fake-эндпоинта
```

## Ограничения и развитие

- **LLM/эмбеддинги**: нужен реальный OpenAI-совместимый endpoint
  (`EMBED_BASE_URL`/`CHAT_BASE_URL`). Без него `doctor` честно сообщает,
  поиск и ответы недоступны, но индексация/БД/бот работают.
- `text-embedding-3-small` заявлен в спецификации; размерность проверяется
  при старте и в `doctor`.
- MMR — жадная аппроксимация на кандидатах (без сторонних библиотек).
- Пути: `duckduckgo` может отдавать пустую выдачу (капча/регион) — фолбэк на
  bing/tavily/serpapi. При желании добавьте `searxng`.
- Чанкинг по эвристикам: «важные определения», списки и код не рвутся по
  смыслу, но идеальное сохранение блоков зависит от разметки исходника.
- Материалы распространяются для обучения/аудита/легального пентеста;
  соблюдайте лицензии источников.