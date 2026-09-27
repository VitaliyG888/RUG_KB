"""Конфигурация приложения: читается из .env (см. .env.example) с значениями
по умолчанию. Каждое значение можно переопределить переменной окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on", "да"}


@dataclass(frozen=True)
class DBConfig:
    host: str = field(default_factory=lambda: _env("RUG_PG_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("RUG_PG_PORT", 5433))
    dbname: str = field(default_factory=lambda: _env("RUG_PG_DB", "rug_kb"))
    user: str = field(default_factory=lambda: _env("RUG_PG_USER", "postgres"))
    password: str = field(default_factory=lambda: _env("RUG_PG_PASSWORD", "postgres"))
    pool_size: int = field(default_factory=lambda: _env_int("RUG_PG_POOL_SIZE", 5))

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password} connect_timeout=10"
        )


@dataclass(frozen=True)
class EmbedConfig:
    base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL", "http://127.0.0.1:8000/v1"))
    api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY", "none"))
    model: str = field(default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small"))
    dimensions: int = field(default_factory=lambda: _env_int("EMBED_DIMENSIONS", 1536))
    batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 16))


@dataclass(frozen=True)
class ChatConfig:
    base_url: str = field(default_factory=lambda: _env("CHAT_BASE_URL", "http://127.0.0.1:8000/v1"))
    api_key: str = field(default_factory=lambda: _env("CHAT_API_KEY", "none"))
    model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: _env_float("CHAT_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _env_int("CHAT_MAX_TOKENS", 1400))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: _env_int("RETRIEVE_TOP_K", 8))
    threshold: float = field(default_factory=lambda: _env_float("RETRIEVE_SCORE_THRESHOLD", 0.25))
    high_confidence: float = field(default_factory=lambda: _env_float("RETRIEVE_HIGH_CONFIDENCE", 0.55))
    min_hits: int = field(default_factory=lambda: _env_int("RETRIEVE_MIN_HITS", 2))
    use_mmr: bool = field(default_factory=lambda: _env_bool("RETRIEVE_MMR", True))
    mmr_lambda: float = field(default_factory=lambda: _env_float("RETRIEVE_MMR_LAMBDA", 0.7))
    hybrid_weight: float = field(default_factory=lambda: _env_float("RETRIEVE_HYBRID_WEIGHT", 0.3))
    cross_lingual: bool = field(default_factory=lambda: _env_bool("CROSS_LINGUAL_FALLBACK", True))


@dataclass(frozen=True)
class WebConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("WEB_ENABLED", True))
    providers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip() for p in _env("WEB_PROVIDERS", "duckduckgo,bing").split(",") if p.strip()
        )
    )
    top_n: int = field(default_factory=lambda: _env_int("WEB_TOP_N", 5))
    tavily_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))
    serpapi_key: str = field(default_factory=lambda: _env("SERPAPI_API_KEY", ""))
    serpapi_engine: str = field(default_factory=lambda: _env("SERPAPI_ENGINE", "google"))


@dataclass(frozen=True)
class BotConfig:
    token: str = field(default_factory=lambda: _env("BOT_TOKEN", ""))
    admin_ids: tuple[int, ...] = field(
        default_factory=lambda: tuple(
            int(x) for x in _env("BOT_ADMIN_IDS", "").split(",") if x.strip().lstrip("-").isdigit()
        )
    )
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 10))
    history_turns: int = field(default_factory=lambda: _env_int("HISTORY_TURNS", 6))
    upload_dir: str = field(default_factory=lambda: _env("UPLOAD_DIR", "uploads"))


@dataclass(frozen=True)
class Config:
    db: DBConfig = field(default_factory=DBConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    web: WebConfig = field(default_factory=WebConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    version: str = field(default_factory=lambda: _env("RUG_VERSION", "0.1.0"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    @property
    def upload_dir(self) -> Path:
        p = Path(self.bot.upload_dir)
        return p if p.is_absolute() else ROOT / p


def load_config() -> Config:
    return Config()