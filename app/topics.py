"""Тематический рубрикатор и извлечение CVE/CWE/URL из текста.

Порядок тем важен: сначала проверяются более специфичные темы.
Если ни одна тема не подошла — fallback по группе «Общие темы».
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# ИЗВЛЕЧЕНИЕ сущностей
# ---------------------------------------------------------------------------
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_CWE_RE = re.compile(r"CWE-\d{1,5}", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)


def extract_cve(text: str) -> list[str]:
    return sorted({m.upper() for m in _CVE_RE.findall(text)})


def extract_cwe(text: str) -> list[str]:
    return sorted({m.upper() for m in _CWE_RE.findall(text)})


def extract_url(text: str) -> str:
    m = _URL_RE.search(text)
    return m.group(0).rstrip(".,;:") if m else ""


# ---------------------------------------------------------------------------
# РУБРИКАТОР
# ---------------------------------------------------------------------------
Topic = tuple[str, str, list[str]]  # (group, topic, keywords)

RUBRICATOR: list[Topic] = [
    # --- Инъекции ---
    ("Инъекции", "SQL injection", ["sql injection", "sql инъекц", "sql-injection", "sqli", "sqli", "union-based", "blind sql"]),
    ("Инъекции", "OS command injection", ["command injection", "командная инъекция", "os command", "oscmd", "system()", "exec("]),
    ("Инъекции", "NoSQL injection", ["nosql injection", "nosql-инъекц", "mongodb injection"]),
    ("Инъекции", "Server-side template injection", ["server-side template", "ssti", "шаблон", "template injection", "freemarker", "jinja2", "twig"]),
    ("Инъекции", "XXE injection", ["xxe", "xml external entity", "внешняя сущность", "external entity"]),
    ("Инъекции", "SSRF", ["ssrf", "server-side request forgery", "серверный запрос"]),
    ("Инъекции", "HTTP request smuggling", ["request smuggling", "smuggling", "te.cl", "te/te", "cl.te"]),
    ("Инъекции", "Prototype pollution", ["prototype pollution", "prototype poisoning", "загрязнение прототипа"]),
    ("Инъекции", "GraphQL API vulnerabilities", ["graphql", "batching", "graphql introspection", "aliasing"]),
    ("Инъекции", "Web LLM attacks", ["llm attack", "prompt injection", "ллм", "llm", "prompt leak", "indirect prompt"]),
    # --- Клиентские атаки ---
    ("Клиентские атаки", "XSS", ["xss", "cross-site scripting", "межсайтовый скриптинг", "скриптинг", "reflected", "stored xss", "polyglot"]),
    ("Клиентские атаки", "CSRF", ["csrf", "cross-site request forgery", "межсайтовая подделка"]),
    ("Клиентские атаки", "Clickjacking", ["clickjacking", "кликджекинг", "frame busting", "x-frame-options", "click jack"]),
    ("Клиентские атаки", "DOM-based vulnerabilities", ["dom-based", "dom xss", "document.location", "innerhtml", "dom clobbering"]),
    ("Клиентские атаки", "CORS", ["cors", "cross-origin resource sharing", "cross-origin", "preflight", "access-control-allow-origin"]),
    ("Клиентские атаки", "WebSockets", ["websocket", "ws://", "wss://", "cross-site websocket"]),
    # --- Доступ и аутентификация ---
    ("Доступ и аутентификация", "Access control vulnerabilities", ["access control", "broken access", "idor", "управление доступом", "privilege escalation", "горизонтал", "вертикал", "небезопасн* прямые ссылки", "insecure direct"]),
    ("Доступ и аутентификация", "Authentication", ["authentication", "аутентификац", "login", "логин", "брутфорс", "brute force", "bruteforce", "credential stuffing", "default credentials", "пароль"]),
    ("Доступ и аутентификация", "OAuth authentication", ["oauth", "oauth2", "authorization code", "implicit flow", "client credentials", "openid", "oidc"]),
    ("Доступ и аутентификация", "JWT", ["jwt", "json web token", "jsonwebtoken", "jwk", "alg=none", "rs256", "hs256"]),
    ("Доступ и аутентификация", "File upload vulnerabilities", ["file upload", "загрузка файла", "upload", "webshell", "mime", "polyglot file", "zip slip"]),
    ("Доступ и аутентификация", "Business logic vulnerabilities", ["business logic", "бизнес-логик", "logic flaw", "логическая уязвимост"]),
    ("Доступ и аутентификация", "Race conditions", ["race condition", "race on", "time-of-check", "toctou", "состояние гонки", "гонка", "turbotube", "single packet attack"]),
    # --- Серверные и инфраструктурные ---
    ("Серверные и инфраструктурные", "Path traversal", ["path traversal", "directory traversal", "обход каталога", "../", "..%2f", "traversal"]),
    ("Серверные и инфраструктурные", "Insecure deserialization", ["deserialization", "десериализац", "serialize", "pickle", "yaml.load", "ysoserial", "rce gadget"]),
    ("Серверные и инфраструктурные", "Information disclosure", ["information disclosure", "раскрытие информации", "disclosure", "утечка", "leak", "verbose error", "stack trace"]),
    ("Серверные и инфраструктурные", "Web cache poisoning", ["cache poisoning", "отравление кэша", "web cache", "cdn cache", "cache key"]),
    ("Серверные и инфраструктурные", "Web cache deception", ["cache deception", "cache poisoning", "web cache deception"]),
    ("Серверные и инфраструктурные", "HTTP Host header attacks", ["host header", "host-header", "host header injection", "absolute url", "virtual host"]),
    # --- Общие темы ---
    ("Общие темы", "Essential skills", ["burp suite", "burpsuite", "recon", "разведка", "osint", "essential skills", "базовые навыки", "fuzzing", "фаззинг", "интерсептор", "intercept"]),
    ("Общие темы", "API testing", ["api testing", "тестирование api", "rest api", "graphql api", "swagger", "openapi", "postman", "api security"]),
    ("Общие темы", "Vulnerabilities", ["vulnerability", "уязвимост", "cve-", "cwe-", "exploit", "эксплойт", "0-day", "zero-day", "уязвимост"]),
    ("Общие темы", "Пентестинг", ["пентест", "pentest", "penetration test", "проникновение", "ручные тест", "методолог"]),
    ("Общие темы", "Инструменты для пентестинга", ["nmap", "burp", "sqlmap", "metasploit", "msfconsole", "nikto", "gobuster", "ffuf", "wfuzz", "hydra", "john", "hashcat", "nessus", "openvas", "aircrack", "wireshark", "ettercap", "bettercap", "mimikatz", "linpeas", "pspy", "subfinder", "amass", "dirb", "cewl", "responder", "impacket", "crackmapexec", "nxc", "bloodhound", "ligolo", "chisel"]),
    ("Общие темы", "Инструменты для взлома", ["взлом", "взлома", "хакер", "hack", "крэк", "crack", "кейген", "keygen", "взломан"]),
    ("Общие темы", "Методы взлома", ["методы взлома", "способы взлома", "техники взлома", "атаки на", "эксплуатация"]),
    ("Общие темы", "Методы пентестинга", ["методы пентестинга", "способы пентестинга", "подход", "методология", "assessment", "testing guide"]),
    ("Общие темы", "Инструменты для проникновения", ["проникновение", "penetration", "продвижение", "pivoting", "пивотинг", "lateral", "горизонтальное передвижение", "c2", "command and control"]),
    ("Общие темы", "Сканеры для нахождения vulnerabilities", ["сканер", "scanner", "скан уязвимост", "vulnerability scan", "retina", "qualys", "acunetix", "owasp zap", "zap", "nuclei", "vikunja"]),
    ("Общие темы", "Database vulnerabilities", ["database vulnerability", "sql server", "postgresql", "oracle database", "mysql", "mongodb", "redis", "база данных", "бд ", "tiberius", "noSQL", "привилегий в бд"]),
    ("Общие темы", "Программирование", ["программирован", "python", "скрипт", "код", "source code", "исходный код", "раскодирование", "декомпил", "reverse engineering", "реверс", "assembler", "asm", "shellcode"]),
]

_FALLBACK_GROUP = "Общие темы"
_FALLBACK_TOPIC = "Vulnerabilities / Уязвимости"
# Тема "инструменты по умолчанию" для технических фрагментов без совпадений
_FALLBACK_TECH = "Программирование"

_last_regex_cache: dict[str, re.Pattern] = {}


def _keyword_re(keyword: str) -> re.Pattern:
    """Собираем regex из ключевого слова (без учёта регистра и окончаний по \b)."""
    pat = _last_regex_cache.get(keyword)
    if pat is None:
        # Для коротких токенов (sql, xss, jwt) не применяем \b — иначе ru-текст не матчится.
        if re.fullmatch(r"[a-zA-Z0-9_./*-]{1,6}", keyword):
            pat = re.compile(re.escape(keyword), re.IGNORECASE)
        else:
            pat = re.compile(r"(?<![a-zA-Zа-яА-ЯёЁ0-9])" + re.escape(keyword) + r"(?![a-zA-Zа-яА-ЯёЁ0-9])", re.IGNORECASE)
        _last_regex_cache[keyword] = pat
    return pat


def classify_topic(text: str) -> str:
    """Возвращает название темы (rubricator) или fallback."""
    low = text
    for _group, topic, keywords in RUBRICATOR:
        for kw in keywords:
            if not kw:
                continue
            try:
                if _keyword_re(kw).search(low):
                    return topic
            except re.error:
                if kw.lower() in low.lower():
                    return topic
    # Технический fallback: есть код/CVE, но не распознано специализированно
    if extract_cve(text) or re.search(r"\b(select|insert|update|curl|wget|nmap|python|import|def |GET |POST )\b", low, re.IGNORECASE):
        return _FALLBACK_TECH
    return _FALLBACK_TOPIC


def groups() -> list[str]:
    seen: list[str] = []
    for g, _t, _k in RUBRICATOR:
        if g not in seen:
            seen.append(g)
    return seen


def topics_by_group() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for g, t, _k in RUBRICATOR:
        out.setdefault(g, [])
        if t not in out[g]:
            out[g].append(t)
    return out