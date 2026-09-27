# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf
import importlib.util

spec = importlib.util.spec_from_file_location("pe", r"C:\DeepSeek\MarWorckspace\pdf_extract.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

FILES = [
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2022\Хакер 2022 07(280).pdf", "2022-07"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2024\Хакер 2024 01(298).pdf", "2024-01"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2026\Хакер 2026 01(322).pdf", "2026-01"),
]

for path, label in FILES:
    d = pymupdf.open(path)
    p = d[5]  # a body page
    t = pe.extract_cid_text(p, pe.font_cmaps(d))
    # Show glyph distribution
    from collections import Counter
    cnt = Counter(ch for ch in t if not ch.isascii())
    print("=== %s page 6 ===" % label)
    print("sample:", repr(t[:300]))
    print("top non-ascii:", cnt.most_common(12))
    # Try reinterpreting as latin-1 bytes -> utf-8
    try:
        t2 = t.encode("latin-1", errors="replace").decode("utf-8", errors="replace")
        print("utf-8 reinterpret (first 200):", t2[:200])
    except Exception as e:
        print("utf8 reinterp err:", e)
    try:
        t3 = t.encode("latin-1", errors="replace").decode("cp1251", errors="replace")
        print("cp1251 reinterpret (first 200):", t3[:200])
    except Exception as e:
        print("cp1251 reinterp err:", e)
    print()