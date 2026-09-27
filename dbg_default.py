# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

FILES = [
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2022\Хакер 2022 07(280).pdf", "2022-07"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2024\Хакер 2024 01(298).pdf", "2024-01"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2026\Хакер 2026 01(322).pdf", "2026-01"),
]
for path, label in FILES:
    d = pymupdf.open(path)
    print("=== %s ===" % label)
    for pno in [2, 5, 8]:
        t = d[pno].get_text("text", sort=True)
        cyr = sum(1 for ch in t if '\u0400' <= ch <= '\u04FF')
        print("  page %d: %d chars, %d cyrillic, sample: %r" % (pno, len(t), cyr, t[:180]))
    print()