# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import importlib.util

spec = importlib.util.spec_from_file_location("pe", r"C:\DeepSeek\MarWorckspace\pdf_extract.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

FILES = [
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2020\Хакер 2020 03(252).pdf", "2020-03"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2022\Хакер 2022 07(280).pdf", "2022-07"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2024\Хакер 2024 01(298).pdf", "2024-01"),
    (r"C:\MyProject\WP_Vuln_scaner\Библиотека\2026\Хакер 2026 01(322).pdf", "2026-01"),
    (r"C:\DeepSeek\MarWorckspace\Библиотека\2015\Хакер 2015 01(192).pdf", "2015-01"),
]
for path, label in FILES:
    import pymupdf
    d = pymupdf.open(path)
    text = pe.extract(d, force_cid=False).replace("\xa0", " ")
    cyr = sum(1 for ch in text if '\u0400' <= ch <= '\u04FF')
    print("=== %s: %d chars, %d cyr ===" % (label, len(text), cyr))
    print(text[:250].replace("\xad", "-"))
    print()