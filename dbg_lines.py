# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf
import importlib.util
spec = importlib.util.spec_from_file_location("pe", r"C:\DeepSeek\MarWorckspace\pdf_extract.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

d = pymupdf.open(r"C:\MyProject\WP_Vuln_scaner\Библиотека\2020\Хакер 2020 03(252).pdf")
cmaps = pe.font_cmaps(d)

for pno in [2, 3, 4, 5]:
    p = d[pno]
    name_map = {}
    for f in p.get_fonts(full=True):
        name_map[f[3]] = cmaps.get(f[0], {})
    tr = p.get_texttrace()
    print("=== page", pno, "===")
    cnt = 0
    for line in tr:
        chars = line.get("chars") or []
        if not chars:
            continue
        cmap = name_map.get(line.get("font"), {})
        s = "".join(pe.decode_char(cmap, t[1], t[0]) for t in chars)
        if s.strip() and cnt < 12:
            print("  y=%.1f font=%s | %s" % (chars[0][2][1], line.get("font"), s[:60]))
            cnt += 1
    print()