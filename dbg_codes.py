# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf
import importlib.util
spec = importlib.util.spec_from_file_location("pe", r"C:\DeepSeek\MarWorckspace\pdf_extract.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

d = pymupdf.open(r"C:\MyProject\WP_Vuln_scaner\Библиотека\2020\Хакер 2020 03(252).pdf")
p = d[2]
cmaps = pe.font_cmaps(d)
name_map = {}
for f in p.get_fonts(full=True):
    name_map[f[3]] = cmaps.get(f[0], {})
for line in p.get_texttrace():
    chars = line.get("chars") or []
    if not chars:
        continue
    cmap = name_map.get(line.get("font"), {})
    s = "".join(pe.decode_char(cmap, t[1], t[0]) for t in chars)
    if "Ред" in s or "кий" in s:
        # show codepoints
        print(line.get("font"), "|", " ".join("U+%04X" % ord(ch) if ord(ch) > 255 else repr(ch) for ch in s[:40]))