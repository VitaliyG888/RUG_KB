# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

d = pymupdf.open(r"C:\MyProject\WP_Vuln_scaner\Библиотека\2020\Хакер 2020 03(252).pdf")
# find the page with "GDPR" article text
for pno in range(d.page_count):
    t = d[pno].get_text("text")
    if "GDPR" in t or "Браузер" in t:
        pass
p = d[2]
tr = p.get_texttrace()
rows = []
for line in tr:
    for tup in line.get("chars") or []:
        gid, cid, origin, bbox = tup[0], tup[1], tup[2], tup[3]
        rows.append((origin[1], origin[0], cid, gid))
rows.sort()

# cluster by y
lines = []
cur = None
buf = []
for y, x, c, g in rows:
    if cur is None or abs(y - cur) > 2.0:
        if buf:
            lines.append((cur, buf))
        cur = y
        buf = []
    buf.append((y, x, c, g))
if buf:
    lines.append((cur, buf))

# print rows that have many chars (body text) with x-gaps
for y, buf in lines:
    if len(buf) < 30:
        continue
    buf.sort(key=lambda t: t[1])
    prev_x = None
    gaps = []
    for y2, x, c, g in buf:
        if prev_x is not None:
            gaps.append(x - prev_x)
        prev_x = x
    # print first 60 chars with gaps
    s = []
    prev_x = None
    for y2, x, c, g in buf[:60]:
        gp = ""
        if prev_x is not None:
            dgap = x - prev_x
            gp = " |%d| " % dgap if dgap > 3.5 else ""
        s.append(gp + chr(c) if 32 <= c < 0xE000 else gp + "?")
        prev_x = x
    print("y~%.0f n=%d" % (y, len(buf)))
    print("  " + "".join(s[:200]))
    break