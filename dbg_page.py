# -*- coding: utf-8 -*-
import pymupdf

d = pymupdf.open(r"C:\MyProject\WP_Vuln_scaner\Библиотека\2020\Хакер 2020 03(252).pdf")
p = d[15]
tr = p.get_texttrace()
rows = []
for line in tr:
    for tup in line.get("chars") or []:
        gid, cid, origin, bbox = tup[0], tup[1], tup[2], tup[3]
        rows.append((origin[1], origin[0], cid))
rows.sort()

def show(rows, tol=2.0):
    cur = None
    buf = []
    out = []
    for y, x, c in rows:
        if cur is None or abs(y - cur) > tol:
            if buf:
                out.append((cur, buf))
            cur = y
            buf = []
        buf.append((y, x, c))
    if buf:
        out.append((cur, buf))
    return out

lines = show(rows)
# print 8 lines with unicode where possible
for y, buf in lines[2:12]:
    s = ""
    for _y, x, c in buf:
        if 32 <= c < 0xE000:
            ch = chr(c)
            s += ch if ch.isprintable() else "?"
        else:
            s += "[%02x]" % c
    print("y~%6.1f n=%d: %s" % (y, len(buf), s))