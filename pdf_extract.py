#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract text from PDFs.

Normal PDFs (with proper ToUnicode CMaps or standard fonts): standard
pymupdf get_text() is fine.

Magazine PDFs from "Хакер" publisher use CID Identity-H fonts with PARTIAL
ToUnicode CMaps, so pymupdf drops Cyrillic chars. This script re-extracts
per-character (gid, cid) via page.get_texttrace() and rebuilds unicode using
the parsed ToUnicode CMap of each font, with identity fallback for ASCII.

Usage:
    python pdf_extract.py <in.pdf> <out.txt> [--force-cid]
"""
import argparse
import re
import sys

import pymupdf


def parse_cmap(cmap_text):
    """Parse a ToUnicode CMap into {cid: unicode_code_point}."""
    mapping = {}
    for blk in re.findall(r"beginbfchar\n(.*?)\nendbfchar", cmap_text, re.S):
        for line in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
            mapping[int(line[0], 16)] = int(line[1], 16)
    for blk in re.findall(r"beginbfrange\n(.*?)\nendbfrange", cmap_text, re.S):
        for line in blk.splitlines():
            line = line.strip()
            m = re.match(
                r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", line)
            if m:
                lo = int(m.group(1), 16)
                hi = int(m.group(2), 16)
                base = int(m.group(3), 16)
                for i in range(lo, hi + 1):
                    mapping[i] = base + (i - lo)
    return mapping


def font_cmaps(doc):
    """Return {font_xref: {cid: codepoint}} for all fonts in doc."""
    cmaps = {}
    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref)
        except Exception:
            continue
        m = re.search(r"ToUnicode\s+(\d+)\s+(\d+)\s+R", obj)
        if not m:
            continue
        try:
            raw = doc.xref_stream(int(m.group(1)))
        except Exception:
            continue
        if raw is None:
            continue
        try:
            cmaps[xref] = parse_cmap(raw.decode("latin1", "replace"))
        except Exception:
            continue
    return cmaps


def decode_char(cmap, cid, gid):
    """Decode one glyph to a unicode char."""
    if cid in cmap:
        return chr(cmap[cid])
    if gid in cmap:
        return chr(cmap[gid])
    # ASCII / Latin-1 identity fallback
    if 32 <= cid <= 0x1FF or 0x3000 <= cid <= 0x33FF:
        ch = chr(cid)
        if ch.isprintable() or ch in " \t":
            return ch
    return "\ufffd"


def extract_cid_text(page, cmaps):
    """Reconstruct page text from texttrace CIDs, grouping chars by baseline."""
    lines_out = []
    try:
        traces = page.get_texttrace()
    except Exception:
        return page.get_text()
    # font name -> xref -> cmap
    name_to_cmap = {}
    for f in page.get_fonts(full=True):
        xref, ext, typ, bname, enc, *_ = f
        name_to_cmap[bname] = cmaps.get(xref, {})
    # Collect (y_baseline, x, char) with small per-char unions; group by y.
    import math
    groups = {}
    for line in traces:
        font = line.get("font")
        cmap = name_to_cmap.get(font, {}) if font else {}
        for tup in line.get("chars") or []:
            gid, cid, origin, bbox = tup[0], tup[1], tup[2], tup[3]
            y = origin[1]
            key = None
            for gk in groups:
                if abs(gk - y) <= 6.0:
                    key = gk
                    break
            if key is None:
                key = y
                groups[key] = []
            groups[key].append((origin[0], decode_char(cmap, cid, gid)))
    for y, chars in sorted(groups.items()):
        chars.sort(key=lambda c: c[0])
        text = "".join(c[1] for c in chars).rstrip()
        if text.strip():
            lines_out.append(text)
    return "\n".join(lines_out)


def extract(doc, force_cid=False, sort=True):
    """Extract whole doc text. Prefers standard extraction; falls back to CID
    rebuild when the result is mostly replacement chars."""
    pages = []
    cmaps = font_cmaps(doc)
    for pno in range(doc.page_count):
        page = doc[pno]
        if force_cid:
            pages.append(extract_cid_text(page, cmaps))
            continue
        t = page.get_text("text", sort=sort)
        if t.count("\ufffd") > max(3, 0.2 * len(t) / 3) and cmaps:
            t2 = extract_cid_text(page, cmaps)
            if t2.count("\ufffd") < t.count("\ufffd"):
                t = t2
        pages.append(t)
    return "\n".join(pages)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("out")
    ap.add_argument("--force-cid", action="store_true")
    args = ap.parse_args(argv)
    doc = pymupdf.open(args.pdf)
    text = extract(doc, force_cid=args.force_cid)
    # normalize non-breaking spaces (magazine typesetting uses them inside words)
    text = text.replace("\xa0", " ")
    # drop soft hyphens (hyphenation artifacts inside words)
    text = text.replace("\xad", "")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print("pages:", doc.page_count, "chars:", len(text))


if __name__ == "__main__":
    main()