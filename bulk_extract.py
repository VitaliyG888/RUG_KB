#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bulk-extract text from all PDFs under a directory into an output dir.

Usage:
    python bulk_extract.py <src_dir> <out_dir> [--force-cid]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pymupdf
from pdf_extract import extract


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--force-cid", action="store_true")
    ap.add_argument("--min-mtime", type=float, default=None,
                    help="skip files older than this mtime (resume support)")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    pdfs = []
    for root, _dirs, files in os.walk(args.src_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    print("found", len(pdfs), "pdfs under", args.src_dir)

    done = 0
    for i, path in enumerate(pdfs):
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, base + ".txt")
        if os.path.exists(out) and os.path.getsize(out) > 2000:
            done += 1
            continue
        try:
            doc = pymupdf.open(path)
            text = extract(doc, force_cid=args.force_cid).replace("\xa0", " ").replace("\xad", "")
            doc.close()
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[{i+1}/{len(pdfs)}] OK {base} ({len(text)} chars)")
        except Exception as e:
            print(f"[{i+1}/{len(pdfs)}] FAIL {base}: {e}")
        done += 1
    print("DONE, processed:", done)


if __name__ == "__main__":
    main()